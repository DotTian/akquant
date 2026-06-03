import os

import numpy as np
import pandas as pd
from typing import Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from akq_module_tusharedatamanager import TushareStockDataManager
from datetime import datetime as dt  # 给 datetime 类起个别名
import warnings
warnings.filterwarnings('ignore')

# 假设使用 AKQuant 框架（也可以独立运行）
from akquant.akquant import Bar
from akquant.backtest import run_backtest
from akquant.ml import SklearnAdapter
from akquant.strategy import Strategy



class FourFactorMLStrategy(Strategy):
    """
    四因子机器学习策略
    因子：趋势动量 + RSI + ATR(波动率) + 成交量
    预测：下一日涨跌
    """
    
    def __init__(self, retrain_frequency: int = 20):
        """
        初始化策略
        
        Args:
            retrain_frequency: 重新训练的间隔（bar数）
        """
        super().__init__()

        # 1. 初始化模型（使用随机森林，对非线性关系适应好）
        # 包在 Adapter 里
        self.model = SklearnAdapter(
            RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=20,
                random_state=42,
                n_jobs=-1
            )
        )
        
        # 配置框架自动管理滚动验证
        self.model.set_validation(
            method="walk_forward",
            train_window=252,
            test_window=50,
            rolling_step=50,
            frequency="1d",
            verbose=True
        )

        # 2. 策略参数
        self.retrain_frequency = retrain_frequency
        self.bar_count = 0
        self.last_train_bar = -retrain_frequency
        self.min_history_for_train = 150  # 最少需要150根bar才能训练
        self.reserve_ratio = 0.1  # 预留10%资金，避免全仓导致的手续费不足问题
        
        # 3. 用于滚动训练的历史数据存储
        self.history_data = []
        
        # 4. 因子参数
        self.momentum_periods = [5, 10, 20]     # 多周期趋势动量
        self.rsi_period = 14
        self.atr_period = 14
        self.volume_period = 20
        
        # 5. 信号阈值（概率大于阈值才交易，减少噪音）
        self.buy_threshold = 0.55
        self.sell_threshold = 0.45
        
        # 6. 日志变量
        self._last_prediction = None
        
        # 确保足够的历史数据
        self.set_history_depth(self.min_history_for_train + 50)
        
        print(f"FourFactorMLStrategy initialized")
        print(f"  - Retrain frequency: {retrain_frequency} bars")
        print(f"  - Buy threshold: {self.buy_threshold}, Sell threshold: {self.sell_threshold}")
    
    def on_start(self):
        """策略启动时执行一次"""
        #self.subscribe(self.symbol)  # 订阅标的
        # warmup_period 已设置，无需再调用 set_history_depth
    
    def compute_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """你的完整因子计算代码"""
        X = pd.DataFrame(index=df.index)
        
        # 动量因子
        for period in self.momentum_periods:
            X[f'momentum_{period}'] = df['close'].pct_change(period)
            ma = df['close'].rolling(period).mean()
            X[f'bias_{period}'] = (df['close'] - ma) / ma
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()
        rs = gain / loss
        X['rsi'] = 100 - (100 / (1 + rs))
        X['rsi_oversold'] = (X['rsi'] < 30).astype(int)
        X['rsi_overbought'] = (X['rsi'] > 70).astype(int)
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        X['atr'] = true_range.rolling(self.atr_period).mean()
        X['atr_pct_change'] = X['atr'].pct_change()
        atr_ma = X['atr'].rolling(50).mean()
        X['atr_ratio'] = X['atr'] / atr_ma
        
        # 成交量
        X['volume'] = df['volume']
        X['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(self.volume_period).mean()
        X['volume_pct_change'] = df['volume'].pct_change()
        X['price_volume_corr'] = df['close'].pct_change().rolling(10).corr(df['volume'].pct_change())
        
        # 增强特征
        X['daily_return'] = df['close'].pct_change()
        X['amplitude'] = (df['high'] - df['low']) / df['close']
        X['is_up'] = (df['close'] > df['open']).astype(int)
        
        # 连续涨跌天数
        X['consecutive_up'] = (X['daily_return'] > 0).astype(int).groupby(
            (X['daily_return'] <= 0).astype(int).cumsum()
        ).cumsum()
        X['consecutive_down'] = (X['daily_return'] < 0).astype(int).groupby(
            (X['daily_return'] >= 0).astype(int).cumsum()
        ).cumsum()
        
        return X
    
    def prepare_features(
        self, df: pd.DataFrame, mode: str = "training"
    ) -> Tuple[Any, Any]:
        """
        准备特征和标签 - AKQuant 框架接口
        """
        # 计算特征
        X = self.compute_factors(df)
        
        # 调试：打印数据形状
        # if mode == "training":
        #     print(f"  [调试] 原始数据形状: {df.shape}")
        #     print(f"  [调试] X形状: {X.shape}, NaN数量: {X.isna().sum().sum()}")
        
        if mode == "inference":
            X_curr = X.iloc[-1:].fillna(0)
            return X_curr, None
        
        # ========== 训练模式 ==========
        # 计算标签
        future_return = df['close'].pct_change().shift(-1)
        y = (future_return > 0).astype(int)
        
        # 只保留有效特征行（至少有一列不是NaN）
        X_clean = X.dropna(how='all')  # 删除全部为NaN的行
        y_clean = y.loc[X_clean.index]  # 对齐
        
        # 再删除标签中的NaN
        valid_idx = y_clean.dropna().index
        X_clean = X_clean.loc[valid_idx]
        y_clean = y_clean.loc[valid_idx]
        
        # 如果还有NaN，用前向填充
        if X_clean.isna().any().any():
            X_clean = X_clean.ffill().fillna(0)  # 或 X_clean.fillna(0)
        
        # 调试输出
        # print(f"  [调试] 训练数据: X形状={X_clean.shape}, 正样本比例={y_clean.mean() if len(y_clean) > 0 else 0:.2%}")
        
        # 检查是否为空
        if len(X_clean) == 0:
            print(f"  [警告] 训练数据为空！请检查数据量是否足够")
            # 返回空的 DataFrame 和 Series（但模型会报错，所以最好检查）
        
        return X_clean, y_clean
    
    
    def on_bar(self, bar: Bar):
        """处理每个Bar"""
        self.bar_count += 1
        
        hist_df = self.get_history_df(count=200)
        if len(hist_df) < 70:  # 增加最小数据要求
            return
        
        if not self.is_model_ready():
            return
    
        # 直接尝试预测，如果模型未训练会抛异常
        try:
            X_curr, _ = self.prepare_features(hist_df, mode="inference")
            
            # 检查 X_curr 是否有效
            if X_curr is None or len(X_curr) == 0:
                return
            
            # 尝试预测
            prob = self.model.predict(X_curr)
            prob = prob[0] if isinstance(prob, (list, np.ndarray)) else prob
            
            # 打印预测结果（调试用，确认是否有输出）
            # if self.bar_count >= 1020:  # 从第一个训练窗口开始打印
            #     print(f"[Bar {self.bar_count}] 概率={prob:.2%}")
            
            # 交易逻辑
            symbol = bar.symbol
            current_pos = self.get_position(symbol)
        
            # 获取当前价格
            current_price = bar.close
            # 获取当前可用现金
            cash = self.get_cash()
            
            # 预留部分资金（避免手续费不足）
            available_cash = cash * (1 - self.reserve_ratio)
            # 计算最大可买股数（向下取整到100的整数倍）
            max_shares = int(available_cash / current_price / 100) * 100
            
            #全仓操作
            if prob > self.buy_threshold:
                if self.get_position(symbol) == 0:
                    self.buy(symbol=symbol, quantity=max_shares)
                    #self.buy_all(symbol=symbol)
                    print(f">>> 买入! Bar={self.bar_count}, 概率={prob:.2%}")
            elif prob < self.sell_threshold:
                pos = self.get_position(symbol)
                if pos > 0:
                    #self.close_position(symbol=symbol)
                    self.sell(symbol=symbol, quantity=pos)
                    print(f">>> 卖出! Bar={self.bar_count}, 概率={prob:.2%}")
                    
        except Exception as e:
            # 只在第一次出错时打印，避免刷屏
            if not hasattr(self, '_error_printed'):
                print(f"[首次错误] Bar {self.bar_count}: {e}")
                self._error_printed = True

# 获取真实A股数据的函数
def get_real_stock_data(symbol="000001", start="2020-01-01", end="2024-12-31"):
    """获取真实A股数据"""
     # 1. 获取数据
    # symbol = "688131"  # 替换为你想测试的股票代码
    # start_date = "20210101" 
    # end_date = "20260601"
    DATA_DIR = "tsdata"  # 数据存储目录
    
    print(f"正在获取 {symbol} 数据...")
    
    mytoken = os.getenv('TUSHARE_TOKEN')
    print('TUSHARE_TOKEN set:', bool(mytoken))

    manager = TushareStockDataManager(
        token= mytoken,  # 替换为你的实际 Token # type: ignore
        data_dir=DATA_DIR,
        request_interval=1.5  # 请求间隔 1.5 秒
    )
    df = manager.get_stock_data(symbol=symbol, start_date=start, end_date=end)
    
    
    print(f"数据获取成功，共 {len(df)} 条记录")
    
    # 确保数据按时间排序
    df = df.sort_index()
    
    print(f"数据获取完成，共{len(df)}个交易日")
    print(f"数据范围：{df.index[0]} 至 {df.index[-1]}")

    # # 重命名列
    # df.columns = ['date', 'open', 'close', 'high', 'low', 'volume', 
    #               'amount', 'amplitude', 'pct_change', 'change', 'turnover']
    # df['date'] = pd.to_datetime(df['date'])
    # df.set_index('date', inplace=True)
    return df

# # 获取真实数据
# df_real = get_real_stock_data("000001", "2020-01-01", "2024-12-31")
# print(f"真实数据: {len(df_real)} 天")

# ========== 独立运行测试（不依赖AKQuant） ==========
def standalone_test():
    """独立测试：不依赖AKQuant框架"""
    print("=" * 60)
    print("四因子机器学习策略 - 独立回测")
    print("=" * 60)

    # 1. 获取真实数据
    df = get_real_stock_data("688131", "2022-01-01", "2026-06-02")

    #df.set_index('date', inplace=True)
    
    print(f"数据量: {len(df)} 天")
    print(f"价格范围: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # 2. 滚动训练-预测回测
    print("\n" + "=" * 60)
    print("开始滚动回测...")
    print("=" * 60)
    
    train_window = 252    # 用1年数据训练
    test_window = 21      # 预测1个月
    buy_threshold = 0.75
    sell_threshold = 0.25
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    strategy = FourFactorMLStrategy()
    
    # 替换策略的模型
    strategy.model = model
    strategy.buy_threshold = buy_threshold
    strategy.sell_threshold = sell_threshold
    
    # 回测记录
    positions = []
    predictions = []
    dates_list = []
    
    for i in range(train_window, len(df) - 1, test_window):
        # 训练集
        train_df = df.iloc[i - train_window:i]
        # 测试集
        test_df = df.iloc[i:i + test_window]
        
        # 准备训练数据
        X_train, y_train = strategy.prepare_data(train_df)
        
        if len(X_train) < 50:
            continue
        
        # 训练
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        
        # 对测试集每个交易日预测
        for j in range(len(test_df) - 1):
            hist_df = df.iloc[:i + j + 1]
            X_pred = strategy.compute_factors(hist_df)
            
            if len(X_pred) > 0:
                X_latest = X_pred.iloc[-1:].fillna(0)
                prob = model.predict_proba(X_latest)[0, 1]
                
                date = test_df.index[j]
                predictions.append(prob)
                dates_list.append(date)
                
                # 模拟交易（简单记录信号）
                if prob > buy_threshold:
                    positions.append(1)  # 买入
                elif prob < sell_threshold:
                    positions.append(-1) # 卖出
                else:
                    positions.append(0)  # 持有
        
        print(f"窗口 {i}: 训练样本={len(X_train)}, 训练准确率={train_acc:.2%}")
    
    # 3. 结果分析
    print("\n" + "=" * 60)
    print("回测结果")
    print("=" * 60)
    
    # 创建结果DataFrame
    results = pd.DataFrame({
        'date': dates_list,
        'prediction': predictions,
        'signal': positions
    })
    
    print(f"预测次数: {len(predictions)}")
    print(f"买入信号次数: {sum(1 for p in positions if p == 1)}")
    print(f"卖出信号次数: {sum(1 for p in positions if p == -1)}")
    print(f"平均预测概率: {np.mean(predictions):.2%}")
    
    # 简单评估预测准确率
    # 获取实际收益率
    actual_returns = df['close'].pct_change().shift(-1)
    
    valid_results = []
    for i, date in enumerate(dates_list):
        if date in actual_returns.index:
            actual = actual_returns.loc[date]
            if not pd.isna(actual):
                valid_results.append({
                    'date': date,
                    'pred_prob': predictions[i],
                    'pred_direction': 1 if predictions[i] > 0.5 else 0,
                    'actual_direction': 1 if actual > 0 else 0,
                    'actual_return': actual
                })
    
    if valid_results:
        valid_df = pd.DataFrame(valid_results)
        accuracy = (valid_df['pred_direction'] == valid_df['actual_direction']).mean()
        print(f"\n预测方向准确率: {accuracy:.2%}")
        
        # 信号收益分析
        buy_returns = valid_df[valid_df['pred_prob'] > buy_threshold]['actual_return']
        sell_returns = valid_df[valid_df['pred_prob'] < sell_threshold]['actual_return']
        
        if len(buy_returns) > 0:
            print(f"买入信号平均收益: {buy_returns.mean():.2%}")
            print(f"买入信号胜率: {(buy_returns > 0).mean():.2%}")
        if len(sell_returns) > 0:
            print(f"卖出信号平均收益: {sell_returns.mean():.2%}")
    
    return results


if __name__ == "__main__":
    # 运行独立测试
    # results = standalone_test()
    # 1. 获取数据
    symbol = "688131"  # 替换为你想测试的股票代码
    start_date = "20210101" 
    end_date = "20260602"
    DATA_DIR = "tsdata"  # 数据存储目录
    
    print(f"正在获取 {symbol} 数据...")
    
    mytoken = os.getenv('TUSHARE_TOKEN')
    print('TUSHARE_TOKEN set:', bool(mytoken))

    manager = TushareStockDataManager(
        token= mytoken,  # 替换为你的实际 Token # type: ignore
        data_dir=DATA_DIR,
        request_interval=1.5  # 请求间隔 1.5 秒
    )
    df = manager.get_stock_data(symbol=symbol, start_date=start_date, end_date=end_date)
    
    
    print(f"数据获取成功，共 {len(df)} 条记录")
    
    # 确保数据按时间排序
    df = df.sort_index()
    
    print(f"数据获取完成，共{len(df)}个交易日")
    print(f"数据范围：{df.index[0]} 至 {df.index[-1]}")
    
    # 2. 运行回测
    result = run_backtest(
        strategy=FourFactorMLStrategy,
        data=df,
        symbols=[symbol],
        initial_cash=100000.0,      # 初始资金10万
        commission_rate=0.0003,      # 万三佣金
        slippage=0.0002,            # 万分之2滑点
        t_plus_one=True,             # A股T+1
        #debug=False                  # 调试模式（开启会打印更多日志）
    )

    # 4. 输出结果
    print("\n=== 回测结果 ===")
    print(result.metrics_df)
    
    # 5. 生成报告
    report_dir = "./reports"
    os.makedirs(report_dir, exist_ok=True)
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"{report_dir}/four_factor_ml_strategy_{symbol}_{timestamp}.html"
    
    result.report(
        filename=report_path,
        title=f"four_factor_ml策略报告 ({symbol})",
        market_data=df,
        include_trade_kline=True
    )
    
    print(f"\n报告已保存至: {report_path}")