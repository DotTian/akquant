import os

import numpy as np
import pandas as pd
from typing import Any, cast
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from akq_module_tusharedatamanager import TushareStockDataManager
import warnings
warnings.filterwarnings('ignore')

# 假设使用 AKQuant 框架（也可以独立运行）
try:
    from akquant.akquant import Bar
    from akquant.backtest import run_backtest
    from akquant.ml import SklearnAdapter
    from akquant.strategy import Strategy
except ImportError:
    print("AKQuant not available, running in standalone mode")
    # 定义简单的基类用于独立运行
    class Strategy:
        def __init__(self):
            self.model = None
            self.history = []
        def buy(self, symbol, size): pass
        def sell(self, symbol, size): pass
        def get_history_df(self, n): return pd.DataFrame()
        def set_history_depth(self, n): pass


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
        self.model = RandomForestClassifier(
            n_estimators=100,      # 100棵决策树
            max_depth=5,           # 限制深度防过拟合
            min_samples_split=20,  # 分裂所需最小样本数
            random_state=42,
            n_jobs=-1              # 并行计算
        )
        
        # 2. 策略参数
        self.retrain_frequency = retrain_frequency
        self.bar_count = 0
        self.last_train_bar = -retrain_frequency
        self.min_history_for_train = 100  # 最少需要100根bar才能训练
        
        # 3. 用于滚动训练的历史数据存储
        self.history_data = []
        
        # 4. 因子参数
        self.momentum_periods = [5, 10, 20]     # 多周期趋势动量
        self.rsi_period = 14
        self.atr_period = 14
        self.volume_period = 20
        
        # 5. 信号阈值（概率大于阈值才交易，减少噪音）
        self.buy_threshold = 0.65
        self.sell_threshold = 0.35
        
        # 6. 日志变量
        self._last_prediction = None
        
        # 确保足够的历史数据
        self.set_history_depth(self.min_history_for_train + 50)
        
        print(f"FourFactorMLStrategy initialized")
        print(f"  - Retrain frequency: {retrain_frequency} bars")
        print(f"  - Buy threshold: {self.buy_threshold}, Sell threshold: {self.sell_threshold}")
    
    
    def compute_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算四个核心因子 + 扩展特征
        
        Args:
            df: 包含 open, high, low, close, volume 的DataFrame
            
        Returns:
            包含所有特征的DataFrame
        """
        X = pd.DataFrame(index=df.index)
        
        # ========== 因子1: 趋势动量 ==========
        # 多周期收益率
        for period in self.momentum_periods:
            X[f'momentum_{period}'] = df['close'].pct_change(period)
        
        # 价格相对于均线的位置（乖离率）
        for period in self.momentum_periods:
            ma = df['close'].rolling(period).mean()
            X[f'bias_{period}'] = (df['close'] - ma) / ma
        
        # ========== 因子2: RSI ==========
        # 计算RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        X['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI的极值信号（超买超卖）
        X['rsi_oversold'] = (X['rsi'] < 30).astype(int)   # 超卖区域
        X['rsi_overbought'] = (X['rsi'] > 70).astype(int) # 超买区域
        
        # ========== 因子3: ATR波动率 ==========
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        X['atr'] = true_range.rolling(window=self.atr_period).mean()
        
        # 波动率变化率
        X['atr_pct_change'] = X['atr'].pct_change()
        
        # 标准化波动率（相对于过去均值）
        atr_ma = X['atr'].rolling(50).mean()
        X['atr_ratio'] = X['atr'] / atr_ma
        
        # ========== 因子4: 成交量 ==========
        X['volume'] = df['volume']
        X['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(self.volume_period).mean()
        X['volume_pct_change'] = df['volume'].pct_change()
        
        # 量价关系
        X['price_volume_corr'] = df['close'].pct_change().rolling(10).corr(df['volume'].pct_change())
        
        # ========== 额外增强特征 ==========
        # 当日涨跌幅
        X['daily_return'] = df['close'].pct_change()
        
        # 振幅
        X['amplitude'] = (df['high'] - df['low']) / df['close']
        
        # 是否收涨
        X['is_up'] = (df['close'] > df['open']).astype(int)
        
        # 连续上涨/下跌天数
        X['consecutive_up'] = (X['daily_return'] > 0).astype(int).groupby(
            (X['daily_return'] <= 0).astype(int).cumsum()
        ).cumsum()
        X['consecutive_down'] = (X['daily_return'] < 0).astype(int).groupby(
            (X['daily_return'] >= 0).astype(int).cumsum()
        ).cumsum()
        
        return X
    
    
    def prepare_data(self, df: pd.DataFrame) -> tuple:
        """
        准备训练数据：计算特征和标签
        
        Args:
            df: 原始行情数据
            
        Returns:
            (X, y) 特征矩阵和标签
        """
        # 计算因子
        X = self.compute_factors(df)
        
        # 计算标签：预测下一日涨跌
        # y = 1 表示下一日上涨，0 表示下跌
        future_return = df['close'].pct_change().shift(-1)
        y = (future_return > 0).astype(int)
        
        # 对齐数据：删除NaN行
        valid_idx = X.dropna().index.intersection(y.dropna().index)
        X_clean = X.loc[valid_idx]
        y_clean = y.loc[valid_idx]
        
        return X_clean, y_clean
    
    
    def train_model(self, df: pd.DataFrame) -> bool:
        """
        训练模型
        
        Args:
            df: 历史行情数据
            
        Returns:
            bool: 训练是否成功
        """
        if len(df) < self.min_history_for_train:
            print(f"  训练数据不足: {len(df)} < {self.min_history_for_train}")
            return False
        
        try:
            # 准备数据
            X, y = self.prepare_data(df)
            
            if len(X) < 50:
                print(f"  有效样本不足: {len(X)}")
                return False
            
            # 训练模型
            self.model.fit(X, y)
            
            # 计算训练集准确率
            train_pred = self.model.predict(X)
            accuracy = accuracy_score(y, train_pred)
            
            print(f"  ✓ 模型训练完成: 样本数={len(X)}, 训练准确率={accuracy:.2%}")
            
            # 可选：打印特征重要性
            if hasattr(self.model, 'feature_importances_'):
                importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                print(f"  特征重要性 Top 5:")
                for _, row in importance.head(5).iterrows():
                    print(f"    - {row['feature']}: {row['importance']:.4f}")
            
            return True
            
        except Exception as e:
            print(f"  ✗ 模型训练失败: {e}")
            return False
    
    
    def predict_next(self, df: pd.DataFrame) -> float:
        """
        预测下一日涨跌概率
        
        Args:
            df: 包含历史数据的DataFrame
            
        Returns:
            float: 上涨概率 (0-1)
        """
        if self.model is None:
            return 0.5
        
        try:
            # 计算最新一行的特征
            X = self.compute_factors(df)
            X_latest = X.iloc[-1:].fillna(0)
            
            # 预测概率
            prob = self.model.predict_proba(X_latest)[0, 1]  # 上涨的概率
            
            return prob
            
        except Exception as e:
            print(f"  预测失败: {e}")
            return 0.5
    
    
    def on_bar(self, bar: Any) -> None:
        """
        处理每个Bar的事件
        """
        self.bar_count += 1
        
        # 获取历史数据
        history_df = self.get_history_df(self.min_history_for_train + 100)
        
        if len(history_df) < self.min_history_for_train:
            return
        
        # 定期重新训练模型
        need_retrain = (self.bar_count - self.last_train_bar) >= self.retrain_frequency
        
        if need_retrain:
            print(f"\n[Bar {self.bar_count}] 重新训练模型...")
            if self.train_model(history_df):
                self.last_train_bar = self.bar_count
        
        # 检查模型是否已训练
        if not hasattr(self.model, 'predict_proba'):
            return
        
        # 预测
        prob = self.predict_next(history_df)
        
        # 每20个bar打印一次预测（避免刷屏）
        if self.bar_count % 20 == 0:
            print(f"[Bar {self.bar_count}] 上涨概率: {prob:.2%}")
        
        # 交易逻辑
        symbol = bar.symbol if hasattr(bar, 'symbol') else 'TEST'
        
        if prob > self.buy_threshold:
            # 上涨概率高 -> 买入
            self.buy(symbol, 100)
            if self._last_prediction != 'buy':
                print(f"  >>> 买入信号: 概率={prob:.2%} > {self.buy_threshold:.0%}")
                self._last_prediction = 'buy'
                
        elif prob < self.sell_threshold:
            # 上涨概率低 -> 卖出
            self.sell(symbol, 100)
            if self._last_prediction != 'sell':
                print(f"  >>> 卖出信号: 概率={prob:.2%} < {self.sell_threshold:.0%}")
                self._last_prediction = 'sell'

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
    
    # # 1. 生成模拟数据
    # np.random.seed(42)
    # n_days = 2000
    # dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    # # 生成随机游走价格
    # returns = np.random.randn(n_days) * 0.02
    # price = 100 * np.exp(np.cumsum(returns))
    
    # # 添加一些趋势和波动特征，使数据更真实
    # trend = np.linspace(0, 0.3, n_days)  # 长期上涨趋势
    # price = price * (1 + trend)
    
    # # 生成OHLC数据
    # df = pd.DataFrame({
    #     'date': dates,
    #     'open': price * (1 + np.random.randn(n_days) * 0.005),
    #     'high': price * (1 + np.abs(np.random.randn(n_days) * 0.01)),
    #     'low': price * (1 - np.abs(np.random.randn(n_days) * 0.01)),
    #     'close': price,
    #     'volume': np.random.randint(1000, 10000, n_days),
    #     'symbol': 'TEST'
    # })

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

import akshare as ak



if __name__ == "__main__":
    # 运行独立测试
    results = standalone_test()