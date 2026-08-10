import akshare as ak
import akquant as aq
import pandas as pd
import numpy as np
from datetime import datetime
import os

from akq_module_tusharedatamanager import TushareStockDataManager    

class BollingerStrategy(aq.Strategy):
    """
    布林线策略
    逻辑：
    - 价格跌破下轨下方5%时，全仓买入
    - 价格突破上轨上方5%时，全仓卖出
    """
    
    def __init__(self, period=20, std_dev=2, buy_threshold=-0.05, sell_threshold=0.05):
        """
        参数说明：
        - period: 布林线周期（默认20）
        - std_dev: 标准差倍数（默认2）
        - buy_threshold: 买入阈值，价格低于下轨的百分比（默认-0.05即-5%）
        - sell_threshold: 卖出阈值，价格高于上轨的百分比（默认0.05即5%）
        """
        super().__init__()
        self.period = period
        self.std_dev = std_dev
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        
        # 计算所需的最少数据量
        self.min_bars = period + 1

        self.set_history_depth(period + 1)
        
        
    def calculate_bollinger(self, close_prices):
        """
        计算布林带
        返回: (上轨, 中轨, 下轨)
        """
        if len(close_prices) < self.period:
            return None, None, None
        
        # 计算中轨（移动平均）
        middle = close_prices[-self.period:].mean()
        
        # 计算标准差
        std = close_prices[-self.period:].std()
        
        # 计算上下轨
        upper = middle + self.std_dev * std
        lower = middle - self.std_dev * std
        
        return upper, middle, lower
    
    def on_bar(self,  bar):
        """每个bar触发"""
        # 获取历史收盘价
        closes = self.get_history(count=self.period + 1, symbol=self.symbol, field="close")
        
        
        if len(closes) < self.min_bars:
            return
        
        # 计算布林带
        upper, middle, lower = self.calculate_bollinger(closes)
        
        if upper is None or lower is None:
            return
        
        current_price = bar.close
        
        # 计算偏离百分比
        upper_deviation = (current_price - upper) / upper
        lower_deviation = (lower - current_price) / lower
        
        # 获取当前持仓
        current_position = self.get_position(bar.symbol)
        has_position = current_position > 0
        #available_cash = self.get_available_cash()
        available_cash = self.get_cash()  # 获取当前可用资金
        
        # 交易逻辑
        # 卖出条件：价格超出上轨sell_threshold以上
        if has_position and upper_deviation >= self.sell_threshold:
            self.close_position(bar.symbol)
            self.log(f"卖出信号: 价格={current_price:.2f}, 上轨={upper:.2f}, 超出={upper_deviation:.2%}")
        
        # 买入条件：价格低于下轨buy_threshold以下
        elif not has_position and lower_deviation >= abs(self.buy_threshold):
            if available_cash > 0:
                # 全仓买入
                shares = int(available_cash / current_price / 100) * 100  # 整手
                if shares > 0:
                    #self.buy(bar.symbol, shares)
                    self.order_target_percent(0.95, bar.symbol)
                    self.log(f"买入信号: 价格={current_price:.2f}, 下轨={lower:.2f}, 低于={lower_deviation:.2%}")


# ========== 运行回测 ==========
if __name__ == "__main__":
    # 1. 获取数据
    symbol = "300001"  # 替换为你想测试的股票代码
    start_date = "20220101" 
    end_date = "20260515"
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
    
    # 额外获取基准数据（例如沪深300）
    benchmark_symbol = "000300.SH"
    print(f"正在获取 {benchmark_symbol} 基准数据...")
    try:
        benchmark_df = manager.get_stock_data(
            symbol=benchmark_symbol,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as exc:
        print(f"基准数据获取失败，使用主数据收益率作为兜底基准: {exc}")
        benchmark_df = None

    if benchmark_df is not None and not benchmark_df.empty:
        benchmark_df.index = pd.to_datetime(benchmark_df.index)
        benchmark_returns = (
            benchmark_df["close"]
            .pct_change()
            .fillna(0.0)
            .rename(benchmark_symbol)
        )
    else:
        benchmark_returns = (
            df["close"]
            .pct_change()
            .fillna(0.0)
            .rename("fallback_benchmark")
        )
    
    print(f"数据获取成功，共 {len(df)} 条记录")
    
    # 2. 创建策略实例
    strategy = BollingerStrategy(
        period=20,
        std_dev=2,
        buy_threshold=-0.0001,
        sell_threshold=0.0001
    )
    
    # 3. 运行回测
    result = aq.run_backtest(
        strategy=strategy,
        data=df,

        initial_cash=1_000_000,      # 初始资金100万
        commission_rate=0.0003,      # 万三佣金
        stamp_tax_rate=0.001,        # 千一印花税（卖出收取）
        min_commission=5.0,          # 最低佣金5元
        t_plus_one=True,             # A股T+1制度
        lot_size=100,                # 最小交易单位100股

        fill_policy={
            "price_basis": "open",   # 使用开盘价
            "bar_offset": 1          # 下根bar成交
        },
        show_progress=True
    )
    
    # 4. 输出结果
    print("\n=== 回测结果 ===")
    print(result.metrics_df)
    
    # 5. 生成报告
    report_dir = "./reports"
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"{report_dir}/bollinger_strategy_{symbol}_{timestamp}.html"
    
    result.report(
        filename=report_path,
        title=f"布林线策略报告 ({symbol})",
        market_data=df,
        include_trade_kline=True,
        benchmark=benchmark_returns,
    )
    
    print(f"\n报告已保存至: {report_path}")