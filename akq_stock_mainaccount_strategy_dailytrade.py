import os

import akshare as ak
import pandas as pd
import numpy as np
from akquant import Strategy
from akquant import run_backtest
from datetime import datetime as dt  # 给 datetime 类起个别名

from akq_module_tusharedatamanager import TushareStockDataManager

class DailyWaveStrategy(Strategy):
    """
    日线波段策略
    
    核心逻辑：
    1. 周线定方向：用周线MA20判断大趋势（周线 > MA20 为多头）
    2. 日线等买点：股价回调至20日或60日均线附近（偏差<3%）
    3. 日线确认信号：出现反包阳线或长下影线时入场
    4. 止盈止损：跌破10日线卖出，或到达前期高点区域止盈
    """
    
    def __init__(self, 
                 fast_ma=10,      # 短期均线（止盈/止损用）
                 medium_ma=20,    # 中期均线（买点1）
                 slow_ma=60,      # 长期均线（买点2）
                 weekly_ma=20,    # 周线均线（趋势判断）
                 deviation=0.03,  # 允许偏离均线的幅度
                 buy_ratio=0.2    # 每次买入比例（20%）
                 ):
        super().__init__()
        self.fast_ma = fast_ma
        self.medium_ma = medium_ma
        self.slow_ma = slow_ma
        self.weekly_ma = weekly_ma
        self.deviation = deviation
        self.buy_ratio = buy_ratio
        # 状态记录
        self.daily_closes = {}      # 存储日线收盘价序列
        self.weekly_closes = {}     # 存储周线收盘价序列
        self.buy_price = 0          # 记录买入价格
        self.entry_reason = ""      # 记录入场原因

        self.set_history_depth(max(fast_ma, medium_ma, slow_ma) + 10)  # 设置历史数据深度，确保足够计算均线和形态
    
    def on_start(self):
        """策略启动时，订阅数据"""
        # 通过上下文获取配置信息
        if hasattr(self.ctx, 'symbols') and self.ctx.symbols:
            self.trade_symbol = self.ctx.symbols[0]
            self.subscribe(self.trade_symbol)
        
    
    # def _get_history(self, n=100, field="close"):
    #     """获取历史数据（避免未来函数）"""
    #     hist = self.get_history(n, self.symbol)
    #     if hist is None or len(hist) == 0:
    #         return None# 方式A：如果只交易一个股票，取列表的第一个
    #     symbols = self.get_parameter('symbols', [])
    #     return hist[field].values
    
    def _is_bullish_trend_weekly(self):
        # 获取足够的日线数据（比如 120 天）
        closes = self.get_history(120,  field="close")
        opens = self.get_history(120,  field="open")
        highs = self.get_history(120,  field="high")
        lows = self.get_history(120,  field="low")
        
        if len(closes) < 120:
            return False
        
        # 构建 DataFrame，需要一个日期序列
        # 注意：get_history 返回的数组是按时间顺序从旧到新的
        import pandas as pd
        from datetime import datetime, timedelta
        
        # 方法A：如果有日期数据，用 bar.datetime
        # 方法B：根据回测频率，假设每个 bar 间隔 1 天
        dates = pd.date_range(end=datetime.now(), periods=len(closes), freq='D')
        
        df = pd.DataFrame({
            'close': closes,
            'open': opens,
            'high': highs,
            'low': lows
        }, index=dates)
        
        # 现在可以用了
        df['week'] = df.index.isocalendar().week
        weekly = df.groupby('week').agg({
            'close': 'last',
            'open': 'first',
            'high': 'max',
            'low': 'min'
        })
        
        # 计算周线均线
        if len(weekly) >= 5:
            ma5 = weekly['close'].rolling(5).mean().iloc[-1]
            ma10 = weekly['close'].rolling(10).mean().iloc[-1]
            return ma5 > ma10
        
        return False
    
    def _get_ma_value(self, period: int) -> float:
        """获取指定周期的均线值"""
        # get_history 返回的是 numpy 数组，直接就是收盘价序列
        hist = self.get_history(period + 10, field="close")
        
        # 检查数据是否足够
        if hist is None or len(hist) < period:
            return None # type: ignore
        
        # hist 直接就是收盘价数组，不需要再取 ['close']
        closes = hist[-period:]  # 取最近 period 个收盘价
        return float(np.mean(closes))
    
    def _is_near_ma(self, price: float, ma_value: float) -> bool:
        """判断价格是否在均线附近（允许偏差）"""
        if ma_value is None or ma_value == 0:
            return False
        deviation_ratio = abs(price - ma_value) / ma_value
        return deviation_ratio <= self.deviation
    
    def _is_reversal_bullish(self) -> bool:
        """判断是否出现反转看涨信号"""
        # 获取最近10天的收盘价
        closes = self.get_history(10, field="close")
        
        if closes is None or len(closes) < 10:
            return False
        
        # numpy 数组直接用下标索引
        today = closes[-1]      # 最新收盘价
        yesterday = closes[-2]  # 昨天收盘价
        day_before = closes[-3] # 前天收盘价
        
        # 反转看涨逻辑：连续下跌后今日收阳
        if day_before > yesterday and today > yesterday:
            return True
        
        return False
    
    def _should_exit(self, current_price: float) -> tuple:
        """
        判断卖出条件
        返回: (是否卖出, 卖出原因)
        """
        # 获取足够的历史数据（需要多个字段）
        closes = self.get_history(self.fast_ma + 30,  field="close")
        highs = self.get_history(self.fast_ma + 30, field="high")
        
        if closes is None or len(closes) < self.fast_ma:
            return False, ""
        
        # 条件1：跌破快线（MA10）
        ma_fast = self._get_ma_value(self.fast_ma)
        if ma_fast and current_price < ma_fast:
            return True, f"跌破MA{self.fast_ma}止盈/止损"
        
        # 条件2：到达前期高点阻力位（近20日最高点附近）
        # highs 是 numpy 数组，直接用 [-20:] 取最后20个，然后用 max()
        if len(highs) >= 20:
            recent_high = highs[-20:].max()  # numpy 数组直接用 .max() 方法
            if current_price >= recent_high * 0.98 and current_price > self.buy_price * 1.05:
                return True, f"触及前期高点{recent_high:.2f}止盈"
        
        # 条件3：亏损超过5%止损
        if self.buy_price > 0 and (current_price - self.buy_price) / self.buy_price < -0.05:
            return True, "亏损超过5%止损"
        
        return False, ""
    
    def on_bar(self, bar):
        """
        每根日线触发一次 - 核心交易逻辑
        """
        current_position = self.get_position(bar.symbol)
        current_price = bar.close
        
        # ========== 持仓时：检查卖出条件 ==========
        if current_position > 0:
            should_exit, exit_reason = self._should_exit(current_price)
            if should_exit:
                self.close_position(bar.symbol)
                self.log(f"[卖出] 价格:{current_price:.2f}, 原因:{exit_reason}, "
                        f"盈亏:{(current_price - self.buy_price)/self.buy_price*100:.2f}%")
            return  # 已处理卖出，本周期不再检查买入
        
        # ========== 空仓时：检查买入条件 ==========
        
        # 条件1：周线趋势必须为多头（大势过滤）
        if not self._is_bullish_trend_weekly():
            self.log("[过滤] 周线非多头趋势，跳过")
            return
        
        # 条件2：价格回调至MA20或MA60附近
        ma20 = self._get_ma_value(self.medium_ma)
        ma60 = self._get_ma_value(self.slow_ma)
        
        is_near_ma20 = self._is_near_ma(current_price, ma20) if ma20 else False
        is_near_ma60 = self._is_near_ma(current_price, ma60) if ma60 else False
        
        if not (is_near_ma20 or is_near_ma60):
            self.log(f"[过滤] 价格{current_price:.2f}未回调至支撑位, MA20:{ma20:.2f}, MA60:{ma60:.2f}")
            return
        
        # 条件3：出现反转K线形态确认
        if not self._is_reversal_bullish():
            self.log("[过滤] 未出现反转形态确认信号")
            return
        
        # 所有条件满足：买入buy_ratio比例的仓位
        self.order_target_percent(self.buy_ratio, bar.symbol, price=current_price)
        self.buy_price = current_price
        
        # 记录入场原因
        entry_ma = "MA20" if is_near_ma20 else "MA60"
        self.log(f"[买入] 价格:{current_price:.2f}, 均线支撑:{entry_ma}, "
                f"MA20:{ma20:.2f}, MA60:{ma60:.2f}")
    
    def on_stop(self):
        """策略结束，输出总结"""
        self.log("策略运行结束")


if __name__ == "__main__":
     # 1. 获取数据
    symbol = "688131"  # 替换为你想测试的股票代码
    start_date = "20210101" 
    end_date = "20260601"
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
        strategy=DailyWaveStrategy,
        data=df,
        symbols=[symbol],
        initial_cash=100000.0,      # 初始资金10万
        commission_rate=0.0003,      # 万三佣金
        slippage=0.0002,  # 万分之2滑点
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
    report_path = f"{report_dir}/dailytrade_strategy_{symbol}_{timestamp}.html"
    
    result.report(
        filename=report_path,
        title=f"dailytrade策略报告 ({symbol})",
        market_data=df,
        include_trade_kline=True
    )
    
    print(f"\n报告已保存至: {report_path}")