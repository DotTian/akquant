"""
日线 MACD + 趋势分类策略

交易逻辑：
1. 使用 TrendClassifier 综合判断当前处于上升 / 震荡 / 下降趋势
2. 上升趋势内，等待 MACD 快线上穿慢线（金叉）买入
3. 下降趋势内，MACD 线下穿（死叉）卖出持仓
4. 震荡趋势不交易
5. 止损：浮亏 5% 立即平仓
6. 止盈：浮盈超过 10% 后，收益回撤 30% 止盈（跟踪止盈）
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime as dt

from akquant import Strategy, run_backtest
from akq_module_trendclassifier import TrendClassifier
from akq_module_tusharedatamanager import TushareStockDataManager


class TrendMacdStrategy(Strategy):
    """
    MACD + 趋势分类策略
    """

    def __init__(self,
                 # MACD 参数
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 # 趋势分类器参数（使用默认值即可）
                 classifier_window: int = 60,   # 分类器需要的历史数据长度
                 # 交易参数
                 buy_ratio: float = 0.8,        # 每次买入仓位比例
                 stop_loss: float = -0.05,      # 止损幅度 -5%
                 take_profit_threshold: float = 0.10,   # 盈利超过10%后启动跟踪止盈
                 trailing_drawdown: float = 0.30        # 收益回撤30%止盈
                 ):
        super().__init__()
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.classifier_window = classifier_window
        self.buy_ratio = buy_ratio
        self.stop_loss = stop_loss
        self.take_profit_threshold = take_profit_threshold
        self.trailing_drawdown = trailing_drawdown

        # 存储状态
        self.buy_price: float = 0.0
        self.highest_price_since_entry: float = 0.0   # 持仓期间最高价（用于跟踪止盈）
        self.entry_bar_index: int = 0

        # 分类器实例
        self.classifier = TrendClassifier()

        # 关键：启用历史数据追踪，确保 get_history 可以调用
        self.set_history_depth(max(self.classifier_window, self.macd_slow + self.macd_signal + 5) + 20)  # 至少 80，设为 200 更安全

    def on_start(self):
        """策略启动时调用"""
        pass

    def _calculate_macd(self, closes: np.ndarray):
        """
        计算 MACD 指标
        返回 (macd_line, signal_line) 两个 numpy 数组，长度与 closes 相同
        """
        if len(closes) < self.macd_slow + 1:
            return None, None

        # 转换为 pandas Series 以便计算 EMA
        close_series = pd.Series(closes)
        ema_fast = close_series.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = close_series.ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()

        return macd_line.values, signal_line.values

    def on_bar(self, bar):
        """
        每根日线触发
        """
        symbol = bar.symbol
        current_price = bar.close
        current_position = self.get_position(symbol)

        # 获取足够的历史数据
        history_len = max(self.classifier_window, self.macd_slow + self.macd_signal + 5)
        closes = self.get_history(history_len, field="close")
        highs = self.get_history(history_len, field="high")
        lows = self.get_history(history_len, field="low")

        if closes is None or len(closes) < history_len:
            return

        # 1. 使用趋势分类器判断当前趋势
        close_series = pd.Series(closes)
        high_series = pd.Series(highs) if highs is not None else None
        low_series = pd.Series(lows) if lows is not None else None
        trend, confidence = self.classifier.classify(close_series, high_series, low_series)

        # 2. 计算 MACD
        macd_line, signal_line = self._calculate_macd(closes)
        if macd_line is None or len(macd_line) < 2:
            return

        # 取最后两天的 MACD 值判断交叉
        macd_today = macd_line[-1]
        macd_yesterday = macd_line[-2]
        signal_today = signal_line[-1]
        signal_yesterday = signal_line[-2]

        # 3. 处理持仓逻辑
        if current_position > 0:
            # 更新最高价（用于跟踪止盈）
            if current_price > self.highest_price_since_entry:
                self.highest_price_since_entry = current_price

            # 只有在有买入记录时才计算盈亏
            if self.buy_price > 0:
                loss_pct = (current_price - self.buy_price) / self.buy_price

                # ---- 止损 ----
                if loss_pct <= self.stop_loss:
                    self.close_position(symbol)
                    self.log(f"[止损] 价格:{current_price:.2f}, 买入价:{self.buy_price:.2f}, 亏损:{loss_pct*100:.2f}%")
                    self._reset_state()
                    return

                # ---- 跟踪止盈 ----
                if loss_pct >= self.take_profit_threshold:
                    # 从最高价回撤幅度
                    drawdown_pct = (self.highest_price_since_entry - current_price) / self.highest_price_since_entry
                    if drawdown_pct >= self.trailing_drawdown:
                        self.close_position(symbol)
                        self.log(f"[跟踪止盈] 价格:{current_price:.2f}, "
                                    f"最高:{self.highest_price_since_entry:.2f}, "
                                    f"回撤:{drawdown_pct*100:.2f}%, 收益:{loss_pct*100:.2f}%")
                        self._reset_state()
                        return

                # ---- 下降趋势内，MACD 死叉卖出 ----
                if trend == "downtrend" and macd_today < signal_today and macd_yesterday >= signal_yesterday:
                    self.close_position(symbol)
                    self.log(f"[趋势死叉卖出] 价格:{current_price:.2f}, MACD死叉, 收益:{loss_pct*100:.2f}%")
                    self._reset_state()
                    return
            else:
                # buy_price == 0 但还有持仓（异常情况），直接平仓
                self.close_position(symbol)
                self.log(f"[异常平仓] 价格:{current_price:.2f}, 买入价记录丢失")
                self._reset_state()
                return

            # 其他情况继续持仓
            return

        # 4. 空仓逻辑
        # 只在上升趋势内交易
        if trend != "uptrend":
            # 可以 debug 记录，但数据量大会刷屏，暂时不记
            return

        # MACD 金叉买入
        if macd_today > signal_today and macd_yesterday <= signal_yesterday:
            # 计算买入股数（按仓位比例）
            self.order_target_percent(self.buy_ratio, symbol, price=current_price)
            self.buy_price = current_price
            self.highest_price_since_entry = current_price
            self.entry_bar_index = len(closes)
            self.log(f"[买入] 价格:{current_price:.2f}, 趋势:{trend}, MACD金叉, 置信度:{confidence:.0%}")

    def _reset_state(self):
        """重置持仓相关状态"""
        self.buy_price = 0.0
        self.highest_price_since_entry = 0.0

    def on_stop(self):
        """策略结束"""
        self.log("策略运行结束")


# ==================== 本地测试 ====================
if __name__ == "__main__":
    # 1. 获取数据
    symbol = "300001"  # 测试股票代码
    start_date = "20100101"
    end_date = "20260616"
    DATA_DIR = "tsdata"

    print(f"获取 {symbol} 数据...")
    mytoken = os.getenv('TUSHARE_TOKEN')
    manager = TushareStockDataManager(token=mytoken, data_dir=DATA_DIR, request_interval=1.5)
    df = manager.get_stock_data(symbol=symbol, start_date=start_date, end_date=end_date)
    df = df.sort_index()
    print(f"数据获取完成，共 {len(df)} 个交易日")

    # 2. 运行回测
    result = run_backtest(
        strategy=TrendMacdStrategy,
        data=df,
        symbols=[symbol],
        initial_cash=100000.0,
        commission_rate=0.0003,
        slippage=0.0002,
        t_plus_one=True,
    )

    # 3. 输出结果
    print("\n=== 回测结果 ===")
    print(result.metrics_df)

    # 4. 生成报告
    report_dir = "./reports"
    os.makedirs(report_dir, exist_ok=True)
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"{report_dir}/trend_macd_strategy_{symbol}_{timestamp}.html"
    result.report(
        filename=report_path,
        title=f"趋势+MACD策略报告 ({symbol})",
        market_data=df,
        include_trade_kline=True
    )
    print(f"\n报告已保存至: {report_path}")