# -*- coding: utf-8 -*-
# akq_stock_mainaccount_strategy.py
# 主账户策略示例，展示如何在主账户中使用策略进行交易

from akquant import Bar, Strategy

class stock_MainAccountStrategy(Strategy):
    """主账户策略示例."""

    def __init__(self) -> None:
        """初始化策略."""
        super().__init__()
        self.warmup_period = 20  # 设置 warmup_period，确保有足够的历史数据计算指标

    def on_bar(self, bar: Bar) -> None:
        """收到 Bar 事件的回调."""
        symbol = bar.symbol
        closes = self.get_history(count=20, symbol=symbol, field="close")
        if len(closes) < 20:
            return

        ma_short = closes[-5:].mean()
        ma_long = closes[-20:].mean()

        pos = self.get_position(symbol)

        if ma_short > ma_long and pos == 0:
            self.order_target_percent(0.95, symbol)
        elif ma_short < ma_long and pos > 0:
            self.close_position(symbol)