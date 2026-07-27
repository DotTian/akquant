"""
混合策略：核心持有 + 布林带波段（akquant 版本）

策略说明：
1. 核心仓位（core）：长期持有，不做频繁调仓。
2. 波段仓位（tactical）：基于布林带回归信号开平仓。
3. 信号规则（与原脚本一致）：
   - 买入信号：昨日收盘 <= 昨日下轨，且今日收盘 > 今日下轨
   - 卖出信号：昨日收盘 >= 昨日上轨，且今日收盘 < 今日上轨

默认配置：core=80%，tactical=20%。
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from akquant import Strategy, run_backtest

from akq_module_tusharedatamanager import TushareStockDataManager


class MixedBollingerStrategy(Strategy):
    """核心持有 + 布林带波段策略。"""

    def __init__(
        self,
        symbols: list[str],
        core_weight: float = 0.8,
        tactical_weight: float = 0.2,
        boll_period: int = 20,
        boll_std: float = 2.0,
    ) -> None:
        super().__init__()
        if not symbols:
            raise ValueError("symbols 不能为空")
        if core_weight < 0 or tactical_weight < 0:
            raise ValueError("core_weight 与 tactical_weight 必须 >= 0")
        if core_weight + tactical_weight > 1.0 + 1e-8:
            raise ValueError("core_weight + tactical_weight 不能超过 1.0")
        if boll_period < 2:
            raise ValueError("boll_period 必须 >= 2")

        self.symbols = sorted({str(s).strip() for s in symbols if str(s).strip()})
        self.core_weight = float(core_weight)
        self.tactical_weight = float(tactical_weight)
        self.boll_period = int(boll_period)
        self.boll_std = float(boll_std)

        self.per_symbol_core_weight = self.core_weight / float(len(self.symbols))
        self.per_symbol_tactical_weight = self.tactical_weight / float(len(self.symbols))

        self.core_initialized: set[str] = set()
        self.tactical_active: dict[str, bool] = {symbol: False for symbol in self.symbols}

        # 需要用到最近 boll_period 根历史 + 1 根昨日比较数据
        self.set_history_depth(self.boll_period + 2)

    def on_start(self) -> None:
        for symbol in self.symbols:
            self.subscribe(symbol)

    def _boll_signal(self, symbol: str, current_close: float) -> int:
        """
        计算布林带买卖信号。

        返回：
        1  -> 买入波段仓位
        -1 -> 卖出波段仓位
        0  -> 无信号
        """
        hist = self.get_history(count=self.boll_period + 1, symbol=symbol, field="close")
        if hist is None or len(hist) < self.boll_period + 1:
            return 0

        closes = pd.Series(list(hist) + [float(current_close)], dtype="float64")
        mid = closes.rolling(self.boll_period).mean()
        std = closes.rolling(self.boll_period).std()
        upper = mid + self.boll_std * std
        lower = mid - self.boll_std * std

        if pd.isna(upper.iloc[-1]) or pd.isna(lower.iloc[-1]):
            return 0

        prev_close = float(closes.iloc[-2])
        curr_close = float(closes.iloc[-1])
        prev_upper = float(upper.iloc[-2])
        prev_lower = float(lower.iloc[-2])
        curr_upper = float(upper.iloc[-1])
        curr_lower = float(lower.iloc[-1])

        if prev_close <= prev_lower and curr_close > curr_lower:
            return 1
        if prev_close >= prev_upper and curr_close < curr_upper:
            return -1
        return 0

    def on_bar(self, bar) -> None:
        symbol = str(bar.symbol)
        if symbol not in self.tactical_active:
            return

        # 首次看到该标的时建立核心仓位。
        if symbol not in self.core_initialized:
            self.order_target_percent(self.per_symbol_core_weight, symbol)
            self.core_initialized.add(symbol)
            self.log(
                f"[{symbol}] 初始化核心仓位: {self.per_symbol_core_weight:.2%}"
            )

        signal = self._boll_signal(symbol=symbol, current_close=float(bar.close))
        tactical_on = self.tactical_active[symbol]

        if signal == 1 and not tactical_on:
            target = self.per_symbol_core_weight + self.per_symbol_tactical_weight
            self.order_target_percent(target, symbol)
            self.tactical_active[symbol] = True
            self.log(
                f"[{symbol}] 波段买入: target={target:.2%}, close={bar.close:.2f}"
            )
        elif signal == -1 and tactical_on:
            target = self.per_symbol_core_weight
            self.order_target_percent(target, symbol)
            self.tactical_active[symbol] = False
            self.log(
                f"[{symbol}] 波段卖出: target={target:.2%}, close={bar.close:.2f}"
            )

    def on_stop(self) -> None:
        active_count = sum(1 for v in self.tactical_active.values() if v)
        self.log(
            f"策略结束: symbols={len(self.symbols)}, tactical_active={active_count}"
        )


def load_market_data(
    symbols: list[str],
    start_date: str,
    end_date: str,
    data_dir: str,
) -> dict[str, pd.DataFrame]:
    """从 Tushare 批量拉取/读取行情数据。"""
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("请先设置环境变量 TUSHARE_TOKEN")

    manager = TushareStockDataManager(
        token=token,
        data_dir=data_dir,
        request_interval=1.5,
    )
    raw = manager.get_multiple_stocks(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        force_update=False,
        adjust="qfq",
        delay_between=0.8,
    )

    data: dict[str, pd.DataFrame] = {}
    for symbol, df in raw.items():
        if df is None or df.empty:
            continue
        data[str(symbol)] = df

    if not data:
        raise RuntimeError("没有加载到可用行情数据")

    return data


def main() -> None:
    symbols = ["688131", "688690"]
    start_date = "20220101"
    end_date = "20260720"
    data_dir = "tsdata"

    core_weight = 0.8
    tactical_weight = 0.2

    data = load_market_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        data_dir=data_dir,
    )
    tradable_symbols = sorted(data.keys())

    strategy = MixedBollingerStrategy(
        symbols=tradable_symbols,
        core_weight=core_weight,
        tactical_weight=tactical_weight,
        boll_period=20,
        boll_std=2.0,
    )

    result = run_backtest(
        strategy=strategy,
        data=data,
        symbols=tradable_symbols,
        initial_cash=1_000_000.0,
        commission_rate=0.00025,
        stamp_tax_rate=0.001,
        min_commission=5.0,
        transfer_fee_rate=0.0,
        t_plus_one=True,
        lot_size=100,
        timezone="Asia/Shanghai",
        fill_policy={
            "price_basis": "open",
            "bar_offset": 1,
        },
        show_progress=True,
    )

    print("\n=== 回测结果 ===")
    print(result.metrics_df)

    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"mixed_bollinger_{core_weight:.0%}_{tactical_weight:.0%}_{timestamp}.html"

    plot_symbol = tradable_symbols[0] if tradable_symbols else None
    result.report(
        filename=str(report_path),
        title="混合策略（核心持有 + 布林带波段）",
        market_data=data,
        plot_symbol=plot_symbol,
        include_trade_kline=True,
        show=False,
    )
    print(f"\n报告已保存至: {report_path}")


if __name__ == "__main__":
    main()
