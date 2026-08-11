"""
科创超跌价值回归策略（入池即买版，独立实现）

核心约束：
1. 股票池来源：复用 StockSelector.select_star_value_reversion（月度）。
2. 买入规则：不使用技术买点，当月入池且仓位未满即直接买入。
3. 卖出边界：仅风险卖出（止损/动态止盈/ST）。
4. 仓位：最多 5 只，每只 20%，不做加减仓。
5. 出池不卖出：月度股票池仅用于新开仓门控。
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from akquant import Strategy, run_backtest

from akq_stock_strategy_star_value_reversion import (
    STAR_VALUE_REVERSION_FILTER_PARAMS,
    build_monthly_star_universe,
    load_benchmark_returns,
    load_market_data,
)


class StarValueReversionDirectBuyStrategy(Strategy):
    """科创超跌价值回归策略（入池即买版）。"""

    def __init__(
        self,
        symbols: list[str],
        monthly_universe: Optional[dict[str, set[str]]] = None,
        st_windows: Optional[dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]] = None,
        position_weight: float = 0.20,
        max_positions: int = 5,
        stop_loss_pct: float = -0.07,
        trailing_start_pct: float = 0.10,
        trailing_drawdown_pct: float = 0.30,
    ) -> None:
        super().__init__()
        if not symbols:
            raise ValueError('symbols 不能为空')
        if position_weight <= 0 or position_weight > 1.0:
            raise ValueError('position_weight 必须在 (0, 1]')
        if max_positions < 1:
            raise ValueError('max_positions 必须 >= 1')

        self.symbols = sorted({str(s).strip() for s in symbols if str(s).strip()})
        self.monthly_universe = monthly_universe or {}
        self.st_windows = st_windows or {}

        self.position_weight = float(position_weight)
        self.max_positions = int(max_positions)
        self.stop_loss_pct = float(stop_loss_pct)
        self.trailing_start_pct = float(trailing_start_pct)
        self.trailing_drawdown_pct = float(trailing_drawdown_pct)

        self.entry_price: dict[str, float] = {}
        self.peak_pnl: dict[str, float] = {}

    def on_start(self) -> None:
        for symbol in self.symbols:
            self.subscribe(symbol)

    @staticmethod
    def _month_key_from_ts(ts_ns: Optional[int]) -> str:
        if ts_ns is None:
            return pd.Timestamp.today().strftime('%Y-%m')
        dt = pd.to_datetime(int(ts_ns), unit='ns', utc=True).tz_convert('Asia/Shanghai')
        return dt.strftime('%Y-%m')

    @staticmethod
    def _trade_day_from_ts(ts_ns: Optional[int]) -> str:
        if ts_ns is None:
            return pd.Timestamp.today().strftime('%Y-%m-%d')
        dt = pd.to_datetime(int(ts_ns), unit='ns', utc=True).tz_convert('Asia/Shanghai')
        return dt.strftime('%Y-%m-%d')

    def _open_symbols(self) -> list[str]:
        return [s for s in self.symbols if self.get_position(s) > 0]

    def _is_st_on_day(self, symbol: str, trade_day: str) -> bool:
        windows = self.st_windows.get(symbol, [])
        if not windows:
            return False
        day = pd.Timestamp(trade_day)
        for start, end in windows:
            if start <= day <= end:
                return True
        return False

    def _close_position_with_reason(self, symbol: str, reason: str, price: float) -> None:
        self.close_position(symbol)
        self.log(f'[{symbol}] {reason}: close={price:.2f}')
        self.entry_price.pop(symbol, None)
        self.peak_pnl.pop(symbol, None)

    def on_bar(self, bar) -> None:
        symbol = str(bar.symbol)
        if symbol not in self.symbols:
            return

        ts_ns = getattr(bar, 'timestamp', None)
        month_key = self._month_key_from_ts(ts_ns)
        trade_day = self._trade_day_from_ts(ts_ns)

        price = float(bar.close)
        position = self.get_position(symbol)

        if position > 0:
            if self._is_st_on_day(symbol, trade_day):
                self._close_position_with_reason(symbol, 'ST 风险卖出', price)
                return

            entry = float(self.entry_price.get(symbol, price))
            pnl = (price - entry) / entry if entry > 0 else 0.0
            peak = max(float(self.peak_pnl.get(symbol, pnl)), pnl)
            self.peak_pnl[symbol] = peak

            if pnl <= self.stop_loss_pct:
                self._close_position_with_reason(symbol, f'硬止损({pnl:.2%})', price)
                return

            if peak >= self.trailing_start_pct and peak > 0:
                retrace = (peak - pnl) / peak
                if retrace >= self.trailing_drawdown_pct:
                    self._close_position_with_reason(
                        symbol,
                        f'动态止盈(peak={peak:.2%}, now={pnl:.2%}, retrace={retrace:.2%})',
                        price,
                    )
                    return
            return

        allowed = self.monthly_universe.get(month_key, set())
        if symbol not in allowed:
            return

        if len(self._open_symbols()) >= self.max_positions:
            return

        self.order_target_percent(self.position_weight, symbol)
        self.entry_price[symbol] = price
        self.peak_pnl[symbol] = 0.0
        self.log(f'[{symbol}] 入池即买入: target={self.position_weight:.2%}, day={trade_day}')


def main() -> None:
    token = os.getenv('TUSHARE_TOKEN')
    if not token:
        raise RuntimeError('请先设置环境变量 TUSHARE_TOKEN')

    start_date = '20240101'
    end_date = '20260811'
    selector_data_dir = 'selector_data'
    market_data_dir = 'tsdata'

    monthly_universe, st_windows = build_monthly_star_universe(
        token=token,
        start_date=start_date,
        end_date=end_date,
        data_dir=selector_data_dir,
        preload=True,
        preload_force=False,
        use_cache=True,
        star_filter_params=STAR_VALUE_REVERSION_FILTER_PARAMS,
    )

    all_symbols = sorted({s for vals in monthly_universe.values() for s in vals})
    if not all_symbols:
        raise RuntimeError('月度股票池为空，无法回测')

    data = load_market_data(
        symbols=all_symbols,
        start_date=start_date,
        end_date=end_date,
        data_dir=market_data_dir,
    )
    benchmark_returns = load_benchmark_returns(
        start_date=start_date,
        end_date=end_date,
        data_dir=market_data_dir,
        fallback_data=data,
    )
    tradable_symbols = sorted(data.keys())

    strategy = StarValueReversionDirectBuyStrategy(
        symbols=tradable_symbols,
        monthly_universe=monthly_universe,
        st_windows=st_windows,
        position_weight=0.20,
        max_positions=5,
        stop_loss_pct=-0.07,
        trailing_start_pct=0.10,
        trailing_drawdown_pct=0.30,
    )

    result = run_backtest(
        strategy=strategy,
        data=data,
        symbols=tradable_symbols,
        initial_cash=1_000_000.0,
        commission_rate=0.0003,
        stamp_tax_rate=0.001,
        transfer_fee_rate=0.0,
        min_commission=5.0,
        t_plus_one=True,
        lot_size=100,
        timezone='Asia/Shanghai',
        fill_policy={
            'price_basis': 'close',
            'temporal': 'same_cycle',
        },
        show_progress=True,
    )

    print('\n=== 回测结果 ===')
    print(result.metrics_df)

    report_dir = Path('reports')
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = report_dir / f'star_value_reversion_direct_buy_{ts}.html'

    result.report(
        filename=str(report_path),
        title='科创超跌价值回归策略报告（入池即买）',
        market_data=data,
        include_trade_kline=True,
        benchmark=benchmark_returns,
    )
    print(f'\n报告已保存至: {report_path}')


if __name__ == '__main__':
    main()
