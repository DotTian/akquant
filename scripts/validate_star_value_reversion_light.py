#!/usr/bin/env python3
"""Lightweight validator for STAR value reversion strategy rules.

This script uses synthetic data and deterministic signal injection.
It is designed for fast rule validation without external data APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Tuple

import pandas as pd
from akquant import run_backtest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from akq_stock_strategy_star_value_reversion import StarValueReversionStrategy


SignalKey = Tuple[str, int]


@dataclass
class CaseResult:
    name: str
    passed: bool
    details: str


class DeterministicStarStrategy(StarValueReversionStrategy):
    """Replace indicator signals with deterministic injected signals."""

    def __init__(self, *args, signal_plan: Dict[SignalKey, dict], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._signal_plan = signal_plan
        self._bar_index: Dict[str, int] = {s: 0 for s in self.symbols}

    def _boll_signals(self, symbol: str, bar) -> tuple[bool, bool]:
        i = self._bar_index.get(symbol, 0)
        cfg = self._signal_plan.get((symbol, i), {})
        return bool(cfg.get('boll_buy', False)), bool(cfg.get('boll_sell', False))

    def _macd_buy_sell(self, symbol: str, bar) -> tuple[bool, bool]:
        i = self._bar_index.get(symbol, 0)
        cfg = self._signal_plan.get((symbol, i), {})
        self._bar_index[symbol] = i + 1
        return bool(cfg.get('macd_buy', False)), bool(cfg.get('macd_sell', False))


def make_price_df(symbol: str, closes: list[float], start: str = '2024-01-01') -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=len(closes), freq='D')
    return pd.DataFrame(
        {
            'open': closes,
            'high': [c * 1.01 for c in closes],
            'low': [c * 0.99 for c in closes],
            'close': closes,
            'volume': [1_000_000.0 for _ in closes],
            'symbol': [symbol for _ in closes],
        },
        index=idx,
    )


def run_case_position_cap() -> CaseResult:
    name = 'position_cap_5'
    symbols = ['688001', '688002', '688003', '688004', '688005', '688006']
    data = {s: make_price_df(s, [100.0, 100.0, 100.0]) for s in symbols}
    signal_plan: Dict[SignalKey, dict] = {(s, 0): {'boll_buy': True} for s in symbols}

    monthly_universe = {'2024-01': set(symbols)}
    strategy = DeterministicStarStrategy(
        symbols=symbols,
        monthly_universe=monthly_universe,
        st_windows={},
        signal_plan=signal_plan,
        max_positions=5,
        position_weight=0.20,
    )

    result = run_backtest(
        data=data,
        strategy=strategy,
        symbols=symbols,
        initial_cash=1_000_000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        show_progress=False,
        fill_policy={'price_basis': 'close', 'bar_offset': 0, 'temporal': 'same_cycle'},
    )

    pos = result.positions
    if pos.empty:
        return CaseResult(name, False, 'positions is empty')

    max_open = 0
    for _, row in pos.iterrows():
        max_open = max(max_open, int((row.fillna(0.0) > 0).sum()))

    return CaseResult(name, max_open <= 5, f'max_open={max_open}')


def run_case_st_snapshot_sell() -> CaseResult:
    name = 'st_snapshot_sell'
    symbol = '688010'
    data = {symbol: make_price_df(symbol, [100.0, 101.0, 102.0, 103.0, 104.0])}
    signal_plan: Dict[SignalKey, dict] = {(symbol, 0): {'boll_buy': True}}

    st_windows = {
        symbol: [
            (pd.Timestamp('2024-01-03'), pd.Timestamp('2024-01-04')),
        ]
    }
    strategy = DeterministicStarStrategy(
        symbols=[symbol],
        monthly_universe={'2024-01': {symbol}},
        st_windows=st_windows,
        signal_plan=signal_plan,
        max_positions=5,
        position_weight=0.20,
    )

    result = run_backtest(
        data=data,
        strategy=strategy,
        symbols=[symbol],
        initial_cash=1_000_000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        show_progress=False,
        fill_policy={'price_basis': 'close', 'bar_offset': 0, 'temporal': 'same_cycle'},
    )

    trades = result.trades_df
    if trades.empty:
        return CaseResult(name, False, 'expected close trade by ST window, got none')

    return CaseResult(name, True, f'closed_trades={len(trades)}')


def run_case_signal_sell_no_trade() -> CaseResult:
    name = 'signal_sell_no_trade'
    symbol = '688020'
    data = {symbol: make_price_df(symbol, [100.0, 101.0, 101.2, 101.1, 101.3, 101.4])}
    signal_plan: Dict[SignalKey, dict] = {
        (symbol, 0): {'macd_buy': True},
        (symbol, 1): {'boll_sell': True, 'macd_sell': True},
        (symbol, 2): {'boll_sell': True},
        (symbol, 3): {'macd_sell': True},
    }

    strategy = DeterministicStarStrategy(
        symbols=[symbol],
        monthly_universe={'2024-01': {symbol}},
        st_windows={},
        signal_plan=signal_plan,
        max_positions=5,
        position_weight=0.20,
        stop_loss_pct=-0.30,
        trailing_start_pct=0.50,
        trailing_drawdown_pct=0.50,
    )

    result = run_backtest(
        data=data,
        strategy=strategy,
        symbols=[symbol],
        initial_cash=1_000_000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        show_progress=False,
        fill_policy={'price_basis': 'close', 'bar_offset': 0, 'temporal': 'same_cycle'},
    )

    trades = result.trades_df
    # 仅信号卖出不应触发平仓，因此期望没有 closed trade。
    if not trades.empty:
        return CaseResult(name, False, f'unexpected closed trades={len(trades)}')

    pos = result.positions
    if pos.empty:
        return CaseResult(name, False, 'positions empty')
    final_pos = float(pos.iloc[-1].get(symbol, 0.0))
    return CaseResult(name, final_pos > 0, f'final_position={final_pos}')


def run_case_pool_ejection_no_sell() -> CaseResult:
    name = 'pool_ejection_no_sell'
    symbol = '688030'
    data = {symbol: make_price_df(symbol, [100.0, 100.2, 100.4, 100.5, 100.6, 100.8], start='2024-01-30')}
    signal_plan: Dict[SignalKey, dict] = {(symbol, 0): {'boll_buy': True}}

    # 仅 2024-01 在池，2024-02 出池；出池后不应自动卖出。
    monthly_universe = {'2024-01': {symbol}, '2024-02': set()}
    strategy = DeterministicStarStrategy(
        symbols=[symbol],
        monthly_universe=monthly_universe,
        st_windows={},
        signal_plan=signal_plan,
        max_positions=5,
        position_weight=0.20,
        stop_loss_pct=-0.30,
        trailing_start_pct=0.50,
        trailing_drawdown_pct=0.50,
    )

    result = run_backtest(
        data=data,
        strategy=strategy,
        symbols=[symbol],
        initial_cash=1_000_000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        show_progress=False,
        fill_policy={'price_basis': 'close', 'bar_offset': 0, 'temporal': 'same_cycle'},
    )

    trades = result.trades_df
    if not trades.empty:
        return CaseResult(name, False, f'unexpected closed trades={len(trades)}')

    pos = result.positions
    if pos.empty:
        return CaseResult(name, False, 'positions empty')
    final_pos = float(pos.iloc[-1].get(symbol, 0.0))
    return CaseResult(name, final_pos > 0, f'final_position={final_pos}')


def run_case_risk_stop_loss() -> CaseResult:
    name = 'risk_stop_loss'
    symbol = '688040'
    data = {symbol: make_price_df(symbol, [100.0, 99.5, 92.0, 91.5])}
    signal_plan: Dict[SignalKey, dict] = {(symbol, 0): {'macd_buy': True}}

    strategy = DeterministicStarStrategy(
        symbols=[symbol],
        monthly_universe={'2024-01': {symbol}},
        st_windows={},
        signal_plan=signal_plan,
        max_positions=5,
        position_weight=0.20,
        stop_loss_pct=-0.07,
    )

    result = run_backtest(
        data=data,
        strategy=strategy,
        symbols=[symbol],
        initial_cash=1_000_000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        show_progress=False,
        fill_policy={'price_basis': 'close', 'bar_offset': 0, 'temporal': 'same_cycle'},
    )

    trades = result.trades_df
    if trades.empty:
        return CaseResult(name, False, 'expected stop-loss close trade, got none')

    ret = float(trades.iloc[-1].get('return_pct', 0.0))
    return CaseResult(name, ret <= -6.5, f'return_pct={ret:.4f}')


def main() -> None:
    cases = [
        run_case_position_cap,
        run_case_st_snapshot_sell,
        run_case_signal_sell_no_trade,
        run_case_pool_ejection_no_sell,
        run_case_risk_stop_loss,
    ]

    results: list[CaseResult] = []
    for fn in cases:
        results.append(fn())

    print('\n=== validate_star_value_reversion_light ===')
    passed = 0
    for r in results:
        status = 'PASS' if r.passed else 'FAIL'
        print(f'[{status}] {r.name}: {r.details}')
        if r.passed:
            passed += 1

    print(f'\nSummary: {passed}/{len(results)} passed')
    if passed != len(results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
