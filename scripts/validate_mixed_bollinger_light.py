#!/usr/bin/env python3
"""Lightweight validator for mixed bollinger + ADX strategy rules.

This script uses synthetic market data and deterministic signal injection,
so it runs fast and does not require Tushare/network dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Tuple

import pandas as pd
from akquant import run_backtest

# Ensure repository root is importable when running from scripts/.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from akq_stock_strategy_mixed_bollinger import MixedBollingerStrategy


SignalKey = Tuple[str, int]


@dataclass
class CaseResult:
    name: str
    passed: bool
    details: str


class DeterministicSignalStrategy(MixedBollingerStrategy):
    """Strategy wrapper that replaces indicator computation with fixed signals."""

    def __init__(self, *args, signal_plan: Dict[SignalKey, dict], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._signal_plan = signal_plan
        self._bar_index: Dict[str, int] = {s: 0 for s in self.symbols}

    def _signal_features(self, symbol: str, bar) -> dict[str, float | bool] | None:
        i = self._bar_index.get(symbol, 0)
        self._bar_index[symbol] = i + 1

        cfg = self._signal_plan.get((symbol, i), {})
        return {
            "boll_buy": bool(cfg.get("boll_buy", False)),
            "boll_sell": bool(cfg.get("boll_sell", False)),
            "adx_ok": bool(cfg.get("adx_ok", False)),
            "adx": float(cfg.get("adx", 0.0)),
            "strength": float(cfg.get("strength", 0.0)),
        }


def make_price_df(symbol: str, closes: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=len(closes), freq="D")
    df = pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [1_000_000.0 for _ in closes],
            "symbol": [symbol for _ in closes],
        },
        index=idx,
    )
    return df


def week_key(day: str) -> str:
    ts = pd.to_datetime(day)
    monday = ts - pd.Timedelta(days=ts.weekday())
    return monday.strftime("%Y-%m-%d")


def run_case_position_and_industry_constraints() -> CaseResult:
    name = "constraints_and_industry_cap"
    symbols = ["A1", "A2", "A3", "B1", "B2"]
    industry_by_symbol = {
        "A1": "半导体",
        "A2": "半导体",
        "A3": "半导体",
        "B1": "化工",
        "B2": "化工",
    }

    data = {s: make_price_df(s, [100.0, 100.0, 100.0]) for s in symbols}
    wk = week_key("2024-01-01")
    weekly_universe = {wk: set(symbols)}

    signal_plan: Dict[SignalKey, dict] = {
        ("A1", 0): {"boll_buy": True, "adx_ok": True, "strength": 0.10, "adx": 25.0},
        ("A2", 0): {"boll_buy": True, "adx_ok": True, "strength": 0.20, "adx": 25.0},
        ("A3", 0): {"boll_buy": True, "adx_ok": True, "strength": 0.90, "adx": 25.0},
        ("B1", 0): {"boll_buy": True, "adx_ok": True, "strength": 0.30, "adx": 25.0},
        ("B2", 0): {"boll_buy": True, "adx_ok": True, "strength": 0.40, "adx": 25.0},
    }

    strategy = DeterministicSignalStrategy(
        symbols=symbols,
        industry_by_symbol=industry_by_symbol,
        weekly_universe=weekly_universe,
        signal_plan=signal_plan,
        position_weight=0.10,
        max_positions=4,
        max_positions_per_industry=2,
        stop_loss_pct=-0.07,
        trailing_start_pct=0.10,
        trailing_drawdown_pct=0.30,
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
        fill_policy={"price_basis": "close", "bar_offset": 0, "temporal": "same_cycle"},
    )

    pos = result.positions
    if pos.empty:
        return CaseResult(name, False, "positions is empty")

    last = pos.iloc[-1].fillna(0.0)
    open_symbols = {s for s in pos.columns if float(last.get(s, 0.0)) > 0}

    max_concurrent = 0
    for _, row in pos.iterrows():
        concurrent = int((row.fillna(0.0) > 0).sum())
        max_concurrent = max(max_concurrent, concurrent)

    industry_counts: Dict[str, int] = {}
    for s in open_symbols:
        ind = industry_by_symbol.get(s, "UNKNOWN")
        industry_counts[ind] = industry_counts.get(ind, 0) + 1

    checks = [
        (max_concurrent <= 4, f"max_concurrent={max_concurrent}"),
        (all(v <= 2 for v in industry_counts.values()), f"industry_counts={industry_counts}"),
        ("A3" in open_symbols, f"open_symbols={sorted(open_symbols)}"),
        ("A1" not in open_symbols, f"open_symbols={sorted(open_symbols)}"),
    ]

    failed = [msg for ok, msg in checks if not ok]
    if failed:
        return CaseResult(name, False, " | ".join(failed))
    return CaseResult(name, True, f"open={sorted(open_symbols)}, industry_counts={industry_counts}")


def run_case_trailing_take_profit() -> CaseResult:
    name = "trailing_take_profit"
    symbol = "TP1"
    closes = [100.0, 102.0, 115.0, 108.0, 108.0]
    data = {symbol: make_price_df(symbol, closes)}

    wk = week_key("2024-01-01")
    signal_plan: Dict[SignalKey, dict] = {
        (symbol, 0): {"boll_buy": True, "adx_ok": True, "strength": 1.0, "adx": 30.0},
    }

    strategy = DeterministicSignalStrategy(
        symbols=[symbol],
        industry_by_symbol={symbol: "半导体"},
        weekly_universe={wk: {symbol}},
        signal_plan=signal_plan,
        position_weight=0.10,
        max_positions=10,
        max_positions_per_industry=3,
        stop_loss_pct=-0.07,
        trailing_start_pct=0.10,
        trailing_drawdown_pct=0.30,
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
        fill_policy={"price_basis": "close", "bar_offset": 0, "temporal": "same_cycle"},
    )

    trades = result.trades_df
    if trades.empty:
        return CaseResult(name, False, "no closed trade found")

    last_trade = trades.iloc[-1]
    ret = float(last_trade.get("return_pct", 0.0))
    # return_pct 在该引擎中使用“百分数”表示（8.0 表示 8%）。
    # 期望在盈利后回撤止盈，因此应为正收益且小于峰值 15%。
    if not (ret > 0 and ret < 15.0):
        return CaseResult(name, False, f"unexpected return_pct={ret:.4f}")
    return CaseResult(name, True, f"return_pct={ret:.4f}, trades={len(trades)}")


def run_case_hard_stop_loss() -> CaseResult:
    name = "hard_stop_loss"
    symbol = "SL1"
    closes = [100.0, 100.0, 93.0, 93.0]
    data = {symbol: make_price_df(symbol, closes)}

    wk = week_key("2024-01-01")
    signal_plan: Dict[SignalKey, dict] = {
        (symbol, 0): {"boll_buy": True, "adx_ok": True, "strength": 1.0, "adx": 30.0},
    }

    strategy = DeterministicSignalStrategy(
        symbols=[symbol],
        industry_by_symbol={symbol: "化工"},
        weekly_universe={wk: {symbol}},
        signal_plan=signal_plan,
        position_weight=0.10,
        max_positions=10,
        max_positions_per_industry=3,
        stop_loss_pct=-0.07,
        trailing_start_pct=0.10,
        trailing_drawdown_pct=0.30,
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
        fill_policy={"price_basis": "close", "bar_offset": 0, "temporal": "same_cycle"},
    )

    trades = result.trades_df
    if trades.empty:
        return CaseResult(name, False, "no closed trade found")

    last_trade = trades.iloc[-1]
    ret = float(last_trade.get("return_pct", 0.0))
    # 允许极小计算误差。
    if ret > -6.9:
        return CaseResult(name, False, f"stop loss not triggered as expected, return_pct={ret:.4f}")
    return CaseResult(name, True, f"return_pct={ret:.4f}, trades={len(trades)}")


def main() -> int:
    cases = [
        run_case_position_and_industry_constraints(),
        run_case_trailing_take_profit(),
        run_case_hard_stop_loss(),
    ]

    failed = [c for c in cases if not c.passed]

    print("=== Lightweight Validation Report ===")
    for c in cases:
        status = "PASS" if c.passed else "FAIL"
        print(f"[{status}] {c.name}: {c.details}")

    if failed:
        print(f"\nValidation failed: {len(failed)} case(s)")
        return 1

    print("\nAll validations passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
