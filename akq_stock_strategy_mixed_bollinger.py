"""
混合策略：基本面股票池 + 布林线与 ADX 共振择时（akquant 版本）

策略要点：
1. 股票池来源：StockSelector 基本面筛选结果。
2. 买入信号：近 N 日触下轨后回到下轨上方即可买入；若同时满足 ADX 趋势确认，则优先级更高。
3. 仓位控制：总计 10 个仓位，每仓 10%，单行业最多 3 只。
4. 同行业候选超限时，按 strength = boll_deviation * adx_value 由强到弱优先。
5. 风控退出：
   - 强制止损：单票亏损达到 -7% 立即平仓。
   - 动态止盈：浮盈达到 +10% 后，若从最高浮盈回撤 30% 则止盈。
6. 普通退出：布林卖出信号触发平仓。
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from akquant import Strategy, run_backtest

from akq_module_stock_selector import StockSelector
from akq_module_tusharedatamanager import TushareStockDataManager


class MixedBollingerStrategy(Strategy):
    """基本面股票池 + 布林与 ADX 共振策略。"""

    def __init__(
        self,
        symbols: list[str],
        industry_by_symbol: Optional[dict[str, str]] = None,
        weekly_universe: Optional[dict[str, set[str]]] = None,
        position_weight: float = 0.10,
        max_positions: int = 10,
        max_positions_per_industry: int = 3,
        boll_period: int = 20,
        boll_std: float = 2.0,
        boll_rebound_lookback: int = 3,
        adx_period: int = 14,
        adx_threshold: float = 15.0,
        weekly_gate_weeks: int = 2,
        stop_loss_pct: float = -0.07,
        trailing_start_pct: float = 0.10,
        trailing_drawdown_pct: float = 0.30,
    ) -> None:
        super().__init__()
        if not symbols:
            raise ValueError("symbols 不能为空")
        if position_weight <= 0 or position_weight > 1.0:
            raise ValueError("position_weight 必须在 (0, 1] 范围内")
        if max_positions < 1:
            raise ValueError("max_positions 必须 >= 1")
        if max_positions_per_industry < 1:
            raise ValueError("max_positions_per_industry 必须 >= 1")
        if boll_period < 2:
            raise ValueError("boll_period 必须 >= 2")
        if boll_rebound_lookback < 1:
            raise ValueError("boll_rebound_lookback 必须 >= 1")
        if adx_period < 2:
            raise ValueError("adx_period 必须 >= 2")
        if weekly_gate_weeks < 1:
            raise ValueError("weekly_gate_weeks 必须 >= 1")

        self.symbols = sorted({str(s).strip() for s in symbols if str(s).strip()})
        self.industry_by_symbol = industry_by_symbol or {}
        self.weekly_universe = weekly_universe
        self.position_weight = float(position_weight)
        self.max_positions = int(max_positions)
        self.max_positions_per_industry = int(max_positions_per_industry)

        self.boll_period = int(boll_period)
        self.boll_std = float(boll_std)
        self.boll_rebound_lookback = int(boll_rebound_lookback)
        self.adx_period = int(adx_period)
        self.adx_threshold = float(adx_threshold)
        self.weekly_gate_weeks = int(weekly_gate_weeks)

        self.stop_loss_pct = float(stop_loss_pct)
        self.trailing_start_pct = float(trailing_start_pct)
        self.trailing_drawdown_pct = float(trailing_drawdown_pct)

        self.entry_price: dict[str, float] = {}
        self.peak_pnl: dict[str, float] = {}
        self.entry_strength: dict[str, float] = {}

        history_need = max(
            self.boll_period + self.boll_rebound_lookback + 2,
            self.adx_period * 2 + 2,
        )
        self.set_history_depth(history_need)

    def on_start(self) -> None:
        for symbol in self.symbols:
            self.subscribe(symbol)

    def _get_industry(self, symbol: str) -> str:
        return str(self.industry_by_symbol.get(symbol, "UNKNOWN"))

    def _open_symbols(self) -> list[str]:
        return [s for s in self.symbols if self.get_position(s) > 0]

    def _open_symbols_in_industry(self, industry: str) -> list[str]:
        return [s for s in self._open_symbols() if self._get_industry(s) == industry]

    @staticmethod
    def _week_key_from_date(day: pd.Timestamp) -> str:
        monday = day - pd.Timedelta(days=day.weekday())
        return monday.strftime("%Y-%m-%d")

    def _get_bar_week_key(self, bar) -> str:
        ts = getattr(bar, "timestamp", None)
        if ts is None:
            return self._week_key_from_date(pd.Timestamp.today())
        day = pd.to_datetime(int(ts), unit="ns", utc=True).tz_convert("Asia/Shanghai").normalize()
        return self._week_key_from_date(day)

    def _get_allowed_symbols(self, week_key: str) -> Optional[set[str]]:
        if self.weekly_universe is None:
            return None

        allowed: set[str] = set()
        base_day = pd.to_datetime(week_key)
        for i in range(self.weekly_gate_weeks):
            wk = (base_day - pd.Timedelta(days=7 * i)).strftime("%Y-%m-%d")
            allowed.update(self.weekly_universe.get(wk, set()))
        return allowed

    def _calc_adx_parts(
        self,
        highs: pd.Series,
        lows: pd.Series,
        closes: pd.Series,
    ) -> tuple[float, float, float]:
        """计算 ADX、+DI、-DI 的最新值。"""
        period = self.adx_period

        up_move = highs.diff()
        down_move = -lows.diff()

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        tr = pd.concat(
            [
                highs - lows,
                (highs - closes.shift()).abs(),
                (lows - closes.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(period).mean()
        plus_di = 100.0 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
        minus_di = 100.0 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))

        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.rolling(period).mean()

        adx_v = float(adx.iloc[-1]) if pd.notna(adx.iloc[-1]) else float("nan")
        plus_v = float(plus_di.iloc[-1]) if pd.notna(plus_di.iloc[-1]) else float("nan")
        minus_v = float(minus_di.iloc[-1]) if pd.notna(minus_di.iloc[-1]) else float("nan")
        return adx_v, plus_v, minus_v

    def _signal_features(self, symbol: str, bar) -> Optional[dict[str, float | bool]]:
        """返回布林与 ADX 特征及共振强度。"""
        count = max(self.boll_period + 1, self.adx_period * 2 + 1)
        closes_hist = self.get_history(count=count, symbol=symbol, field="close")
        highs_hist = self.get_history(count=count, symbol=symbol, field="high")
        lows_hist = self.get_history(count=count, symbol=symbol, field="low")

        if (
            closes_hist is None
            or highs_hist is None
            or lows_hist is None
            or len(closes_hist) < count
            or len(highs_hist) < count
            or len(lows_hist) < count
        ):
            return None

        closes = pd.Series(list(closes_hist) + [float(bar.close)], dtype="float64")
        highs = pd.Series(list(highs_hist) + [float(bar.high)], dtype="float64")
        lows = pd.Series(list(lows_hist) + [float(bar.low)], dtype="float64")

        mid = closes.rolling(self.boll_period).mean()
        std = closes.rolling(self.boll_period).std()
        upper = mid + self.boll_std * std
        lower = mid - self.boll_std * std

        if pd.isna(upper.iloc[-1]) or pd.isna(lower.iloc[-1]):
            return None

        prev_close = float(closes.iloc[-2])
        curr_close = float(closes.iloc[-1])
        prev_upper = float(upper.iloc[-2])
        curr_upper = float(upper.iloc[-1])
        curr_lower = float(lower.iloc[-1])

        rebound_slice = slice(-(self.boll_rebound_lookback + 1), -1)
        recent_touch_lower = bool((closes.iloc[rebound_slice] <= lower.iloc[rebound_slice]).any())
        boll_buy = recent_touch_lower and curr_close > curr_lower
        boll_sell = prev_close >= prev_upper and curr_close < curr_upper

        adx_val, plus_di, minus_di = self._calc_adx_parts(highs, lows, closes)
        adx_ok = (
            np.isfinite(adx_val)
            and np.isfinite(plus_di)
            and np.isfinite(minus_di)
            and adx_val >= self.adx_threshold
            and plus_di > minus_di
        )

        boll_deviation = max(
            0.0,
            (curr_lower - prev_close) / max(abs(curr_lower), 1e-9),
        )
        strength = boll_deviation * max(adx_val, 0.0) if np.isfinite(adx_val) else 0.0

        return {
            "boll_buy": boll_buy,
            "boll_sell": boll_sell,
            "adx_ok": bool(adx_ok),
            "adx": float(adx_val) if np.isfinite(adx_val) else 0.0,
            "strength": float(strength),
            "resonance": bool(boll_buy and adx_ok),
        }

    def _reset_symbol_state(self, symbol: str) -> None:
        self.entry_price.pop(symbol, None)
        self.peak_pnl.pop(symbol, None)
        self.entry_strength.pop(symbol, None)

    def _close_symbol(self, symbol: str, reason: str, price: Optional[float] = None) -> None:
        self.close_position(symbol)
        if price is None:
            self.log(f"[{symbol}] {reason}")
        else:
            self.log(f"[{symbol}] {reason}: close={price:.2f}")
        self._reset_symbol_state(symbol)

    def _open_position(self, symbol: str, strength: float, price: float) -> None:
        self.order_target_percent(self.position_weight, symbol)
        self.entry_price[symbol] = float(price)
        self.peak_pnl[symbol] = 0.0
        self.entry_strength[symbol] = float(strength)
        self.log(
            f"[{symbol}] 共振买入: target={self.position_weight:.2%}, "
            f"strength={strength:.6f}, close={price:.2f}"
        )

    def _try_open_with_constraints(self, symbol: str, strength: float, price: float) -> None:
        if self.get_position(symbol) > 0:
            return

        open_symbols = self._open_symbols()
        if len(open_symbols) >= self.max_positions:
            return

        industry = self._get_industry(symbol)
        same_industry_open = self._open_symbols_in_industry(industry)

        if len(same_industry_open) < self.max_positions_per_industry:
            self._open_position(symbol, strength, price)
            return

        weakest_symbol = min(
            same_industry_open,
            key=lambda s: self.entry_strength.get(s, -1.0),
        )
        weakest_strength = self.entry_strength.get(weakest_symbol, -1.0)

        if strength <= weakest_strength:
            self.log(
                f"[{symbol}] 行业超限跳过: industry={industry}, "
                f"strength={strength:.6f} <= weakest={weakest_strength:.6f}"
            )
            return

        self._close_symbol(
            weakest_symbol,
            f"行业替换卖出(强度被超越 {weakest_strength:.6f}->{strength:.6f})",
        )
        self._open_position(symbol, strength, price)

    def on_bar(self, bar) -> None:
        symbol = str(bar.symbol)
        if symbol not in self.symbols:
            return

        week_key = self._get_bar_week_key(bar)
        allowed_symbols = self._get_allowed_symbols(week_key)

        features = self._signal_features(symbol, bar)
        if features is None:
            return

        price = float(bar.close)
        position = self.get_position(symbol)

        if position > 0:
            entry = float(self.entry_price.get(symbol, price))
            pnl = (price - entry) / entry if entry > 0 else 0.0

            peak = max(float(self.peak_pnl.get(symbol, pnl)), pnl)
            self.peak_pnl[symbol] = peak

            if pnl <= self.stop_loss_pct:
                self._close_symbol(symbol, f"强制止损({pnl:.2%})", price)
                return

            if peak >= self.trailing_start_pct and peak > 0:
                retrace = (peak - pnl) / peak
                if retrace >= self.trailing_drawdown_pct:
                    self._close_symbol(
                        symbol,
                        f"动态止盈(peak={peak:.2%}, now={pnl:.2%}, retrace={retrace:.2%})",
                        price,
                    )
                    return

            if bool(features["boll_sell"]):
                self._close_symbol(symbol, "布林卖出", price)
                return

            return

        if bool(features["boll_buy"]) and (allowed_symbols is None or symbol in allowed_symbols):
            # 共振信号在仓位冲突时优先：给强度一个固定加成。
            base_strength = float(features["strength"])
            priority_boost = 1.0 if bool(features.get("resonance", False)) else 0.0
            self._try_open_with_constraints(
                symbol=symbol,
                strength=base_strength + priority_boost,
                price=price,
            )

    def on_stop(self) -> None:
        open_symbols = self._open_symbols()
        self.log(
            f"策略结束: universe={len(self.symbols)}, open_positions={len(open_symbols)}"
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


def build_fundamental_universe(
    token: str,
    trade_date: str,
    data_dir: str = "selector_data",
) -> pd.DataFrame:
    """使用 StockSelector 构建基本面股票池。"""
    selector = StockSelector(
        token=token,
        data_dir=data_dir,
        request_interval=0.32,
    )
    selected = selector.select(trade_date=trade_date, verbose=True)
    if selected is None or selected.empty:
        raise RuntimeError("基本面选股结果为空")
    return selected


def build_weekly_universe(
    token: str,
    start_date: str,
    end_date: str,
    data_dir: str = "selector_data",
    preload: bool = True,
) -> tuple[dict[str, set[str]], dict[str, str]]:
    """按周构建基本面候选池：仅影响买入门控，不强制卖出。"""

    def _run_with_heartbeat(task_name: str, fn, *args, heartbeat_sec: int = 30, **kwargs):
        started = time.time()
        stop_event = threading.Event()

        def _heartbeat() -> None:
            while not stop_event.wait(heartbeat_sec):
                elapsed = time.time() - started
                print(f"   ... {task_name} 仍在运行，已耗时 {elapsed:.0f}s")

        t = threading.Thread(target=_heartbeat, daemon=True)
        t.start()
        try:
            return fn(*args, **kwargs)
        finally:
            stop_event.set()
            t.join(timeout=0.2)

    selector = StockSelector(
        token=token,
        data_dir=data_dir,
        request_interval=0.32,
    )

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    weekly_dates = list(pd.date_range(start=start_ts, end=end_ts, freq="W-MON"))
    if not weekly_dates or weekly_dates[0].normalize() != start_ts.normalize():
        weekly_dates.insert(0, start_ts.normalize())

    print("\n" + "=" * 70)
    print("📦 构建周度基本面股票池")
    print(f"区间: {start_date} -> {end_date}, 周期数: {len(weekly_dates)}")
    print("=" * 70)

    if preload:
        print("\n⏳ [预加载] 首次运行会较慢，正在预加载基础数据缓存...")
        _run_with_heartbeat(
            task_name="预加载缓存",
            fn=selector.preload_all_data,
            start_date=start_date,
            end_date=end_date,
        )

    universe_by_week: dict[str, set[str]] = {}
    industry_by_symbol: dict[str, str] = {}

    t_all = time.time()
    for idx, dt in enumerate(weekly_dates, start=1):
        t_week = time.time()
        td = dt.strftime("%Y%m%d")
        print(f"\n⏳ [周度选股 {idx}/{len(weekly_dates)}] {td} 开始...")
        # 首周打开详细日志，后续以摘要进度为主。
        df = _run_with_heartbeat(
            task_name=f"周度选股 {idx}/{len(weekly_dates)} ({td})",
            fn=selector.select,
            trade_date=td,
            verbose=(idx == 1),
        )
        week_key = MixedBollingerStrategy._week_key_from_date(dt.normalize())
        if df is None or df.empty:
            universe_by_week[week_key] = set()
            elapsed = time.time() - t_week
            print(f"✅ [周度选股 {idx}/{len(weekly_dates)}] {td} 完成: 0 只, 耗时 {elapsed:.1f}s")
            continue

        symbols = set(df["symbol"].astype(str).str.zfill(6).tolist())
        universe_by_week[week_key] = symbols
        for _, row in df[["symbol", "industry"]].iterrows():
            industry_by_symbol[str(row["symbol"]).zfill(6)] = str(row["industry"])

        elapsed = time.time() - t_week
        print(
            f"✅ [周度选股 {idx}/{len(weekly_dates)}] {td} 完成: "
            f"{len(symbols)} 只, 耗时 {elapsed:.1f}s"
        )

    if not universe_by_week:
        raise RuntimeError("周度基本面股票池为空")

    print(
        f"\n🎯 周度股票池构建完成: {len(universe_by_week)} 周, "
        f"累计标的 {len({s for v in universe_by_week.values() for s in v})} 只, "
        f"总耗时 {time.time() - t_all:.1f}s"
    )

    return universe_by_week, industry_by_symbol


def main() -> None:
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("请先设置环境变量 TUSHARE_TOKEN")

    start_date = "20220101"
    end_date = "20260727"
    data_dir = "tsdata"

    weekly_universe, industry_by_symbol = build_weekly_universe(
        token=token,
        start_date=start_date,
        end_date=end_date,
        preload=True,
    )

    symbols = sorted({s for syms in weekly_universe.values() for s in syms})
    if not symbols:
        raise RuntimeError("周度基本面股票池未产出可交易标的")

    data = load_market_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        data_dir=data_dir,
    )
    tradable_symbols = sorted(data.keys())
    if not tradable_symbols:
        raise RuntimeError("基本面股票池在回测区间内无可交易数据")

    strategy = MixedBollingerStrategy(
        symbols=tradable_symbols,
        industry_by_symbol=industry_by_symbol,
        weekly_universe=weekly_universe,
        position_weight=0.10,
        max_positions=10,
        max_positions_per_industry=3,
        boll_period=20,
        boll_std=2.0,
        boll_rebound_lookback=3,
        adx_period=14,
        adx_threshold=15.0,
        weekly_gate_weeks=2,
        stop_loss_pct=-0.07,
        trailing_start_pct=0.10,
        trailing_drawdown_pct=0.30,
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
    report_path = report_dir / f"mixed_bollinger_adx_{timestamp}.html"

    plot_symbol = tradable_symbols[0] if tradable_symbols else None
    result.report(
        filename=str(report_path),
        title="基本面 + 布林/ADX 共振策略",
        market_data=data,
        plot_symbol=plot_symbol,
        include_trade_kline=True,
        show=False,
    )
    print(f"\n报告已保存至: {report_path}")


if __name__ == "__main__":
    main()
