"""
科创超跌价值回归策略（独立实现）

核心约束：
1. 股票池来源：StockSelector.select_star_value_reversion（月度）。
2. 买入信号：BOLL 或 MACD 任一买入信号触发即可买入。
3. 卖出边界：仅风险卖出（止损/动态止盈/ST），信号卖出只提示不执行。
4. 仓位：最多 5 只，每只 20%，不做加减仓。
5. 出池不卖出：月度股票池仅用于新开仓门控。
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from akquant import Strategy, run_backtest

from akq_module_stock_selector import StockSelector
from akq_module_tusharedatamanager import TushareStockDataManager


# 科创选股参数覆盖：默认保持当前设定；按需在这里统一调整。
STAR_VALUE_REVERSION_FILTER_PARAMS: dict[str, object] = {
    'enable_tech_filter': False,
}


class StarValueReversionStrategy(Strategy):
    """科创超跌价值回归策略。"""

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
        boll_period: int = 20,
        boll_std: float = 2.0,
        boll_rebound_lookback: int = 3,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
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

        self.boll_period = int(boll_period)
        self.boll_std = float(boll_std)
        self.boll_rebound_lookback = int(boll_rebound_lookback)
        self.macd_fast = int(macd_fast)
        self.macd_slow = int(macd_slow)
        self.macd_signal = int(macd_signal)

        self.entry_price: dict[str, float] = {}
        self.peak_pnl: dict[str, float] = {}

        history_need = max(
            self.boll_period + self.boll_rebound_lookback + 2,
            self.macd_slow + self.macd_signal + 10,
        )
        self.set_history_depth(history_need)

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

    def _boll_signals(self, symbol: str, bar) -> tuple[bool, bool]:
        count = self.boll_period + self.boll_rebound_lookback + 2
        closes_hist = self.get_history(count=count, symbol=symbol, field='close')
        if closes_hist is None or len(closes_hist) < count:
            return False, False

        closes = pd.Series(list(closes_hist) + [float(bar.close)], dtype='float64')
        mid = closes.rolling(self.boll_period).mean()
        std = closes.rolling(self.boll_period).std()
        upper = mid + self.boll_std * std
        lower = mid - self.boll_std * std

        if pd.isna(upper.iloc[-1]) or pd.isna(lower.iloc[-1]):
            return False, False

        prev_close = float(closes.iloc[-2])
        curr_close = float(closes.iloc[-1])
        prev_upper = float(upper.iloc[-2])
        curr_upper = float(upper.iloc[-1])
        curr_lower = float(lower.iloc[-1])

        rebound_slice = slice(-(self.boll_rebound_lookback + 1), -1)
        recent_touch_lower = bool((closes.iloc[rebound_slice] <= lower.iloc[rebound_slice]).any())

        boll_buy = recent_touch_lower and curr_close > curr_lower
        boll_sell = prev_close >= prev_upper and curr_close < curr_upper
        return boll_buy, boll_sell

    def _macd_buy_sell(self, symbol: str, bar) -> tuple[bool, bool]:
        count = self.macd_slow + self.macd_signal + 8
        closes_hist = self.get_history(count=count, symbol=symbol, field='close')
        if closes_hist is None or len(closes_hist) < count:
            return False, False

        closes = np.asarray(list(closes_hist) + [float(bar.close)], dtype='float64')
        close_series = pd.Series(closes)

        ema_fast = close_series.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = close_series.ewm(span=self.macd_slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=self.macd_signal, adjust=False).mean()

        if len(dif) < 2 or len(dea) < 2:
            return False, False

        macd_buy = bool(dif.iloc[-2] <= dea.iloc[-2] and dif.iloc[-1] > dea.iloc[-1])
        macd_sell = bool(dif.iloc[-2] >= dea.iloc[-2] and dif.iloc[-1] < dea.iloc[-1])
        return macd_buy, macd_sell

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

        boll_buy, boll_sell = self._boll_signals(symbol, bar)
        macd_buy, macd_sell = self._macd_buy_sell(symbol, bar)

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

            if boll_sell or macd_sell:
                self.log(
                    f'[{symbol}] 卖出信号提示(不执行): '
                    f'boll_sell={boll_sell}, macd_sell={macd_sell}, day={trade_day}'
                )
            return

        allowed = self.monthly_universe.get(month_key, set())
        if symbol not in allowed:
            return

        if len(self._open_symbols()) >= self.max_positions:
            if boll_buy or macd_buy:
                self.log(f'[{symbol}] 买入信号被忽略(仓位已满): day={trade_day}')
            return

        if boll_buy or macd_buy:
            self.order_target_percent(self.position_weight, symbol)
            self.entry_price[symbol] = price
            self.peak_pnl[symbol] = 0.0
            self.log(
                f'[{symbol}] 买入: target={self.position_weight:.2%}, '
                f'boll_buy={boll_buy}, macd_buy={macd_buy}, day={trade_day}'
            )


def build_monthly_star_universe(
    token: str,
    start_date: str,
    end_date: str,
    data_dir: str = 'selector_data',
    preload: bool = True,
    preload_force: bool = False,
    use_cache: bool = True,
    enable_tech_filter: Optional[bool] = None,
    star_filter_params: Optional[dict[str, object]] = None,
) -> tuple[dict[str, set[str]], dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]]:
    """构建月度科创股票池。"""
    selector = StockSelector(
        token=token,
        data_dir=data_dir,
        request_interval=0.32,
    )

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    monthly_dates = list(pd.date_range(start=start_ts, end=end_ts, freq='MS'))

    # 首次统一预加载：把潜在候选标的日线一次性拉到 end_date，避免月度循环中反复补拉。
    if preload:
        selector.preload_star_value_reversion_data(
            start_date=start_date,
            end_date=end_date,
            force=preload_force,
            verbose=True,
        )

    effective_filter_params = dict(star_filter_params or {})
    if enable_tech_filter is not None:
        effective_filter_params['enable_tech_filter'] = bool(enable_tech_filter)

    cache_suffix = f'{start_date}_{end_date}'
    if effective_filter_params:
        cfg_text = json.dumps(effective_filter_params, sort_keys=True, ensure_ascii=True)
        cfg_key = hashlib.md5(cfg_text.encode('utf-8')).hexdigest()[:10]
        cache_suffix = f'{cache_suffix}_cfg_{cfg_key}'
    cache_dir = Path(data_dir) / 'monthly_star_universe_cache' / cache_suffix
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)

    universe: dict[str, set[str]] = {}
    st_windows: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}

    for dt in monthly_dates:
        td = dt.strftime('%Y%m%d')
        month_key = dt.strftime('%Y-%m')
        cache_file = cache_dir / f'{td}.csv'

        df = pd.DataFrame()
        if use_cache and cache_file.exists():
            try:
                df = pd.read_csv(cache_file, dtype={'symbol': 'string'})
            except Exception:
                df = pd.DataFrame()

        if df.empty:
            df = selector.select_star_value_reversion(
                trade_date=td,
                verbose=True,
                filter_params=effective_filter_params,
            )
            if use_cache:
                save_df = pd.DataFrame()
                if df is not None and not df.empty:
                    keep_cols = [c for c in ['symbol', 'name', 'industry', 'pe_ttm', 'peg'] if c in df.columns]
                    if keep_cols:
                        save_df = df[keep_cols].copy()
                save_df.to_csv(cache_file, index=False, encoding='utf-8')

        if df is None or df.empty:
            universe[month_key] = set()
            continue

        symbols = set(df['symbol'].astype(str).str.zfill(6).tolist())
        universe[month_key] = symbols

    # 生成按日 ST 快照区间：只要交易日落入 ST 区间，即触发即时卖出。
    all_symbols = sorted({s for vals in universe.values() for s in vals})
    for sym in all_symbols:
        st_windows[sym] = []
        ts_code = selector._to_ts_code(sym)
        try:
            selector._wait()
            nc = selector.pro.namechange(ts_code=ts_code)
        except Exception:
            continue
        if nc is None or nc.empty:
            continue

        work = nc.copy()
        if 'name' not in work.columns:
            continue
        work = work[work['name'].astype(str).str.contains(r'\*?ST', regex=True, na=False)].copy()
        if work.empty:
            continue

        # 兼容不同字段口径：优先 start_date/end_date，其次 ann_date。
        far_future = pd.Timestamp('2099-12-31')
        for _, row in work.iterrows():
            start_raw = row.get('start_date')
            end_raw = row.get('end_date')
            ann_raw = row.get('ann_date')

            start = pd.to_datetime(str(start_raw), errors='coerce') if start_raw is not None else pd.NaT
            if pd.isna(start):
                start = pd.to_datetime(str(ann_raw), errors='coerce') if ann_raw is not None else pd.NaT
            if pd.isna(start):
                continue

            end = pd.to_datetime(str(end_raw), errors='coerce') if end_raw is not None else pd.NaT
            if pd.isna(end):
                end = far_future

            if end < start:
                continue
            st_windows[sym].append((start.normalize(), end.normalize()))

        st_windows[sym].sort(key=lambda x: x[0])

    return universe, st_windows


def load_market_data(
    symbols: list[str],
    start_date: str,
    end_date: str,
    data_dir: str,
) -> dict[str, pd.DataFrame]:
    """批量加载行情：优先本地缓存，仅对缺失标的补拉 API。"""
    token = os.getenv('TUSHARE_TOKEN')
    if not token:
        raise RuntimeError('请先设置环境变量 TUSHARE_TOKEN')

    manager = TushareStockDataManager(
        token=token,
        data_dir=data_dir,
        request_interval=1.5,
    )
    # 1) 先走本地缓存，避免全量回测阶段触发不必要的 API 请求。
    raw = manager.get_multiple_stocks(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        force_update=False,
        adjust='qfq',
        delay_between=0.0,
        allow_api=False,
    )

    missing_symbols: list[str] = []
    for symbol, df in raw.items():
        if df is None or df.empty:
            missing_symbols.append(str(symbol))

    # 2) 仅补拉本地缺失部分，减少请求量与等待时间。
    if missing_symbols:
        print(f'本地缓存缺失 {len(missing_symbols)} 只，开始补拉 Tushare...')
        fetched = manager.get_multiple_stocks(
            symbols=missing_symbols,
            start_date=start_date,
            end_date=end_date,
            force_update=False,
            adjust='qfq',
            delay_between=0.5,
            allow_api=True,
        )
        raw.update(fetched)

    data: dict[str, pd.DataFrame] = {}
    for symbol, df in raw.items():
        if df is None or df.empty:
            continue
        data[str(symbol)] = df

    if not data:
        raise RuntimeError('没有加载到可用行情数据')
    return data


def load_benchmark_returns(
    start_date: str,
    end_date: str,
    data_dir: str,
    fallback_data: dict[str, pd.DataFrame],
) -> pd.Series:
    """加载科创综指基准收益率，失败时回退到策略样本收益率。"""
    token = os.getenv('TUSHARE_TOKEN')
    if not token:
        raise RuntimeError('请先设置环境变量 TUSHARE_TOKEN')

    benchmark_symbol = '000680.SH'  # 科创综指
    manager = TushareStockDataManager(
        token=token,
        data_dir=data_dir,
        request_interval=1.5,
    )

    try:
        benchmark_df = manager.get_stock_data(
            symbol=benchmark_symbol,
            start_date=start_date,
            end_date=end_date,
            force_update=False,
            adjust='qfq',
        )
    except Exception as exc:
        print(f'基准数据获取失败，使用策略样本收益率兜底: {exc}')
        benchmark_df = None

    if benchmark_df is not None and not benchmark_df.empty and 'close' in benchmark_df.columns:
        benchmark_df.index = pd.to_datetime(benchmark_df.index)
        return benchmark_df['close'].pct_change().fillna(0.0).rename(benchmark_symbol)

    first_symbol = sorted(fallback_data.keys())[0]
    fb = fallback_data[first_symbol].copy()
    fb.index = pd.to_datetime(fb.index)
    print(f'基准数据为空，回退使用样本标的 {first_symbol} 收益率作为基准。')
    return fb['close'].pct_change().fillna(0.0).rename('fallback_benchmark')


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

    strategy = StarValueReversionStrategy(
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
    report_path = report_dir / f'star_value_reversion_{ts}.html'

    result.report(
        filename=str(report_path),
        title='科创超跌价值回归策略报告',
        market_data=data,
        include_trade_kline=True,
        benchmark=benchmark_returns,
    )
    print(f'\n报告已保存至: {report_path}')


if __name__ == '__main__':
    main()
