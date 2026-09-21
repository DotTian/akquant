"""
A股高股息低风险长线策略（独立实现，低耦合版）

设计原则：
1. 不修改现有策略与选股模块行为。
2. 新增独立文件实现，复用已有数据管理与选股基础能力。
3. 月度更新股票池，季度调仓，非必要不卖出（长持导向）。

说明：
- 本版优先保证可运行与低耦合，作为 MVP。
- 若后续需要引入组合级实时回撤闸门（<=5%）的精细化实现，可能需要确认
  Strategy 运行时是否开放账户净值序列接口；如无接口，需要做轻量架构扩展。
"""

from __future__ import annotations

import json
import os
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tushare as ts
from akquant import Strategy, run_backtest

from akq_module_stock_selector import StockSelector
from akq_module_tusharedatamanager import TushareStockDataManager


@dataclass
class HighDividendConfig:
    """高股息低风险筛选参数。"""

    min_dividend_yield_3y_avg: float = 3.0
    pe_quantile_max: float = 0.70
    vol_quantile_max: float = 0.50
    roe_vol_quantile_max: float = 0.50
    min_dividend_years: int = 3
    max_industry_rank: int = 5
    industry_allow_list: tuple[str, ...] = (
        '家用电器', '食品饮料', '银行', '保险', '煤炭', '电力', '公用事业', '交通运输', '通信', '石油石化'
    )
    min_pe_history_samples: int = 60
    min_return_samples: int = 120


def _to_month_key(date_like: str | pd.Timestamp) -> str:
    dt = pd.to_datetime(date_like)
    return dt.strftime('%Y-%m')


def _safe_percentile_rank(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    return series.rank(pct=True, method='average')


class HighDividendDataAdapter:
    """Tushare 适配层，包含字段兼容与轻量缓存。"""

    def __init__(self, token: str, request_interval: float = 0.35):
        ts.set_token(token)
        self.pro = ts.pro_api()
        self.request_interval = float(request_interval)
        self._daily_basic_cache: dict[str, pd.DataFrame] = {}
        self._roe_cache: dict[str, pd.DataFrame] = {}

    def get_daily_basic(self, ts_code: str) -> pd.DataFrame:
        if ts_code in self._daily_basic_cache:
            return self._daily_basic_cache[ts_code]

        fields_candidates = [
            'ts_code,trade_date,pe_ttm,total_mv,dv_ttm,dv_ratio',
            'ts_code,trade_date,pe_ttm,total_mv,dv_ratio',
            'ts_code,trade_date,pe_ttm,total_mv',
        ]

        for fields in fields_candidates:
            try:
                df = self.pro.daily_basic(
                    ts_code=ts_code,
                    start_date='20160101',
                    end_date=datetime.now().strftime('%Y%m%d'),
                    fields=fields,
                )
                if df is not None:
                    out = df.copy()
                    if not out.empty:
                        out['trade_date'] = pd.to_datetime(out['trade_date'], format='%Y%m%d', errors='coerce')
                        out = out.sort_values('trade_date')
                    self._daily_basic_cache[ts_code] = out
                    return out
            except Exception:
                continue

        self._daily_basic_cache[ts_code] = pd.DataFrame()
        return self._daily_basic_cache[ts_code]

    def get_roe_history(self, ts_code: str) -> pd.DataFrame:
        if ts_code in self._roe_cache:
            return self._roe_cache[ts_code]

        fields_candidates = [
            'ts_code,end_date,roe',
            'ts_code,end_date,roe,roe_yearly',
        ]

        for fields in fields_candidates:
            try:
                df = self.pro.fina_indicator(
                    ts_code=ts_code,
                    start_date='20160101',
                    end_date=datetime.now().strftime('%Y%m%d'),
                    fields=fields,
                )
                if df is not None:
                    out = df.copy()
                    if not out.empty:
                        out['end_date'] = pd.to_datetime(out['end_date'], format='%Y%m%d', errors='coerce')
                        out = out.sort_values('end_date')
                    self._roe_cache[ts_code] = out
                    return out
            except Exception:
                continue

        self._roe_cache[ts_code] = pd.DataFrame()
        return self._roe_cache[ts_code]


def build_monthly_high_dividend_universe(
    token: str,
    start_date: str,
    end_date: str,
    data_dir: str = 'selector_data',
    market_data_dir: str = 'tsdata',
    config: Optional[HighDividendConfig] = None,
    use_cache: bool = True,
    verbose: bool = True,
) -> dict[str, set[str]]:
    """构建月度高股息低风险股票池。"""
    cfg = config or HighDividendConfig()
    selector = StockSelector(token=token, data_dir=data_dir, request_interval=0.32)
    adapter = HighDividendDataAdapter(token=token, request_interval=0.35)
    dm = TushareStockDataManager(token=token, data_dir=market_data_dir, request_interval=1.2)

    monthly_dates = pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), freq='MS')

    cfg_text = json.dumps(cfg.__dict__, ensure_ascii=True, sort_keys=True)
    cfg_key = hashlib.md5(cfg_text.encode('utf-8')).hexdigest()[:10]
    cache_dir = Path(data_dir) / 'monthly_high_dividend_universe_cache' / f'{start_date}_{end_date}_{cfg_key}'
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)

    universe: dict[str, set[str]] = {}

    for dt in monthly_dates:
        trade_date = dt.strftime('%Y%m%d')
        month_key = dt.strftime('%Y-%m')
        cache_file = cache_dir / f'{trade_date}.csv'

        if use_cache and cache_file.exists():
            try:
                cached = pd.read_csv(cache_file, dtype={'symbol': 'string'})
                universe[month_key] = set(cached['symbol'].astype(str).str.zfill(6).tolist())
                continue
            except Exception:
                pass

        # 先复用现有基础筛选链，降低坏数据概率与耦合成本。
        base_df = selector.select(trade_date=trade_date, verbose=False)
        if base_df is None or base_df.empty:
            universe[month_key] = set()
            if use_cache:
                pd.DataFrame({'symbol': []}).to_csv(cache_file, index=False)
            continue

        work = base_df.copy()
        work['symbol'] = work['symbol'].astype(str).str.zfill(6)

        # 行业白名单（跨周期导向）：允许传空列表跳过该过滤。
        allow_set = set(cfg.industry_allow_list)
        if allow_set:
            work = work[work['industry'].astype(str).isin(allow_set)].copy()
            if work.empty:
                universe[month_key] = set()
                if use_cache:
                    pd.DataFrame({'symbol': []}).to_csv(cache_file, index=False)
                continue

        # 行业内市值前五
        work['industry_mcap_rank'] = work.groupby('industry')['market_cap'].rank(method='first', ascending=False)
        work = work[work['industry_mcap_rank'] <= cfg.max_industry_rank].copy()
        if work.empty:
            universe[month_key] = set()
            if use_cache:
                pd.DataFrame({'symbol': []}).to_csv(cache_file, index=False)
            continue

        # 扩展股息、PE历史分位、3年波动率、ROE波动率
        rows: list[dict[str, object]] = []
        for _, row in work.iterrows():
            symbol = str(row['symbol']).zfill(6)
            ts_code = selector._to_ts_code(symbol)

            db = adapter.get_daily_basic(ts_code)
            if db is None or db.empty:
                continue

            db_upto = db[db['trade_date'] <= pd.to_datetime(trade_date)]
            if db_upto.empty:
                continue

            pe_hist = db_upto['pe_ttm'].dropna()
            pe_hist = pe_hist[pe_hist > 0]
            if len(pe_hist) < int(cfg.min_pe_history_samples):
                continue

            curr_pe = float(pe_hist.iloc[-1])
            pe_q = float((pe_hist <= curr_pe).mean())

            recent_3y = db_upto[db_upto['trade_date'] >= (pd.to_datetime(trade_date) - pd.DateOffset(years=3))]

            # 股息率优先使用 dv_ttm；若不可用则回退 dv_ratio。
            div_col = None
            for candidate_col in ['dv_ttm', 'dv_ratio']:
                if candidate_col in recent_3y.columns:
                    non_na = pd.to_numeric(recent_3y[candidate_col], errors='coerce').dropna()
                    if not non_na.empty:
                        div_col = candidate_col
                        break

            if div_col is None:
                continue

            div_recent = pd.to_numeric(recent_3y.get(div_col), errors='coerce').dropna()
            if div_recent.empty:
                continue
            div_recent = div_recent[div_recent >= 0]
            if div_recent.empty:
                continue

            div_3y_avg = float(div_recent.mean())

            yearly_obs = (
                recent_3y.assign(year=recent_3y['trade_date'].dt.year)
                .groupby('year')[div_col]
                .mean()
                .dropna()
            )
            div_years = int((yearly_obs > 0).sum())

            # 3年波动率
            start_3y = (pd.to_datetime(trade_date) - pd.DateOffset(years=3)).strftime('%Y%m%d')
            k = dm.get_stock_data(
                symbol=symbol,
                start_date=start_3y,
                end_date=trade_date,
                force_update=False,
                adjust='qfq',
            )
            if k is None or k.empty or 'close' not in k.columns:
                continue
            ret = pd.to_numeric(k['close'], errors='coerce').pct_change().dropna()
            if len(ret) < int(cfg.min_return_samples):
                continue
            vol_3y = float(ret.std() * np.sqrt(252))

            # ROE 波动率
            roe_df = adapter.get_roe_history(ts_code)
            if roe_df is None or roe_df.empty or 'roe' not in roe_df.columns:
                continue
            roe_recent = roe_df[roe_df['end_date'] >= (pd.to_datetime(trade_date) - pd.DateOffset(years=3))]
            roe_vals = pd.to_numeric(roe_recent['roe'], errors='coerce').dropna()
            if len(roe_vals) < 4:
                continue
            roe_vol = float(roe_vals.std())

            rows.append(
                {
                    'symbol': symbol,
                    'industry': str(row['industry']),
                    'market_cap': float(row['market_cap']),
                    'pe_ttm': curr_pe,
                    'pe_quantile': pe_q,
                    'dividend_yield_3y_avg': div_3y_avg,
                    'dividend_positive_years': div_years,
                    'vol_3y': vol_3y,
                    'roe_vol_3y': roe_vol,
                }
            )

        ext = pd.DataFrame(rows)
        if ext.empty:
            universe[month_key] = set()
            if use_cache:
                pd.DataFrame({'symbol': []}).to_csv(cache_file, index=False)
            continue

        # 分位标准化后执行阈值过滤
        ext['vol_quantile'] = ext.groupby('industry')['vol_3y'].transform(_safe_percentile_rank)
        ext['roe_vol_quantile'] = ext.groupby('industry')['roe_vol_3y'].transform(_safe_percentile_rank)

        filtered = ext[
            (ext['dividend_yield_3y_avg'] >= cfg.min_dividend_yield_3y_avg)
            & (ext['dividend_positive_years'] >= cfg.min_dividend_years)
            & (ext['pe_quantile'] <= cfg.pe_quantile_max)
            & (ext['vol_quantile'] <= cfg.vol_quantile_max)
            & (ext['roe_vol_quantile'] <= cfg.roe_vol_quantile_max)
        ].copy()

        # 简单综合分：高股息 + 低PE + 低波动
        if not filtered.empty:
            filtered['score'] = (
                filtered['dividend_yield_3y_avg'].rank(pct=True)
                + (1.0 - filtered['pe_quantile'])
                + (1.0 - filtered['vol_quantile'])
                + (1.0 - filtered['roe_vol_quantile'])
            )
            filtered = filtered.sort_values(['score', 'dividend_yield_3y_avg'], ascending=False)

        symbols = set(filtered['symbol'].astype(str).str.zfill(6).tolist()) if not filtered.empty else set()
        universe[month_key] = symbols

        if verbose:
            print(
                f'[Universe] {month_key}: base={len(base_df)}, after_industry_rank={len(work)}, '
                f'extended={len(ext)}, final={len(symbols)}'
            )

        if use_cache:
            save = filtered[['symbol', 'industry', 'market_cap', 'dividend_yield_3y_avg', 'score']].copy() if not filtered.empty else pd.DataFrame({'symbol': []})
            save.to_csv(cache_file, index=False, encoding='utf-8')

    return universe


class HighDividendLowRiskLongTermStrategy(Strategy):
    """高股息低风险长线策略（MVP）。"""

    def __init__(
        self,
        symbols: list[str],
        monthly_universe: Optional[dict[str, set[str]]] = None,
        position_weight: float = 0.08,
        max_positions: int = 12,
        stop_loss_pct: float = -0.10,
        trailing_start_pct: float = 0.20,
        trailing_drawdown_pct: float = 0.35,
    ) -> None:
        super().__init__()
        if not symbols:
            raise ValueError('symbols 不能为空')
        if position_weight <= 0 or position_weight > 1.0:
            raise ValueError('position_weight 必须在 (0, 1]')

        self.symbols = sorted({str(s).strip() for s in symbols if str(s).strip()})
        self.monthly_universe = monthly_universe or {}
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
    def _is_quarter_rebalance_month(month_key: str) -> bool:
        month = int(month_key.split('-')[1])
        return month in {1, 4, 7, 10}

    def _open_symbols(self) -> list[str]:
        return [s for s in self.symbols if self.get_position(s) > 0]

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
        price = float(bar.close)
        position = self.get_position(symbol)

        # 已持仓：仅做底线风险管理，不做频繁信号化卖出。
        if position > 0:
            entry = float(self.entry_price.get(symbol, price))
            pnl = (price - entry) / entry if entry > 0 else 0.0
            peak = max(float(self.peak_pnl.get(symbol, pnl)), pnl)
            self.peak_pnl[symbol] = peak

            if pnl <= self.stop_loss_pct:
                self._close_position_with_reason(symbol, f'底线止损({pnl:.2%})', price)
                return

            if peak >= self.trailing_start_pct and peak > 0:
                retrace = (peak - pnl) / peak
                if retrace >= self.trailing_drawdown_pct:
                    self._close_position_with_reason(
                        symbol,
                        f'利润回撤止盈(peak={peak:.2%}, now={pnl:.2%}, retrace={retrace:.2%})',
                        price,
                    )
                    return
            return

        # 空仓：仅在季度调仓月允许新开仓，降低换手。
        if not self._is_quarter_rebalance_month(month_key):
            return

        allowed = self.monthly_universe.get(month_key, set())
        if symbol not in allowed:
            return

        if len(self._open_symbols()) >= self.max_positions:
            return

        self.order_target_percent(self.position_weight, symbol)
        self.entry_price[symbol] = price
        self.peak_pnl[symbol] = 0.0
        self.log(f'[{symbol}] 长线建仓: target={self.position_weight:.2%}, month={month_key}')


def load_market_data(
    token: str,
    symbols: list[str],
    start_date: str,
    end_date: str,
    data_dir: str,
) -> dict[str, pd.DataFrame]:
    """批量加载行情：优先本地缓存，仅对缺失标的补拉 API。"""
    manager = TushareStockDataManager(token=token, data_dir=data_dir, request_interval=1.2)

    raw = manager.get_multiple_stocks(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        force_update=False,
        adjust='qfq',
        delay_between=0.0,
        allow_api=False,
    )

    missing = [str(sym) for sym, df in raw.items() if df is None or df.empty]
    if missing:
        print(f'本地缓存缺失 {len(missing)} 只，开始补拉 Tushare...')
        fetched = manager.get_multiple_stocks(
            symbols=missing,
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
    token: str,
    start_date: str,
    end_date: str,
    data_dir: str,
    fallback_data: dict[str, pd.DataFrame],
) -> pd.Series:
    """加载基准收益率，失败时回退到样本收益率。"""
    benchmark_symbol = '000300.SH'  # 沪深300
    manager = TushareStockDataManager(token=token, data_dir=data_dir, request_interval=1.2)

    try:
        benchmark_df = manager.get_stock_data(
            symbol=benchmark_symbol,
            start_date=start_date,
            end_date=end_date,
            force_update=False,
            adjust='qfq',
        )
    except Exception as exc:
        print(f'基准获取失败，回退样本收益率: {exc}')
        benchmark_df = None

    if benchmark_df is not None and not benchmark_df.empty and 'close' in benchmark_df.columns:
        benchmark_df.index = pd.to_datetime(benchmark_df.index)
        return benchmark_df['close'].pct_change().fillna(0.0).rename(benchmark_symbol)

    first_symbol = sorted(fallback_data.keys())[0]
    fb = fallback_data[first_symbol].copy()
    fb.index = pd.to_datetime(fb.index)
    return fb['close'].pct_change().fillna(0.0).rename('fallback_benchmark')


def main() -> None:
    # 运行参数：直接在此处配置，不依赖环境变量。
    token = os.getenv('TUSHARE_TOKEN')
    if not token:
        raise RuntimeError('请先设置环境变量 TUSHARE_TOKEN')

    start_date = '20220101'
    end_date = datetime.now().strftime('%Y%m%d')
    quick_mode = False
    if quick_mode:
        # 快速模式默认只跑最近约14个月，优先验证是否可跑通。
        end_dt = pd.to_datetime(end_date)
        start_date = (end_dt - pd.DateOffset(months=14)).strftime('%Y%m%d')
        print(f'quick_mode=True 已启用，区间调整为 {start_date} -> {end_date}')

    selector_data_dir = 'selector_data'
    market_data_dir = 'tsdata'

    cfg = HighDividendConfig(
        min_dividend_yield_3y_avg=3.0,
        pe_quantile_max=0.70,
        vol_quantile_max=0.50,
        roe_vol_quantile_max=0.50,
        min_dividend_years=3,
        max_industry_rank=5,
        min_pe_history_samples=40 if quick_mode else 60,
        min_return_samples=80 if quick_mode else 120,
    )

    monthly_universe = build_monthly_high_dividend_universe(
        token=token,
        start_date=start_date,
        end_date=end_date,
        data_dir=selector_data_dir,
        market_data_dir=market_data_dir,
        config=cfg,
        use_cache=True,
        verbose=True,
    )

    non_empty_months = sum(1 for v in monthly_universe.values() if v)
    all_symbols = sorted({s for vals in monthly_universe.values() for s in vals})
    if not all_symbols:
        print('严格参数下月度股票池为空，自动切换到宽松参数重试...')
        relaxed_cfg = HighDividendConfig(
            min_dividend_yield_3y_avg=1.5,
            pe_quantile_max=0.80,
            vol_quantile_max=0.70,
            roe_vol_quantile_max=0.70,
            min_dividend_years=1,
            max_industry_rank=8,
            industry_allow_list=(),
            min_pe_history_samples=30,
            min_return_samples=80,
        )
        monthly_universe = build_monthly_high_dividend_universe(
            token=token,
            start_date=start_date,
            end_date=end_date,
            data_dir=selector_data_dir,
            market_data_dir=market_data_dir,
            config=relaxed_cfg,
            use_cache=True,
            verbose=True,
        )
        non_empty_months = sum(1 for v in monthly_universe.values() if v)
        all_symbols = sorted({s for vals in monthly_universe.values() for s in vals})
        if not all_symbols:
            raise RuntimeError('月度股票池仍为空：请先检查 Tushare 股息字段权限/字段覆盖，再调整阈值。')

    print(f'股票池构建完成: 月份总数={len(monthly_universe)}, 非空月份={non_empty_months}, 去重标的数={len(all_symbols)}')

    data = load_market_data(
        token=token,
        symbols=all_symbols,
        start_date=start_date,
        end_date=end_date,
        data_dir=market_data_dir,
    )
    tradable_symbols = sorted(data.keys())

    strategy = HighDividendLowRiskLongTermStrategy(
        symbols=tradable_symbols,
        monthly_universe=monthly_universe,
        position_weight=0.08,
        max_positions=12,
        stop_loss_pct=-0.10,
        trailing_start_pct=0.20,
        trailing_drawdown_pct=0.35,
    )

    benchmark_returns = load_benchmark_returns(
        token=token,
        start_date=start_date,
        end_date=end_date,
        data_dir=market_data_dir,
        fallback_data=data,
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
    ts_now = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = report_dir / f'high_dividend_low_risk_{ts_now}.html'

    result.report(
        filename=str(report_path),
        title='A股高股息低风险长线策略报告（MVP）',
        market_data=data,
        include_trade_kline=True,
        benchmark=benchmark_returns,
    )
    print(f'\n报告已保存至: {report_path}')


if __name__ == '__main__':
    main()
