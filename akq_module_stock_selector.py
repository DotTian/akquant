"""
医药行业选股策略模块

选股条件（12条）：
  1. 行业筛选（医药相关行业，参数化，可组合多个或单个行业）
  2. 毛利率 > 行业均值 + 20%
  3. 近3年毛利率趋势斜率 > 0（线性回归）
  4. 研发费用 / 营收 >= 5%
  5. 排除上一年年报净利润为负
  6. 排除 ST / 退市
  7. 排除 全体股东质押 > 50% 或 商誉/净资产 > 30%
  8. 过去60个交易日平均日成交额 > 5000万
  9. 排除 60日平均振幅 < 1%
 10. 市值 20亿 ~ 300亿
 11. PE_TTM < 自身历史 80% 分位
 12. PEG 在 0.3 ~ 1.2 之间

用法:
    from akq_module_stock_selector import StockSelector
    sel = StockSelector(token='xxx')
    df = sel.select(trade_date='20240701')

月度回测:
    python akq_module_stock_selector.py
"""

import os
import json
import logging
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import tushare as ts

from akq_module_tusharedatamanager import TushareStockDataManager
from akq_module_stockinfo import StockInfoManager
from akq_module_fundamentals import FundamentalsManager

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# ============================================================================
# 默认医药相关行业列表（可通过 discover_industries() 动态发现后确认）
# ============================================================================
DEFAULT_MEDICAL_INDUSTRIES = [
    '化学制药', '生物制药', '医疗保健', '医药商业', '医疗器械',
    '中药', '医药', '医疗服务', '医药流通', '疫苗', '创新药',
    '原料药', '血液制品', '体外诊断', '基因测序', 'CRO',
]


class StockSelector:
    """
    A股选股器 —— 基于基本面 + 技术面多条件过滤

    Parameters
    ----------
    token : str
        Tushare Pro Token
    industries : List[str] or None
        目标行业列表，None 则使用默认医药行业
    data_dir : str
        缓存数据根目录
    request_interval : float
        API 请求间隔（秒）
    """

    def __init__(
        self,
        token: str,
        industries: Optional[List[str]] = None,
        data_dir: str = 'selector_data',
        request_interval: float = 1.2,
    ):
        self.token = token
        self.industries = industries or DEFAULT_MEDICAL_INDUSTRIES
        self.request_interval = request_interval
        self._last_req = 0

        # 数据目录
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.cache_dir = self.data_dir / 'cache'
        self.cache_dir.mkdir(exist_ok=True)

        # 初始化 tushare
        ts.set_token(token)
        self.pro = ts.pro_api()

        # 子模块
        self.dm = TushareStockDataManager(
            token=token,
            data_dir=str(self.data_dir / 'daily_data'),
            request_interval=request_interval,
        )
        self.sim = StockInfoManager(
            token=token,
            data_dir=str(self.data_dir / 'stock_info'),
            request_interval=request_interval,
            auto_update=False,
        )
        self.fm = FundamentalsManager(
            token=token,
            cache_dir=str(self.data_dir / 'fundamentals'),
            request_interval=request_interval,
        )

        # 内存缓存
        self._daily_basic_cache: Dict[str, pd.DataFrame] = {}
        self._pledge_cache: Dict[str, Optional[float]] = {}
        self._bs_cache: Dict[str, pd.DataFrame] = {}

        logger.info(
            f'StockSelector 初始化完成，目标行业: {len(self.industries)} 个'
        )

    # ========================================================================
    # 工具方法
    # ========================================================================
    def _wait(self):
        elapsed = time.time() - self._last_req
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)
        self._last_req = time.time()

    @staticmethod
    def _to_ts_code(symbol: str) -> str:
        code = str(symbol).zfill(6)
        if code.startswith(('688', '600', '601', '603', '605')):
            return f'{code}.SH'
        return f'{code}.SZ'

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f'{key}.parquet'

    def _load_pickle_cache(self, key: str) -> Optional[pd.DataFrame]:
        p = self._cache_path(key)
        if not p.exists():
            return None
        mtime = datetime.fromtimestamp(p.stat().st_mtime)
        if datetime.now() - mtime > timedelta(hours=24):
            return None
        try:
            return pd.read_parquet(p)
        except Exception:
            return None

    def _save_pickle_cache(self, key: str, df: pd.DataFrame):
        df.to_parquet(self._cache_path(key), index=False)

    # ========================================================================
    # 行业发现
    # ========================================================================
    def discover_industries(self, keyword: str = '医药') -> pd.DataFrame:
        """
        扫描全市场行业，返回包含关键词的行业及股票数量，
        帮助用户确认最终行业列表。

        Parameters
        ----------
        keyword : str
            搜索关键词，支持多个用 | 分隔，如 '医药|制药|医疗|生物|疫苗|CRO'

        Returns
        -------
        pd.DataFrame : columns=['industry', 'count']
        """
        df_all = self.sim.get_all_stocks_info(force_update=False)
        if df_all is None or df_all.empty:
            logger.error('无法获取股票基本信息')
            return pd.DataFrame()

        # 统计每个行业的股票数量
        industry_counts = (
            df_all.groupby('industry')
            .size()
            .reset_index(name='count')
            .sort_values('count', ascending=False)
        )

        # 筛选含关键词的行业
        mask = industry_counts['industry'].str.contains(
            keyword, case=False, regex=True, na=False
        )
        matched = industry_counts[mask].copy()
        matched['total_in_industry'] = matched['count'].sum()

        logger.info(
            f'关键词 "{keyword}" 匹配到 {len(matched)} 个行业，'
            f'共 {matched["count"].sum()} 只股票'
        )
        print(f'\n{"="*60}')
        print(f'🔍 关键词 "{keyword}" 匹配的行业：')
        print(f'{"="*60}')
        for _, row in matched.iterrows():
            print(f'  {row["industry"]:<20s}  {row["count"]:>5d} 只')

        return matched

    # ========================================================================
    # 数据获取扩展（市值、质押、商誉、历史PE、历史毛利率）
    # ========================================================================
    def _get_daily_basic(self, symbol: str) -> Optional[pd.DataFrame]:
        """获取 daily_basic（含 total_mv, pe_ttm）"""
        if symbol in self._daily_basic_cache:
            return self._daily_basic_cache[symbol]

        ck = f'daily_basic_{symbol}'
        df = self._load_pickle_cache(ck)
        if df is not None:
            self._daily_basic_cache[symbol] = df
            return df

        self._wait()
        ts_code = self._to_ts_code(symbol)
        try:
            df = self.pro.daily_basic(
                ts_code=ts_code,
                start_date='20180101',
                end_date=datetime.now().strftime('%Y%m%d'),
                fields='ts_code,trade_date,pe_ttm,total_mv,pe,pb',
            )
            if df is not None and not df.empty:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                self._save_pickle_cache(ck, df)
                self._daily_basic_cache[symbol] = df
                return df
        except Exception as e:
            logger.warning(f'daily_basic {symbol} 失败: {e}')
        return None

    def get_market_cap(self, symbol: str, trade_date: str) -> Optional[float]:
        """获取市值（亿元），tushare total_mv 单位万元"""
        df = self._get_daily_basic(symbol)
        if df is None or df.empty:
            return None
        target = pd.to_datetime(trade_date.replace('-', ''))
        df = df[df['trade_date'] <= target].sort_values('trade_date')
        if df.empty:
            return None
        mv = df.iloc[-1].get('total_mv')
        if mv is not None and pd.notna(mv) and float(mv) > 0:
            return round(float(mv) / 10000, 2)
        return None

    def get_current_pe_ttm(self, symbol: str, trade_date: str) -> Optional[float]:
        """获取最近 PE_TTM"""
        df = self._get_daily_basic(symbol)
        if df is None or df.empty:
            return None
        target = pd.to_datetime(trade_date.replace('-', ''))
        df = df[df['trade_date'] <= target].sort_values('trade_date')
        if df.empty:
            return None
        pe = df.iloc[-1].get('pe_ttm')
        if pe is not None and pd.notna(pe) and float(pe) > 0:
            return float(pe)
        return None

    def get_pe_80th_percentile(self, symbol: str) -> Optional[float]:
        """PE_TTM 历史 80% 分位"""
        df = self._get_daily_basic(symbol)
        if df is None or df.empty:
            return None
        pe = df['pe_ttm'].dropna()
        pe = pe[pe > 0]
        if len(pe) < 20:
            return None
        return float(np.percentile(pe, 80))

    def get_pledge_ratio(self, symbol: str) -> Optional[float]:
        """全体股东质押比例 (%)"""
        if symbol in self._pledge_cache:
            return self._pledge_cache[symbol]

        ck = f'pledge_{symbol}'
        df = self._load_pickle_cache(ck)
        if df is not None:
            val = None if df.empty else df.iloc[0].get('pledge_ratio')
            self._pledge_cache[symbol] = float(val) if val is not None and pd.notna(val) else None
            return self._pledge_cache[symbol]

        self._wait()
        ts_code = self._to_ts_code(symbol)
        try:
            df = self.pro.pledge_stat(ts_code=ts_code)
            if df is not None and not df.empty:
                self._save_pickle_cache(ck, df)
                val = df.iloc[0].get('pledge_ratio')
                result = float(val) if val is not None and pd.notna(val) else None
            else:
                self._save_pickle_cache(ck, pd.DataFrame({'cached': [True]}))
                result = None
            self._pledge_cache[symbol] = result
            return result
        except Exception as e:
            logger.warning(f'pledge_stat {symbol} 失败: {e}')
            self._pledge_cache[symbol] = None
            return None

    def _get_balance_sheet(self, symbol: str) -> Optional[pd.DataFrame]:
        """资产负债表"""
        if symbol in self._bs_cache:
            return self._bs_cache[symbol]

        ck = f'bs_{symbol}'
        df = self._load_pickle_cache(ck)
        if df is not None:
            self._bs_cache[symbol] = df
            return df

        self._wait()
        ts_code = self._to_ts_code(symbol)
        try:
            df = self.pro.balancesheet(
                ts_code=ts_code,
                start_date='20180101',
                end_date=datetime.now().strftime('%Y%m%d'),
                fields='ts_code,end_date,goodwill,total_hldr_eqy_exc_min_int',
                report_type='1',
            )
            if df is not None and not df.empty:
                df['end_date'] = pd.to_datetime(df['end_date'])
                self._save_pickle_cache(ck, df)
                self._bs_cache[symbol] = df
                return df
        except Exception as e:
            logger.warning(f'balancesheet {symbol} 失败: {e}')
        return None

    def get_goodwill_ratio(self, symbol: str) -> Optional[float]:
        """商誉 / 归母净资产 (%)"""
        df = self._get_balance_sheet(symbol)
        if df is None or df.empty:
            return None
        latest = df.sort_values('end_date', ascending=False).iloc[0]
        gw = latest.get('goodwill')
        eq = latest.get('total_hldr_eqy_exc_min_int')
        if (
            gw is not None
            and eq is not None
            and pd.notna(gw)
            and pd.notna(eq)
            and eq > 0
        ):
            return round(float(gw) / float(eq) * 100, 2)
        return None

    def get_historical_gross_margins(
        self, symbol: str
    ) -> Optional[List[Dict]]:
        """获取历史毛利率序列 [{end_date, gross_margin}, ...]"""
        df = self.fm.get_fina_indicator(symbol)
        if df is None or df.empty:
            return None
        if 'grossprofit_margin' not in df.columns:
            return None
        df = df[['end_date', 'grossprofit_margin']].dropna().sort_values('end_date')
        return [
            {'end_date': row['end_date'], 'gross_margin': float(row['grossprofit_margin'])}
            for _, row in df.iterrows()
        ] or None

    def calc_gross_margin_slope(self, symbol: str, years: int = 3) -> Optional[float]:
        """近 N 年毛利率线性回归斜率"""
        margins = self.get_historical_gross_margins(symbol)
        if margins is None or len(margins) < 3:
            return None
        cutoff = datetime.now() - timedelta(days=years * 365)
        recent = [m for m in margins if m['end_date'] > pd.Timestamp(cutoff)]
        if len(recent) < 3:
            return None
        x = np.arange(len(recent))
        y = np.array([m['gross_margin'] for m in recent], dtype=float)
        valid = np.isfinite(y)
        if valid.sum() < 3:
            return None
        slope, _ = np.polyfit(x[valid], y[valid], 1)
        return round(float(slope), 4)

    # ========================================================================
    # 60 日成交额 & 振幅
    # ========================================================================
    def _calc_turnover_amplitude(
        self, symbol: str, trade_date: str, n_days: int = 60
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        返回 (日均成交额_万元, 平均振幅_%)
        日均成交额 = mean(volume * close) 近似
        日均振幅   = mean((high - low) / pre_close) * 100
        """
        end = trade_date.replace('-', '')
        # 往前推足够多的自然日
        start = (
            pd.to_datetime(trade_date) - timedelta(days=n_days * 3)
        ).strftime('%Y%m%d')

        try:
            df = self.dm.get_stock_data(
                symbol=symbol, start_date=start, end_date=end, adjust='qfq'
            )
        except Exception as e:
            logger.debug(f'{symbol} 日线获取失败: {e}')
            return None, None

        if df is None or df.empty:
            return None, None

        # 取最近 n_days 个交易日
        df = df.sort_index().tail(n_days)
        if len(df) < n_days * 0.8:  # 至少 80% 数据
            return None, None

        # 成交额 = volume × close（粗略，单位：元）
        if 'volume' in df.columns and 'close' in df.columns:
            avg_amount = (df['volume'] * df['close']).mean()
            avg_amount_wan = round(float(avg_amount) / 10000, 2)  # 万元
        else:
            avg_amount_wan = None

        # 日均振幅
        if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            df['amplitude'] = (df['high'] - df['low']) / df['close'].shift(1) * 100
            avg_amp = float(df['amplitude'].dropna().mean())
        else:
            avg_amp = None

        return avg_amount_wan, avg_amp

    # ========================================================================
    # 上一年年报是否亏损
    # ========================================================================
    def is_last_fiscal_year_loss(self, symbol: str, trade_date: str) -> Optional[bool]:
        """
        判断最近一个完整财年年报是否亏损。
        trade_date 对应日期，找该日期之前最近一个 12-31 年报的 n_income_attr_p。
        """
        df = self.fm.get_income(symbol)
        if df is None or df.empty:
            return None
        target = pd.to_datetime(trade_date.replace('-', ''))
        # 年报通常 end_date 是 12-31，取 target 之前最近的年报
        df = df[df['end_date'] <= target].sort_values('end_date')
        if df.empty:
            return None
        # 取最近一条
        latest = df.iloc[-1]
        ni = latest.get('n_income_attr_p')
        if ni is not None and pd.notna(ni):
            return float(ni) < 0
        return None

    # ========================================================================
    # 主筛选方法
    # ========================================================================
    def select(self, trade_date: str, verbose: bool = True) -> pd.DataFrame:
        """
        按全部条件筛选股票。

        Parameters
        ----------
        trade_date : str
            选股日期，如 '20240701' 或 '2024-07-01'
        verbose : bool
            是否打印每步过滤信息

        Returns
        -------
        pd.DataFrame : 符合条件的股票，含关键指标列
        """
        trade_date = trade_date.replace('-', '')
        t0 = time.time()

        if verbose:
            print(f'\n{"="*70}')
            print(f'📊 开始选股 [{trade_date}]')
            print(f'{"="*70}')

        # ── Step 0: 获取全市场股票列表 ──
        all_stocks = self.sim.get_all_stocks_info(force_update=False)
        total = len(all_stocks)
        if verbose:
            print(f'\n[0] 全市场股票: {total} 只')

        # 转为方便处理的格式
        candidates = all_stocks.reset_index()  # symbol 变成列
        candidates = candidates[['symbol', 'name', 'industry', 'list_date']].copy()
        candidates['symbol'] = candidates['symbol'].astype(str).str.zfill(6)

        # ── Step 1: 行业筛选 ──
        before = len(candidates)
        candidates = candidates[
            candidates['industry'].isin(self.industries)
        ].copy()
        if verbose:
            print(f'[1] 行业筛选 → {before} → {len(candidates)} (过滤 {before - len(candidates)})')

        if candidates.empty:
            logger.warning('行业筛选后无股票')
            return pd.DataFrame()

        # ── Step 6: ST / 退市（提前做，减少后续请求） ──
        symbols = candidates['symbol'].tolist()
        st_flags = {}
        delisted_flags = {}
        for sym in symbols:
            status = self.fm.get_listing_status(sym)
            st_flags[sym] = status.get('is_st', False)
            delisted_flags[sym] = status.get('is_delisted', False)
        candidates['is_st'] = candidates['symbol'].map(st_flags)
        candidates['is_delisted'] = candidates['symbol'].map(delisted_flags)
        before = len(candidates)
        candidates = candidates[~candidates['is_st'] & ~candidates['is_delisted']].copy()
        if verbose:
            print(f'[2] 排除 ST/退市 → {before} → {len(candidates)}')

        if candidates.empty:
            return pd.DataFrame()

        # ── Step 10: 市值 20-300 亿 ──
        mkt_caps = {}
        for sym in candidates['symbol']:
            mkt_caps[sym] = self.get_market_cap(sym, trade_date)
        candidates['market_cap'] = candidates['symbol'].map(mkt_caps)
        before = len(candidates)
        candidates = candidates[
            candidates['market_cap'].notna()
            & (candidates['market_cap'] >= 20)
            & (candidates['market_cap'] <= 300)
        ].copy()
        if verbose:
            print(f'[3] 市值 20-300亿 → {before} → {len(candidates)}')

        if candidates.empty:
            return pd.DataFrame()

        # ── Step 8 & 9: 60日成交额 & 振幅 ──
        amt_dict = {}
        amp_dict = {}
        for i, sym in enumerate(candidates['symbol']):
            if verbose and (i % 20 == 0):
                print(f'  计算成交额/振幅: {i+1}/{len(candidates)}')
            amt, amp = self._calc_turnover_amplitude(sym, trade_date, n_days=60)
            amt_dict[sym] = amt
            amp_dict[sym] = amp
        candidates['avg_amount_60d'] = candidates['symbol'].map(amt_dict)
        candidates['avg_amplitude_60d'] = candidates['symbol'].map(amp_dict)

        # 排除振幅 < 1%
        before = len(candidates)
        candidates = candidates[
            candidates['avg_amplitude_60d'].notna()
            & (candidates['avg_amplitude_60d'] >= 1.0)
        ].copy()
        if verbose:
            print(f'[4] 排除 60日均振幅<1% → {before} → {len(candidates)}')

        # 成交额 > 5000万
        before = len(candidates)
        candidates = candidates[
            candidates['avg_amount_60d'].notna()
            & (candidates['avg_amount_60d'] > 5000)
        ].copy()
        if verbose:
            print(f'[5] 60日均成交额>5000万 → {before} → {len(candidates)}')

        if candidates.empty:
            return pd.DataFrame()

        # ── Step 5: 上一年年报非亏损 ──
        loss_flags = {}
        for sym in candidates['symbol']:
            loss_flags[sym] = self.is_last_fiscal_year_loss(sym, trade_date)
        candidates['is_loss_last_year'] = candidates['symbol'].map(loss_flags)
        before = len(candidates)
        candidates = candidates[
            ~candidates['is_loss_last_year'].fillna(True)
        ].copy()
        if verbose:
            print(f'[6] 排除上年度亏损 → {before} → {len(candidates)}')

        if candidates.empty:
            return pd.DataFrame()

        # ── Step 7a: 质押 ≤ 50% ──
        pledge_ratios = {}
        for sym in candidates['symbol']:
            pledge_ratios[sym] = self.get_pledge_ratio(sym)
        candidates['pledge_ratio'] = candidates['symbol'].map(pledge_ratios)
        before = len(candidates)
        candidates = candidates[
            candidates['pledge_ratio'].isna()
            | (candidates['pledge_ratio'] <= 50)
        ].copy()
        if verbose:
            print(f'[7] 排除质押>50% → {before} → {len(candidates)}')

        if candidates.empty:
            return pd.DataFrame()

        # ── Step 7b: 商誉 ≤ 30% ──
        gw_ratios = {}
        for i, sym in enumerate(candidates['symbol']):
            if verbose and (i % 20 == 0):
                print(f'  计算商誉: {i+1}/{len(candidates)}')
            gw_ratios[sym] = self.get_goodwill_ratio(sym)
        candidates['goodwill_ratio'] = candidates['symbol'].map(gw_ratios)
        before = len(candidates)
        candidates = candidates[
            candidates['goodwill_ratio'].isna()
            | (candidates['goodwill_ratio'] <= 30)
        ].copy()
        if verbose:
            print(f'[8] 排除商誉>30% → {before} → {len(candidates)}')

        if candidates.empty:
            return pd.DataFrame()

        # ── 获取毛利率、研发占比等基本面指标 ──
        gross_margins = {}
        rd_ratios = {}
        gm_slopes = {}
        for i, sym in enumerate(candidates['symbol']):
            if verbose and (i % 10 == 0):
                print(f'  获取基本面数据: {i+1}/{len(candidates)}')
            metrics = self.fm.get_all_metrics(sym, trade_date)
            gross_margins[sym] = metrics.get('gross_margin')
            rd_ratios[sym] = metrics.get('rd_ratio')
            # 毛利率 3 年趋势
            gm_slopes[sym] = self.calc_gross_margin_slope(sym, years=3)

        candidates['gross_margin'] = candidates['symbol'].map(gross_margins)
        candidates['rd_ratio'] = candidates['symbol'].map(rd_ratios)
        candidates['gm_slope'] = candidates['symbol'].map(gm_slopes)

        # ── Step 2: 毛利率 > 行业均值 + 20% ──
        # 先计算行业均值
        industry_avg_gm = {}
        for ind in candidates['industry'].unique():
            ind_mask = candidates['industry'] == ind
            ind_gm = candidates.loc[ind_mask, 'gross_margin'].dropna()
            if len(ind_gm) > 0:
                industry_avg_gm[ind] = ind_gm.mean()
            else:
                industry_avg_gm[ind] = None

        candidates['industry_avg_gm'] = candidates['industry'].map(industry_avg_gm)
        before = len(candidates)
        candidates = candidates[
            candidates['gross_margin'].notna()
            & candidates['industry_avg_gm'].notna()
            & (candidates['gross_margin'] > candidates['industry_avg_gm'] + 20)
        ].copy()
        if verbose:
            print(
                f'[9] 毛利率>行业均值+20% → {before} → {len(candidates)}'
            )

        if candidates.empty:
            return pd.DataFrame()

        # ── Step 3: 3年毛利率趋势斜率 > 0 ──
        before = len(candidates)
        candidates = candidates[
            candidates['gm_slope'].notna() & (candidates['gm_slope'] > 0)
        ].copy()
        if verbose:
            print(f'[10] 3年毛利率趋势>0 → {before} → {len(candidates)}')

        if candidates.empty:
            return pd.DataFrame()

        # ── Step 4: 研发占比 ≥ 5% ──
        before = len(candidates)
        candidates = candidates[
            candidates['rd_ratio'].notna() & (candidates['rd_ratio'] >= 5)
        ].copy()
        if verbose:
            print(f'[11] 研发占比≥5% → {before} → {len(candidates)}')

        if candidates.empty:
            return pd.DataFrame()

        # ── Step 11: PE < 历史 80% 分位 ──
        pe_80 = {}
        pe_current = {}
        for i, sym in enumerate(candidates['symbol']):
            if verbose and (i % 10 == 0):
                print(f'  计算 PE 分位: {i+1}/{len(candidates)}')
            pe_80[sym] = self.get_pe_80th_percentile(sym)
            pe_current[sym] = self.get_current_pe_ttm(sym, trade_date)
        candidates['pe_80th'] = candidates['symbol'].map(pe_80)
        candidates['pe_ttm'] = candidates['symbol'].map(pe_current)
        before = len(candidates)
        candidates = candidates[
            candidates['pe_ttm'].notna()
            & candidates['pe_80th'].notna()
            & (candidates['pe_ttm'] < candidates['pe_80th'])
        ].copy()
        if verbose:
            print(f'[12] PE<80%分位 → {before} → {len(candidates)}')

        if candidates.empty:
            return pd.DataFrame()

        # ── Step 12: PEG 0.3 ~ 1.2 ──
        peg_dict = {}
        growth_dict = {}
        for i, sym in enumerate(candidates['symbol']):
            if verbose and (i % 10 == 0):
                print(f'  计算 PEG: {i+1}/{len(candidates)}')
            peg, growth = self.fm.calculate_peg(sym, trade_date)
            peg_dict[sym] = peg
            growth_dict[sym] = growth
        candidates['peg'] = candidates['symbol'].map(peg_dict)
        candidates['profit_growth'] = candidates['symbol'].map(growth_dict)
        before = len(candidates)
        candidates = candidates[
            candidates['peg'].notna()
            & (candidates['peg'] >= 0.3)
            & (candidates['peg'] <= 1.2)
        ].copy()
        if verbose:
            print(f'[13] PEG 0.3~1.2 → {before} → {len(candidates)}')

        # ── 整理输出列 ──
        out_cols = [
            'symbol', 'name', 'industry',
            'gross_margin', 'industry_avg_gm', 'gm_slope',
            'rd_ratio', 'peg', 'profit_growth',
            'pe_ttm', 'pe_80th', 'market_cap',
            'avg_amount_60d', 'avg_amplitude_60d',
            'pledge_ratio', 'goodwill_ratio',
        ]
        result = candidates[out_cols].reset_index(drop=True)
        result.insert(0, 'trade_date', trade_date)

        elapsed = time.time() - t0
        if verbose:
            print(f'\n✅ 选股完成: {len(result)} 只，耗时 {elapsed:.1f}s')
            print(f'{"="*70}\n')

        return result

    # ========================================================================
    # 月度回测
    # ========================================================================
    def run_monthly(
        self,
        start_date: str = '20240101',
        end_date: str = '20260713',
        output_excel: Optional[str] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        月度选股回测：每月1号选股，汇总结果。

        Parameters
        ----------
        start_date : str
            开始日期 YYYYMMDD 或 YYYY-MM-DD
        end_date : str
            结束日期 YYYYMMDD 或 YYYY-MM-DD
        output_excel : str or None
            输出 Excel 文件路径，None 则自动命名为
            stock_selection_YYYYMMDD_HHMMSS.xlsx
        verbose : bool
            是否打印详细信息

        Returns
        -------
        pd.DataFrame : 所有月份合并结果
        """
        start_date = start_date.replace('-', '')
        end_date = end_date.replace('-', '')

        # 生成每月1号日期列表
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        dates = pd.date_range(start=start_dt, end=end_dt, freq='MS')  # Month Start

        all_results = []
        summary_rows = []

        for dt in dates:
            td = dt.strftime('%Y%m%d')
            if verbose:
                print(f'\n{"#"*70}')
                print(f'### 月度选股: {td}')
                print(f'{"#"*70}')

            try:
                df_month = self.select(trade_date=td, verbose=verbose)
                df_month['month'] = dt.strftime('%Y-%m')
                all_results.append(df_month)
                summary_rows.append({
                    'month': dt.strftime('%Y-%m'),
                    'trade_date': td,
                    'count': len(df_month),
                    'symbols': ','.join(df_month['symbol'].tolist()) if not df_month.empty else '',
                    'names': ','.join(df_month['name'].tolist()) if not df_month.empty else '',
                })
            except Exception as e:
                logger.error(f'月度选股 {td} 异常: {e}', exc_info=True)
                summary_rows.append({
                    'month': dt.strftime('%Y-%m'),
                    'trade_date': td,
                    'count': 0,
                    'symbols': f'ERROR: {e}',
                    'names': '',
                })

        # 合并所有结果
        if all_results:
            df_all = pd.concat(all_results, ignore_index=True)
        else:
            df_all = pd.DataFrame()

        # 汇总表
        df_summary = pd.DataFrame(summary_rows)

        # 打印汇总
        if verbose and not df_summary.empty:
            print(f'\n{"="*70}')
            print('📋 月度选股汇总')
            print(f'{"="*70}')
            for _, row in df_summary.iterrows():
                print(
                    f'  {row["month"]} | {row["trade_date"]} | '
                    f'选出 {row["count"]} 只'
                )
                if row['count'] > 0:
                    print(f'    {row["names"]}')
            print(f'\n总计: {df_summary["count"].sum()} 只次')

        # 导出 Excel
        if output_excel is None:
            output_excel = (
                f'stock_selection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
            )

        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            df_summary.to_excel(writer, sheet_name='月度汇总', index=False)
            if not df_all.empty:
                df_all.to_excel(writer, sheet_name='详细结果', index=False)
                # 每月一个 sheet
                for month, group in df_all.groupby('month'):
                    sheet_name = str(month)[:7]  # 如 '2024-01'
                    group.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f'\n📁 结果已保存至: {output_excel}')
        return df_all


# ============================================================================
# 主程序
# ============================================================================
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    token = os.getenv('TUSHARE_TOKEN')
    if not token:
        print('❌ 请设置环境变量 TUSHARE_TOKEN')
        print('   export TUSHARE_TOKEN=your_token_here')
        exit(1)

    # ====== Step A: 发现行业（首次运行建议开启） ======
    sel = StockSelector(token=token, request_interval=1.0)

    print('\n🔍 发现医药相关行业...')
    industry_df = sel.discover_industries(keyword='医药|制药|医疗|生物|疫苗|CRO|中药|原料药|血液|体外|基因')
    print(f'\n共发现 {len(industry_df)} 个相关行业')
    print('请确认上述行业后，将其作为 industries 参数传入 StockSelector')
    print('当前默认行业:')
    for ind in DEFAULT_MEDICAL_INDUSTRIES:
        print(f'  - {ind}')

    # ====== Step B: 月度回测选股 ======
    print('\n' + '='*70)
    print('🚀 开始月度选股回测: 2024-01-01 → 2026-07-13')
    print('='*70)

    sel = StockSelector(
        token=token,
        industries=DEFAULT_MEDICAL_INDUSTRIES,
        request_interval=1.0,
    )

    df_results = sel.run_monthly(
        start_date='20240101',
        end_date='20260713',
        output_excel='stock_selection_results.xlsx',
        verbose=True,
    )

    if not df_results.empty:
        print(f'\n✅ 全部完成！共 {len(df_results)} 条选股记录')
    else:
        print('\n⚠️ 未选出任何股票')