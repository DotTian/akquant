"""
五行业选股策略模块

选股条件（10条）：
    1. 行业筛选（五行业配置 + 电池关键词补充）
    2. 毛利率 > 行业均值 + 20%
    3. 近3年毛利率趋势斜率 > 0（线性回归）
    4. 排除上一年年报净利润为负
    5. 排除 ST / 退市
    6. 排除 全体股东质押 > 50% 或 商誉/净资产 > 30%
    7. 过去60个交易日平均日成交额 > 5000万
    8. 排除 60日平均振幅 < 1%
    9. 市值 20亿 ~ 500亿
 10. PE_TTM < 自身历史 80% 分位

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
from typing import Any, Optional, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import tushare as ts

from akq_module_tusharedatamanager import TushareStockDataManager
from akq_module_stockinfo import StockInfoManager
from akq_module_fundamentals import FundamentalsManager

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# ============================================================================
# 五行业配置（来源于外部脚本口径）
# ============================================================================
FIVE_INDUSTRY_GROUPS: Dict[str, List[str]] = {
    '医疗/医药': ['医疗保健', '化学制药', '生物制药', '中成药', '医药商业'],
    '半导体': ['半导体'],
    '电池(锂电)': ['电气设备', '汽车配件', '化工原料', '小金属', '汽车整车'],
    '有色金属': ['小金属', '工业金属', '贵金属', '能源金属', '铝', '铜', '铅锌', '黄金'],
    '化工': ['化工制品', '化工原料', '塑料', '化纤', '日用化工', '橡胶', '化工机械', '农药化肥'],
}

BATTERY_KEYWORDS = [
    '电池', '锂', '宁德', '比亚迪', '恩捷', '天赐', '璞泰来',
    '新宙邦', '当升', '容百', '华友', '寒锐', '科达利',
    '星源材质', '德方纳米', '富临精工', '杉杉', '中材科技',
    '天奈', '嘉元', '诺德', '振华新材', '长远锂科', '格林美',
    '赣锋锂业', '天齐锂业', '融捷', '盛新', '永兴', '江特',
]

# 科创超跌价值回归筛选参数：默认值保持当前口径；可通过 overrides 或调用参数覆盖。
STAR_VALUE_REVERSION_FILTER_DEFAULTS: Dict[str, Any] = {
    'listing_days_min': 182,
    'listing_days_max': 1095, #730 = 2year
    'drawdown_from_post_list_high_max': -0.50,
    'revenue_growth_since_list_min': 0.0,
    'gross_margin_growth_since_list_min': 0.0,
    'pledge_ratio_max': 50.0,
    'goodwill_ratio_max': 30.0,
    'debt_ratio_max': 70.0,
    'amp_20d_max': 0.20, #0.20 振幅
    'vol_ratio_20_60_max': 0.80, #缩量
    'avg_turnover_60d_min': 1.0, #平均换手率
    'enable_tech_filter': True, #技术面筛选开关
}

STAR_VALUE_REVERSION_FILTER_OVERRIDES: Dict[str, Any] = {}

# mixed bollinger 对应的基础选股参数：默认值保持当前口径；支持统一覆盖。
MIXED_BOLLINGER_FILTER_DEFAULTS: Dict[str, Any] = {
    'market_cap_min': 20.0,
    'market_cap_max': 500.0,
    'turnover_window_days': 60,
    'avg_amplitude_60d_min': 1.0,
    'avg_amount_60d_min': 5000.0,
    'pledge_ratio_max': 50.0,
    'goodwill_ratio_max': 30.0,
    'gross_margin_industry_premium': 20.0,
    'gm_slope_min': 0.0,
    'pe_history_percentile': 80.0,
    'pe_history_min_samples': 20,
}

MIXED_BOLLINGER_FILTER_OVERRIDES: Dict[str, Any] = {}


class StockSelector:
    """
    A股选股器 —— 基于基本面 + 技术面多条件过滤

    Parameters
    ----------
    token : str
        Tushare Pro Token
    industries : List[str] or None
        目标行业列表；None 则使用五行业默认配置
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
        request_interval: float = 0.05,
        industry_groups: Optional[Dict[str, List[str]]] = None,
        battery_keywords: Optional[List[str]] = None,
    ):
        self.token = token
        self.industry_groups = industry_groups or FIVE_INDUSTRY_GROUPS
        self.battery_keywords = battery_keywords or BATTERY_KEYWORDS
        self.industries = industries or sorted(
            {
                ind for ind_list in self.industry_groups.values() for ind in ind_list
            }
        )
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
        self._star_daily_basic_cache: Dict[str, pd.DataFrame] = {}
        self._pledge_cache: Dict[str, Optional[float]] = {}
        self._bs_cache: Dict[str, pd.DataFrame] = {}
        self._star_bs_debt_cache: Dict[str, pd.DataFrame] = {}
        self._listing_status_cache: Dict[str, dict] = {}
        self._kline_cache: Dict[str, pd.DataFrame] = {}
        self._kline_no_data_symbols: Set[str] = set()
        self._daily_data_local_only: bool = False
        self._turnover_amp_cache: Dict[str, pd.DataFrame] = {}
        self._goodwill_ratio_cache: Dict[str, Optional[float]] = {}
        self._fina_static_cache: Dict[str, dict] = {}
        self._income_annual_loss_cache: Dict[str, pd.DataFrame] = {}

        logger.info(
            f'StockSelector 初始化完成，目标行业: {len(self.industries)} 个'
        )

    def _apply_industry_filter(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """五行业筛选：行业匹配 + 电池关键词补充。"""
        if candidates.empty:
            return candidates

        industry_set = set(self.industries)
        industry_mask = candidates['industry'].isin(industry_set)
        keyword_mask = candidates['name'].astype(str).apply(
            lambda n: any(k in n for k in self.battery_keywords)
        )
        return candidates[industry_mask | keyword_mask].copy()

    @staticmethod
    def _normalize_symbol(symbol: object) -> str:
        code = str(symbol).strip().upper()
        if '.' in code:
            code = code.split('.', 1)[0]
        return code.zfill(6)

    @classmethod
    def _is_bj_symbol(cls, symbol: object) -> bool:
        code = cls._normalize_symbol(symbol)
        return code.startswith(('43', '83', '87', '88', '92'))

    @classmethod
    def _is_star_688_symbol(cls, symbol: object) -> bool:
        code = cls._normalize_symbol(symbol)
        return code.startswith('688')

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
        if symbol is None:
            return ''
        code = str(symbol).strip().upper()
        if '.' in code:
            code = code.split('.', 1)[0]
        code = code.zfill(6)
        if code.startswith(('43', '83', '87', '88', '92')):
            return f'{code}.BJ'
        if code.startswith(('688', '600', '601', '603', '605')):
            return f'{code}.SH'
        return f'{code}.SZ'

    @staticmethod
    def _resolve_star_value_reversion_filter_params(
        filter_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params = dict(STAR_VALUE_REVERSION_FILTER_DEFAULTS)
        params.update({k: v for k, v in STAR_VALUE_REVERSION_FILTER_OVERRIDES.items() if v is not None})
        if filter_params:
            params.update({k: v for k, v in filter_params.items() if v is not None})
        return params

    @staticmethod
    def _resolve_mixed_bollinger_filter_params(
        filter_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params = dict(MIXED_BOLLINGER_FILTER_DEFAULTS)
        params.update({k: v for k, v in MIXED_BOLLINGER_FILTER_OVERRIDES.items() if v is not None})
        if filter_params:
            params.update({k: v for k, v in filter_params.items() if v is not None})
        return params

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f'{key}.parquet'

    def _load_pickle_cache(self, key: str) -> Optional[pd.DataFrame]:
        p = self._cache_path(key)
        if not p.exists():
            return None
        # 永久有效，不自动过期（需手动清除缓存目录来强制更新）
        try:
            return pd.read_parquet(p)
        except Exception:
            return None

    def _save_pickle_cache(self, key: str, df: pd.DataFrame):
        df.to_parquet(self._cache_path(key), index=False)

    def _get_star_daily_basic(self, symbol: str) -> Optional[pd.DataFrame]:
        """获取科创策略专用 daily_basic（含换手率字段）。"""
        if symbol in self._star_daily_basic_cache:
            return self._star_daily_basic_cache[symbol]

        ck = f'star_vr_daily_basic_{symbol}'
        df = self._load_pickle_cache(ck)
        if df is not None and not df.empty:
            try:
                df = df.copy()
                if 'trade_date' in df.columns:
                    df['trade_date'] = pd.to_datetime(df['trade_date'])
                self._star_daily_basic_cache[symbol] = df
                return df
            except Exception:
                pass

        self._wait()
        ts_code = self._to_ts_code(symbol)
        try:
            df = self.pro.daily_basic(
                ts_code=ts_code,
                start_date='20180101',
                end_date=datetime.now().strftime('%Y%m%d'),
                fields='ts_code,trade_date,pe_ttm,total_mv,turnover_rate,turnover_rate_f',
            )
            if df is not None and not df.empty:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                self._save_pickle_cache(ck, df)
                self._star_daily_basic_cache[symbol] = df
                return df
        except Exception as e:
            logger.warning(f'star daily_basic {symbol} 失败: {e}')
        return None

    def _get_balance_sheet_for_debt(self, symbol: str) -> Optional[pd.DataFrame]:
        """获取资产负债率所需字段。"""
        if symbol in self._star_bs_debt_cache:
            return self._star_bs_debt_cache[symbol]

        ck = f'star_vr_bs_debt_{symbol}'
        cached = self._load_pickle_cache(ck)
        if cached is not None and not cached.empty:
            cached = cached.copy()
            cached['end_date'] = pd.to_datetime(cached['end_date'])
            self._star_bs_debt_cache[symbol] = cached
            return cached

        self._wait()
        ts_code = self._to_ts_code(symbol)
        try:
            df = self.pro.balancesheet(
                ts_code=ts_code,
                start_date='20180101',
                end_date=datetime.now().strftime('%Y%m%d'),
                fields='ts_code,end_date,total_assets,total_liab',
                report_type='1',
            )
            if df is not None and not df.empty:
                df['end_date'] = pd.to_datetime(df['end_date'])
                self._save_pickle_cache(ck, df)
                self._star_bs_debt_cache[symbol] = df
                return df
        except Exception as e:
            logger.warning(f'star balancesheet debt {symbol} 失败: {e}')
        return None

    def _calc_listing_days(self, list_date: object, trade_date: str) -> Optional[int]:
        if list_date is None or pd.isna(list_date):
            return None
        try:
            ld = pd.to_datetime(list_date)
            td = pd.to_datetime(trade_date.replace('-', ''))
            return int((td - ld).days)
        except Exception:
            return None

    def _calc_post_list_drawdown(self, symbol: str, trade_date: str, list_date: object) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """返回 (drawdown, latest_close, post_list_high)。"""
        if list_date is None or pd.isna(list_date):
            return None, None, None

        start = pd.to_datetime(list_date).strftime('%Y%m%d')
        end = trade_date.replace('-', '')
        df = self._get_star_kline(symbol=symbol, start_date=start, end_date=end)

        if df is None or df.empty or 'close' not in df.columns:
            return None, None, None

        k = df.sort_index().copy()
        latest_close = float(k['close'].iloc[-1])
        post_list_high = float(k['close'].max())
        if post_list_high <= 0:
            return None, latest_close, None
        drawdown = (latest_close - post_list_high) / post_list_high
        return float(drawdown), latest_close, post_list_high

    def _calc_star_tech_metrics(self, symbol: str, trade_date: str) -> dict:
        """计算科创策略技术过滤指标。"""
        result: dict[str, Optional[float]] = {
            'amp_20d': None,
            'avg_vol_20d': None,
            'avg_vol_60d': None,
            'vol_ratio_20_60': None,
            'avg_turnover_60d': None,
        }

        end = trade_date.replace('-', '')
        start = (pd.to_datetime(end) - timedelta(days=220)).strftime('%Y%m%d')
        df = self._get_star_kline(symbol=symbol, start_date=start, end_date=end)

        if df is None or df.empty:
            return result

        k = df.sort_index().copy()
        tail20 = k.tail(20)
        tail60 = k.tail(60)
        if len(tail20) >= 20 and 'close' in tail20.columns:
            c_min = float(tail20['close'].min())
            c_max = float(tail20['close'].max())
            if c_min > 0:
                result['amp_20d'] = (c_max - c_min) / c_min

        if len(tail20) >= 20 and 'volume' in tail20.columns:
            result['avg_vol_20d'] = float(tail20['volume'].mean())
        if len(tail60) >= 60 and 'volume' in tail60.columns:
            result['avg_vol_60d'] = float(tail60['volume'].mean())

        avg20 = result.get('avg_vol_20d')
        avg60 = result.get('avg_vol_60d')
        if avg20 is not None and avg60 is not None and avg60 > 0:
            result['vol_ratio_20_60'] = float(avg20 / avg60)

        db = self._get_star_daily_basic(symbol)
        if db is None or db.empty:
            return result

        td = pd.to_datetime(end)
        d = db[db['trade_date'] <= td].sort_values('trade_date').copy()
        if d.empty:
            return result

        turn_col = 'turnover_rate_f' if 'turnover_rate_f' in d.columns else 'turnover_rate'
        if turn_col in d.columns:
            t60 = d.tail(60)[turn_col].dropna()
            if len(t60) >= 30:
                result['avg_turnover_60d'] = float(t60.mean())

        return result

    def _get_star_kline(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """统一获取科创策略所需日线，优先内存缓存，再读本地/按需增量。"""
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        cached = self._kline_cache.get(symbol)
        if cached is not None and not cached.empty:
            c = cached.sort_index()
            if c.index.min() <= start_dt and c.index.max() >= end_dt:
                return c[(c.index >= start_dt) & (c.index <= end_dt)]

        try:
            df = self.dm.get_stock_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjust='qfq',
                allow_api=not self._daily_data_local_only,
            )
        except Exception:
            return None

        if df is None or df.empty:
            return None

        self._kline_cache[symbol] = df
        k = df.sort_index()
        return k[(k.index >= start_dt) & (k.index <= end_dt)]

    def _calc_debt_ratio(self, symbol: str, trade_date: str) -> Optional[float]:
        """资产负债率 (%)."""
        df = self._get_balance_sheet_for_debt(symbol)
        if df is None or df.empty:
            return None
        td = pd.to_datetime(trade_date.replace('-', ''))
        d = df[df['end_date'] <= td].sort_values('end_date')
        if d.empty:
            return None
        last = d.iloc[-1]
        ta = last.get('total_assets')
        tl = last.get('total_liab')
        if ta is None or tl is None or pd.isna(ta) or pd.isna(tl) or float(ta) <= 0:
            return None
        return round(float(tl) / float(ta) * 100.0, 2)

    def _calc_rd_to_revenue_ratio(self, symbol: str, trade_date: str) -> Optional[float]:
        """研发费用/营收占比 (%), 提示项。"""
        df = self.fm.get_income(symbol)
        if df is None or df.empty:
            return None
        td = pd.to_datetime(trade_date.replace('-', ''))
        d = df[df['end_date'] <= td].sort_values('end_date')
        if d.empty:
            return None
        latest = d.iloc[-1]
        rev = latest.get('total_revenue')
        rd = latest.get('rd_exp')
        if rev is None or rd is None or pd.isna(rev) or pd.isna(rd) or float(rev) <= 0:
            return None
        return round(float(rd) / float(rev) * 100.0, 2)

    def _calc_growth_since_listing(self, symbol: str, trade_date: str, list_date: object) -> tuple[Optional[float], Optional[float]]:
        """返回 (营收增长率%, 毛利率增量百分点)."""
        if list_date is None or pd.isna(list_date):
            return None, None

        td = pd.to_datetime(trade_date.replace('-', ''))
        ld = pd.to_datetime(list_date)

        income = self.fm.get_income(symbol)
        if income is None or income.empty:
            return None, None
        annual_income = income[
            (income['end_date'].dt.month == 12)
            & (income['end_date'].dt.day == 31)
            & (income['end_date'] <= td)
        ].sort_values('end_date')
        if annual_income.empty:
            return None, None

        first_income = annual_income[annual_income['end_date'] >= ld]
        if first_income.empty:
            return None, None
        first_income_row = first_income.iloc[0]
        latest_income_row = annual_income.iloc[-1]

        first_rev = first_income_row.get('total_revenue')
        latest_rev = latest_income_row.get('total_revenue')
        revenue_growth = None
        if (
            first_rev is not None and latest_rev is not None
            and pd.notna(first_rev) and pd.notna(latest_rev)
            and float(first_rev) > 0
        ):
            revenue_growth = (float(latest_rev) - float(first_rev)) / float(first_rev) * 100.0

        fina = self.fm.get_fina_indicator(symbol)
        if fina is None or fina.empty:
            return revenue_growth, None
        annual_fina = fina[
            (fina['end_date'].dt.month == 12)
            & (fina['end_date'].dt.day == 31)
            & (fina['end_date'] <= td)
        ].sort_values('end_date')
        if annual_fina.empty:
            return revenue_growth, None

        first_fina = annual_fina[annual_fina['end_date'] >= ld]
        if first_fina.empty:
            return revenue_growth, None
        first_gm = first_fina.iloc[0].get('grossprofit_margin')
        latest_gm = annual_fina.iloc[-1].get('grossprofit_margin')
        gm_growth = None
        if first_gm is not None and latest_gm is not None and pd.notna(first_gm) and pd.notna(latest_gm):
            gm_growth = float(latest_gm) - float(first_gm)

        return revenue_growth, gm_growth

    def _build_turnover_amplitude_series(self, symbol: str, n_days: int = 60) -> Optional[pd.DataFrame]:
        """基于单票日线一次性预计算滚动 60 日成交额/振幅特征序列。"""
        if n_days != 60:
            return None

        ck = f'turnover_amp_{n_days}_{symbol}'
        if symbol in self._turnover_amp_cache:
            return self._turnover_amp_cache[symbol]

        cached = self._load_pickle_cache(ck)
        if cached is not None and not cached.empty:
            try:
                cached = cached.copy()
                cached['trade_date'] = pd.to_datetime(cached['trade_date'])
                cached = cached.sort_values('trade_date').set_index('trade_date')
                self._turnover_amp_cache[symbol] = cached
                return cached
            except Exception:
                pass

        df = self._kline_cache.get(symbol)
        if df is None or df.empty:
            return None

        k = df.copy()
        if not k.index.is_monotonic_increasing:
            k = k.sort_index()

        if 'volume' not in k.columns or 'close' not in k.columns:
            return None

        # 成交额（万元）
        amount_wan = (k['volume'] * 100 * k['close']) / 10000.0
        avg_amount_60d = amount_wan.rolling(n_days, min_periods=max(10, int(n_days * 0.8))).mean()

        # 振幅（%）
        if 'high' in k.columns and 'low' in k.columns:
            amplitude = ((k['high'] - k['low']) / k['close'].shift(1)) * 100.0
            avg_amp_60d = amplitude.rolling(n_days, min_periods=max(10, int(n_days * 0.8))).mean()
        else:
            avg_amp_60d = pd.Series(index=k.index, dtype='float64')

        feat = pd.DataFrame(
            {
                'avg_amount_60d': avg_amount_60d,
                'avg_amplitude_60d': avg_amp_60d,
            },
            index=k.index,
        )
        feat.index.name = 'trade_date'
        feat = feat.reset_index()
        self._save_pickle_cache(ck, feat)
        feat = feat.set_index('trade_date')
        self._turnover_amp_cache[symbol] = feat
        return feat

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

    def _calculate_peg(self, symbol: str, pe_ttm: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
        """
        计算 PEG = PE_TTM / 净利润增长率。
        pe_ttm 从外部统一传入（避免内部再调 get_current_pe_ttm）。
        """
        if pe_ttm is None:
            return None, None

        # 净利润同比增长率（复用 FundamentalsManager 的 TTM 增长计算）
        growth = self.fm._calc_ttm_growth(symbol)
        if growth is None:
            # 回退到单季同比
            df_fina = self.fm.get_fina_indicator(symbol)
            if df_fina is not None and not df_fina.empty:
                latest = df_fina.sort_values('end_date', ascending=False).iloc[0]
                g = latest.get('q_profit_yoy')
                if g is not None and pd.notna(g):
                    growth = float(g)

        if growth is None or growth <= 0:
            return None, None

        try:
            peg = round(float(pe_ttm) / float(growth), 2)
            return peg, round(float(growth), 2)
        except (ZeroDivisionError, ValueError):
            return None, None

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
        if symbol in self._goodwill_ratio_cache:
            return self._goodwill_ratio_cache[symbol]

        ck = f'goodwill_ratio_{symbol}'
        cached = self._load_pickle_cache(ck)
        if cached is not None and not cached.empty:
            val = cached.iloc[0].get('goodwill_ratio')
            result = None if val is None or pd.isna(val) else float(val)
            self._goodwill_ratio_cache[symbol] = result
            return result

        df = self._get_balance_sheet(symbol)
        if df is None or df.empty:
            self._goodwill_ratio_cache[symbol] = None
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
            result = round(float(gw) / float(eq) * 100, 2)
            self._goodwill_ratio_cache[symbol] = result
            self._save_pickle_cache(ck, pd.DataFrame([{'goodwill_ratio': result}]))
            return result
        self._goodwill_ratio_cache[symbol] = None
        self._save_pickle_cache(ck, pd.DataFrame([{'goodwill_ratio': np.nan}]))
        return None

    def _get_fina_static_features(self, symbol: str, years: int = 3) -> dict:
        """财务静态特征：最新毛利率、近N年毛利率斜率（按股票预计算并缓存）。"""
        if symbol in self._fina_static_cache:
            return self._fina_static_cache[symbol]

        ck = f'fina_static_{symbol}_{years}y'
        cached = self._load_pickle_cache(ck)
        if cached is not None and not cached.empty:
            latest_gm_raw = cached.iloc[0].get('latest_gross_margin')
            slope_raw = cached.iloc[0].get('gm_slope')
            feat = {
                'latest_gross_margin': None if latest_gm_raw is None or pd.isna(latest_gm_raw) else float(latest_gm_raw),
                'gm_slope': None if slope_raw is None or pd.isna(slope_raw) else float(slope_raw),
            }
            self._fina_static_cache[symbol] = feat
            return feat

        feat: Dict[str, Optional[float]] = {'latest_gross_margin': None, 'gm_slope': None}
        df = self.fm.get_fina_indicator(symbol)
        if df is None or df.empty or ('grossprofit_margin' not in df.columns):
            self._fina_static_cache[symbol] = feat
            self._save_pickle_cache(
                ck,
                pd.DataFrame([{'latest_gross_margin': np.nan, 'gm_slope': np.nan}]),
            )
            return feat

        d = df[['end_date', 'grossprofit_margin']].dropna().copy()
        if d.empty:
            self._fina_static_cache[symbol] = feat
            self._save_pickle_cache(
                ck,
                pd.DataFrame([{'latest_gross_margin': np.nan, 'gm_slope': np.nan}]),
            )
            return feat

        d['end_date'] = pd.to_datetime(d['end_date'])
        d = d.sort_values('end_date')

        latest_val = d.iloc[-1]['grossprofit_margin']
        if pd.notna(latest_val):
            feat['latest_gross_margin'] = float(latest_val)

        cutoff = datetime.now() - timedelta(days=years * 365)
        recent = d[d['end_date'] > pd.Timestamp(cutoff)]
        if len(recent) >= 3:
            x = np.arange(len(recent))
            y = recent['grossprofit_margin'].astype(float).to_numpy()
            valid = np.isfinite(y)
            if valid.sum() >= 3:
                slope, _ = np.polyfit(x[valid], y[valid], 1)
                feat['gm_slope'] = round(float(slope), 4)

        self._fina_static_cache[symbol] = feat
        self._save_pickle_cache(
            ck,
            pd.DataFrame(
                [
                    {
                        'latest_gross_margin': feat['latest_gross_margin'] if feat['latest_gross_margin'] is not None else np.nan,
                        'gm_slope': feat['gm_slope'] if feat['gm_slope'] is not None else np.nan,
                    }
                ]
            ),
        )
        return feat

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
        feat = self._get_fina_static_features(symbol, years=years)
        val = feat.get('gm_slope')
        return None if val is None else float(val)

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
        end_dt = pd.to_datetime(end)

        # 优先读取“下载后一次性预计算”的滚动特征缓存。
        feat = self._build_turnover_amplitude_series(symbol, n_days=n_days)
        if feat is not None and (not feat.empty):
            hist = feat[feat.index <= end_dt]
            if not hist.empty:
                row = hist.iloc[-1]
                amt = row.get('avg_amount_60d')
                amp = row.get('avg_amplitude_60d')
                amt_v = None if pd.isna(amt) else round(float(amt), 2)
                amp_v = None if pd.isna(amp) else float(amp)
                return amt_v, amp_v

        # 往前推足够多的自然日
        start = (
            pd.to_datetime(trade_date) - timedelta(days=n_days * 3)
        ).strftime('%Y%m%d')

        df = self._kline_cache.get(symbol)
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)

        # 对已确认“无日线数据”的股票直接跳过，避免每月重复触发 API。
        if symbol in self._kline_no_data_symbols and (df is None or df.empty):
            return None, None

        # 若内存缓存未覆盖目标区间，则仅拉到 trade_date，避免无意义补“今天”导致空数据重试。
        if df is None or df.empty or df.index.min() > start_dt or df.index.max() < end_dt:
            cache_end = end
            try:
                df = self.dm.get_stock_data(
                    symbol=symbol,
                    start_date=start,
                    end_date=cache_end,
                    adjust='qfq',
                    allow_api=not self._daily_data_local_only,
                )
                if df is None or df.empty:
                    # 本地模式下不将空窗口视为永久无数据，避免后续月份被误伤。
                    if not self._daily_data_local_only:
                        self._kline_no_data_symbols.add(symbol)
                    return None, None
                self._kline_cache[symbol] = df
                self._kline_no_data_symbols.discard(symbol)
                self._turnover_amp_cache.pop(symbol, None)
                # 新日线覆盖后重建特征缓存（一次性），后续周度只读。
                self._build_turnover_amplitude_series(symbol, n_days=n_days)
            except Exception as e:
                logger.debug(f'{symbol} 日线获取失败: {e}')
                if ('空数据' in str(e)) or ('empty' in str(e).lower()):
                    self._kline_no_data_symbols.add(symbol)
                return None, None

        if df is None or df.empty:
            return None, None

        # 先裁到目标窗口，再取末端 n_days。
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
        if df.empty:
            return None, None

        # 取最近 n_days 个交易日
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        df = df.tail(n_days)
        if len(df) < n_days * 0.8:  # 至少 80% 数据
            return None, None

        # 成交额 = volume(手) × 100 × close (单位：元)
        # Tushare 的 vol 字段单位是「手」，需要 ×100 转换为「股」
        if 'volume' in df.columns and 'close' in df.columns:
            avg_amount = (df['volume'] * 100 * df['close']).mean()
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
        if symbol in self._income_annual_loss_cache:
            annual = self._income_annual_loss_cache[symbol]
        else:
            ck = f'income_annual_loss_{symbol}'
            cached = self._load_pickle_cache(ck)
            if cached is not None and not cached.empty and {'end_date', 'is_loss'}.issubset(set(cached.columns)):
                annual = cached.copy()
                annual['end_date'] = pd.to_datetime(annual['end_date'])
                annual = annual.sort_values('end_date').reset_index(drop=True)
            else:
                df = self.fm.get_income(symbol)
                if df is None or df.empty:
                    return None
                cols = [c for c in ['end_date', 'n_income_attr_p'] if c in df.columns]
                if len(cols) < 2:
                    return None
                annual = df[cols].copy()
                annual['end_date'] = pd.to_datetime(annual['end_date'])
                # 仅使用年报，降低噪声与重复计算。
                annual = annual[
                    (annual['end_date'].dt.month == 12)
                    & (annual['end_date'].dt.day == 31)
                ].copy()
                if annual.empty:
                    return None
                annual['is_loss'] = annual['n_income_attr_p'].apply(
                    lambda v: np.nan if (v is None or pd.isna(v)) else (float(v) < 0)
                )
                annual = annual[['end_date', 'is_loss']].sort_values('end_date').reset_index(drop=True)
                self._save_pickle_cache(ck, annual)
            self._income_annual_loss_cache[symbol] = annual

        if annual is None or annual.empty:
            return None

        target = pd.to_datetime(trade_date.replace('-', ''))
        end_vals = annual['end_date'].to_numpy()
        pos = int(np.searchsorted(end_vals, np.datetime64(target), side='right')) - 1
        if pos < 0:
            return None
        loss_val = annual.iloc[pos].get('is_loss')
        if loss_val is None or pd.isna(loss_val):
            return None
        return bool(loss_val)

    # ========================================================================
    # ST/退市检查（独立实现，带频率控制 + 内存缓存 + 磁盘缓存）
    # ========================================================================
    def _get_listing_status(self, symbol: str) -> dict:
        """
        检查 ST 及退市状态，带频率控制 + 内存缓存 + 磁盘缓存。
        """
        # 1. 内存缓存
        if symbol in self._listing_status_cache:
            return self._listing_status_cache[symbol]

        # 2. 磁盘缓存
        ck = f'listing_status_{symbol}'
        cached = self._load_pickle_cache(ck)
        if cached is not None and not cached.empty:
            result = {
                'is_st': bool(cached.iloc[0].get('is_st', False)),
                'st_reason': str(cached.iloc[0].get('st_reason', '')),
                'is_delisted': bool(cached.iloc[0].get('is_delisted', False)),
                'delist_date': str(cached.iloc[0].get('delist_date', '')),
            }
            self._listing_status_cache[symbol] = result
            return result

        # 3. 调 API
        ts_code = self._to_ts_code(symbol)
        result = {'is_st': False, 'st_reason': '', 'is_delisted': False, 'delist_date': ''}

        self._wait()
        try:
            nc = self.pro.namechange(ts_code=ts_code)
            if nc is not None and not nc.empty:
                st_rows = nc[nc['name'].str.contains(r'ST', regex=True)]
                if not st_rows.empty:
                    result['is_st'] = True
                    latest = st_rows.sort_values('ann_date', ascending=False).iloc[0]
                    result['st_reason'] = str(latest.get('reason', ''))
        except Exception as e:
            logger.warning(f'namechange {symbol} 失败: {e}')

        self._wait()
        try:
            sb = self.pro.stock_basic(ts_code=ts_code, fields='ts_code,name,delist_date')
            if sb is not None and not sb.empty:
                dl_date = sb.iloc[0].get('delist_date')
                if pd.notna(dl_date):
                    result['is_delisted'] = True
                    result['delist_date'] = str(dl_date)
        except Exception as e:
            logger.warning(f'stock_basic {symbol} 失败: {e}')

        # 4. 保存到磁盘和内存缓存
        self._save_pickle_cache(ck, pd.DataFrame([result]))
        self._listing_status_cache[symbol] = result
        return result

    # ========================================================================
    # 主筛选方法
    # ========================================================================
    def select(
        self,
        trade_date: str,
        verbose: bool = True,
        filter_params: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
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
        params = self._resolve_mixed_bollinger_filter_params(filter_params)

        market_cap_min = float(params['market_cap_min'])
        market_cap_max = float(params['market_cap_max'])
        turnover_window_days = int(params['turnover_window_days'])
        avg_amplitude_60d_min = float(params['avg_amplitude_60d_min'])
        avg_amount_60d_min = float(params['avg_amount_60d_min'])
        pledge_ratio_max = float(params['pledge_ratio_max'])
        goodwill_ratio_max = float(params['goodwill_ratio_max'])
        gross_margin_industry_premium = float(params['gross_margin_industry_premium'])
        gm_slope_min = float(params['gm_slope_min'])
        pe_history_percentile = float(params['pe_history_percentile'])
        pe_history_min_samples = int(params['pe_history_min_samples'])

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

        # ── Step 1: 行业筛选（五行业 + 电池关键词） ──
        before = len(candidates)
        candidates = self._apply_industry_filter(candidates)
        if verbose:
            print(f'[1] 行业筛选 → {before} → {len(candidates)} (过滤 {before - len(candidates)})')

        if candidates.empty:
            logger.warning('行业筛选后无股票')
            return pd.DataFrame()

        # ── Step 6: ST / 退市（提前做，减少后续请求） ──
        # 使用自带频率控制的 _get_listing_status，内部每次调用 2 个 API 均有 _wait()
        # 注意：448只股票 × 2API × 1.2s间隔 ≈ 18分钟，有进度提示不会卡死
        symbols = candidates['symbol'].tolist()
        st_flags = {}
        delisted_flags = {}
        total_sym = len(symbols)
        for idx, sym in enumerate(symbols):
            if verbose and (idx % 10 == 0):
                print(f'  检查ST/退市: {idx+1}/{total_sym} ({((idx+1)/total_sym)*100:.0f}%)')
            status = self._get_listing_status(sym)
            st_flags[sym] = status.get('is_st', False)
            delisted_flags[sym] = status.get('is_delisted', False)
        if verbose:
            print(f'  检查ST/退市: {total_sym}/{total_sym} (100%) 完成')
        candidates['is_st'] = candidates['symbol'].map(st_flags)
        candidates['is_delisted'] = candidates['symbol'].map(delisted_flags)
        before = len(candidates)
        candidates = candidates[~candidates['is_st'] & ~candidates['is_delisted']].copy()
        if verbose:
            print(f'[2] 排除 ST/退市 → {before} → {len(candidates)}')

        if candidates.empty:
            return pd.DataFrame()

        # ── 统一加载 daily_basic（一次性拉取全部 PE/市值字段到内存）──
        db_data = {}
        for i, sym in enumerate(candidates['symbol']):
            if verbose and (i % 20 == 0):
                print(f'  加载 daily_basic: {i+1}/{len(candidates)}')
            db_data[sym] = self._get_daily_basic(sym)

        # ── Step 10: 市值 20-500 亿 ──
        mkt_caps = {}
        for sym in candidates['symbol']:
            df = db_data.get(sym)
            if df is None or df.empty:
                mkt_caps[sym] = None
                continue
            target_dt = pd.to_datetime(trade_date)
            df_hist = df[df['trade_date'] <= target_dt].sort_values('trade_date')
            if df_hist.empty:
                mkt_caps[sym] = None
                continue
            mv = df_hist.iloc[-1].get('total_mv')
            if mv is not None and pd.notna(mv) and float(mv) > 0:
                mkt_caps[sym] = round(float(mv) / 10000, 2)
            else:
                mkt_caps[sym] = None
        candidates['market_cap'] = candidates['symbol'].map(mkt_caps)
        before = len(candidates)
        candidates = candidates[
            candidates['market_cap'].notna()
            & (candidates['market_cap'] >= market_cap_min)
            & (candidates['market_cap'] <= market_cap_max)
        ].copy()
        if verbose:
            print(f'[3] 市值 20-500亿 → {before} → {len(candidates)}')

        if candidates.empty:
            return pd.DataFrame()

        # ── Step 8 & 9: 60日成交额 & 振幅 ──
        amt_dict = {}
        amp_dict = {}
        for i, sym in enumerate(candidates['symbol']):
            if verbose and (i % 20 == 0):
                print(f'  计算成交额/振幅: {i+1}/{len(candidates)}')
            amt, amp = self._calc_turnover_amplitude(sym, trade_date, n_days=turnover_window_days)
            amt_dict[sym] = amt
            amp_dict[sym] = amp
        candidates['avg_amount_60d'] = candidates['symbol'].map(amt_dict)
        candidates['avg_amplitude_60d'] = candidates['symbol'].map(amp_dict)

        # 排除振幅 < 1%
        before = len(candidates)
        candidates = candidates[
            candidates['avg_amplitude_60d'].notna()
            & (candidates['avg_amplitude_60d'] >= avg_amplitude_60d_min)
        ].copy()
        if verbose:
            print(f'[4] 排除 60日均振幅<1% → {before} → {len(candidates)}')

        # 成交额 > 5000万
        before = len(candidates)
        candidates = candidates[
            candidates['avg_amount_60d'].notna()
            & (candidates['avg_amount_60d'] > avg_amount_60d_min)
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
        keep_non_loss = ~candidates['is_loss_last_year'].fillna(True).astype(bool)
        candidates = candidates[keep_non_loss].copy()
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
            | (candidates['pledge_ratio'] <= pledge_ratio_max)
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
            | (candidates['goodwill_ratio'] <= goodwill_ratio_max)
        ].copy()
        if verbose:
            print(f'[8] 排除商誉>30% → {before} → {len(candidates)}')

        if candidates.empty:
            return pd.DataFrame()

        # ── 获取毛利率等基本面指标 ──
        gross_margins = {}
        gm_slopes = {}
        for i, sym in enumerate(candidates['symbol']):
            if verbose and (i % 10 == 0):
                print(f'  获取基本面数据: {i+1}/{len(candidates)}')
            feat = self._get_fina_static_features(sym, years=3)
            gross_margins[sym] = feat.get('latest_gross_margin')
            gm_slopes[sym] = feat.get('gm_slope')

        candidates['gross_margin'] = candidates['symbol'].map(gross_margins)
        candidates['gm_slope'] = candidates['symbol'].map(gm_slopes)

        # ── Step 2: 毛利率 > 行业均值 + 20% ──
        # 先计算行业均值
        industry_avg_gm = {}
        for ind in candidates['industry'].unique():
            ind_mask = candidates['industry'] == ind
            ind_gm = pd.Series(candidates.loc[ind_mask, 'gross_margin']).dropna()
            if len(ind_gm) > 0:
                industry_avg_gm[ind] = ind_gm.mean()
            else:
                industry_avg_gm[ind] = None

        candidates['industry_avg_gm'] = candidates['industry'].map(industry_avg_gm)
        before = len(candidates)
        candidates = candidates[
            candidates['gross_margin'].notna()
            & candidates['industry_avg_gm'].notna()
            & (candidates['gross_margin'] > candidates['industry_avg_gm'] + gross_margin_industry_premium)
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
            candidates['gm_slope'].notna() & (candidates['gm_slope'] > gm_slope_min)
        ].copy()
        if verbose:
            print(f'[10] 3年毛利率趋势>0 → {before} → {len(candidates)}')

        if candidates.empty:
            return pd.DataFrame()

        # ── Step 11: PE < 历史 80% 分位（统一从 db_data 提取）──
        pe_80 = {}
        pe_current = {}
        for sym in candidates['symbol']:
            df = db_data.get(sym)
            if df is None or df.empty:
                pe_80[sym] = None
                pe_current[sym] = None
                continue
            # PE 历史 80% 分位
            pe_series = df['pe_ttm'].dropna()
            pe_series = pe_series[pe_series > 0]
            if len(pe_series) >= pe_history_min_samples:
                pe_80[sym] = float(np.percentile(pe_series, pe_history_percentile))
            else:
                pe_80[sym] = None
            # 当前 PE（trade_date 及之前最近交易日）
            target_dt = pd.to_datetime(trade_date)
            df_hist = df[df['trade_date'] <= target_dt].sort_values('trade_date')
            if not df_hist.empty:
                pe_val = df_hist.iloc[-1].get('pe_ttm')
                if pe_val is not None and pd.notna(pe_val) and float(pe_val) > 0:
                    pe_current[sym] = float(pe_val)
                else:
                    pe_current[sym] = None
            else:
                pe_current[sym] = None
        candidates['pe_80th'] = candidates['symbol'].map(pe_80)
        candidates['pe_ttm'] = candidates['symbol'].map(pe_current)
        before = len(candidates)
        candidates = candidates[
            candidates['pe_ttm'].notna()
            & candidates['pe_80th'].notna()
            & (candidates['pe_ttm'] < candidates['pe_80th'])
        ].copy()
        if verbose:
            print(f'[11] PE<80%分位 → {before} → {len(candidates)}')

        peg_map: Dict[str, Optional[float]] = {}
        for sym in candidates['symbol']:
            peg_val, _ = self._calculate_peg(sym, pe_current.get(sym))
            peg_map[sym] = peg_val
        candidates['peg'] = candidates['symbol'].map(peg_map)

        # ── 整理输出列 ──
        out_cols = [
            'symbol', 'name', 'industry',
            'gross_margin', 'industry_avg_gm', 'gm_slope',
            'pe_ttm', 'peg', 'pe_80th', 'market_cap',
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

    def select_star_value_reversion(
        self,
        trade_date: str,
        verbose: bool = True,
        enable_tech_filter: Optional[bool] = None,
        filter_params: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """科创超跌价值回归选股入口（不影响现有 select 流程）。"""
        trade_date = trade_date.replace('-', '')
        t0 = time.time()
        params = self._resolve_star_value_reversion_filter_params(filter_params)
        if enable_tech_filter is not None:
            params['enable_tech_filter'] = bool(enable_tech_filter)

        listing_days_min = int(params['listing_days_min'])
        listing_days_max = int(params['listing_days_max'])
        drawdown_max = float(params['drawdown_from_post_list_high_max'])
        rev_growth_min = float(params['revenue_growth_since_list_min'])
        gm_growth_min = float(params['gross_margin_growth_since_list_min'])
        pledge_ratio_max = float(params['pledge_ratio_max'])
        goodwill_ratio_max = float(params['goodwill_ratio_max'])
        debt_ratio_max = float(params['debt_ratio_max'])
        amp_20d_max = float(params['amp_20d_max'])
        vol_ratio_20_60_max = float(params['vol_ratio_20_60_max'])
        avg_turnover_60d_min = float(params['avg_turnover_60d_min'])
        tech_filter_on = bool(params.get('enable_tech_filter', True))

        if verbose:
            print(f'\n{"="*70}')
            print(f'🚀 科创超跌价值回归选股 [{trade_date}]')
            print(f'{"="*70}')

        all_stocks = self.sim.get_all_stocks_info(force_update=False)
        candidates = all_stocks.reset_index()
        candidates = candidates[['symbol', 'name', 'industry', 'list_date']].copy()
        candidates['symbol'] = candidates['symbol'].astype(str).str.zfill(6)
        candidates['list_date'] = pd.to_datetime(candidates['list_date'], errors='coerce')

        before = len(candidates)
        candidates = candidates[candidates['symbol'].map(self._is_star_688_symbol)].copy()
        if verbose:
            print(f'[1] 仅保留688 → {before} → {len(candidates)}')
        if candidates.empty:
            return pd.DataFrame()

        # 上市 0.5~2 年
        candidates['listing_days'] = candidates['list_date'].map(lambda d: self._calc_listing_days(d, trade_date))
        before = len(candidates)
        candidates = candidates[
            candidates['listing_days'].notna()
            & (candidates['listing_days'] >= listing_days_min)
            & (candidates['listing_days'] <= listing_days_max)
        ].copy()
        if verbose:
            print(f'[2] 上市182~730天 → {before} → {len(candidates)}')
        if candidates.empty:
            return pd.DataFrame()

        # ST/退市
        st_map: Dict[str, bool] = {}
        delisted_map: Dict[str, bool] = {}
        for sym in candidates['symbol']:
            status = self._get_listing_status(sym)
            st_map[sym] = bool(status.get('is_st', False))
            delisted_map[sym] = bool(status.get('is_delisted', False))
        candidates['is_st'] = candidates['symbol'].map(st_map)
        candidates['is_delisted'] = candidates['symbol'].map(delisted_map)
        before = len(candidates)
        candidates = candidates[~candidates['is_st'] & ~candidates['is_delisted']].copy()
        if verbose:
            print(f'[3] 排除ST/退市 → {before} → {len(candidates)}')
        if candidates.empty:
            return pd.DataFrame()

        # 深跌回撤
        drawdown_map: Dict[str, Optional[float]] = {}
        close_map: Dict[str, Optional[float]] = {}
        high_map: Dict[str, Optional[float]] = {}
        for i, row in candidates.iterrows():
            _ = i
            sym = str(row['symbol'])
            dd, close_v, high_v = self._calc_post_list_drawdown(sym, trade_date, row['list_date'])
            drawdown_map[sym] = dd
            close_map[sym] = close_v
            high_map[sym] = high_v
        candidates['drawdown_from_post_list_high'] = candidates['symbol'].map(drawdown_map)
        candidates['latest_close'] = candidates['symbol'].map(close_map)
        candidates['post_list_high'] = candidates['symbol'].map(high_map)
        before = len(candidates)
        candidates = candidates[
            candidates['drawdown_from_post_list_high'].notna()
            & (candidates['drawdown_from_post_list_high'] <= drawdown_max)
        ].copy()
        if verbose:
            print(f'[4] 回撤>=50% → {before} → {len(candidates)}')
        if candidates.empty:
            return pd.DataFrame()

        # 不亏损
        loss_map = {sym: self.is_last_fiscal_year_loss(sym, trade_date) for sym in candidates['symbol']}
        candidates['is_loss_last_year'] = candidates['symbol'].map(loss_map)
        before = len(candidates)
        candidates = candidates[~candidates['is_loss_last_year'].fillna(True).astype(bool)].copy()
        if verbose:
            print(f'[5] 最近完整年报不亏损 → {before} → {len(candidates)}')
        if candidates.empty:
            return pd.DataFrame()

        # 上市后营收/毛利率增长
        rev_growth_map: Dict[str, Optional[float]] = {}
        gm_growth_map: Dict[str, Optional[float]] = {}
        for _, row in candidates.iterrows():
            sym = str(row['symbol'])
            rev_growth, gm_growth = self._calc_growth_since_listing(sym, trade_date, row['list_date'])
            rev_growth_map[sym] = rev_growth
            gm_growth_map[sym] = gm_growth
        candidates['revenue_growth_since_list'] = candidates['symbol'].map(rev_growth_map)
        candidates['gross_margin_growth_since_list'] = candidates['symbol'].map(gm_growth_map)
        before = len(candidates)
        candidates = candidates[
            candidates['revenue_growth_since_list'].notna()
            & (candidates['revenue_growth_since_list'] > rev_growth_min)
            & candidates['gross_margin_growth_since_list'].notna()
            & (candidates['gross_margin_growth_since_list'] > gm_growth_min)
        ].copy()
        if verbose:
            print(f'[6] 上市后营收/毛利率增长 → {before} → {len(candidates)}')
        if candidates.empty:
            return pd.DataFrame()

        # 质押 / 商誉 / 负债率
        pledge_map = {sym: self.get_pledge_ratio(sym) for sym in candidates['symbol']}
        goodwill_map = {sym: self.get_goodwill_ratio(sym) for sym in candidates['symbol']}
        debt_map = {sym: self._calc_debt_ratio(sym, trade_date) for sym in candidates['symbol']}
        candidates['pledge_ratio'] = candidates['symbol'].map(pledge_map)
        candidates['goodwill_ratio'] = candidates['symbol'].map(goodwill_map)
        candidates['debt_ratio'] = candidates['symbol'].map(debt_map)
        before = len(candidates)
        candidates = candidates[
            (candidates['pledge_ratio'].isna() | (candidates['pledge_ratio'] < pledge_ratio_max))
            & (candidates['goodwill_ratio'].isna() | (candidates['goodwill_ratio'] < goodwill_ratio_max))
            & candidates['debt_ratio'].notna()
            & (candidates['debt_ratio'] < debt_ratio_max)
        ].copy()
        if verbose:
            print(f'[7] 质押/商誉/负债率 → {before} → {len(candidates)}')
        if candidates.empty:
            return pd.DataFrame()

        # 技术过滤
        tech_map: Dict[str, dict] = {}
        for sym in candidates['symbol']:
            tech_map[sym] = self._calc_star_tech_metrics(sym, trade_date)
        candidates['amp_20d'] = candidates['symbol'].map(lambda s: tech_map.get(s, {}).get('amp_20d'))
        candidates['vol_ratio_20_60'] = candidates['symbol'].map(lambda s: tech_map.get(s, {}).get('vol_ratio_20_60'))
        candidates['avg_turnover_60d'] = candidates['symbol'].map(lambda s: tech_map.get(s, {}).get('avg_turnover_60d'))
        before = len(candidates)
        if tech_filter_on:
            candidates = candidates[
                candidates['amp_20d'].notna()
                & (candidates['amp_20d'] <= amp_20d_max)
                & candidates['vol_ratio_20_60'].notna()
                & (candidates['vol_ratio_20_60'] < vol_ratio_20_60_max)
                & candidates['avg_turnover_60d'].notna()
                & (candidates['avg_turnover_60d'] > avg_turnover_60d_min)
            ].copy()
            if verbose:
                print(f'[8] 底部振幅/缩量/流动性 → {before} → {len(candidates)}')
            if candidates.empty:
                return pd.DataFrame()
        elif verbose:
            print(f'[8] 底部振幅/缩量/流动性 → 已跳过，保留 {before} 只')

        # 提示指标
        pe_map = {sym: self.get_current_pe_ttm(sym, trade_date) for sym in candidates['symbol']}
        peg_map: Dict[str, Optional[float]] = {}
        rd_ratio_map: Dict[str, Optional[float]] = {}
        for sym in candidates['symbol']:
            peg_val, _ = self._calculate_peg(sym, pe_map.get(sym))
            peg_map[sym] = peg_val
            rd_ratio_map[sym] = self._calc_rd_to_revenue_ratio(sym, trade_date)
        candidates['pe_ttm'] = candidates['symbol'].map(pe_map)
        candidates['peg'] = candidates['symbol'].map(peg_map)
        candidates['rd_to_revenue_ratio'] = candidates['symbol'].map(rd_ratio_map)

        out_cols = [
            'symbol', 'name', 'industry', 'list_date', 'listing_days',
            'latest_close', 'post_list_high', 'drawdown_from_post_list_high',
            'revenue_growth_since_list', 'gross_margin_growth_since_list',
            'pledge_ratio', 'goodwill_ratio', 'debt_ratio',
            'amp_20d', 'vol_ratio_20_60', 'avg_turnover_60d',
            'pe_ttm', 'peg', 'rd_to_revenue_ratio',
        ]
        result = candidates[out_cols].reset_index(drop=True)
        result.insert(0, 'trade_date', trade_date)

        if verbose:
            elapsed = time.time() - t0
            print(f'\n✅ 科创超跌选股完成: {len(result)} 只，耗时 {elapsed:.1f}s')
            print(f'{"="*70}\n')

        return result

    def preload_star_value_reversion_data(
        self,
        start_date: str,
        end_date: str,
        force: bool = False,
        verbose: bool = True,
    ) -> int:
        """科创超跌策略专用预加载：首次统一拉满全区间，后续月度选股只读缓存。"""
        start_date = start_date.replace('-', '')
        end_date = end_date.replace('-', '')

        all_stocks = self.sim.get_all_stocks_info(force_update=False)
        candidates = all_stocks.reset_index()[['symbol', 'list_date']].copy()
        candidates['symbol'] = candidates['symbol'].astype(str).str.zfill(6)
        candidates['list_date'] = pd.to_datetime(candidates['list_date'], errors='coerce')

        # 对于回测区间 [start_date, end_date]，潜在入池标的上市日期范围：
        # [start_date-730天, end_date-182天]
        s = pd.to_datetime(start_date)
        e = pd.to_datetime(end_date)
        list_min = s - pd.Timedelta(days=730)
        list_max = e - pd.Timedelta(days=182)

        candidates = candidates[
            candidates['symbol'].map(self._is_star_688_symbol)
            & candidates['list_date'].notna()
            & (candidates['list_date'] >= list_min)
            & (candidates['list_date'] <= list_max)
        ].copy()

        def _star_cache_ready() -> bool:
            """快速判断科创策略预加载缓存是否完整。"""
            md = self.dm.metadata or {}
            end_dt = pd.to_datetime(end_date)
            for _, row in candidates.iterrows():
                sym = str(row['symbol'])
                list_dt = pd.to_datetime(row['list_date'])
                k_start_dt = max(list_dt, list_min)

                # 1) 日线元数据覆盖检查（优先走 metadata，避免逐文件读取）
                meta = md.get(sym)
                if not meta:
                    return False
                last_date = meta.get('last_date')
                first_date = meta.get('first_date')
                if not last_date or not first_date:
                    return False
                try:
                    last_dt = pd.to_datetime(str(last_date))
                    first_dt = pd.to_datetime(str(first_date))
                except Exception:
                    return False
                if last_dt < end_dt or first_dt > k_start_dt:
                    return False

                # 2) 基础缓存文件覆盖检查
                if not self._cache_path(f'listing_status_{sym}').exists():
                    return False
                if not self._cache_path(f'star_vr_daily_basic_{sym}').exists():
                    return False
                if not self._cache_path(f'pledge_{sym}').exists():
                    return False
                if not self._cache_path(f'goodwill_ratio_{sym}').exists():
                    return False
                if not self._cache_path(f'star_vr_bs_debt_{sym}').exists():
                    return False

                # 3) 财务缓存覆盖检查
                fm_sym = self.fm.normalize_symbol(sym)
                if not (self.fm.cache_dir / f'income_{fm_sym}.parquet').exists():
                    return False
                if not (self.fm.cache_dir / f'fina_{fm_sym}.parquet').exists():
                    return False
            return True

        symbols = candidates['symbol'].tolist()
        total = len(symbols)
        if total == 0:
            if verbose:
                print('⚠️ 科创预加载无候选标的，跳过。')
            return 0

        if (not force) and _star_cache_ready():
            self._daily_data_local_only = True
            if verbose:
                print('\n✅ 科创预加载缓存已完整，跳过首次预加载。')
                print(f'   区间: {start_date} -> {end_date}, 标的数: {total}')
                print('   后续月度选股直接使用本地缓存。\n')
            return total

        if verbose:
            print('\n' + '=' * 70)
            print('🚚 科创策略首次预加载（统一全区间）')
            print(f'区间: {start_date} -> {end_date}, 标的数: {total}')
            print('=' * 70)

        for idx, row in candidates.reset_index(drop=True).iterrows():
            sym = str(row['symbol'])
            if verbose and (idx % 20 == 0):
                print(f'  预加载进度: {idx+1}/{total} ({(idx+1)/total*100:.0f}%)')

            list_dt = pd.to_datetime(row['list_date'])
            k_start = max(list_dt, list_min).strftime('%Y%m%d')

            # 基础与财务缓存
            _ = self._get_listing_status(sym)
            _ = self._get_star_daily_basic(sym)
            _ = self.get_pledge_ratio(sym)
            _ = self.get_goodwill_ratio(sym)
            _ = self._get_balance_sheet_for_debt(sym)
            _ = self.fm.get_income(sym)
            _ = self.fm.get_fina_indicator(sym)

            # 日线一次性拉到回测结束日，避免月度循环中反复增量补拉
            try:
                k = self.dm.get_stock_data(
                    symbol=sym,
                    start_date=k_start,
                    end_date=end_date,
                    force_update=force,
                    adjust='qfq',
                    allow_api=True,
                )
                if k is not None and not k.empty:
                    self._kline_cache[sym] = k
            except Exception:
                pass

        self._daily_data_local_only = True
        if verbose:
            print(f'✅ 科创预加载完成: {total}/{total} (100%)')
            print('   后续月度选股将优先使用本地缓存。\n')
        return total

    def _all_caches_exist(self, symbols: List[str]) -> bool:
        """检查所有核心缓存文件是否存在（任一缺失则返回 False）"""
        cache_keys = ['listing_status', 'daily_basic', 'pledge', 'bs', 'fina', 'income']
        for sym in symbols:
            fm_symbol = self.fm.normalize_symbol(sym)
            for prefix in cache_keys:
                p = self._cache_path(f'{prefix}_{sym}')
                # 注意：FundamentalsManager 的缓存路径不在 self.cache_dir 下
                # 所以我们只检查 StockSelector 自己管理的缓存
                if prefix in ('listing_status', 'daily_basic', 'pledge', 'bs'):
                    if not p.exists():
                        return False
                # fm 的缓存文件名不同，单独检查
                else:
                    fm_path = self.fm.cache_dir / f'{prefix}_{fm_symbol}.parquet'
                    if not fm_path.exists():
                        return False
        return True

    def _warmup_memory_caches(self, symbols: List[str]):
        """
        预热内存缓存：将磁盘上的持久化缓存一次性读入内存字典，
        后续 select() 中所有方法直接命中内存，零磁盘 I/O。
        """
        print(f'  🔥 预热内存缓存 ({len(symbols)} 只)...')
        for idx, sym in enumerate(symbols):
            if idx % 50 == 0:
                print(f'    {idx+1}/{len(symbols)} ({(idx+1)/len(symbols)*100:.0f}%)')
            # 触发方法内部的内存缓存填充（从磁盘读取一次）
            _ = self._get_listing_status(sym)   # 内存+磁盘 → 入 mem
            _ = self._get_daily_basic(sym)      # 内存+磁盘 → 入 mem
            _ = self.get_pledge_ratio(sym)      # 内存+磁盘 → 入 mem
            _ = self._get_balance_sheet(sym)    # 内存+磁盘 → 入 mem
            _ = self.fm.get_fina_indicator(sym) # fm 内存+磁盘 → 入 fm mem
            _ = self.fm.get_income(sym)         # fm 内存+磁盘 → 入 fm mem
            _ = self.get_goodwill_ratio(sym)
            _ = self._get_fina_static_features(sym, years=3)
            _ = self.is_last_fiscal_year_loss(sym, datetime.now().strftime('%Y%m%d'))
            _ = self._build_turnover_amplitude_series(sym, n_days=60)
        print(f'    {len(symbols)}/{len(symbols)} (100%) 完成')

    # ========================================================================
    # 批量预加载
    # ========================================================================
    def preload_all_data(self, sectors: Optional[List[str]] = None, force: bool = False, start_date: str = '20180101', end_date: Optional[str] = None) -> int:
        """
        一次性预加载目标行业所有股票的全部数据到磁盘缓存。
        后续 select() 将直接读取缓存，几乎不调 API。

        Parameters
        ----------
        sectors : List[str] or None
            行业列表，None 则使用 self.industries
        force : bool
            强制重新拉取（即使缓存已存在）
        start_date : str
            日线数据起始日期 (YYYYMMDD)，默认 '20180101'
        end_date : str or None
            日线数据结束日期 (YYYYMMDD)，None 则为当天

        Returns
        -------
        int : 预加载的股票数量
        """
        all_stocks = self.sim.get_all_stocks_info(force_update=False)
        candidates = all_stocks.reset_index()
        if sectors is not None:
            candidates = candidates[candidates['industry'].isin(sectors)]
        else:
            candidates = self._apply_industry_filter(candidates)
        bj_count = int(candidates['symbol'].map(self._is_bj_symbol).sum()) if not candidates.empty else 0
        if bj_count > 0:
            candidates = candidates[~candidates['symbol'].map(self._is_bj_symbol)].copy()
        symbols = (
            candidates['symbol']
            .map(self._normalize_symbol)
            .tolist()
        )
        total = len(symbols)

        # 快速检查：如果所有缓存都存在且 force=False，只预热内存，不调 API
        if not force and self._all_caches_exist(symbols):
            print(f'\n✅ 所有缓存已存在（{total} 只），跳过 API 预加载。')
            self._warmup_memory_caches(symbols)
            self._daily_data_local_only = True
            print(f'   如需强制更新请设置 preload_all_data(force=True) 或删除缓存目录。\n')
            return total

        print(f'\n🚀 预加载 {total} 只股票数据到本地缓存...')
        if bj_count > 0:
            print(f'   已在预加载前排除北交所股票: {bj_count} 只')

        # 1. ST/退市
        print('  [1/7] 预加载 ST/退市状态...')
        for idx, sym in enumerate(symbols):
            if idx % 20 == 0:
                print(f'    {idx+1}/{total} ({(idx+1)/total*100:.0f}%)')
            _ = self._get_listing_status(sym)
        print(f'    {total}/{total} (100%) 完成')

        # 2. daily_basic（市值 / PE）
        print('  [2/7] 预加载 daily_basic（市值/PE）...')
        for idx, sym in enumerate(symbols):
            if idx % 20 == 0:
                print(f'    {idx+1}/{total} ({(idx+1)/total*100:.0f}%)')
            _ = self._get_daily_basic(sym)
        print(f'    {total}/{total} (100%) 完成')

        # 3. 质押
        print('  [3/7] 预加载 股东质押比例...')
        for idx, sym in enumerate(symbols):
            if idx % 20 == 0:
                print(f'    {idx+1}/{total} ({(idx+1)/total*100:.0f}%)')
            _ = self.get_pledge_ratio(sym)
        print(f'    {total}/{total} (100%) 完成')

        # 4. 资产负债表（商誉）
        print('  [4/7] 预加载 资产负债表（商誉）...')
        for idx, sym in enumerate(symbols):
            if idx % 20 == 0:
                print(f'    {idx+1}/{total} ({(idx+1)/total*100:.0f}%)')
            _ = self._get_balance_sheet(sym)
        print(f'    {total}/{total} (100%) 完成')

        # 5. 财务指标（毛利率）
        print('  [5/7] 预加载 财务指标（毛利率）...')
        for idx, sym in enumerate(symbols):
            if idx % 20 == 0:
                print(f'    {idx+1}/{total} ({(idx+1)/total*100:.0f}%)')
            _ = self.fm.get_fina_indicator(sym)
            _ = self._get_fina_static_features(sym, years=3)
        print(f'    {total}/{total} (100%) 完成')

        # 6. 利润表（净利润）
        print('  [6/7] 预加载 利润表（净利润）...')
        for idx, sym in enumerate(symbols):
            if idx % 20 == 0:
                print(f'    {idx+1}/{total} ({(idx+1)/total*100:.0f}%)')
            _ = self.fm.get_income(sym)
            _ = self.is_last_fiscal_year_loss(sym, datetime.now().strftime('%Y%m%d'))
        print(f'    {total}/{total} (100%) 完成')

        # 7. 日线数据（成交额/振幅）
        print('  [7/7] 预加载 日线数据（成交额/振幅）...')
        for idx, sym in enumerate(symbols):
            if idx % 20 == 0:
                print(f'    {idx+1}/{total} ({(idx+1)/total*100:.0f}%)')
            try:
                df_kline = self.dm.get_stock_data(
                    symbol=sym,
                    start_date=start_date,
                    end_date=end_date or datetime.now().strftime('%Y%m%d'),
                    adjust='qfq',
                )
                if df_kline is None or df_kline.empty:
                    self._kline_no_data_symbols.add(sym)
                else:
                    self._kline_cache[sym] = df_kline
                    self._kline_no_data_symbols.discard(sym)
                    self._turnover_amp_cache.pop(sym, None)
                    self._build_turnover_amplitude_series(sym, n_days=60)
            except Exception:
                pass
        print(f'    {total}/{total} (100%) 完成')

        # 预加载完成后，选股阶段只读本地日线，避免再次触发 Tushare。
        self._daily_data_local_only = True

        print(f'\n✅ 预加载全部完成！共 {total} 只股票，后续选股将直接使用本地缓存。\n')
        return total

    # ========================================================================
    # 月度回测
    # ========================================================================
    def run_monthly(
        self,
        start_date: str = '20240101',
        end_date: str = '20260713',
        output_excel: Optional[str] = None,
        verbose: bool = True,
        preload: bool = True,
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
        preload : bool
            是否预加载数据

        Returns
        -------
        pd.DataFrame : 所有月份合并结果
        """
        start_date = start_date.replace('-', '')
        end_date = end_date.replace('-', '')

        # ── 预加载：一次性拉取全部数据到本地缓存 ──
        if preload:
            self.preload_all_data(sectors=self.industries, start_date=start_date, end_date=end_date)

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

        output_path = Path(output_excel)
        if not output_path.is_absolute() and output_path.parent == Path('.'):
            output_path = Path(__file__).resolve().parent / 'reports' / output_path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_summary.to_excel(writer, sheet_name='月度汇总', index=False)
            if not df_all.empty:
                df_all.to_excel(writer, sheet_name='详细结果', index=False)
                # 每月一个 sheet
                for month, group in df_all.groupby('month'):
                    sheet_name = str(month)[:7]  # 如 '2024-01'
                    group.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f'\n📁 结果已保存至: {output_path}')
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
    # sel = StockSelector(token=token, request_interval=0.4)

    # print('\n🔍 发现医药相关行业...')
    # industry_df = sel.discover_industries(keyword='医药|制药|医疗|生物|疫苗|CRO|中药|原料药|血液|体外|基因')
    # print(f'\n共发现 {len(industry_df)} 个相关行业')
    # print('请确认上述行业后，将其作为 industries 参数传入 StockSelector')
    # print('当前默认行业:')
    # for ind in sorted({x for v in FIVE_INDUSTRY_GROUPS.values() for x in v}):
    #     print(f'  - {ind}')

    # ====== Step B: 月度回测选股 ======
    print('\n' + '='*70)
    print('🚀 开始月度选股回测: 2026-01-03 → 2026-07-13')
    print('='*70)

    sel = StockSelector(
        token=token,
        # industries=sorted({x for v in FIVE_INDUSTRY_GROUPS.values() for x in v}),
        #industries=['医疗保健', '化学制药', '生物制药', '医药商业'],
        # 关注四个大行业，半导体，医药，金属，化工
        industries=['半导体', '元器件','医疗保健', '化学制药', '生物制药', '医药商业', '工业金属','小金属','贵金属','能源金属', '化学制品', '化学原料' ],
        request_interval=0.32,
    )

    # 生成带日期时间的报告文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report_dir = Path(__file__).resolve().parent / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f'stock_selection_results_{timestamp}.xlsx'

    df_results = sel.run_monthly(
        start_date='20260101',
        end_date='20260715',
        output_excel=str(report_path),
        verbose=True,
        #preload=False
    )

    if not df_results.empty:
        print(f'\n✅ 全部完成！共 {len(df_results)} 条选股记录')
    else:
        print('\n⚠️ 未选出任何股票')