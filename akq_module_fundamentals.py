"""
基本面数据模块 - 获取股票毛利率、研发占比、亏损状态、ST/退市、PEG 等

依赖 Tushare，通过缓存方式减少重复请求
"""

import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from pathlib import Path

import pandas as pd
import tushare as ts

logger = logging.getLogger(__name__)


class FundamentalsManager:
    """
    基本面数据管理器，负责从 Tushare 获取并缓存基本面指标

    用法:
        fm = FundamentalsManager(token='your_token', cache_dir='fundamentals_cache')
        metrics = fm.get_all_metrics(symbol='600519', trade_date='20251231')
    """

    # 字段分类
    FIELDS_INCOME = ['end_date', 'n_income_attr_p', 'total_revenue', 'rd_exp']
    FIELDS_FINA_INDICATOR = ['end_date', 'grossprofit_margin', 'q_profit_yoy', 'netprofit_margin']

    def __init__(self,
                 token: str,
                 cache_dir: str = "fundamentals_cache",
                 request_interval: float = 1.5):
        self.token = token
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.request_interval = request_interval

        ts.set_token(token)
        self._pro = ts.pro_api()

        # 内存缓存（避免同一次运行重复请求）
        self._basic_info_cache: Dict[str, dict] = {}

    # ─── 工具方法 ───
    @staticmethod
    def to_ts_code(symbol: str) -> str:
        code = str(symbol).zfill(6)
        if code.startswith(('688', '600', '601', '603', '605')):
            return f"{code}.SH"
        else:
            return f"{code}.SZ"

    def _load_cache(self, key: str) -> Optional[pd.DataFrame]:
        path = self.cache_dir / f"{key}.parquet"
        if not path.exists():
            return None
        # 缓存有效期 24 小时
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        if datetime.now() - mtime > timedelta(hours=24):
            return None
        try:
            return pd.read_parquet(path)
        except Exception:
            return None

    def _save_cache(self, key: str, df: pd.DataFrame):
        df.to_parquet(self.cache_dir / f"{key}.parquet", index=False)

    # ─── 获取利润表数据 (含净利润/营收/研发费用) ───
    def get_income(self, symbol: str) -> Optional[pd.DataFrame]:
        ts_code = self.to_ts_code(symbol)
        cache_key = f"income_{symbol}"
        df = self._load_cache(cache_key)
        if df is not None:
            return df
        time.sleep(self.request_interval)
        try:
            df = self._pro.income(
                ts_code=ts_code,
                start_date='20180101',
                end_date=datetime.now().strftime('%Y%m%d'),
                fields=','.join(self.FIELDS_INCOME),
                report_type='1'  # 只取合并报表，避免重复行
            )
            if df is not None and not df.empty:
                df['end_date'] = pd.to_datetime(df['end_date'])
                self._save_cache(cache_key, df)
                return df
        except Exception as e:
            logger.warning(f"获取 {symbol} income 失败: {e}")
        return None

    # ─── 获取财务指标数据 (含毛利率/净利率/净利润增长率) ───
    def get_fina_indicator(self, symbol: str) -> Optional[pd.DataFrame]:
        ts_code = self.to_ts_code(symbol)
        cache_key = f"fina_{symbol}"
        df = self._load_cache(cache_key)
        if df is not None:
            return df
        time.sleep(self.request_interval)
        try:
            df = self._pro.fina_indicator(
                ts_code=ts_code,
                start_date='20180101',
                end_date=datetime.now().strftime('%Y%m%d'),
                fields=','.join(self.FIELDS_FINA_INDICATOR)
            )
            if df is not None and not df.empty:
                df['end_date'] = pd.to_datetime(df['end_date'])
                self._save_cache(cache_key, df)
                return df
        except Exception as e:
            logger.warning(f"获取 {symbol} fina_indicator 失败: {e}")
        return None

    # ─── 获取 ST/退市状态 ───
    def get_listing_status(self, symbol: str) -> Dict[str, str]:
        """
        返回 {'is_st': bool, 'st_reason': str, 'is_delisted': bool, 'delist_date': str}
        """
        ts_code = self.to_ts_code(symbol)
        result = {'is_st': False, 'st_reason': '', 'is_delisted': False, 'delist_date': ''}

        # 统一使用 ts_code 作为缓存 key
        cache_key = ts_code
        if cache_key in self._basic_info_cache:
            cached = self._basic_info_cache[cache_key]
            if 'listing_status' in cached:
                return cached['listing_status']

        try:
            # ST 检查: namechange 接口（更宽松的 ST 匹配）
            nc = self._pro.namechange(ts_code=ts_code)
            if nc is not None and not nc.empty:
                st_rows = nc[nc['name'].str.contains(r'ST', regex=True)]
                if not st_rows.empty:
                    result['is_st'] = True
                    latest = st_rows.sort_values('ann_date', ascending=False).iloc[0]
                    result['st_reason'] = str(latest.get('reason', ''))

            # 退市检查: stock_basic
            sb = self._pro.stock_basic(ts_code=ts_code, fields='ts_code,name,delist_date')
            if sb is not None and not sb.empty:
                dl_date = sb.iloc[0]['delist_date']
                if pd.notna(dl_date):
                    result['is_delisted'] = True
                    result['delist_date'] = str(dl_date)
        except Exception as e:
            logger.warning(f"获取 {symbol} 上市状态失败: {e}")

        self._basic_info_cache.setdefault(cache_key, {})['listing_status'] = result
        return result

    # ─── 计算 PEG ───
    def calculate_peg(self, symbol: str, trade_date: str) -> tuple[Optional[float], Optional[float]]:
        """
        PEG = PE_TTM / 净利润同比增长率

        优先使用 TTM 滚动净利润增长率，回退到单季同比 (q_profit_yoy)
        """
        ts_code = self.to_ts_code(symbol)

        # 获取 PE_TTM
        time.sleep(self.request_interval)
        try:
            df_db = self._pro.daily_basic(
                ts_code=ts_code,
                trade_date=trade_date.replace('-', ''),
                fields='ts_code,trade_date,pe_ttm'
            )
            if df_db is None or df_db.empty:
                return None, None
            pe_ttm = df_db.iloc[0].get('pe_ttm')
            if not pe_ttm or pe_ttm <= 0:
                return None, None
        except Exception as e:
            logger.warning(f"获取 {symbol} PE_TTM 失败: {e}")
            return None, None

        # 净利润增长率
        growth = self._calc_ttm_growth(symbol)
        if growth is None:
            # 回退：单季同比
            df_fina = self.get_fina_indicator(symbol)
            if df_fina is not None and not df_fina.empty:
                latest = df_fina.sort_values('end_date', ascending=False).iloc[0]
                g = latest.get('q_profit_yoy')
                if g is not None and pd.notna(g):
                    growth = float(g)

        if not growth or growth <= 0:
            return None, None

        try:
            # Tushare 的 growth 字段已是百分比数值（如 15 表示 15%），无需再 ×100
            peg = float(pe_ttm) / float(growth)
            return round(peg, 2), round(growth, 2)
        except (ZeroDivisionError, ValueError):
            return None, None

    def _calc_ttm_growth(self, symbol: str) -> Optional[float]:
        """基于 income 表计算滚动12个月净利润同比增长率"""
        df = self.get_income(symbol)
        if df is None or df.empty:
            return None

        if 'n_income_attr_p' not in df.columns:
            return None

        df_ni = df[['end_date', 'n_income_attr_p']].dropna().sort_values('end_date')
        if len(df_ni) < 8:
            return None

        ttm_ni = df_ni.tail(4)['n_income_attr_p'].sum()
        prev_ttm_ni = df_ni.iloc[-8:-4]['n_income_attr_p'].sum()

        if prev_ttm_ni <= 0:
            return None

        return (ttm_ni - prev_ttm_ni) / abs(prev_ttm_ni) * 100

    # ─── 一站式获取所有基本面指标 ───
    def get_all_metrics(self, symbol: str, trade_date: str) -> Dict:
        """
        获取单只股票的全部基本面指标，返回 dict:

        {
            'gross_margin': float or None,       # 毛利率 (%)
            'rd_ratio': float or None,            # 研发费用占营收比 (%)
            'net_margin': float or None,          # 净利率 (%)
            'is_loss': bool,                      # 最近期是否亏损
            'is_st': bool,                        # 是否 ST
            'st_reason': str,                     # ST 原因
            'is_delisted': bool,                  # 是否已退市
            'delist_date': str,                   # 退市日期
            'peg': float or None,                 # PEG
            'pe_ttm': float or None,              # PE(TTM)
            'profit_growth': float or None        # 净利润同比增长率 (%)
        }
        """
        result = {
            'gross_margin': None,
            'rd_ratio': None,
            'net_margin': None,
            'is_loss': False,
            'is_st': False,
            'st_reason': '',
            'is_delisted': False,
            'delist_date': '',
            'peg': None,
            'pe_ttm': None,
            'profit_growth': None
        }

        # 财务指标
        df_fina = self.get_fina_indicator(symbol)
        if df_fina is not None and not df_fina.empty:
            latest = df_fina.sort_values('end_date', ascending=False).iloc[0]
            result['gross_margin'] = float(latest['grossprofit_margin']) if pd.notna(latest.get('grossprofit_margin')) else None
            result['net_margin'] = float(latest['netprofit_margin']) if pd.notna(latest.get('netprofit_margin')) else None
            result['profit_growth'] = float(latest['q_profit_yoy']) if pd.notna(latest.get('q_profit_yoy')) else None

        # 利润表 (营收/研发/净利润)
        df_income = self.get_income(symbol)
        if df_income is not None and not df_income.empty:
            latest_i = df_income.sort_values('end_date', ascending=False).iloc[0]
            rev = latest_i.get('total_revenue')
            rd = latest_i.get('rd_exp')
            if rev and rd and rev > 0:
                result['rd_ratio'] = round(float(rd) / float(rev) * 100, 2)

            ni = latest_i.get('n_income_attr_p')
            if ni is not None and pd.notna(ni) and float(ni) < 0:
                result['is_loss'] = True

        # ST/退市状态
        status = self.get_listing_status(symbol)
        result.update(status)

        # PEG
        result['peg'] = self.calculate_peg(symbol, trade_date)

        return result


# ─── 简单自测 ───
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    token = os.getenv('TUSHARE_TOKEN')
    if not token:
        print("请设置 TUSHARE_TOKEN 环境变量")
        exit(1)

    fm = FundamentalsManager(token=token, cache_dir='fundamentals_cache', request_interval=0.5)

    # 测试几只不同行业的股票
    test_stocks = [
        ('600519', '20251231'),   # 贵州茅台 - 消费
        ('688981', '20251231'),   # 中芯国际 - 半导体
        ('300750', '20251231'),   # 宁德时代 - 新能源
        ('600868', '20251231'),   # ST 梅雁 (如有 ST)
    ]

    for sym, date in test_stocks:
        print(f"\n{'='*60}")
        print(f"📊 {sym} ({date})")
        m = fm.get_all_metrics(sym, date)
        for k, v in m.items():
            print(f"  {k}: {v}")
