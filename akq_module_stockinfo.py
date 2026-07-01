"""
股票基本信息管理模块
- 获取全部A股基本信息（代码、名称、行业、上市日期等）
- 提供中文名称查询API
- 本地Parquet缓存，支持增量更新

使用方法:
    from akq_module_stockinfo import StockInfoManager
    manager = StockInfoManager(token='your_token')
    name = manager.get_stock_name('300724')
    print(name)  # 捷佳伟创
"""

import json
import os
import time
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, List, Union

import pandas as pd
import tushare as ts

logger = logging.getLogger(__name__)


class StockInfoManager:
    """
    A股股票基本信息管理器
    - 使用 tushare stock_basic 接口获取所有股票基本信息
    - 缓存为 Parquet 文件，支持每日自动检查更新
    - 提供 get_stock_name() / get_stock_info() 等API
    """

    def __init__(self,
                 token: str,
                 data_dir: str = "stock_info",
                 request_interval: float = 1.2,
                 auto_update: bool = True):
        """
        初始化

        Parameters:
        -----------
        token : str
            Tushare Pro Token
        data_dir : str
            缓存数据目录
        request_interval : float
            请求间隔(秒)
        auto_update : bool
            初始化时是否自动检查并更新缓存（若缓存超过24小时）
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.request_interval = request_interval
        self.last_request_time = 0

        # 初始化 tushare
        ts.set_token(token)
        self.pro = ts.pro_api()

        # 缓存文件路径
        self.cache_file = self.data_dir / "all_stocks_info.parquet"
        self.meta_file = self.data_dir / "metadata.json"
        self.metadata = self._load_metadata()

        # 缓存 DataFrame（懒加载）
        self._df: Optional[pd.DataFrame] = None

        # 如果启用自动更新且缓存过期，则更新
        if auto_update and self._should_update():
            logger.info("缓存已过期，自动更新股票基本信息...")
            self.refresh_cache()

        logger.info(f"StockInfoManager 初始化完成，缓存文件: {self.cache_file}")

    # ==================== 元数据管理 ====================
    def _load_metadata(self) -> dict:
        if self.meta_file.exists():
            return json.loads(self.meta_file.read_text(encoding='utf-8'))
        return {}

    def _save_metadata(self):
        self.meta_file.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding='utf-8')

    def _should_update(self) -> bool:
        """判断缓存是否需要更新（超过24小时）"""
        last_update = self.metadata.get("last_update")
        if not last_update:
            return True
        last_dt = datetime.fromisoformat(last_update)
        return (datetime.now() - last_dt).total_seconds() > 86400  # 24h

    # ==================== 请求控制 ====================
    def _wait_if_needed(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)
        self.last_request_time = time.time()

    # ==================== 核心数据获取 ====================
    def _fetch_all_stocks(self, max_retries: int = 3) -> pd.DataFrame:
        """
        从 tushare 获取全部A股基本信息（一次性获取，不分页）
        """
        for attempt in range(max_retries):
            try:
                self._wait_if_needed()
                logger.info("正在从 tushare 获取全部股票基本信息...")

                # stock_basic 接口：fields 可按需选择
                df = self.pro.stock_basic(
                    fields="ts_code,symbol,name,area,industry,market,list_date,is_hs,curr_type,exchange,delist_date"
                )

                if df is None or df.empty:
                    raise ValueError("tushare stock_basic 返回空数据")

                # 处理 symbol（去掉 .SH/.SZ 后缀）
                df['symbol'] = df['ts_code'].str.replace(r'\.(SH|SZ)$', '', regex=True)

                # 处理 list_date 为日期类型
                if 'list_date' in df.columns:
                    df['list_date'] = pd.to_datetime(df['list_date'], errors='coerce')

                # symbol 作为索引（保持和主数据一致）
                df = df.set_index('symbol')

                logger.info(f"成功获取 {len(df)} 只股票基本信息")
                time.sleep(0.5)  # 额外等待
                return df

            except Exception as e:
                logger.warning(f"获取失败 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 3)
                else:
                    logger.error("获取全部股票基本信息最终失败")
                    raise

    # ==================== 缓存管理 ====================
    def refresh_cache(self) -> pd.DataFrame:
        """
        强制从 tushare 刷新全部缓存并保存到本地
        """
        df = self._fetch_all_stocks()
        self._save_cache(df)
        self._df = df
        return df

    def _save_cache(self, df: pd.DataFrame):
        """保存到 Parquet 并更新元数据"""
        df.to_parquet(self.cache_file, index=True)
        self.metadata["last_update"] = datetime.now().isoformat()
        self.metadata["total_stocks"] = len(df)
        self._save_metadata()
        logger.info(f"股票基本信息缓存已保存: {len(df)} 只股票")

    def _load_cache(self) -> pd.DataFrame:
        """从本地 Parquet 加载缓存"""
        if self.cache_file.exists():
            df = pd.read_parquet(self.cache_file)
            logger.debug(f"从缓存加载 {len(df)} 只股票基本信息")
            return df
        return pd.DataFrame()

    def _get_df(self) -> pd.DataFrame:
        """获取 DataFrame（懒加载）"""
        if self._df is None:
            if self.cache_file.exists():
                self._df = self._load_cache()
            else:
                logger.info("本地无缓存，从 tushare 获取全量数据...")
                self._df = self.refresh_cache()
        return self._df

    # ==================== 公开 API ====================
    def get_all_stocks_info(self, force_update: bool = False) -> pd.DataFrame:
        """
        获取所有股票基本信息 DataFrame

        Parameters:
        -----------
        force_update : bool
            是否强制从 tushare 更新
        """
        if force_update:
            return self.refresh_cache()
        return self._get_df()

    def get_stock_name(self, symbol: Union[str, int]) -> Optional[str]:
        """
        根据股票代码（纯数字）获取中文名称

        Parameters:
        -----------
        symbol : str or int
            股票代码，如 "300724" 或 300724

        Returns:
        --------
        Optional[str] : 中文名称，未找到返回 None
        """
        symbol = str(symbol).zfill(6)
        df = self._get_df()
        if symbol in df.index:
            return df.loc[symbol, 'name']
        logger.warning(f"未找到股票 {symbol} 的基本信息")
        return None

    def get_stock_info(self, symbol: Union[str, int]) -> Optional[Dict]:
        """
        获取单只股票的详细信息（dict）

        Parameters:
        -----------
        symbol : str or int
            股票代码

        Returns:
        --------
        Optional[Dict] : 包含 ts_code, name, industry, list_date 等字段
        """
        symbol = str(symbol).zfill(6)
        df = self._get_df()
        if symbol in df.index:
            row = df.loc[symbol]
            return row.to_dict()
        logger.warning(f"未找到股票 {symbol} 的基本信息")
        return None

    def search_stock(self, keyword: str) -> pd.DataFrame:
        """
        根据关键词搜索股票（模糊匹配 code 或 name）

        Parameters:
        -----------
        keyword : str
            搜索关键词（可以是代码片段或名称片段）

        Returns:
        --------
        pd.DataFrame : 匹配的股票信息
        """
        df = self._get_df()
        # 匹配代码或名称
        mask = df.index.str.contains(keyword, case=False) | df['name'].str.contains(keyword, case=False)
        result = df[mask].copy()
        # 重置索引以便显示 symbol 列
        result = result.reset_index()
        return result

    def get_stocks_by_industry(self, industry: str) -> pd.DataFrame:
        """
        根据行业获取股票列表

        Parameters:
        -----------
        industry : str
            行业名称（如 "专用机械"）

        Returns:
        --------
        pd.DataFrame : 该行业的所有股票
        """
        df = self._get_df()
        return df[df['industry'] == industry].copy()

    def get_stocks_by_area(self, area: str) -> pd.DataFrame:
        """
        根据地域获取股票列表

        Parameters:
        -----------
        area : str
            地区名称（如 "深圳"）

        Returns:
        --------
        pd.DataFrame : 该地区的所有股票
        """
        df = self._get_df()
        return df[df['area'] == area].copy()

    def get_cache_info(self) -> dict:
        """获取缓存状态信息"""
        info = {
            "cache_exists": self.cache_file.exists(),
            "last_update": self.metadata.get("last_update"),
            "total_stocks": self.metadata.get("total_stocks", 0),
            "cache_size_mb": round(self.cache_file.stat().st_size / (1024 * 1024), 2) if self.cache_file.exists() else 0,
        }
        return info

    def clear_cache(self):
        """清除本地缓存"""
        if self.cache_file.exists():
            self.cache_file.unlink()
        if self.meta_file.exists():
            self.meta_file.unlink()
        self._df = None
        self.metadata = {}
        logger.info("已清除全部股票基本信息缓存")


# ==================== 使用示例 ====================
if __name__ == "__main__":
    import os

    mytoken = os.getenv('TUSHARE_TOKEN')
    if not mytoken:
        print("请设置 TUSHARE_TOKEN 环境变量")
        exit(1)

    manager = StockInfoManager(
        token=mytoken,
        data_dir="stock_info",
        request_interval=1.2,
        auto_update=False  # 首次手动更新
    )

    # 首次获取（会请求 tushare 并缓存）
    df = manager.get_all_stocks_info(force_update=False)
    print(f"共 {len(df)} 只股票")
    print(df[['name', 'industry', 'list_date']].head(10))

    # 测试查询中文名称
    print("\n查询股票名称:")
    print("300724 ->", manager.get_stock_name("300724"))
    print("000001 ->", manager.get_stock_name("000001"))
    print("600519 ->", manager.get_stock_name("600519"))

    # 测试搜索
    print("\n搜索 '捷佳':")
    print(manager.search_stock("捷佳"))

    # 测试按行业查询
    print("\n行业 '专用机械' 的股票数量:", len(manager.get_stocks_by_industry("专用机械")))

    # 查看缓存信息
    print("\n缓存信息:", manager.get_cache_info())