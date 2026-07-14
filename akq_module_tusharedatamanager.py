import pandas as pd
import tushare as ts
import numpy as np
from pathlib import Path
import time
import random
from datetime import datetime, timedelta
from typing import Optional, Union, List, Dict
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TushareStockDataManager:
    """
    Tushare 数据管理器（支持动态获取 + Parquet 缓存）
    接口与 AKShare 版本保持一致，方便切换
    """
    
    def __init__(self, token: str, data_dir: str = "stock_data", request_interval: float = 1.5):
        """
        初始化 Tushare 数据管理器
        
        Parameters:
        -----------
        token : str
            Tushare Pro 的 API Token
        data_dir : str
            数据存储目录
        request_interval : float
            请求间隔（秒），避免触发频率限制
        """
        import tushare as ts
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.request_interval = request_interval
        self.last_request_time = 0
        
        # 初始化 Tushare
        ts.set_token(token)
        self.pro = ts.pro_api()
        
        # 缓存元数据文件
        self.meta_file = self.data_dir / "metadata.json"
        self.metadata = self._load_metadata()
        
        logger.info(f"Tushare 数据管理器初始化完成，数据目录: {self.data_dir}")
    
    def _load_metadata(self) -> dict:
        """加载元数据（记录每只股票的最新日期等信息）"""
        if self.meta_file.exists():
            import json
            with open(self.meta_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """保存元数据"""
        import json
        with open(self.meta_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
    
    def _wait_if_needed(self):
        """控制请求频率"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_interval:
            wait_time = self.request_interval - elapsed
            time.sleep(wait_time)
        self.last_request_time = time.time()
    
    def _convert_symbol(self, symbol: str) -> str:
        """
        将纯数字代码转换为 Tushare 格式
        示例：'688131' -> '688131.SH'
              '000001' -> '000001.SZ'
              '920017' -> '920017.BJ'
        """
        code = str(symbol).zfill(6)
        
        # 北交所（43/83/87/88/92 开头）
        if code.startswith(('43', '83', '87', '88', '92')):
            return f"{code}.BJ"
        # 上海市场：688(科创板)、600/601/603/605(主板)
        if code.startswith(('688', '600', '601', '603', '605')):
            return f"{code}.SH"
        # 深圳市场：000/001/002(主板)、003(中小板)、300/301(创业板)
        return f"{code}.SZ"
    
    # def _convert_to_akquant_format(self, df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     将 Tushare 返回的 DataFrame 转换为 akquant 标准格式
    #     - 列名转为中文（与 AKShare 保持一致）
    #     - 日期设为索引并排序
    #     """
    #     if df is None or df.empty:
    #         return pd.DataFrame()
        
    #     # 列名映射（Tushare -> 中文标准格式）
    #     column_mapping = {
    #         'trade_date': '日期',
    #         'open': '开盘',
    #         'high': '最高',
    #         'low': '最低',
    #         'close': '收盘',
    #         'vol': '成交量',
    #         'amount': '成交额'
    #     }
        
    #     # 只映射存在的列
    #     existing_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
    #     df = df.rename(columns=existing_mapping)
        
    #     # 处理日期列
    #     if '日期' in df.columns:
    #         df['日期'] = pd.to_datetime(df['日期'])
    #         df = df.set_index('日期')
        
    #     # 按时间升序排序（Tushare 默认是倒序）
    #     df = df.sort_index()
        
    #     return df
    def _convert_to_akquant_format(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        将 Tushare 返回的 DataFrame 转换为 akquant 标准格式
        - 列名转为英文小写
        - 日期设为索引并排序
        - ts_code 转为 symbol（去掉后缀）
        """
        if df is None or df.empty:
            return pd.DataFrame()
        
        df = df.copy()
        
        # 1. 列名映射（Tushare -> 英文标准格式）
        column_mapping = {
            'trade_date': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'vol': 'volume',
            'amount': 'amount',
            'ts_code': 'symbol'
        }
        
        # 只映射存在的列
        existing_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=existing_mapping)
        
        # 2. 处理 symbol 列（去掉 .SH/.SZ/.BJ 后缀）
        if 'symbol' in df.columns:
            df['symbol'] = df['symbol'].str.replace(r'\.(SH|SZ|BJ)$', '', regex=True)
        elif symbol:
            df['symbol'] = symbol
        
        # 3. 处理日期列
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        
        # 4. 按时间升序排序（Tushare 默认是倒序）
        df = df.sort_index()
        
        # 5. 确保数据类型正确
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 6. 验证必需列
        required = ['open', 'high', 'low', 'close', 'volume', 'symbol']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"缺少必需列: {missing}")
        
        return df
    
    def _fetch_from_tushare(self, symbol: str, start_date: str, end_date: str, 
                            adjust: str = 'qfq', max_retries: int = 3) -> pd.DataFrame:
        """
        从 Tushare 获取数据（带重试机制）
        
        Parameters:
        -----------
        symbol : str
            股票代码（纯数字）
        start_date : str
            开始日期，格式 YYYYMMDD 或 YYYY-MM-DD
        end_date : str
            结束日期，格式 YYYYMMDD 或 YYYY-MM-DD
        adjust : str
            复权类型：'qfq'(前复权), 'hfq'(后复权), None(不复权)
        max_retries : int
            最大重试次数
        """
        # 标准化日期格式
        start_date = start_date.replace('-', '')
        end_date = end_date.replace('-', '')
        
        ts_code = self._convert_symbol(symbol)
        
        for attempt in range(max_retries):
            try:
                # 请求频率控制
                self._wait_if_needed()
                
                logger.debug(f"请求 Tushare: {ts_code}, {start_date} - {end_date}")
                
                # 根据复权类型选择不同的接口
                if adjust == 'qfq':
                    # 使用 pro_bar 接口获取前复权数据（推荐）
                    df = ts.pro_bar(
                        ts_code=ts_code,
                        start_date=start_date,
                        end_date=end_date,
                        adj='qfq',
                        freq='D'
                    )
                elif adjust == 'hfq':
                    df = ts.pro_bar(
                        ts_code=ts_code,
                        start_date=start_date,
                        end_date=end_date,
                        adj='hfq',
                        freq='D'
                    )
                else:
                    # 不复权，使用 daily 接口
                    df = self.pro.daily(
                        ts_code=ts_code,
                        start_date=start_date,
                        end_date=end_date
                    )
                
                if df is None or df.empty:
                    raise ValueError(f"Tushare 返回空数据: {ts_code}")
                
                # 转换为标准格式
                df = self._convert_to_akquant_format(df)
                
                if df.empty:
                    raise ValueError(f"数据转换后为空: {ts_code}")
                
                logger.info(f"成功获取 {symbol} 数据: {len(df)} 条, "
                           f"日期范围 {df.index.min().strftime('%Y-%m-%d')} - {df.index.max().strftime('%Y-%m-%d')}")
                
                # 额外添加短暂延迟，避免连续请求
                time.sleep(0.5)
                
                return df
                
            except Exception as e:
                logger.warning(f"获取失败 (尝试 {attempt+1}/{max_retries}): {symbol}, 错误: {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 1  # 1, 2, 3 秒
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"获取 {symbol} 数据最终失败")
                    raise
    
    def get_stock_data(self, symbol: str, start_date: str, end_date: str, 
                        force_update: bool = False, adjust: str = 'qfq') -> pd.DataFrame:
        """
        动态获取股票数据（核心方法）
        - 优先读取本地 Parquet 缓存
        - 如果数据不存在或需要更新，则增量获取
        - 保存到 Parquet 文件
    
        Parameters:
        -----------
        symbol : str
            股票代码（纯数字，如 '688131'）
        start_date : str
            开始日期，格式 YYYYMMDD 或 YYYY-MM-DD
        end_date : str
            结束日期，格式 YYYYMMDD 或 YYYY-MM-DD
        force_update : bool
            是否强制更新全部数据
        adjust : str
            复权类型：'qfq'(前复权), 'hfq'(后复权), None(不复权)
    
        Returns:
        --------
        pd.DataFrame : 股票数据，索引为日期，列为中文
        """
        # 标准化日期
        start_date_clean = start_date.replace('-', '')
        end_date_clean = end_date.replace('-', '')
        target_start = pd.to_datetime(start_date_clean)
        target_end = pd.to_datetime(end_date_clean)

        file_path = self.data_dir / f"{symbol}.parquet"

        # 1. 强制更新：重新获取全部数据
        if force_update:
            logger.info(f"强制更新 {symbol} 全部数据...")
            df_new = self._fetch_from_tushare(symbol, start_date_clean, end_date_clean, adjust=adjust)
            self._save_data(df_new, file_path, symbol)
            return df_new

        # 2. 文件不存在：首次获取全部数据
        if not file_path.exists():
            logger.info(f"首次获取 {symbol} 数据...")
            df_new = self._fetch_from_tushare(symbol, start_date_clean, end_date_clean, adjust=adjust)
            self._save_data(df_new, file_path, symbol)
            return df_new

        # 3. 文件存在：读取已有数据
        df_existing = pd.read_parquet(file_path)
        first_existing = df_existing.index.min()
        last_existing = df_existing.index.max()
        logger.info(f"读取已有 {symbol} 数据: {len(df_existing)} 条, "
                    f"日期范围: {first_existing.strftime('%Y-%m-%d')} - {last_existing.strftime('%Y-%m-%d')}")

        need_save = False

        # 4. 检查是否需要补充早期数据（start_date 早于现有数据的最早日期）
        if first_existing > target_start:
            missing_start = target_start.strftime("%Y%m%d")
            missing_end = (first_existing - pd.Timedelta(days=1)).strftime("%Y%m%d")
            logger.info(f"需要补充早期数据: {missing_start} 至 {missing_end}")
            try:
                df_early = self._fetch_from_tushare(symbol, missing_start, missing_end, adjust=adjust)
                if not df_early.empty:
                    df_existing = pd.concat([df_early, df_existing])
                    df_existing = df_existing[~df_existing.index.duplicated(keep='last')]
                    df_existing = df_existing.sort_index()
                    need_save = True
                    logger.info(f"早期数据补充完成: 新增 {len(df_early)} 条")
            except Exception as e:
                logger.warning(f"补充早期数据失败: {e}，继续使用现有数据")

        # 5. 检查是否需要补充最新数据（end_date 晚于现有数据的最晚日期）
        if last_existing < target_end:
            missing_start = (last_existing + pd.Timedelta(days=1)).strftime("%Y%m%d")
            missing_end = target_end.strftime("%Y%m%d")
            logger.info(f"需要补充最新数据: {missing_start} 至 {missing_end}")
            try:
                df_late = self._fetch_from_tushare(symbol, missing_start, missing_end, adjust=adjust)
                if not df_late.empty:
                    df_existing = pd.concat([df_existing, df_late])
                    df_existing = df_existing[~df_existing.index.duplicated(keep='last')]
                    df_existing = df_existing.sort_index()
                    need_save = True
                    logger.info(f"最新数据补充完成: 新增 {len(df_late)} 条")
            except Exception as e:
                logger.warning(f"补充最新数据失败: {e}，返回现有数据范围")

        # 6. 如有补充，保存更新后的数据
        if need_save:
            self._save_data(df_existing, file_path, symbol)

        # 7. 按请求日期裁剪最终返回
        return self._trim_by_date(df_existing, start_date_clean, end_date_clean)
    
    def _save_data(self, df: pd.DataFrame, file_path: Path, symbol: str):
        """保存数据到 Parquet 并更新元数据"""
        df.to_parquet(file_path, index=True)
        
        # 更新元数据
        self.metadata[symbol] = {
            'last_date': df.index.max().strftime("%Y-%m-%d"),
            'first_date': df.index.min().strftime("%Y-%m-%d"),
            'total_rows': len(df),
            'last_update': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'data_source': 'tushare'
        }
        self._save_metadata()
        logger.info(f"数据已保存: {file_path}")
    
    def _trim_by_date(self, df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """按日期范围裁剪数据"""
        start = pd.to_datetime(start_date.replace('-', ''))
        end = pd.to_datetime(end_date.replace('-', ''))
        return df[(df.index >= start) & (df.index <= end)]
    
    def get_multiple_stocks(self, symbols: List[str], start_date: str, end_date: str,
                           force_update: bool = False, adjust: str = 'qfq',
                           delay_between: float = 2.0) -> Dict[str, pd.DataFrame]:
        """
        批量获取多只股票数据
        
        Parameters:
        -----------
        symbols : List[str]
            股票代码列表
        start_date : str
            开始日期
        end_date : str
            结束日期
        force_update : bool
            是否强制更新
        adjust : str
            复权类型
        delay_between : float
            每只股票之间的延迟（秒）
        
        Returns:
        --------
        Dict[str, pd.DataFrame] : 股票代码到数据的映射
        """
        results = {}
        
        for i, symbol in enumerate(symbols):
            logger.info(f"\n处理 ({i+1}/{len(symbols)}): {symbol}")
            
            try:
                df = self.get_stock_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    force_update=force_update,
                    adjust=adjust
                )
                results[symbol] = df
                
                # 股票间延迟
                if i < len(symbols) - 1:
                    logger.info(f"等待 {delay_between} 秒后继续...")
                    time.sleep(delay_between)
                    
            except Exception as e:
                logger.error(f"处理 {symbol} 失败: {e}")
                results[symbol] = pd.DataFrame()
        
        return results
    
    def update_all_to_latest(self, symbols: List[str], days_back: int = 5) -> Dict[str, pd.DataFrame]:
        """
        将指定股票更新到最新交易日
        
        Parameters:
        -----------
        symbols : List[str]
            股票代码列表
        days_back : int
            往回追溯的天数（用于计算开始日期）
        
        Returns:
        --------
        Dict[str, pd.DataFrame] : 更新后的数据
        """
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
        
        return self.get_multiple_stocks(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            force_update=False
        )
    
    def get_data_info(self, symbol: str) -> dict:
        """获取某只股票的缓存信息"""
        return self.metadata.get(symbol, {})
    
    def clear_cache(self, symbol: Optional[str] = None):
        """
        清除缓存
        
        Parameters:
        -----------
        symbol : Optional[str]
            股票代码，如果为 None 则清除所有缓存
        """
        if symbol:
            file_path = self.data_dir / f"{symbol}.parquet"
            if file_path.exists():
                file_path.unlink()
            if symbol in self.metadata:
                del self.metadata[symbol]
            logger.info(f"已清除 {symbol} 的缓存")
        else:
            # 清除所有 parquet 文件
            for f in self.data_dir.glob("*.parquet"):
                f.unlink()
            self.metadata = {}
            logger.info("已清除所有缓存")
        
        self._save_metadata()


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 1. 初始化（替换为你的 Tushare Token）
    manager = TushareStockDataManager(
        token='your_tushare_token_here',  # 替换为你的实际 Token
        data_dir='tushare_stock_data',
        request_interval=1.5  # 请求间隔 1.5 秒
    )
    
    # 2. 获取单只股票数据（自动处理缓存和增量更新）
    df = manager.get_stock_data(
        symbol='688131',
        start_date='20250101',
        end_date='20260531',
        force_update=False,  # 首次会自动下载，后续只增量
        adjust='qfq'  # 前复权
    )
    print(f"\n获取到 {len(df)} 条数据")
    print(df.head())
    
    # 3. 批量获取多只股票
    symbols = ['688131', '000001', '600000']
    results = manager.get_multiple_stocks(
        symbols=symbols,
        start_date='20250101',
        end_date='20260531',
        delay_between=3.0  # 每只股票间隔3秒
    )
    
    # 4. 查看缓存信息
    for symbol in symbols:
        info = manager.get_data_info(symbol)
        print(f"{symbol}: {info}")
    
    # 5. 增量更新到最新
    # manager.update_all_to_latest(symbols, days_back=10)
    
    # 6. 可选：清除缓存重新获取
    # manager.clear_cache('688131')