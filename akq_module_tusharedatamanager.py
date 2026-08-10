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
        self._trade_cal_cache = {}
        
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

    def _infer_exchange(self, symbol: str) -> Optional[str]:
        """根据股票代码推断交易所代码。"""
        code = self._normalize_symbol(symbol)
        if code.startswith(('43', '83', '87', '88', '92')):
            return 'BSE'
        if code.startswith(('688', '600', '601', '603', '605')):
            return 'SSE'
        return 'SZSE'

    def _align_trade_window(self, symbol: str, start_date: str, end_date: str) -> tuple[str, str]:
        """将请求窗口对齐到交易日；若区间内没有交易日，则返回空区间。"""
        start_date = start_date.replace('-', '')
        end_date = end_date.replace('-', '')
        if not hasattr(self, 'pro') or self.pro is None:
            return start_date, end_date

        exchange = self._infer_exchange(symbol)
        cache_key = (exchange, start_date, end_date)
        cached = self._trade_cal_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            self._wait_if_needed()
            cal_df = self.pro.trade_cal(
                exchange=exchange,
                start_date=start_date,
                end_date=end_date,
                is_open='1',
            )
        except Exception as e:
            logger.debug(f'获取交易日历失败 {symbol}: {e}')
            return start_date, end_date

        if cal_df is None or cal_df.empty:
            result = (start_date, end_date)
            self._trade_cal_cache[cache_key] = result
            return result

        trade_dates = pd.to_datetime(cal_df['cal_date']).dt.strftime('%Y%m%d').tolist()
        if not trade_dates:
            result = (start_date, end_date)
            self._trade_cal_cache[cache_key] = result
            return result

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        trade_dt = pd.to_datetime(trade_dates)
        valid = trade_dt[(trade_dt >= start_dt) & (trade_dt <= end_dt)]
        if valid.empty:
            result = ('', '')
            self._trade_cal_cache[cache_key] = result
            return result

        aligned_start = valid.min().strftime('%Y%m%d')
        aligned_end = valid.max().strftime('%Y%m%d')
        result = (aligned_start, aligned_end)
        self._trade_cal_cache[cache_key] = result
        return result
    
    @staticmethod
    def _is_empty_data_error(err: Exception) -> bool:
        msg = str(err)
        return ('空数据' in msg) or ('empty' in msg.lower())

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """规范化股票代码，去掉已有市场后缀后再转成 6 位数字代码。"""
        if symbol is None:
            return ''
        code = str(symbol).strip().upper()
        if '.' in code:
            code = code.split('.', 1)[0]
        return code.zfill(6)

    def _convert_symbol(self, symbol: str) -> str:
        """
        将股票代码转换为 Tushare 格式。
        兼容输入：'688131'、'688131.SH'、'920017.BJ'、'000300.SH'。
        """
        if symbol is None:
            return ""

        raw_symbol = str(symbol).strip().upper()
        if "." in raw_symbol:
            exchange = raw_symbol.split(".", 1)[1]
            if exchange in {"SH", "SZ", "BJ"}:
                return raw_symbol

        code = self._normalize_symbol(raw_symbol)

        # 指数/基准类代码保留常见交易所后缀，避免 000300.SH 被错误映射成 000300.SZ
        if code in {"000001", "000300", "000688", "000905", "000016", "000036", "000050", "000061", "000063"}:
            return f"{code}.SH"
        if code in {"399001", "399005", "399006", "399300", "399905", "399550"}:
            return f"{code}.SZ"

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
    
    def _is_index_symbol(self, symbol: str) -> bool:
        """判断是否为指数/基准类代码。"""
        if symbol is None:
            return False
        code = self._normalize_symbol(str(symbol))
        return code in {
            '000001', '000300', '000688', '000905', '000016', '000036', '000050',
            '000061', '000063', '399001', '399005', '399006', '399300', '399905',
            '399550', '399001', '399006', '399300'
        }

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
                if self._is_index_symbol(symbol):
                    params = dict(
                        ts_code=ts_code,
                        start_date=start_date,
                        end_date=end_date,
                        freq='D',
                        asset='I',
                    )
                    if adjust == 'qfq':
                        params['adj'] = 'qfq'
                    elif adjust == 'hfq':
                        params['adj'] = 'hfq'
                    df = ts.pro_bar(**params)
                elif adjust == 'qfq':
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
                
                return df
                
            except Exception as e:
                if self._is_empty_data_error(e):
                    # 空数据在部分标的/区间属于常见情况，避免刷屏告警。
                    logger.debug(
                        f"获取空数据 (尝试 {attempt+1}/{max_retries}): {symbol}, 错误: {e}"
                    )
                else:
                    logger.warning(f"获取失败 (尝试 {attempt+1}/{max_retries}): {symbol}, 错误: {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 1  # 1, 2, 3 秒
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    if self._is_empty_data_error(e):
                        logger.debug(f"获取 {symbol} 数据最终为空")
                    else:
                        logger.error(f"获取 {symbol} 数据最终失败")
                    raise
    
    def get_stock_data(self, symbol: str, start_date: str, end_date: str,
                        force_update: bool = False, adjust: str = 'qfq',
                        allow_api: bool = True) -> pd.DataFrame:
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
        allow_api : bool
            是否允许访问 Tushare。False 时仅使用本地缓存，不做任何补拉。
    
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

        # 缓存命中快速路径：区间已被本地数据覆盖时，直接裁剪返回，避免 trade_cal/API。
        if (not force_update) and file_path.exists():
            df_existing = pd.read_parquet(file_path)
            if not isinstance(df_existing.index, pd.DatetimeIndex):
                df_existing.index = pd.to_datetime(df_existing.index)
            first_existing = df_existing.index.min()
            last_existing = df_existing.index.max()

            if first_existing <= target_start and last_existing >= target_end:
                logger.debug(
                    f"缓存命中 {symbol}: {target_start.strftime('%Y-%m-%d')} - "
                    f"{target_end.strftime('%Y-%m-%d')}"
                )
                return self._trim_by_date(df_existing, start_date_clean, end_date_clean)

            logger.debug(f"读取已有 {symbol} 数据: {len(df_existing)} 条, "
                         f"日期范围: {first_existing.strftime('%Y-%m-%d')} - {last_existing.strftime('%Y-%m-%d')}")

            # 本地模式：区间不全时也不补拉，直接返回现有区间的裁剪结果。
            if not allow_api:
                return self._trim_by_date(df_existing, start_date_clean, end_date_clean)

            aligned_start, aligned_end = self._align_trade_window(symbol, start_date_clean, end_date_clean)
            if aligned_start == '' and aligned_end == '':
                logger.info(f'{symbol} 请求区间中无交易日，跳过获取')
                return pd.DataFrame()
            start_date_clean = aligned_start or start_date_clean
            end_date_clean = aligned_end or end_date_clean
            target_start = pd.to_datetime(start_date_clean)
            target_end = pd.to_datetime(end_date_clean)

            need_save = False

            # 4. 检查是否需要补充早期数据（start_date 早于现有数据的最早日期）
            if first_existing > target_start:
                missing_start = target_start.strftime("%Y%m%d")
                missing_end = (first_existing - pd.Timedelta(days=1)).strftime("%Y%m%d")
                logger.debug(f"需要补充早期数据: {missing_start} 至 {missing_end}")
                try:
                    if pd.to_datetime(missing_start) > pd.to_datetime(missing_end):
                        df_early = pd.DataFrame()
                    else:
                        # 补数据阶段快速失败，避免长时间重试阻塞批处理。
                        df_early = self._fetch_from_tushare(
                            symbol,
                            missing_start,
                            missing_end,
                            adjust=adjust,
                            max_retries=1,
                        )
                    if not df_early.empty:
                        df_existing = pd.concat([df_early, df_existing])
                        df_existing = df_existing[~df_existing.index.duplicated(keep='last')]
                        df_existing = df_existing.sort_index()
                        need_save = True
                        logger.debug(f"早期数据补充完成: 新增 {len(df_early)} 条")
                except Exception as e:
                    if self._is_empty_data_error(e):
                        logger.debug(f"补充早期数据为空: {symbol}，继续使用现有数据")
                    else:
                        logger.warning(f"补充早期数据失败: {e}，继续使用现有数据")

            # 5. 检查是否需要补充最新数据（end_date 晚于现有数据的最晚日期）
            if last_existing < target_end:
                missing_start = (last_existing + pd.Timedelta(days=1)).strftime("%Y%m%d")
                missing_end = target_end.strftime("%Y%m%d")
                logger.debug(f"需要补充最新数据: {missing_start} 至 {missing_end}")
                try:
                    if pd.to_datetime(missing_start) > pd.to_datetime(missing_end):
                        df_late = pd.DataFrame()
                    else:
                        # 补数据阶段快速失败，避免对空窗口做 3 次重试。
                        df_late = self._fetch_from_tushare(
                            symbol,
                            missing_start,
                            missing_end,
                            adjust=adjust,
                            max_retries=1,
                        )
                    if not df_late.empty:
                        df_existing = pd.concat([df_existing, df_late])
                        df_existing = df_existing[~df_existing.index.duplicated(keep='last')]
                        df_existing = df_existing.sort_index()
                        need_save = True
                        logger.debug(f"最新数据补充完成: 新增 {len(df_late)} 条")
                except Exception as e:
                    if self._is_empty_data_error(e):
                        logger.debug(f"补充最新数据为空: {symbol}，返回现有数据范围")
                    else:
                        logger.warning(f"补充最新数据失败: {e}，返回现有数据范围")

            # 6. 如有补充，保存更新后的数据
            if need_save:
                self._save_data(df_existing, file_path, symbol)

            # 7. 按请求日期裁剪最终返回
            return self._trim_by_date(df_existing, start_date_clean, end_date_clean)

        if not allow_api:
            return pd.DataFrame()

        aligned_start, aligned_end = self._align_trade_window(symbol, start_date_clean, end_date_clean)
        if aligned_start == '' and aligned_end == '':
            logger.info(f'{symbol} 请求区间中无交易日，跳过获取')
            return pd.DataFrame()
        start_date_clean = aligned_start or start_date_clean
        end_date_clean = aligned_end or end_date_clean
        target_start = pd.to_datetime(start_date_clean)
        target_end = pd.to_datetime(end_date_clean)

        # 1. 强制更新：重新获取全部数据
        if force_update:
            logger.info(f"强制更新 {symbol} 全部数据...")
            df_new = self._fetch_from_tushare(symbol, start_date_clean, end_date_clean, adjust=adjust)
            self._save_data(df_new, file_path, symbol)
            return df_new

        # 2. 文件不存在：首次获取全部数据
        logger.info(f"首次获取 {symbol} 数据...")
        df_new = self._fetch_from_tushare(symbol, start_date_clean, end_date_clean, adjust=adjust)
        self._save_data(df_new, file_path, symbol)
        return df_new
    
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