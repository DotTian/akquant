# -*- coding: utf-8 -*-
# akq_stock_mainaccount_datamanager.py
# 股票数据管理器，负责动态获取和更新股票数据

from random import random

import pandas as pd
import akshare as ak
from pathlib import Path
import time
import datetime
import random

class StockDataManagerak:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def get_stock_data(self, symbol, start_date, end_date, force_update=False):
        """
        动态获取股票数据
        - 默认读取已有 parquet 文件
        - 如果文件不存在或需要更新，则增量获取
        """
        # 文件路径
        file_path = self.data_dir / f"{symbol}.parquet"
        
        # 1. 如果强制更新，重新获取全部数据
        if force_update:
            print(f"强制更新 {symbol} 全部数据...")
            df_new = self._fetch_from_source(symbol, start_date, end_date)
            self._save_data(df_new, file_path)
            return df_new
        
        # 2. 如果文件不存在，获取全部数据
        if not file_path.exists():
            print(f"首次获取 {symbol} 数据...")
            df_new = self._fetch_from_source(symbol, start_date, end_date)
            self._save_data(df_new, file_path)
            return df_new
        
        # 3. 文件存在，读取现有数据
        df_existing = pd.read_parquet(file_path)
        print(f"读取已有 {symbol} 数据: {len(df_existing)} 条, 最新日期: {df_existing['日期'].max()}")
        
        # 4. 检查是否需要更新
        latest_date = pd.to_datetime(df_existing['日期'].max())
        target_end_date = pd.to_datetime(end_date)
        
        # 如果已有数据已经包含到目标日期，无需更新
        if latest_date >= target_end_date and not force_update:
            print(f"{symbol} 数据已是最新，无需更新")
            return df_existing
        
        # 5. 增量获取新数据
        new_start_date = (latest_date + datetime.timedelta(days=1)).strftime("%Y%m%d")
        if new_start_date <= end_date:
            print(f"增量获取 {symbol} 数据: {new_start_date} 至 {end_date}")
            try:
                df_incremental = self._fetch_from_source(symbol, new_start_date, end_date)
                
                # 去重合并
                df_combined = pd.concat([df_existing, df_incremental], ignore_index=True)
                df_combined = df_combined.drop_duplicates(subset=['日期'], keep='last')
                df_combined = df_combined.sort_values('日期')
                
                self._save_data(df_combined, file_path)
                print(f"更新完成: 新增 {len(df_incremental)} 条，总计 {len(df_combined)} 条")
                return df_combined
            except Exception as e:
                print(f"增量更新失败: {e}，返回已有数据")
                return df_existing
        else:
            print(f"{symbol} 数据已是最新")
            return df_existing
    
    def _fetch_from_source(self, symbol, start_date, end_date, max_retries=3):
        """从数据源获取数据（带重试机制）"""
        for attempt in range(max_retries):
            try:
                # 每次请求前等待 3-5 秒（关键！）
                wait_time = random.uniform(3, 5)
                print(f"等待 {wait_time:.1f} 秒后发起请求...")
                time.sleep(wait_time)

                # 随机延迟避免被封
                # time.sleep(1)
                
                df = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq"
                )

                # df = ak.stock_zh_a_hist_tx(
                #     symbol=symbol,
                #     #period="daily",
                #     start_date=start_date,
                #     end_date=end_date,
                #     adjust="qfq"
                # )
                
                if df is not None and not df.empty:
                    # 格式化日期列
                    df['日期'] = pd.to_datetime(df['日期'])
                    return df
                else:
                    raise ValueError("返回数据为空")
                    
            except Exception as e:
                print(f"获取失败 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    raise
    
    def _save_data(self, df, file_path):
        """保存数据到 parquet"""
        df.to_parquet(file_path, index=False)
        print(f"数据已保存: {file_path}")

# # 使用示例
# manager = StockDataManager()

# # 自动处理：有则读，无则取，有更新则增量获取
# df = manager.get_stock_data(
#     symbol="688131",
#     start_date="20250101",
#     end_date="20260531"
# )