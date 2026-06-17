import os

import tushare as ts
import time
import random
import pandas as pd
from pathlib import Path    

class StockDataManagerts:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Tushare 初始化 - 从环境变量获取 token
        token = os.getenv('TUSHARE_TOKEN')
        print('TUSHARE_TOKEN set:', bool(token))
        ts.set_token(token)

        self.pro = ts.pro_api()
    
    def _fetch_from_source(self, symbol: str, start_date: str, end_date: str, max_retries: int = 3):
        """
        使用 Tushare 获取数据（替代原来的 AKShare）
        """
        # 请求间隔（避免被限制）
        time.sleep(random.uniform(1, 3))
        
        for attempt in range(max_retries):
            try:
                ts_code = self._convert_symbol(symbol)
                
                # 获取日线数据（前复权）
                # 注意：根据你的付费等级，调整 fields 和复权方式
                df = self.pro.daily(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                    adj='qfq'  # 如果付费等级支持，可直接用此参数
                )
                
                if df is None or df.empty:
                    raise ValueError(f"未获取到数据: {symbol}")
                
                # 转换为 akquant 标准格式
                df = self._to_akquant_format(df)
                return df
                
            except Exception as e:
                print(f"获取失败 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))  # 指数退避
                else:
                    raise
    
    def _convert_symbol(self, symbol: str) -> str:
        """代码格式转换"""
        code = str(symbol).zfill(6)
        if code.startswith(('688', '600', '601', '603', '605')):
            return f"{code}.SH"
        return f"{code}.SZ"
    
    def _to_akquant_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换为 akquant 标准格式"""
        mapping = {'trade_date': 'date', 'vol': 'volume'}
        df = df.rename(columns=mapping)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        return df