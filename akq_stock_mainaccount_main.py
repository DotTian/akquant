# -*- coding: utf-8 -*-
# akq_stock_mainaccount_main.py
# 主账户策略示例，展示如何在主账户中使用策略进行交易


from typing import Any, List

import akquant as aq
import akshare as ak
import numpy as np
import pandas as pd

import os
from pathlib import Path
import time
from datetime import datetime
import webbrowser
import tushare as ts

from akq_stock_mainaccount_strategy import stock_MainAccountStrategy
#from akq_stock_mainaccount_datamanager_ak import StockDataManagerak # akshare数据管理器
#from akq_stock_mainaccount_datamanager_ts import StockDataManagerts # tushare数据管理器-临时不带文件存储
from akq_module_tusharedatamanager import TushareStockDataManager # tushare数据管理器-带文件存储

if __name__ == "__main__":
    # 配置参数
    #SYMBOL = "688131"  # 标的代码 (如 "600000" 或 "688131")
    SYMBOL = "688131"  # 标的代码 (如 "600000" 或 "688131")
    START_DATE = "20240101"
    END_DATE = "20260527" #结束日期，建议设置前一交易日避免重复获取， tushare收盘后才会发布当天的数据
    DATA_DIR = "tsdata"  # 数据存储目录
    file_path = f"{DATA_DIR}/{SYMBOL}.parquet"
    
    # manager = StockDataManager(data_dir=DATA_DIR)
    # df = manager.get_stock_data(symbol=SYMBOL, start_date=START_DATE, end_date=END_DATE)

    mytoken = os.getenv('TUSHARE_TOKEN')
    print('TUSHARE_TOKEN set:', bool(mytoken))

    manager = TushareStockDataManager(
        token= mytoken,  # 替换为你的实际 Token # type: ignore
        data_dir=DATA_DIR,
        request_interval=1.5  # 请求间隔 1.5 秒
    )
    df = manager.get_stock_data(symbol=SYMBOL, start_date=START_DATE, end_date=END_DATE)

    # 创建主账户策略实例
    strategy = stock_MainAccountStrategy()

    # 运行回测
    #result = aq.run_backtest(strategy=strategy, data=df)
    result = aq.run_backtest(
        strategy=strategy,
        data=df,
        
        # A股关键配置
        initial_cash=1_000_000,
        commission_rate=0.0003,      # 万三佣金
        stamp_tax_rate=0.001,        # 千一印花税（卖出收取）
        min_commission=5.0,          # 最低5元佣金
        t_plus_one=True,             # ✅ A股是T+1
        lot_size=100,                # ✅ A股最小交易单位100股
        
        # 撮合规则（常用：收盘信号，下根bar开盘成交）
        fill_policy={
            "price_basis": "open",   # 用开盘价成交
            "bar_offset": 1,         # 下根bar
            #"temporal": "same_cycle" #成交已经明确要跨到下一根 bar 了，所以 same_cycle 被引擎忽略
        },
        
        # 基础风控
        risk_config={
            "max_position_pct": 0.95,   # 单票最大仓位95%
            #"max_drawdown": 0.3,        # 回撤30%停止 akquant的回撤风控目前不太适合A股，建议在策略中自行实现回撤控制逻辑
        },
        
        # 时间范围
        start_time=START_DATE,
        end_time=END_DATE,
        timezone="Asia/Shanghai",
        
        # 预热期（如果策略需要计算均线）
        warmup_period=20,
        
        show_progress=True
    )

    # 输出回测结果
    # print(result)

    # 保存报告
    # 获取当前时间戳（格式：20250114_153045）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 确保 report 目录存在
    report_dir = "./reports"
    os.makedirs(report_dir, exist_ok=True)

    # 生成文件名
    report_filename = f"akq_stock_mainaccount_strategy_report_{SYMBOL}_{timestamp}.html"
    report_path = os.path.join(report_dir, report_filename)

    result.report(
        filename=report_path,
        title=f"策略报告 ({SYMBOL})",
        market_data=df,
        include_trade_kline=True
    )
    # 保存报告后自动在浏览器打开
    webbrowser.open(f"file://{os.path.abspath(report_path)}")  