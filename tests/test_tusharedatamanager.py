from types import SimpleNamespace

import pandas as pd

from akq_module_tusharedatamanager import TushareStockDataManager


def test_align_trading_window_skips_non_trading_days():
    mgr = TushareStockDataManager.__new__(TushareStockDataManager)
    mgr._trade_cal_cache = {}
    mgr.pro = SimpleNamespace(
        trade_cal=lambda exchange, start_date, end_date, **kwargs: pd.DataFrame(
            {
                'cal_date': ['20240101', '20240103', '20240104'],
                'is_open': [1, 1, 1],
            }
        )
    )

    start, end = mgr._align_trade_window('920300', '20240102', '20240103')

    assert start == '20240103'
    assert end == '20240103'
