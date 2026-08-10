from types import SimpleNamespace

import pandas as pd

from akq_module_tusharedatamanager import TushareStockDataManager
import akq_module_tusharedatamanager as manager_module


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


def test_fetch_from_tushare_uses_index_asset_for_benchmark_symbols(monkeypatch):
    mgr = TushareStockDataManager.__new__(TushareStockDataManager)
    mgr._trade_cal_cache = {}
    mgr.pro = SimpleNamespace()
    mgr._wait_if_needed = lambda: None
    mgr._convert_to_akquant_format = lambda df, symbol=None: df.copy()

    captured = {}

    def fake_pro_bar(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame(
            {
                'ts_code': ['000300.SH'],
                'trade_date': ['20220104'],
                'open': [4000.0],
                'high': [4010.0],
                'low': [3990.0],
                'close': [4005.0],
                'pre_close': [3990.0],
                'change': [15.0],
                'pct_chg': [0.38],
                'vol': [100000.0],
                'amount': [100000000.0],
            }
        )

    monkeypatch.setattr(manager_module.ts, 'pro_bar', fake_pro_bar)

    df = mgr._fetch_from_tushare('000300.SH', '20220101', '20220104', adjust='qfq')

    assert not df.empty
    assert captured['asset'] == 'I'
    assert captured['ts_code'] == '000300.SH'
