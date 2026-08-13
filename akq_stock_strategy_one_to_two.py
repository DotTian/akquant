"""
一进二打板策略（纯日线版，akquant 框架）
数据源：Tushare（通过外部获取后传入 run_backtest）
策略逻辑：首板涨停 → 次日高开买入 → 未连板则尾盘卖出 / 止损
"""

import logging
import os
from typing import Tuple

import pandas as pd

from akquant import Strategy
from akquant import run_backtest


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OneToTwoStrategy(Strategy):
    def __init__(self,
                 high_open_range: Tuple[float, float] = (0.03, 0.07),
                 volume_ratio_min: float = 0.8,
                 stop_loss_pct: float = -0.05,
                 position_pct: float = 0.2,
                 max_positions: int = 5):
        """
        初始化策略参数
        """
        super().__init__()
        self.high_open_range = high_open_range
        self.volume_ratio_min = volume_ratio_min
        self.stop_loss_pct = stop_loss_pct
        self.position_pct = position_pct
        self.max_positions = max_positions

        # 持仓记录（symbol -> 买入价）
        self.buy_prices: dict = {}
        self.set_history_depth(10)

    def _get_limit_price(self, symbol: str, prev_close: float) -> float:
        """计算涨停价（区分板块：主板10%，双创20%，北交所30%）"""
        if symbol.startswith('688') or symbol.startswith('689') or symbol.startswith('300') or symbol.startswith('301'):
            rate = 1.2
        elif symbol.startswith('8'):
            rate = 1.3
        else:
            rate = 1.1
        return round(prev_close * rate, 2)

    def _is_limit_up(self, symbol: str, close: float, prev_close: float) -> bool:
        """判断是否涨停"""
        if prev_close is None or prev_close == 0:
            return False
        limit_price = self._get_limit_price(symbol, prev_close)
        return close >= limit_price - 0.01

    def _should_buy_today(self, bar) -> bool:
        """
        判断今天是否可以买入（需要昨日为涨停首板）
        """
        # 获取至少3根K线的收盘价（get_history从旧到新）
        closes = self.get_history(3, field="close")
        if closes is None or len(closes) < 3:
            return False

        yesterday_close = closes[-2]   # 昨天收盘
        day_before_close = closes[-3]  # 前天收盘

        # 1. 昨天必须是涨停（首板）
        if not self._is_limit_up(bar.symbol, yesterday_close, day_before_close):
            return False

        # 1.5 排除一字板：昨日开盘=涨停价 且 最低=最高
        highs = self.get_history(3, field="high")
        lows = self.get_history(3, field="low")
        opens = self.get_history(3, field="open")
        if opens is not None and highs is not None and lows is not None and len(opens) >= 2:
            yesterday_open = opens[-2]
            yesterday_high = highs[-2]
            yesterday_low = lows[-2]
            yesterday_limit = self._get_limit_price(bar.symbol, day_before_close)
            if yesterday_open >= yesterday_limit - 0.01 and yesterday_low == yesterday_high:
                return False

        # 2. 今天开盘高开幅度
        open_pct = (bar.open - yesterday_close) / yesterday_close
        if not (self.high_open_range[0] <= open_pct <= self.high_open_range[1]):
            return False

        # 3. 今天开盘不能涨停（否则买不到）
        limit_price = self._get_limit_price(bar.symbol, yesterday_close)
        if bar.open >= limit_price - 0.01:
            return False

        # 4. 成交量放大（今日成交量 vs 昨日成交量）
        volumes = self.get_history(2, field="volume")
        if volumes is not None and len(volumes) >= 2:
            if volumes[-1] < volumes[-2] * self.volume_ratio_min:
                return False

        return True

    def _should_exit(self, symbol: str, bar) -> Tuple[bool, str]:
        """
        判断是否卖出
        返回 (是否卖出, 原因)
        """
        buy_price = self.buy_prices.get(symbol)
        if buy_price is None:
            return False, ""

        # 1. 止损检查（当日最低价低于买入价的 95%）
        if bar.low <= buy_price * (1 + self.stop_loss_pct):
            return True, "止损(-5%)"

        # 2. 检查今日是否涨停
        closes = self.get_history(2, field="close")
        if closes is not None and len(closes) >= 2:
            yesterday_close = closes[-2]
            if self._is_limit_up(bar.symbol, bar.close, yesterday_close):
                return False, ""  # 涨停继续持有

        # 3. 未涨停，尾盘卖出
        return True, "未连板(尾盘卖出)"

    def on_bar(self, bar):
        """
        每根日线触发一次
        """
        symbol = bar.symbol
        position = self.get_position(symbol)
        current_price = bar.close

        # ========== 有持仓时：检查卖出 ==========
        if position > 0:
            should_exit, reason = self._should_exit(symbol, bar)
            if should_exit:
                self.close_position(symbol)
                profit = (current_price - self.buy_prices[symbol]) / self.buy_prices[symbol]
                logger.info(f"[卖出] {symbol} 价格:{current_price:.2f}, 原因:{reason}, "
                            f"盈亏:{profit*100:.2f}%")
                self.buy_prices.pop(symbol, None)
            return

        # ========== 空仓时：检查买入 ==========
        if len(self.buy_prices) >= self.max_positions:
            return

        if not self._should_buy_today(bar):
            return

        # 满足条件，以开盘价买入
        buy_price = bar.open
        self.order_target_percent(self.position_pct, symbol, price=buy_price)
        self.buy_prices[symbol] = buy_price
        logger.info(f"[买入] {symbol} 价格:{buy_price:.2f}, 日期:{bar.datetime}")

    def on_stop(self):
        """策略结束，清理"""
        logger.info("一进二策略运行结束")


def load_a_share_universe(
    token: str,
    start_date: str,
    end_date: str,
    data_dir: str = 'stock_info_unadj',
) -> list[str]:
    """加载回测区间内有效的沪深股票，排除北交所和 ST 股票。"""
    from akq_module_stockinfo import StockInfoManager

    manager = StockInfoManager(
        token=token,
        data_dir=data_dir,
        request_interval=1.2,
        auto_update=False,
    )
    stocks = manager.get_all_stocks_info(list_status='ALL')
    if stocks.empty:
        raise RuntimeError('股票基础信息为空')

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    filtered = stocks.copy()
    filtered = filtered[filtered['exchange'].isin(['SSE', 'SZSE'])]
    filtered = filtered[
        ~filtered['name'].fillna('').str.contains('ST', case=False, regex=False)
    ]

    list_dates = pd.to_datetime(filtered['list_date'], errors='coerce')
    delist_dates = pd.to_datetime(filtered['delist_date'], errors='coerce')
    filtered = filtered[(list_dates <= end) & (delist_dates.isna() | (delist_dates >= start))]

    symbols = sorted(filtered.index.astype(str).str.zfill(6).unique().tolist())
    if not symbols:
        raise RuntimeError('指定回测区间内没有符合条件的沪深股票')

    logger.info('全市场股票池: %s 只', len(symbols))
    return symbols


def load_market_data(
    symbols: list[str],
    start_date: str,
    end_date: str,
    data_dir: str = 'tsdata_unadj',
) -> dict[str, pd.DataFrame]:
    """优先读取本地缓存，仅为缺失标的补拉不复权日线数据。"""
    from akq_module_tusharedatamanager import TushareStockDataManager

    token = os.getenv('TUSHARE_TOKEN')
    if not token:
        raise RuntimeError('请先设置环境变量 TUSHARE_TOKEN')

    manager = TushareStockDataManager(
        token=token,
        data_dir=data_dir,
        request_interval=1.5,
    )
    raw = manager.get_multiple_stocks(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        adjust=None,
        delay_between=0.0,
        allow_api=False,
    )
    missing_symbols = [
        symbol for symbol, df in raw.items() if df is None or df.empty
    ]
    if missing_symbols:
        logger.info('本地缓存缺失 %s 只，开始补拉不复权行情', len(missing_symbols))
        fetched = manager.get_multiple_stocks(
            symbols=missing_symbols,
            start_date=start_date,
            end_date=end_date,
            adjust=None,
            delay_between=0.0,
            allow_api=True,
        )
        raw.update(fetched)

    data = {
        str(symbol): df.sort_index()
        for symbol, df in raw.items()
        if df is not None and not df.empty
    }
    if not data:
        raise RuntimeError('没有加载到可用行情数据')

    logger.info('可用于回测的标的: %s 只', len(data))
    return data


if __name__ == "__main__":
    # ================== 数据准备 ==================
    TOKEN = os.getenv('TUSHARE_TOKEN')
    if not TOKEN:
        raise ValueError("请设置环境变量 TUSHARE_TOKEN")

    start_date = "20260101"
    end_date = "20260812"
    stock_list = load_a_share_universe(
        token=TOKEN,
        start_date=start_date,
        end_date=end_date,
    )
    stock_data_dict = load_market_data(
        symbols=stock_list,
        start_date=start_date,
        end_date=end_date,
    )
    symbols = sorted(stock_data_dict)

    # ================== 运行回测 ==================
    result = run_backtest(
        strategy=OneToTwoStrategy,
        data=stock_data_dict,
        symbols=symbols,
        initial_cash=1000000.0,
        commission_rate=0.0005,   # 万分之五
        slippage=0.0002,          # 万分之二
        t_plus_one=True,
    )

    print("\n=== 回测结果 ===")
    print(result.metrics_df)

    # 保存报告
    report_dir = "./reports"
    os.makedirs(report_dir, exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"{report_dir}/one_to_two_market_{timestamp}.html"
    result.report(
        filename=report_path,
        title="一进二策略全市场组合报告",
        market_data=stock_data_dict,
        include_trade_kline=True
    )
    print(f"\n报告已保存至: {report_path}")