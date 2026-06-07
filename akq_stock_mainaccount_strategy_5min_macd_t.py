import os

import pandas as pd
import numpy as np
from akquant import Strategy
from akquant import run_backtest
from datetime import datetime as dt

from akq_module_tusharedatamanager import TushareStockDataManager


class FiveMinMacdTStrategy(Strategy):
    """
    5分钟线 MACD 做T策略

    核心逻辑：
    1. 底仓持有：确保有底仓可做T（没有底仓时先建立底仓）
    2. MACD金叉 + 放量 → 买入（做正T）
    3. MACD死叉 + 缩量 → 卖出（T出）
    4.  单次T仓位控制在底仓的 30%~50%，快进快出
    5. 严格止损：单笔T亏损超过 0.5% 立即止损
    6. 日内限制：单日最多做 2~3 次T
    """

    def __init__(
        self,
        fast_period: int = 12,          # MACD 快线周期
        slow_period: int = 26,          # MACD 慢线周期
        signal_period: int = 9,        # MACD 信号线周期
        base_position_ratio: float = 0.3,  # 底仓占总资金比例
        t_ratio: float = 0.4,         # 每次做T使用的仓位占底仓的比例
        stop_loss_pct: float = 0.005, # T止损比例 (0.5%)
        take_profit_pct: float = 0.015,# T止盈比例 (1.5%)
        max_daily_trades: int = 3,   # 单日最大做T次数
        volume_ratio_threshold: float = 1.2, # 放量阈值（相对于过去20根5分钟均量）
    ):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.base_position_ratio = base_position_ratio
        self.t_ratio = t_ratio
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_daily_trades = max_daily_trades
        self.volume_ratio_threshold = volume_ratio_threshold

        # 状态记录
        self.daily_trade_count = 0          # 当日已做T次数
        self.last_trade_date = None          # 上次交易日
        self.t_entry_price = 0              # T买入价格
        self.t_entry_time = None            # T买入时间
        self.has_base_position = False      # 是否有底仓
        self.base_position_cost = 0          # 底仓成本

        # 确保有足够历史数据计算MACD和成交量均线
        required_depth = max(slow_period + signal_period, 30) + 20
        self.set_history_depth(required_depth)

        print(f"FiveMinMacdTStrategy 已初始化")
        print(f"  MACD参数: ({fast_period}, {slow_period}, {signal_period})")
        print(f"  底仓比例: {base_position_ratio:.0%}, T仓位比例: {t_ratio:.0%}")
        print(f"  止盈: {take_profit_pct:.2%}, 止损: {stop_loss_pct:.2%}")

    def on_start(self):
        """策略启动"""
        if hasattr(self.ctx, 'symbols') and self.ctx.symbols:
            self.trade_symbol = self.ctx.symbols[0]
            self.subscribe(self.trade_symbol)

    # ──────────────── MACD 计算 ────────────────
    def _calc_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """计算 EMA"""
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        alpha = 2.0 / (period + 1)
        result = np.empty_like(data, dtype=float)
        # SMA 作为初始值
        result[:period - 1] = np.nan
        result[period - 1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        return result

    def _calc_macd(self) -> dict:
        """计算当前 MACD 指标"""
        closes = self.get_history(self.slow_period + self.signal_period + 5, field="close")
        if closes is None or len(closes) < self.slow_period + self.signal_period:
            return {}

        ema_fast = self._calc_ema(closes, self.fast_period)
        ema_slow = self._calc_ema(closes, self.slow_period)

        dif = ema_fast - ema_slow
        dea = self._calc_ema(dif, self.signal_period)
        histogram = 2 * (dif - dea)  # MACD 柱

        return {
            'dif': dif,
            'dea': dea,
            'histogram': histogram,
            'dif_last': dif[-1],
            'dea_last': dea[-1],
            'hist_last': histogram[-1],
            'dif_prev': dif[-2],
            'dea_prev': dea[-2],
            'hist_prev': histogram[-2],
        }

    # ──────────────── 量能判断 ────────────────
    def _is_volume_surge(self) -> bool:
        """判断当前5分钟是否放量"""
        volumes = self.get_history(25, field="volume")
        if volumes is None or len(volumes) < 21:
            return False
        current_vol = volumes[-1]
        avg_vol_20 = np.mean(volumes[-21:-1])
        if avg_vol_20 == 0:
            return False
        return current_vol / avg_vol_20 >= self.volume_ratio_threshold

    def _is_volume_shrink(self) -> bool:
        """判断当前5分钟是否缩量"""
        volumes = self.get_history(25, field="volume")
        if volumes is None or len(volumes) < 21:
            return False
        current_vol = volumes[-1]
        avg_vol_20 = np.mean(volumes[-21:-1])
        if avg_vol_20 == 0:
            return False
        return current_vol / avg_vol_20 < 0.8

    # ──────────────── 信号判断 ────────────────
    def _is_golden_cross(self) -> bool:
        """MACD 金叉：DIF 上穿 DEA（当前 > 前一）"""
        macd = self._calc_macd()
        if not macd:
            return False
        # 前一周期 DIF <= DEA，当前 DIF > DEA
        return (
            macd['dif_prev'] <= macd['dea_prev']
            and macd['dif_last'] > macd['dea_last']
        )

    def _is_dead_cross(self) -> bool:
        """MACD 死叉：DIF 下穿 DEA（当前 < 前一）"""
        macd = self._calc_macd()
        if not macd:
            return False
        return (
            macd['dif_prev'] >= macd['dea_prev']
            and macd['dif_last'] < macd['dea_last']
        )

    def _is_macd_bullish(self) -> bool:
        """MACD 整体多头：DIF > DEA 且柱状线在零轴上方"""
        macd = self._calc_macd()
        if not macd:
            return False
        return macd['dif_last'] > macd['dea_last'] and macd['hist_last'] > 0

    # ──────────────── 日线判断（新增日期） ────────────────
    def _get_current_date(self) -> str:
        """获取当前 bar 的日期"""
        closes = self.get_history(2,  field="close")
        # 距离上次调用可能不定，返回一个简单的 date 标识
        # 实际框架中应使用 bar.datetime 或 bar.date
        return ""

    # ──────────────── 核心交易逻辑 ────────────────
    def on_bar(self, bar):
        """每根5分钟K线触发"""
        current_price = bar.close
        current_position = self.get_position(bar.symbol)
        date_str = str(bar.datetime.date()) if hasattr(bar, 'datetime') else ""

        # ── 重置每日计数 ──
        if date_str and date_str != self.last_trade_date:
            self.daily_trade_count = 0
            self.last_trade_date = date_str

        # ========== 场景0: 无底仓，先建底仓 ==========
        if not self.has_base_position and current_position == 0:
            macd = self._calc_macd()
            if macd and macd['hist_last'] > 0:
                # 在MACD零轴上方建底仓
                self.order_target_percent(
                    self.base_position_ratio, bar.symbol, price=current_price
                )
                self.has_base_position = True
                self.base_position_cost = current_price
                self.log(
                    f"[建底仓] 价格:{current_price:.2f}, 仓位:{self.base_position_ratio:.0%}"
                )
            return

        # 更新底仓状态
        if current_position > 0 and not self.has_base_position:
            self.has_base_position = True
            self.base_position_cost = current_price

        # ========== 场景1: 持有底仓 + T仓位 → 检查T卖出条件 ==========
        if self.t_entry_price > 0:
            # --- 止损 ---
            if (current_price - self.t_entry_price) / self.t_entry_price <= -self.stop_loss_pct:
                self._sell_t_position(bar.symbol, current_price, "止损")
                return

            # --- 止盈 ---
            if (current_price - self.t_entry_price) / self.t_entry_price >= self.take_profit_pct:
                self._sell_t_position(bar.symbol, current_price, "止盈")
                return

            # --- MACD死叉+缩量卖出 ---
            if self._is_dead_cross() and self._is_volume_shrink():
                self._sell_t_position(bar.symbol, current_price, "死叉缩量")
                return

        # ========== 场景2: 持有底仓，无T仓位 → 检查T买入条件 ==========
        if self.has_base_position and self.t_entry_price == 0:
            # 日限制检查
            if self.daily_trade_count >= self.max_daily_trades:
                return

            # 条件: MACD金叉 + 放量 + DIF在零轴上方（趋势多头）
            golden_cross = self._is_golden_cross()
            volume_surge = self._is_volume_surge()

            if golden_cross and volume_surge and self._is_macd_bullish():
                # 已持有底仓，开多T仓位
                pos = self.get_position(bar.symbol)
                if pos > 0:
                    # 获取可用现金，计算T仓位金额
                    cash = self.get_cash()
                    total_value = cash + pos * current_price
                    t_amount = total_value * self.base_position_ratio * self.t_ratio
                    t_shares = int(t_amount / current_price / 100) * 100

                    if t_shares >= 100:
                        self.buy(bar.symbol, quantity=t_shares, price=current_price)
                        self.t_entry_price = current_price
                        self.daily_trade_count += 1
                        self.log(
                            f"[T买入] 价格:{current_price:.2f}, "
                            f"股数:{t_shares}, 原因:MACD金叉+放量, "
                            f"今日第{self.daily_trade_count}次T"
                        )

    def _sell_t_position(self, symbol: str, current_price: float, reason: str):
        """卖出T仓位"""
        pos = self.get_position(symbol)
        base_shares = self._get_base_shares()
        t_shares = pos - base_shares if pos > base_shares else 0

        if t_shares > 0:
            self.sell(symbol, quantity=t_shares, price=current_price)
            pnl = (current_price - self.t_entry_price) / self.t_entry_price
            self.log(
                f"[T卖出] 价格:{current_price:.2f}, 原因:{reason}, "
                f"盈亏:{pnl:.2%}, 今日累计{self.daily_trade_count}次T"
            )
            self.t_entry_price = 0

    def _get_base_shares(self) -> int:
        """获取底仓对应的股数（估算）"""
        pos = self.get_position(self.trade_symbol)
        # 简单按底仓比例反推，实际取决于建仓时价格
        return int(pos * (1 - self.t_ratio)) if pos > 0 else 0

    def on_stop(self):
        """策略结束"""
        self.log("FiveMinMacdTStrategy 策略运行结束")


# ========== 独立运行 ==========
if __name__ == "__main__":
    # 1. 获取数据
    symbol = "688131"     # 替换为你想测试的股票代码
    start_date = "20250101"
    end_date = "20260601"
    DATA_DIR = "tsdata"

    print(f"正在获取 {symbol} 5分钟数据...")

    mytoken = os.getenv('TUSHARE_TOKEN')
    print('TUSHARE_TOKEN set:', bool(mytoken))

    manager = TushareStockDataManager(
        token=mytoken,   # type: ignore
        data_dir=DATA_DIR,
        request_interval=1.5,
    )

    # 注意：这里需要获取5分钟线数据，视 TushareStockDataManager 是否支持
    # 若不支持5分钟，请替换为你的数据源
    # df = manager.get_minute_data(symbol=symbol, start_date=start_date, end_date=end_date, freq='5min')
    #
    # 暂时复用日线作为演示，实际使用时请替换为5分钟数据源
    df = manager.get_stock_data(symbol=symbol, start_date=start_date, end_date=end_date)

    print(f"数据获取成功，共 {len(df)} 条记录")
    df = df.sort_index()
    print(f"数据范围：{df.index[0]} 至 {df.index[-1]}")

    # 2. 运行回测
    result = run_backtest(
        strategy=FiveMinMacdTStrategy,
        data=df,
        symbols=[symbol],
        initial_cash=100000.0,
        commission_rate=0.0003,
        slippage=0.0002,
        t_plus_one=True,
    )

    # 3. 输出结果
    print("\n=== 回测结果 ===")
    print(result.metrics_df)

    # 4. 生成报告
    report_dir = "./reports"
    os.makedirs(report_dir, exist_ok=True)
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"{report_dir}/5min_macd_t_strategy_{symbol}_{timestamp}.html"

    result.report(
        filename=report_path,
        title=f"5分钟MACD做T策略报告 ({symbol})",
        market_data=df,
        include_trade_kline=True,
    )

    print(f"\n报告已保存至: {report_path}")