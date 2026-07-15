"""
月度 Excel 选股调仓策略（展示版）

策略规则：
1. 仅使用 on_daily_rebalance，不使用 on_bar 择时。
2. 每月首个交易日执行一次调仓。
3. 从 Excel 的“详细结果”sheet 读取当月股票池，按出现顺序取前 N 只。
4. 当月无记录时清仓。
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from akquant import Strategy, run_backtest

from akq_module_tusharedatamanager import TushareStockDataManager


class MonthlyExcelRebalanceStrategy(Strategy):
    """每月首个交易日按 Excel 目标池调仓。"""

    def __init__(
        self,
        excel_path: str,
        symbols: list[str],
        sheet_name: str = "详细结果",
        top_n: int = 5,
        target_exposure: float = 0.95,
    ) -> None:
        super().__init__()
        self.excel_path = Path(excel_path)
        self.sheet_name = sheet_name
        self.top_n = int(top_n)
        self.target_exposure = float(target_exposure)
        self.symbols = sorted({str(s).strip() for s in symbols if str(s).strip()})

        self.month_to_symbols: dict[str, list[str]] = {}
        self.last_rebalance_month: str | None = None
        self.rebalance_history: list[dict[str, Any]] = []

    def on_start(self) -> None:
        """启动时加载月度目标池并订阅全部可交易标的。"""
        self.month_to_symbols = load_monthly_selection(
            excel_path=str(self.excel_path),
            sheet_name=self.sheet_name,
            top_n=self.top_n,
        )

        for symbol in self.symbols:
            self.subscribe(symbol)

        self.log(
            f"monthly selector loaded: months={len(self.month_to_symbols)}, "
            f"universe={len(self.symbols)}, top_n={self.top_n}"
        )

    def on_daily_rebalance(self, trading_date: Any, timestamp: int) -> None:
        """交易日级调仓：仅在每月第一天触发一次。"""
        _ = timestamp
        month_key = pd.Timestamp(trading_date).strftime("%Y-%m")
        if self.last_rebalance_month == month_key:
            return

        self.last_rebalance_month = month_key
        selected = self.month_to_symbols.get(month_key, [])
        selected = [s for s in selected if s in self.symbols]

        if not selected:
            self.order_target_weights(target_weights={}, liquidate_unmentioned=True)
            self.rebalance_history.append(
                {
                    "trading_date": str(trading_date),
                    "month": month_key,
                    "selected": [],
                    "target_weights": {},
                    "action": "liquidate",
                }
            )
            self.log(f"rebalance {trading_date}: month={month_key} missing -> liquidate")
            return

        each_weight = self.target_exposure / float(len(selected))
        target_weights = {symbol: each_weight for symbol in selected}
        self.order_target_weights(
            target_weights=target_weights,
            liquidate_unmentioned=True,
            rebalance_tolerance=0.01,
        )

        self.rebalance_history.append(
            {
                "trading_date": str(trading_date),
                "month": month_key,
                "selected": selected,
                "target_weights": target_weights,
                "action": "rebalance",
            }
        )
        self.log(f"rebalance {trading_date}: month={month_key} targets={target_weights}")


def load_monthly_selection(
    excel_path: str,
    sheet_name: str = "详细结果",
    top_n: int = 5,
) -> dict[str, list[str]]:
    """读取月度选股结果并整理为 month -> symbols。"""
    path = Path(excel_path)
    if not path.exists():
        raise FileNotFoundError(f"Excel not found: {path}")

    df = pd.read_excel(path, sheet_name=sheet_name)
    required = {"month", "symbol"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    work = df.copy()
    work = work.dropna(subset=["month", "symbol"]).copy()
    work["month"] = pd.to_datetime(work["month"], errors="coerce").dt.strftime("%Y-%m")
    work["symbol"] = work["symbol"].astype(str).str.replace(r"\.(SH|SZ|BJ)$", "", regex=True)
    work["symbol"] = work["symbol"].str.strip().str.zfill(6)
    work = work.dropna(subset=["month"])  # 丢弃不可解析月份
    work = work[work["symbol"] != ""]

    month_to_symbols: dict[str, list[str]] = {}
    for month, group in work.groupby("month", sort=True):
        ordered = list(dict.fromkeys(group["symbol"].tolist()))
        month_to_symbols[str(month)] = ordered[:top_n]

    if not month_to_symbols:
        raise ValueError("No monthly selections loaded from Excel")

    return month_to_symbols


def collect_universe_from_selection(month_to_symbols: dict[str, list[str]]) -> list[str]:
    """从月度映射中提取全量标的池。"""
    universe: list[str] = []
    for symbols in month_to_symbols.values():
        for symbol in symbols:
            if symbol not in universe:
                universe.append(symbol)
    return universe


def infer_backtest_window(month_to_symbols: dict[str, list[str]]) -> tuple[str, str]:
    """基于月度键推导回测日期窗口。"""
    months = sorted(month_to_symbols.keys())
    first_month = pd.to_datetime(months[0] + "-01")
    last_month = pd.to_datetime(months[-1] + "-01") + pd.offsets.MonthEnd(1)
    start_date = first_month.strftime("%Y%m%d")
    end_date = last_month.strftime("%Y%m%d")
    return start_date, end_date


def load_market_data(
    symbols: list[str],
    start_date: str,
    end_date: str,
    data_dir: str,
) -> dict[str, pd.DataFrame]:
    """批量加载标的行情，自动使用本地缓存并按需增量更新。"""
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("Environment variable TUSHARE_TOKEN is required")

    manager = TushareStockDataManager(
        token=token,
        data_dir=data_dir,
        request_interval=1.5,
    )
    raw = manager.get_multiple_stocks(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        force_update=False,
        adjust="qfq",
        delay_between=0.8,
    )

    result: dict[str, pd.DataFrame] = {}
    for symbol, df in raw.items():
        if df is None or df.empty:
            continue
        result[symbol] = df

    if not result:
        raise RuntimeError("No market data loaded for any symbol")

    return result


def main() -> None:
    """执行月度调仓回测。"""
    excel_path = "reports/stock_selection_results_20260715_215442.xlsx"
    sheet_name = "详细结果"
    top_n = 5
    data_dir = "tsdata"

    month_to_symbols = load_monthly_selection(
        excel_path=excel_path,
        sheet_name=sheet_name,
        top_n=top_n,
    )
    symbols = collect_universe_from_selection(month_to_symbols)
    start_date, end_date = infer_backtest_window(month_to_symbols)

    print(f"excel: {excel_path}")
    print(f"months: {len(month_to_symbols)}")
    print(f"universe symbols: {len(symbols)}")
    print(f"window: {start_date} -> {end_date}")

    data = load_market_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        data_dir=data_dir,
    )
    tradable_symbols = sorted(data.keys())
    print(f"tradable symbols: {len(tradable_symbols)}")

    monthly_strategy = MonthlyExcelRebalanceStrategy(
        excel_path=excel_path,
        sheet_name=sheet_name,
        symbols=tradable_symbols,
        top_n=top_n,
        target_exposure=0.95,
    )

    result = run_backtest(
        strategy=monthly_strategy,
        data=data,
        symbols=tradable_symbols,
        initial_cash=1_000_000.0,
        commission_rate=0.0003,
        stamp_tax_rate=0.001,
        transfer_fee_rate=0.0,
        min_commission=5.0,
        t_plus_one=True,
        lot_size=100,
        timezone="Asia/Shanghai",
        fill_policy={
            "price_basis": "close",
            "temporal": "same_cycle",
        },
        show_progress=True,
    )

    print("\n=== 回测结果 ===")
    print(result.metrics_df)

    rebalance_history = getattr(monthly_strategy, "rebalance_history", [])
    if rebalance_history:
        print("\n=== 月度调仓轨迹 ===")
        for item in rebalance_history:
            print(
                f"{item['trading_date']} | month={item['month']} | "
                f"action={item['action']} | selected={item['selected']}"
            )

    if not result.positions.empty:
        print("\n=== 最终持仓 ===")
        print(result.positions.iloc[-1])

    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"monthly_excel_rebalance_strategy_{timestamp}.html"

    plot_symbol = tradable_symbols[0] if tradable_symbols else None
    result.report(
        filename=str(report_path),
        title="月度Excel调仓策略报告",
        market_data=data,
        plot_symbol=plot_symbol,
        include_trade_kline=True,
        show=False,
    )
    print(f"\n报告已保存至: {report_path}")

    print("\n完成时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
