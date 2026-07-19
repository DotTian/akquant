#!/usr/bin/env python3
"""Minimal runner for DailyReportGenerator.

Priority of configuration sources: CLI > ENV > default.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_SYMBOLS_FILE = "./scripts/dailyreport_symbols.example.txt"
DEFAULT_OUTPUT_DIR = "./reports"
DEFAULT_DATA_DIR = "tsdata"
DEFAULT_STOCK_INFO_DIR = "stock_info"
DEFAULT_REQUEST_INTERVAL = 0.5

ENV_TOKEN = "TUSHARE_TOKEN"
ENV_SYMBOLS_FILE = "DAILYREPORT_SYMBOLS_FILE"
ENV_START_DATE = "DAILYREPORT_START_DATE"
ENV_END_DATE = "DAILYREPORT_END_DATE"
ENV_OUTPUT_DIR = "DAILYREPORT_OUTPUT_DIR"
ENV_OUTPUT_NAME = "DAILYREPORT_OUTPUT_NAME"
ENV_DATA_DIR = "DAILYREPORT_DATA_DIR"
ENV_STOCK_INFO_DIR = "DAILYREPORT_STOCK_INFO_DIR"
ENV_REQUEST_INTERVAL = "DAILYREPORT_REQUEST_INTERVAL"

_SYMBOL_PATTERN = re.compile(r"^[A-Za-z0-9.]+$")


def _pick(cli_value: Optional[str], env_name: str, default: Optional[str] = None) -> Optional[str]:
    if cli_value is not None:
        return cli_value
    env_value = os.getenv(env_name)
    if env_value not in (None, ""):
        return env_value
    return default


def _pick_float(cli_value: Optional[float], env_name: str, default: float) -> float:
    if cli_value is not None:
        return float(cli_value)
    env_value = os.getenv(env_name)
    if env_value not in (None, ""):
        try:
            return float(env_value)
        except ValueError as exc:
            raise ValueError(f"Invalid float for {env_name}: {env_value}") from exc
    return float(default)


def _normalize_symbol(raw_symbol: str) -> Optional[str]:
    text = raw_symbol.strip().upper()
    if not text:
        return None

    if text.startswith("#"):
        return None

    if not _SYMBOL_PATTERN.match(text):
        return None

    core = text.split(".", 1)[0]
    if not core.isdigit() or len(core) > 6:
        return None

    return core.zfill(6)


def _load_symbols(symbols_file: Path) -> list[str]:
    if not symbols_file.exists():
        raise FileNotFoundError(f"Symbols file not found: {symbols_file}")

    ordered = OrderedDict()
    invalid_count = 0

    for line in symbols_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        symbol = _normalize_symbol(stripped)
        if symbol is None:
            invalid_count += 1
            logger.warning("Skip invalid symbol line: %s", stripped)
            continue

        ordered[symbol] = True

    symbols = list(ordered.keys())
    if not symbols:
        raise ValueError(f"No valid symbols loaded from: {symbols_file}")

    if invalid_count > 0:
        logger.warning("Ignored %d invalid symbol lines", invalid_count)

    logger.info("Loaded %d unique symbols from %s", len(symbols), symbols_file)
    return symbols


def _resolve_output_path(output_dir: Path, output_name: Optional[str]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_name:
        name = output_name if output_name.endswith(".html") else f"{output_name}.html"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"daily_report_{timestamp}.html"

    return output_dir / name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run daily stock report generation")
    parser.add_argument("--token", help=f"Tushare token (fallback env: {ENV_TOKEN})")
    parser.add_argument(
        "--symbols-file",
        help=(
            "Path to symbols txt file (one symbol per line, # for comments) "
            f"(fallback env: {ENV_SYMBOLS_FILE}, default: {DEFAULT_SYMBOLS_FILE})"
        ),
    )
    parser.add_argument(
        "--start-date",
        help=f"Start date (YYYYMMDD or YYYY-MM-DD), fallback env: {ENV_START_DATE}",
    )
    parser.add_argument(
        "--end-date",
        help=f"End date (YYYYMMDD or YYYY-MM-DD), fallback env: {ENV_END_DATE}",
    )
    parser.add_argument(
        "--output-dir",
        help=f"Output directory (fallback env: {ENV_OUTPUT_DIR}, default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output-name",
        help=(
            "Output file name (with or without .html). "
            f"Fallback env: {ENV_OUTPUT_NAME}."
        ),
    )
    parser.add_argument(
        "--data-dir",
        help=f"Data directory (fallback env: {ENV_DATA_DIR}, default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--stock-info-dir",
        help=(
            "Stock info directory "
            f"(fallback env: {ENV_STOCK_INFO_DIR}, default: {DEFAULT_STOCK_INFO_DIR})"
        ),
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        help=(
            "Request interval in seconds "
            f"(fallback env: {ENV_REQUEST_INTERVAL}, default: {DEFAULT_REQUEST_INTERVAL})"
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    token = _pick(args.token, ENV_TOKEN)
    if not token:
        logger.error("Missing token. Provide --token or set %s", ENV_TOKEN)
        return 2

    symbols_file_raw = _pick(args.symbols_file, ENV_SYMBOLS_FILE, DEFAULT_SYMBOLS_FILE)
    if not symbols_file_raw:
        logger.error("Missing symbols file path")
        return 2

    start_date = _pick(args.start_date, ENV_START_DATE)
    if not start_date:
        logger.error("Missing start date. Provide --start-date or set %s", ENV_START_DATE)
        return 2

    end_date = _pick(args.end_date, ENV_END_DATE)
    output_dir = Path(_pick(args.output_dir, ENV_OUTPUT_DIR, DEFAULT_OUTPUT_DIR) or DEFAULT_OUTPUT_DIR)
    output_name = _pick(args.output_name, ENV_OUTPUT_NAME)
    data_dir = _pick(args.data_dir, ENV_DATA_DIR, DEFAULT_DATA_DIR) or DEFAULT_DATA_DIR
    stock_info_dir = (
        _pick(args.stock_info_dir, ENV_STOCK_INFO_DIR, DEFAULT_STOCK_INFO_DIR)
        or DEFAULT_STOCK_INFO_DIR
    )
    request_interval = _pick_float(
        args.request_interval,
        ENV_REQUEST_INTERVAL,
        DEFAULT_REQUEST_INTERVAL,
    )

    symbols_file = Path(symbols_file_raw)
    symbols = _load_symbols(symbols_file)
    output_path = _resolve_output_path(output_dir, output_name)

    logger.info("Start daily report run")
    logger.info("Symbols file: %s", symbols_file)
    logger.info("Date range: %s -> %s", start_date, end_date or "today")
    logger.info("Output path: %s", output_path)

    from akq_module_dailyreport import DailyReportGenerator

    generator = DailyReportGenerator(
        token=token,
        data_dir=data_dir,
        stock_info_dir=stock_info_dir,
        request_interval=request_interval,
    )

    final_output = generator.generate_report(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        output_path=str(output_path),
    )
    logger.info("Daily report generated: %s", final_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
