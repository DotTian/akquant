# Daily Report Runner (Minimal)

This runner avoids using `akq_module_dailyreport.py` main block and supports
configuration priority:

1. CLI arguments
2. Environment variables
3. Default values

## Files

- `scripts/run_daily_report.py`: execution entrypoint
- `scripts/dailyreport_symbols.example.txt`: sample symbols file

## Environment variables

- `TUSHARE_TOKEN` (required unless `--token` is provided)
- `DAILYREPORT_SYMBOLS_FILE`
- `DAILYREPORT_START_DATE`
- `DAILYREPORT_END_DATE`
- `DAILYREPORT_OUTPUT_DIR`
- `DAILYREPORT_OUTPUT_NAME`
- `DAILYREPORT_DATA_DIR`
- `DAILYREPORT_STOCK_INFO_DIR`
- `DAILYREPORT_REQUEST_INTERVAL`

## Minimal run

```bash
python scripts/run_daily_report.py \
  --symbols-file scripts/dailyreport_symbols.example.txt \
  --start-date 20260101
```

## Env-based run

```bash
export TUSHARE_TOKEN=your_token
export DAILYREPORT_SYMBOLS_FILE=scripts/dailyreport_symbols.example.txt
export DAILYREPORT_START_DATE=20260101
python scripts/run_daily_report.py
```

## Symbol file format

- One symbol per line
- Empty lines are ignored
- Lines starting with `#` are ignored
- Duplicate symbols are deduplicated while preserving order
- Invalid symbol lines are skipped with warning logs

## Output behavior

- Default output directory: `./reports`
- Default output file: `daily_report_YYYYMMDD_HHMMSS.html`
- `--output-name` supports names with or without `.html`
