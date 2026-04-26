# Project Structure

The repository is organized so package code, live trading code, configuration, research scripts, and generated artifacts are easy to tell apart.

## Top-Level Layout

- `src/options_backtest/`: installable backtesting library and CLI package.
- `trader/`: live trading engine, dashboard, exchange clients, and local state handling.
- `configs/`: versioned strategy and runtime configuration files.
- `scripts/`: runnable operational helpers and research scripts, grouped by intent.
- `tests/`: pytest suites grouped by package or runtime area.
- `docs/`: project notes and operator-facing documentation.
- `deploy/`, `monitor/`: deployment and monitoring assets.
- `data/`, `reports/`, `logs/`, `tmp/`, `.tmp/`, `test_logs/`: generated or local runtime artifacts and should stay out of source control.

## Scripts

Keep only stable, user-facing entrypoints directly under `scripts/`.

- `scripts/build_hourly_parquet.py`: primary hourly options parquet builder.
- `scripts/bybit_cc_recommender_panel.py`: standalone Streamlit recommendation panel.
- `scripts/backtest_weekend_vol.py`: direct weekend-vol backtest runner.
- `scripts/data/`: downloaders, data filters, and derived dataset builders.
- `scripts/research/`: parameter sweeps, comparisons, and exploratory strategy studies.
- `scripts/debug/`: diagnostics, profiling, and exchange/API probes.
- `scripts/reporting/`: report generation and payoff/analysis plotting helpers.
- `scripts/maintenance/`: one-off maintenance or demo-data utilities.

## Tests

Tests mirror the major runtime areas:

- `tests/options_backtest/`: backtesting library coverage.
- `tests/trader/`: live-trading and dashboard coverage.

New tests should live beside the area they exercise rather than at the root of `tests/`.
