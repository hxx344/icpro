# Scripts Layout

This directory is grouped by intent so the primary entrypoints stay visible and one-off research work does not bury them.

- `scripts/build_hourly_parquet.py`: main hourly parquet build entrypoint
- `scripts/bybit_cc_recommender_panel.py`: standalone Streamlit recommendation panel
- `scripts/backtest_weekend_vol.py`: direct weekend vol backtest runner
- `scripts/data/`: data download, filtering, and derived parquet builders
- `scripts/research/`: parameter sweeps, strategy comparisons, and exploratory studies
- `scripts/debug/`: ad hoc diagnostics, profiling, and exchange/API probes
- `scripts/reporting/`: report generation and payoff/analysis plotting helpers
- `scripts/maintenance/`: one-off repository or demo-data maintenance helpers

If you add a new script, keep stable user-facing entrypoints at the top level. Put support scripts under the narrowest matching folder so the top level stays reserved for commands people run directly.
