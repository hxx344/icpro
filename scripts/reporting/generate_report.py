"""Run a backtest for a config and save metrics + plots to `reports/`.

Usage: python scripts/reporting/generate_report.py configs/backtest/naked_call_usd.yaml
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.analytics.metrics import compute_metrics, print_metrics
from options_backtest.analytics.plotting import generate_all_plots
from options_backtest.cli import _load_strategy


def main(cfg_path: str):
    cfg = Config.from_yaml(Path(cfg_path))
    strategy = _load_strategy(cfg.strategy.name, cfg.strategy.params)

    engine = BacktestEngine(cfg, strategy)
    results = engine.run()

    metrics = compute_metrics(results)

    out_dir = Path(cfg.report.output_dir or "reports").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / f"{cfg.backtest.name.replace(' ', '_')}_metrics.json"
    with metrics_path.open("w", encoding="utf8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print_metrics(metrics)

    # Generate plots into the report dir
    paths = generate_all_plots(results, str(out_dir))
    for p in paths:
        print(f"Saved plot: {p}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/reporting/generate_report.py <config.yaml>")
        raise SystemExit(1)
    main(sys.argv[1])
