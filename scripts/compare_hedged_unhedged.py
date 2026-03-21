"""Run backtests for hedged vs unhedged variants and save comparison outputs.

Usage: python scripts/compare_hedged_unhedged.py configs/backtest/naked_call_usd.yaml
"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.analytics.metrics import compute_metrics, print_metrics
from options_backtest.analytics.plotting import generate_all_plots
from options_backtest.cli import _load_strategy


def run_and_save(cfg: Config, label: str, out_root: Path):
    cfg_copy = deepcopy(cfg)
    out_dir = out_root / label
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure explicit boolean flag exists
    cfg_copy.strategy.params.setdefault("buy_protective_call", False)

    strategy = _load_strategy(cfg_copy.strategy.name, cfg_copy.strategy.params)
    engine = BacktestEngine(cfg_copy, strategy)
    results = engine.run()

    metrics = compute_metrics(results)

    # Save metrics JSON
    metrics_path = out_dir / f"metrics_{label}.json"
    with metrics_path.open("w", encoding="utf8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Save plots
    plot_paths = generate_all_plots(results, str(out_dir))

    return metrics, plot_paths


def summarize_metrics(metrics: dict) -> dict:
    # Robust extraction of common fields
    def g(*keys):
        for k in keys:
            if k in metrics:
                return metrics[k]
        return None

    return {
        "final_equity": g("final_equity", "Final Equity", "final_equity_eth"),
        "total_return": g("total_return", "Total Return", "total_return_pct"),
        "final_usd": g("final_usd", "Final USD", "final_usd_spot"),
        "final_hedged_usd": g("final_hedged_usd", "Final Hedged USD", "final_hedged_usd"),
        "total_trades": g("total_trades", "Total Trades", "trades"),
        "max_drawdown": g("max_drawdown", "Max Drawdown", "max_drawdown_pct"),
    }


def main(cfg_path: str):
    cfg = Config.from_yaml(Path(cfg_path))
    base_out = Path(cfg.report.output_dir or "reports") / "comparison"
    base_out.mkdir(parents=True, exist_ok=True)

    # Run unhedged (override)
    cfg_unhedged = deepcopy(cfg)
    cfg_unhedged.strategy.params = dict(cfg_unhedged.strategy.params)
    cfg_unhedged.strategy.params["buy_protective_call"] = False
    metrics_unhedged, plots_unhedged = run_and_save(cfg_unhedged, "unhedged", base_out)

    # Update todo

    # Run hedged
    cfg_hedged = deepcopy(cfg)
    cfg_hedged.strategy.params = dict(cfg_hedged.strategy.params)
    cfg_hedged.strategy.params["buy_protective_call"] = True
    metrics_hedged, plots_hedged = run_and_save(cfg_hedged, "hedged", base_out)

    # Write combined comparison JSON
    combined = {"unhedged": metrics_unhedged, "hedged": metrics_hedged}
    with (base_out / "comparison_metrics.json").open("w", encoding="utf8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    # Create simple CSV summary
    csv_path = base_out / "comparison_summary.csv"
    import csv

    rows = []
    header = ["case", "final_equity", "total_return", "final_usd", "final_hedged_usd", "total_trades", "max_drawdown"]
    rows.append(header)

    for label, metrics in [("unhedged", metrics_unhedged), ("hedged", metrics_hedged)]:
        s = summarize_metrics(metrics)
        rows.append([
            label,
            s.get("final_equity"),
            s.get("total_return"),
            s.get("final_usd"),
            s.get("final_hedged_usd"),
            s.get("total_trades"),
            s.get("max_drawdown"),
        ])

    with csv_path.open("w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print("Saved comparison to:")
    print(" -", base_out / "comparison_metrics.json")
    print(" -", csv_path)
    print("Plots saved under:")
    print(" -", base_out / "unhedged")
    print(" -", base_out / "hedged")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/compare_hedged_unhedged.py <config.yaml>")
        raise SystemExit(1)
    main(sys.argv[1])
