from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from options_backtest.analytics.metrics import compute_metrics
from options_backtest.cli import _load_strategy
from options_backtest.engine.backtest import BacktestEngine
from scripts.research.run_cached_hourly_backtest import (
    _apply_initial_usd_balance,
    detect_hourly_coverage,
    load_run_config,
)

TRADER_CONFIG = REPO_ROOT / "configs" / "trader" / "weekend_vol_btc.yaml"
OUTPUT_DIR = REPO_ROOT / "reports" / "validation" / "weekend_vol_lower_delta_3y"
TARGET_DELTAS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]


def run_one(target_delta: float) -> dict:
    loaded = load_run_config(TRADER_CONFIG)
    cfg = loaded.cfg
    coverage = detect_hourly_coverage(cfg.backtest.underlying)
    start_date = coverage.start_date
    end_date = coverage.end_date

    cfg.backtest.name = f"WeekendVol lower-delta 3y | delta={target_delta:.2f}"
    cfg.backtest.start_date = start_date
    cfg.backtest.end_date = end_date
    cfg.backtest.option_data_source = "options_hourly"
    cfg.backtest.option_snapshot_pick = "close"
    cfg.backtest.show_progress = False
    cfg.report.generate_plots = False
    cfg.report.output_dir = str(OUTPUT_DIR / f"delta_{int(target_delta * 100):02d}")

    cfg.strategy.params["target_delta"] = float(target_delta)
    cfg.strategy.params["target_call_delta"] = float(target_delta)
    cfg.strategy.params["target_put_delta"] = float(target_delta)

    initial_spot = _apply_initial_usd_balance(
        cfg,
        loaded.source_type,
        start_date,
        end_date,
    )

    strategy = _load_strategy(cfg.strategy.name, cfg.strategy.params)
    engine = BacktestEngine(cfg, strategy)
    started = time.perf_counter()
    results = engine.run()
    elapsed = time.perf_counter() - started
    metrics = compute_metrics(results)

    return {
        "target_delta": target_delta,
        "start_date": start_date,
        "end_date": end_date,
        "initial_spot": initial_spot,
        "total_return_pct": float(metrics.get("total_return", 0.0)) * 100.0,
        "annualized_return_pct": float(metrics.get("annualized_return", 0.0)) * 100.0,
        "max_drawdown_pct": float(metrics.get("max_drawdown", 0.0)) * 100.0,
        "sharpe": float(metrics.get("sharpe_ratio", 0.0)),
        "profit_factor": float(metrics.get("profit_factor", 0.0)),
        "win_rate_pct": float(metrics.get("win_rate", 0.0)) * 100.0,
        "total_trades": int(metrics.get("total_trades", 0)),
        "final_equity": float(metrics.get("final_equity", 0.0)),
        "elapsed_sec": elapsed,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    print(f"Config: {TRADER_CONFIG}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Target deltas: {TARGET_DELTAS}")
    print("=" * 100)

    for idx, target_delta in enumerate(TARGET_DELTAS, start=1):
        row = run_one(target_delta)
        rows.append(row)
        print(
            f"[{idx}/{len(TARGET_DELTAS)}] "
            f"delta={row['target_delta']:.2f} | "
            f"Ret={row['total_return_pct']:.2f}% | "
            f"Ann={row['annualized_return_pct']:.2f}% | "
            f"DD={row['max_drawdown_pct']:.2f}% | "
            f"Sharpe={row['sharpe']:.2f} | "
            f"PF={row['profit_factor']:.2f} | "
            f"WR={row['win_rate_pct']:.2f}% | "
            f"Trades={row['total_trades']} | "
            f"Final={row['final_equity']:.2f} | "
            f"Time={row['elapsed_sec']:.2f}s"
        )

    df = pd.DataFrame(rows).sort_values(["target_delta"], ascending=[True])
    csv_path = OUTPUT_DIR / "summary.csv"
    json_path = OUTPUT_DIR / "summary.json"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(df.to_dict(orient="records"), indent=2, ensure_ascii=False), encoding="utf-8")

    best_return = df.sort_values(["total_return_pct"], ascending=[False]).iloc[0].to_dict()
    best_sharpe = df.sort_values(["sharpe", "total_return_pct"], ascending=[False, False]).iloc[0].to_dict()

    print("=" * 100)
    print(
        f"Best Return: delta={best_return['target_delta']:.2f} | "
        f"Ret={best_return['total_return_pct']:.2f}% | DD={best_return['max_drawdown_pct']:.2f}% | "
        f"Sharpe={best_return['sharpe']:.2f} | Final={best_return['final_equity']:.2f}"
    )
    print(
        f"Best Sharpe: delta={best_sharpe['target_delta']:.2f} | "
        f"Sharpe={best_sharpe['sharpe']:.2f} | Ret={best_sharpe['total_return_pct']:.2f}% | "
        f"DD={best_sharpe['max_drawdown_pct']:.2f}% | Final={best_sharpe['final_equity']:.2f}"
    )
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
