"""Test directional widening of delta tolerance for the current trader strategy.

Base setting:
- target_delta = 0.45
- lower diff = 0.15
- upper diff = 0.15

Two directional sweeps are compared over the latest 1-year window:
1. Widen downward only: allow smaller deltas, keep upper diff at 0.15
2. Widen upward only: allow larger deltas, keep lower diff at 0.15
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
from loguru import logger

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
    DEFAULT_TRADER_INITIAL_USD,
    _apply_initial_usd_balance,
    detect_hourly_coverage,
    load_run_config,
)

TRADER_CONFIG = REPO_ROOT / "configs" / "trader" / "weekend_vol_btc.yaml"
OUTPUT_PATH = REPO_ROOT / "reports" / "current_trader_diff_directional_last_year.json"
TARGET_DELTA = 0.45
BASE_LOWER = 0.15
BASE_UPPER = 0.15
SWEEP_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25]


def run_one(label: str, lower_diff: float, upper_diff: float, start_date: str, end_date: str) -> dict:
    loaded = load_run_config(TRADER_CONFIG)
    cfg = loaded.cfg
    cfg.backtest.name = f"Current trader {label}"
    cfg.backtest.start_date = start_date
    cfg.backtest.end_date = end_date
    cfg.backtest.show_progress = False
    cfg.strategy.params["target_delta"] = TARGET_DELTA
    cfg.strategy.params["target_call_delta"] = TARGET_DELTA
    cfg.strategy.params["target_put_delta"] = TARGET_DELTA
    cfg.strategy.params["max_delta_diff"] = max(lower_diff, upper_diff)
    cfg.strategy.params["max_delta_diff_lower"] = lower_diff
    cfg.strategy.params["max_delta_diff_upper"] = upper_diff
    _apply_initial_usd_balance(cfg, loaded.source_type, start_date, end_date)

    strategy = _load_strategy(cfg.strategy.name, cfg.strategy.params)
    engine = BacktestEngine(cfg, strategy)

    t0 = time.perf_counter()
    results = engine.run()
    elapsed = time.perf_counter() - t0
    metrics = compute_metrics(results)

    return {
        "label": label,
        "lower_diff": lower_diff,
        "upper_diff": upper_diff,
        "allowed_delta_range": [round(TARGET_DELTA - lower_diff, 4), round(TARGET_DELTA + upper_diff, 4)],
        "total_return_pct": float(metrics.get("total_return", 0.0)) * 100.0,
        "annualized_return_pct": float(metrics.get("annualized_return", 0.0)) * 100.0,
        "max_drawdown_pct": float(metrics.get("max_drawdown", 0.0)) * 100.0,
        "sharpe": float(metrics.get("sharpe_ratio", 0.0)),
        "win_rate_pct": float(metrics.get("win_rate", 0.0)) * 100.0,
        "profit_factor": float(metrics.get("profit_factor", 0.0)),
        "total_trades": int(metrics.get("total_trades", 0)),
        "final_equity": float(metrics.get("final_equity", 0.0)),
        "total_fees": float(metrics.get("total_fees", 0.0)),
        "elapsed_sec": elapsed,
    }


def main() -> None:
    loaded = load_run_config(TRADER_CONFIG)
    coverage = detect_hourly_coverage(loaded.cfg.backtest.underlying)
    end_ts = coverage.end_ts.normalize()
    start_ts = end_ts - pd.Timedelta(days=364)
    start_date = start_ts.strftime("%Y-%m-%d")
    end_date = end_ts.strftime("%Y-%m-%d")

    rows = []
    print("Directional delta-diff widening test (recent 1y)")
    print(f"Period: {start_date} -> {end_date}")
    print(f"Target delta fixed at: {TARGET_DELTA:.2f}")
    print(f"Base range: [{TARGET_DELTA-BASE_LOWER:.2f}, {TARGET_DELTA+BASE_UPPER:.2f}]")
    print(f"Initial USD fixed at: {DEFAULT_TRADER_INITIAL_USD:,.0f}\n")

    print("Downward widening only (allow smaller deltas, upper fixed at 0.15)")
    print(f"{'Lower':>6s} {'Upper':>6s} {'Range':>15s} {'Ret':>8s} {'MaxDD':>8s} {'Sharpe':>7s} {'Trades':>6s}")
    print("=" * 78)
    for lower in SWEEP_VALUES:
        row = run_one(f"down_l{lower:.2f}_u{BASE_UPPER:.2f}", lower, BASE_UPPER, start_date, end_date)
        rows.append(row)
        print(
            f"{lower:>6.2f} {BASE_UPPER:>6.2f} "
            f"[{row['allowed_delta_range'][0]:.2f},{row['allowed_delta_range'][1]:.2f}]".rjust(15)
            + f" {row['total_return_pct']:>+7.2f}% {row['max_drawdown_pct']:>7.2f}% {row['sharpe']:>7.2f} {row['total_trades']:>6d}"
        )

    print("\nUpward widening only (allow larger deltas, lower fixed at 0.15)")
    print(f"{'Lower':>6s} {'Upper':>6s} {'Range':>15s} {'Ret':>8s} {'MaxDD':>8s} {'Sharpe':>7s} {'Trades':>6s}")
    print("=" * 78)
    for upper in SWEEP_VALUES:
        row = run_one(f"up_l{BASE_LOWER:.2f}_u{upper:.2f}", BASE_LOWER, upper, start_date, end_date)
        rows.append(row)
        print(
            f"{BASE_LOWER:>6.2f} {upper:>6.2f} "
            f"[{row['allowed_delta_range'][0]:.2f},{row['allowed_delta_range'][1]:.2f}]".rjust(15)
            + f" {row['total_return_pct']:>+7.2f}% {row['max_drawdown_pct']:>7.2f}% {row['sharpe']:>7.2f} {row['total_trades']:>6d}"
        )

    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "period": {"start_date": start_date, "end_date": end_date},
                "target_delta": TARGET_DELTA,
                "base_lower": BASE_LOWER,
                "base_upper": BASE_UPPER,
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved: {OUTPUT_PATH.as_posix()}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    main()
