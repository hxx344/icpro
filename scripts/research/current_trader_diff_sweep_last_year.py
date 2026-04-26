"""Sweep `max_delta_diff` around the current live trader value over the latest year."""
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
OUTPUT_PATH = REPO_ROOT / "reports" / "current_trader_diff_sweep_last_year.json"
DIFFS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
CURRENT_TARGET_DELTA = 0.45


def run_one(max_delta_diff: float, start_date: str, end_date: str) -> dict:
    loaded = load_run_config(TRADER_CONFIG)
    cfg = loaded.cfg
    cfg.backtest.name = f"Current trader diff={max_delta_diff:.2f}"
    cfg.backtest.start_date = start_date
    cfg.backtest.end_date = end_date
    cfg.backtest.show_progress = False
    cfg.strategy.params["target_delta"] = CURRENT_TARGET_DELTA
    cfg.strategy.params["target_call_delta"] = CURRENT_TARGET_DELTA
    cfg.strategy.params["target_put_delta"] = CURRENT_TARGET_DELTA
    cfg.strategy.params["max_delta_diff"] = max_delta_diff
    _apply_initial_usd_balance(cfg, loaded.source_type, start_date, end_date)

    strategy = _load_strategy(cfg.strategy.name, cfg.strategy.params)
    engine = BacktestEngine(cfg, strategy)

    t0 = time.perf_counter()
    results = engine.run()
    elapsed = time.perf_counter() - t0
    metrics = compute_metrics(results)

    return {
        "max_delta_diff": max_delta_diff,
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
    print("Current trader strategy max_delta_diff sweep (recent 1y)")
    print(f"Period: {start_date} -> {end_date}")
    print(f"Target delta fixed at: {CURRENT_TARGET_DELTA:.2f}")
    print(f"Initial USD fixed at: {DEFAULT_TRADER_INITIAL_USD:,.0f}")
    print(f"{'Diff':>6s} {'Ret':>8s} {'MaxDD':>8s} {'Sharpe':>7s} {'WR':>6s} {'PF':>6s} {'Trades':>6s} {'Fees':>9s} {'Time':>6s}")
    print("=" * 88)

    for diff in DIFFS:
        row = run_one(diff, start_date, end_date)
        rows.append(row)
        print(
            f"{diff:>6.2f} {row['total_return_pct']:>+7.2f}% {row['max_drawdown_pct']:>7.2f}% "
            f"{row['sharpe']:>7.2f} {row['win_rate_pct']:>5.1f}% {row['profit_factor']:>6.2f} "
            f"{row['total_trades']:>6d} {row['total_fees']:>9.0f} {row['elapsed_sec']:>5.1f}s"
        )

    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "period": {"start_date": start_date, "end_date": end_date},
                "target_delta": CURRENT_TARGET_DELTA,
                "initial_usd": DEFAULT_TRADER_INITIAL_USD,
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    best_sharpe = max(rows, key=lambda r: r["sharpe"])
    best_return = max(rows, key=lambda r: r["total_return_pct"])
    best_dd = max(rows, key=lambda r: r["max_drawdown_pct"])
    print("\nBest return:")
    print(
        f"  diff={best_return['max_delta_diff']:.2f} Ret={best_return['total_return_pct']:.2f}% "
        f"DD={best_return['max_drawdown_pct']:.2f}% Sharpe={best_return['sharpe']:.2f}"
    )
    print("Best sharpe:")
    print(
        f"  diff={best_sharpe['max_delta_diff']:.2f} Sharpe={best_sharpe['sharpe']:.2f} "
        f"Ret={best_sharpe['total_return_pct']:.2f}% DD={best_sharpe['max_drawdown_pct']:.2f}%"
    )
    print("Best drawdown:")
    print(
        f"  diff={best_dd['max_delta_diff']:.2f} DD={best_dd['max_drawdown_pct']:.2f}% "
        f"Ret={best_dd['total_return_pct']:.2f}% Sharpe={best_dd['sharpe']:.2f}"
    )
    print(f"\nSaved: {OUTPUT_PATH.as_posix()}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    main()
