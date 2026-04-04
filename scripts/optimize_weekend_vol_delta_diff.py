"""Grid-optimize `target_delta` and `max_delta_diff` for the current trader config.

Uses the live strategy config in `configs/trader/weekend_vol_btc.yaml` as the base,
keeps every other parameter unchanged, and produces two optimization sets:

1. Full cache coverage (current ~3-year window)
2. Recent 1-year window
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from options_backtest.analytics.metrics import compute_metrics
from options_backtest.cli import _load_strategy
from options_backtest.engine.backtest import BacktestEngine
from scripts.run_cached_hourly_backtest import detect_hourly_coverage, load_run_config

BASE_CONFIG = REPO_ROOT / "configs" / "trader" / "weekend_vol_btc.yaml"
OUTPUT_ROOT = REPO_ROOT / "reports" / "optimizations" / "weekend_vol_delta_diff"

TARGET_DELTAS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
MAX_DELTA_DIFFS = [0.10, 0.15, 0.20, 0.25]


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _build_periods() -> list[dict[str, str]]:
    loaded = load_run_config(BASE_CONFIG)
    underlying = str(loaded.cfg.backtest.underlying or "BTC").upper()
    coverage = detect_hourly_coverage(underlying)

    full_start = coverage.start_date
    full_end = coverage.end_date

    recent_end_ts = coverage.end_ts.normalize()
    recent_start_ts = recent_end_ts - pd.Timedelta(days=364)

    return [
        {
            "label": "full_3y",
            "title": "Full cache coverage",
            "start_date": full_start,
            "end_date": full_end,
        },
        {
            "label": "recent_1y",
            "title": "Recent 1-year window",
            "start_date": recent_start_ts.strftime("%Y-%m-%d"),
            "end_date": recent_end_ts.strftime("%Y-%m-%d"),
        },
    ]


def _make_config(start_date: str, end_date: str, target_delta: float, max_delta_diff: float):
    loaded = load_run_config(BASE_CONFIG)
    cfg = loaded.cfg
    cfg.backtest.start_date = start_date
    cfg.backtest.end_date = end_date
    cfg.backtest.option_data_source = "options_hourly"
    cfg.backtest.show_progress = False
    cfg.strategy.params["target_delta"] = float(target_delta)
    cfg.strategy.params["target_call_delta"] = float(target_delta)
    cfg.strategy.params["target_put_delta"] = float(target_delta)
    cfg.strategy.params["max_delta_diff"] = float(max_delta_diff)
    return cfg


def _run_one(period: dict[str, str], target_delta: float, max_delta_diff: float) -> dict[str, Any]:
    cfg = _make_config(period["start_date"], period["end_date"], target_delta, max_delta_diff)
    cfg.backtest.name = (
        f"WeekendVol current-config opt | {period['label']} | "
        f"delta={target_delta:.2f} diff={max_delta_diff:.2f}"
    )
    cfg.report.output_dir = str(OUTPUT_ROOT / period["label"] / "runs" / f"d{int(target_delta*100):02d}_md{int(max_delta_diff*100):02d}")

    strategy = _load_strategy(cfg.strategy.name, cfg.strategy.params)
    engine = BacktestEngine(cfg, strategy)

    t0 = time.perf_counter()
    results = engine.run()
    elapsed = time.perf_counter() - t0
    metrics = compute_metrics(results)

    total_return = float(metrics.get("total_return", 0.0)) * 100.0
    annualized_return = float(metrics.get("annualized_return", 0.0)) * 100.0
    max_drawdown = float(metrics.get("max_drawdown", 0.0)) * 100.0
    sharpe = float(metrics.get("sharpe_ratio", 0.0))
    profit_factor = float(metrics.get("profit_factor", 0.0))
    win_rate = float(metrics.get("win_rate", 0.0)) * 100.0
    total_trades = int(metrics.get("total_trades", 0))
    final_equity = float(metrics.get("final_equity", 0.0))
    total_fees = float(metrics.get("total_fees", 0.0))
    calmar_like = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
    return_over_dd = total_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    return {
        "period": period["label"],
        "start_date": period["start_date"],
        "end_date": period["end_date"],
        "target_delta": target_delta,
        "max_delta_diff": max_delta_diff,
        "total_return_pct": total_return,
        "annualized_return_pct": annualized_return,
        "max_drawdown_pct": max_drawdown,
        "sharpe": sharpe,
        "calmar_like": calmar_like,
        "return_over_dd": return_over_dd,
        "profit_factor": profit_factor,
        "win_rate_pct": win_rate,
        "total_trades": total_trades,
        "final_equity": final_equity,
        "total_fees": total_fees,
        "elapsed_sec": elapsed,
    }


def _save_period_outputs(period: dict[str, str], rows: list[dict[str, Any]]) -> dict[str, Path]:
    out_dir = OUTPUT_ROOT / period["label"]
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows).sort_values(
        ["sharpe", "calmar_like", "total_return_pct"],
        ascending=[False, False, False],
    )
    csv_path = out_dir / "grid_results.csv"
    json_path = out_dir / "grid_results.json"
    summary_path = out_dir / "summary.json"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    _atomic_write_json(json_path, df.to_dict(orient="records"))

    summary = {
        "period": period,
        "grid": {
            "target_deltas": TARGET_DELTAS,
            "max_delta_diffs": MAX_DELTA_DIFFS,
            "total_combinations": len(rows),
        },
        "best_by_sharpe": df.sort_values(["sharpe", "calmar_like", "total_return_pct"], ascending=[False, False, False]).iloc[0].to_dict() if not df.empty else None,
        "best_by_calmar_like": df.sort_values(["calmar_like", "sharpe", "total_return_pct"], ascending=[False, False, False]).iloc[0].to_dict() if not df.empty else None,
        "best_by_total_return": df.sort_values(["total_return_pct", "sharpe"], ascending=[False, False]).iloc[0].to_dict() if not df.empty else None,
        "top10_by_sharpe": df.sort_values(["sharpe", "calmar_like", "total_return_pct"], ascending=[False, False, False]).head(10).to_dict(orient="records"),
        "top10_by_total_return": df.sort_values(["total_return_pct", "sharpe"], ascending=[False, False]).head(10).to_dict(orient="records"),
    }
    _atomic_write_json(summary_path, summary)
    return {"csv": csv_path, "json": json_path, "summary": summary_path}


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    periods = _build_periods()
    combos = list(itertools.product(TARGET_DELTAS, MAX_DELTA_DIFFS))

    print(f"Base config: {BASE_CONFIG}")
    print(f"target_delta grid: {TARGET_DELTAS}")
    print(f"max_delta_diff grid: {MAX_DELTA_DIFFS}")
    print(f"combinations per period: {len(combos)}")

    all_summary: list[dict[str, Any]] = []

    for period in periods:
        print("=" * 100)
        print(f"Period: {period['title']}  {period['start_date']} -> {period['end_date']}")
        rows: list[dict[str, Any]] = []
        start_t = time.perf_counter()

        for idx, (target_delta, max_delta_diff) in enumerate(combos, start=1):
            row = _run_one(period, target_delta, max_delta_diff)
            rows.append(row)
            avg_t = (time.perf_counter() - start_t) / idx
            eta_min = avg_t * (len(combos) - idx) / 60.0
            print(
                f"[{idx:>2}/{len(combos)}] "
                f"delta={target_delta:.2f} diff={max_delta_diff:.2f} -> "
                f"Ret={row['total_return_pct']:>8.2f}% DD={row['max_drawdown_pct']:>7.2f}% "
                f"Sharpe={row['sharpe']:>5.2f} PF={row['profit_factor']:>5.2f} "
                f"WR={row['win_rate_pct']:>6.2f}% Trades={row['total_trades']:>4} "
                f"({row['elapsed_sec']:.2f}s, ETA {eta_min:.1f}m)"
            )

        elapsed = time.perf_counter() - start_t
        paths = _save_period_outputs(period, rows)
        df = pd.DataFrame(rows)
        best_sharpe = df.sort_values(["sharpe", "calmar_like", "total_return_pct"], ascending=[False, False, False]).iloc[0].to_dict()
        best_return = df.sort_values(["total_return_pct", "sharpe"], ascending=[False, False]).iloc[0].to_dict()
        print(
            f"Best Sharpe: delta={best_sharpe['target_delta']:.2f} diff={best_sharpe['max_delta_diff']:.2f} "
            f"Sharpe={best_sharpe['sharpe']:.2f} Ret={best_sharpe['total_return_pct']:.2f}% DD={best_sharpe['max_drawdown_pct']:.2f}%"
        )
        print(
            f"Best Return: delta={best_return['target_delta']:.2f} diff={best_return['max_delta_diff']:.2f} "
            f"Ret={best_return['total_return_pct']:.2f}% Sharpe={best_return['sharpe']:.2f} DD={best_return['max_drawdown_pct']:.2f}%"
        )
        print(f"Saved: {paths['csv']}")
        print(f"Saved: {paths['summary']}")
        print(f"Elapsed: {elapsed/60:.1f} min")

        all_summary.append(
            {
                "period": period,
                "best_by_sharpe": best_sharpe,
                "best_by_total_return": best_return,
                "output_paths": {k: str(v) for k, v in paths.items()},
                "elapsed_sec": elapsed,
            }
        )

    _atomic_write_json(OUTPUT_ROOT / "combined_summary.json", all_summary)
    print("=" * 100)
    print(f"Combined summary saved: {OUTPUT_ROOT / 'combined_summary.json'}")


if __name__ == "__main__":
    main()
