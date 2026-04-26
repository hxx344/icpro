from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd

from scripts.explore_weekend_vol_controls import run_variant, select_best

OUTPUT_DIR = Path("reports/optimizations/weekend_vol_tail_controls")
OUTPUT_JSON = OUTPUT_DIR / "tail_experiments.json"
OUTPUT_CSV = OUTPUT_DIR / "tail_experiments.csv"
OUTPUT_REPORT = OUTPUT_DIR / "tail_report.json"

BASE_STACK = {
    "entry_time_utc": "18:00",
    "entry_realized_vol_lookback_hours": 24,
    "entry_realized_vol_max": 1.2,
    "stop_loss_pct": 200.0,
}


def _atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _candidate_rows() -> list[tuple[str, dict]]:
    return [
        ("base_best", dict(BASE_STACK)),
        ("sat08_tp60_half", {
            **BASE_STACK,
            "early_profit_close_day": "saturday",
            "early_profit_close_time_utc": "08:00",
            "early_profit_take_profit_pct": 60.0,
            "early_profit_close_fraction": 0.5,
        }),
        ("sat08_tp60_75", {
            **BASE_STACK,
            "early_profit_close_day": "saturday",
            "early_profit_close_time_utc": "08:00",
            "early_profit_take_profit_pct": 60.0,
            "early_profit_close_fraction": 0.75,
        }),
        ("sat08_tp40_half", {
            **BASE_STACK,
            "early_profit_close_day": "saturday",
            "early_profit_close_time_utc": "08:00",
            "early_profit_take_profit_pct": 40.0,
            "early_profit_close_fraction": 0.5,
        }),
        ("sat08_dir4_full", {
            **BASE_STACK,
            "directional_adjust_day": "saturday",
            "directional_adjust_time_utc": "08:00",
            "directional_adjust_move_pct": 0.04,
            "directional_adjust_close_fraction": 1.0,
        }),
        ("sat08_dir5_full", {
            **BASE_STACK,
            "directional_adjust_day": "saturday",
            "directional_adjust_time_utc": "08:00",
            "directional_adjust_move_pct": 0.05,
            "directional_adjust_close_fraction": 1.0,
        }),
        ("sat08_dir6_full", {
            **BASE_STACK,
            "directional_adjust_day": "saturday",
            "directional_adjust_time_utc": "08:00",
            "directional_adjust_move_pct": 0.06,
            "directional_adjust_close_fraction": 1.0,
        }),
        ("sat08_dir4_half", {
            **BASE_STACK,
            "directional_adjust_day": "saturday",
            "directional_adjust_time_utc": "08:00",
            "directional_adjust_move_pct": 0.04,
            "directional_adjust_close_fraction": 0.5,
        }),
        ("sat08_dir5_half", {
            **BASE_STACK,
            "directional_adjust_day": "saturday",
            "directional_adjust_time_utc": "08:00",
            "directional_adjust_move_pct": 0.05,
            "directional_adjust_close_fraction": 0.5,
        }),
    ]


def build_combo_candidates() -> list[tuple[str, dict]]:
    return [
        ("combo_tp60half_dir4full", {
            **BASE_STACK,
            "early_profit_close_day": "saturday",
            "early_profit_close_time_utc": "08:00",
            "early_profit_take_profit_pct": 60.0,
            "early_profit_close_fraction": 0.5,
            "directional_adjust_day": "saturday",
            "directional_adjust_time_utc": "08:00",
            "directional_adjust_move_pct": 0.04,
            "directional_adjust_close_fraction": 1.0,
        }),
        ("combo_tp60half_dir5full", {
            **BASE_STACK,
            "early_profit_close_day": "saturday",
            "early_profit_close_time_utc": "08:00",
            "early_profit_take_profit_pct": 60.0,
            "early_profit_close_fraction": 0.5,
            "directional_adjust_day": "saturday",
            "directional_adjust_time_utc": "08:00",
            "directional_adjust_move_pct": 0.05,
            "directional_adjust_close_fraction": 1.0,
        }),
        ("combo_tp40half_dir4full", {
            **BASE_STACK,
            "early_profit_close_day": "saturday",
            "early_profit_close_time_utc": "08:00",
            "early_profit_take_profit_pct": 40.0,
            "early_profit_close_fraction": 0.5,
            "directional_adjust_day": "saturday",
            "directional_adjust_time_utc": "08:00",
            "directional_adjust_move_pct": 0.04,
            "directional_adjust_close_fraction": 1.0,
        }),
        ("combo_tp60half_dir4half", {
            **BASE_STACK,
            "early_profit_close_day": "saturday",
            "early_profit_close_time_utc": "08:00",
            "early_profit_take_profit_pct": 60.0,
            "early_profit_close_fraction": 0.5,
            "directional_adjust_day": "saturday",
            "directional_adjust_time_utc": "08:00",
            "directional_adjust_move_pct": 0.04,
            "directional_adjust_close_fraction": 0.5,
        }),
        ("combo_tp60_75_dir4full", {
            **BASE_STACK,
            "early_profit_close_day": "saturday",
            "early_profit_close_time_utc": "08:00",
            "early_profit_take_profit_pct": 60.0,
            "early_profit_close_fraction": 0.75,
            "directional_adjust_day": "saturday",
            "directional_adjust_time_utc": "08:00",
            "directional_adjust_move_pct": 0.04,
            "directional_adjust_close_fraction": 1.0,
        }),
    ]


def _annotate(row: dict, overrides: dict) -> dict:
    row = dict(row)
    for key in [
        "early_profit_close_day",
        "early_profit_close_time_utc",
        "directional_adjust_day",
        "directional_adjust_time_utc",
    ]:
        row[key] = str(overrides.get(key, "") or "")
    for key in [
        "early_profit_take_profit_pct",
        "early_profit_close_fraction",
        "directional_adjust_move_pct",
        "directional_adjust_close_fraction",
    ]:
        row[key] = float(overrides.get(key, 0.0) or 0.0)
    return row


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    print("=== Phase 1: partial profit + directional tail cut ===")
    for name, overrides in _candidate_rows():
        row = _annotate(run_variant(name, overrides), overrides)
        all_rows.append(row)
        print(
            f"{name:<24} Ann={row['annualized_return_pct']:>7.2f}% DD={row['max_drawdown_pct']:>6.2f}% "
            f"Sharpe={row['sharpe']:>4.2f} Calmar={row['calmar_like']:>5.2f} Trades={row['trades']:>4}"
        )

    baseline = next(r for r in all_rows if r["name"] == "base_best")
    best_single = select_best(all_rows, baseline)

    print("\nBest single tail control:", best_single["name"] if best_single else "n/a")
    print("\n=== Phase 2: stacked tail controls ===")
    for name, overrides in build_combo_candidates():
        row = _annotate(run_variant(name, overrides), overrides)
        all_rows.append(row)
        print(
            f"{name:<24} Ann={row['annualized_return_pct']:>7.2f}% DD={row['max_drawdown_pct']:>6.2f}% "
            f"Sharpe={row['sharpe']:>4.2f} Calmar={row['calmar_like']:>5.2f} Trades={row['trades']:>4}"
        )

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["calmar_like", "sharpe", "annualized_return_pct"], ascending=[False, False, False]).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    _atomic_write_json(OUTPUT_JSON, df.to_dict(orient="records"))

    report = {
        "baseline": baseline,
        "best_single_tail": best_single,
        "best_overall": select_best(all_rows, baseline),
        "top10": df.head(10).to_dict(orient="records"),
    }
    _atomic_write_json(OUTPUT_REPORT, report)
    print("\nSaved:")
    print(f"- {OUTPUT_CSV}")
    print(f"- {OUTPUT_JSON}")
    print(f"- {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
