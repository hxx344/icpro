from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd

from scripts.explore_weekend_vol_controls import run_variant

OUTPUT_DIR = Path("reports/optimizations/weekend_vol_novel")
OUTPUT_JSON = OUTPUT_DIR / "novel_experiments.json"
OUTPUT_CSV = OUTPUT_DIR / "novel_experiments.csv"
OUTPUT_REPORT = OUTPUT_DIR / "novel_report.json"

BASE_STACK = {
    "entry_time_utc": "18:00",
    "entry_realized_vol_lookback_hours": 24,
    "entry_realized_vol_max": 1.2,
    "stop_loss_pct": 200.0,
    "early_profit_close_day": "saturday",
    "early_profit_close_time_utc": "08:00",
    "early_profit_take_profit_pct": 60.0,
    "early_profit_close_fraction": 0.5,
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


def _candidates() -> list[tuple[str, dict, str]]:
    return [
        ("base_best_4x", dict(BASE_STACK), "baseline"),
        (
            "reactive_hedge_4pct_d10_half",
            {
                **BASE_STACK,
                "reactive_hedge_move_pct": 0.04,
                "reactive_hedge_delta": 0.10,
                "reactive_hedge_fraction": 0.5,
            },
            "reactive_hedge",
        ),
        (
            "reactive_hedge_5pct_d05_full",
            {
                **BASE_STACK,
                "reactive_hedge_move_pct": 0.05,
                "reactive_hedge_delta": 0.05,
                "reactive_hedge_fraction": 1.0,
            },
            "reactive_hedge",
        ),
        (
            "condwing_rv100_w03",
            {
                **BASE_STACK,
                "conditional_wing_rv_threshold": 1.00,
                "conditional_wing_delta": 0.03,
            },
            "conditional_wing",
        ),
        (
            "condwing_rv100_w05",
            {
                **BASE_STACK,
                "conditional_wing_rv_threshold": 1.00,
                "conditional_wing_delta": 0.05,
            },
            "conditional_wing",
        ),
        (
            "sat08_shorthold",
            {
                "entry_time_utc": "18:00",
                "entry_realized_vol_lookback_hours": 24,
                "entry_realized_vol_max": 1.2,
                "stop_loss_pct": 200.0,
                "close_day": "saturday",
                "close_time_utc": "08:00",
                "expire_day": "saturday",
            },
            "hold_horizon",
        ),
        (
            "sat20_shorthold",
            {
                "entry_time_utc": "18:00",
                "entry_realized_vol_lookback_hours": 24,
                "entry_realized_vol_max": 1.2,
                "stop_loss_pct": 200.0,
                "close_day": "saturday",
                "close_time_utc": "20:00",
                "expire_day": "sunday",
            },
            "hold_horizon",
        ),
    ]


def _annotate(row: dict, overrides: dict, family: str) -> dict:
    row = dict(row)
    row["family"] = family
    for key in [
        "reactive_hedge_move_pct",
        "reactive_hedge_delta",
        "reactive_hedge_fraction",
        "conditional_wing_rv_threshold",
        "conditional_wing_delta",
    ]:
        row[key] = float(overrides.get(key, 0.0) or 0.0)
    for key in ["close_day", "close_time_utc", "expire_day"]:
        row[key] = str(overrides.get(key, row.get(key, "")) or "")
    return row


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for name, overrides, family in _candidates():
        row = _annotate(run_variant(name, overrides), overrides, family)
        rows.append(row)
        print(
            f"{name:<28} Ann={row['annualized_return_pct']:>7.2f}% DD={row['max_drawdown_pct']:>6.2f}% "
            f"Sharpe={row['sharpe']:>4.2f} Calmar={row['calmar_like']:>5.2f} Trades={row['trades']:>4}"
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["calmar_like", "annualized_return_pct"], ascending=[False, False]).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    _atomic_write_json(OUTPUT_JSON, df.to_dict(orient="records"))

    baseline = next(row for row in rows if row["name"] == "base_best_4x")
    report = {
        "baseline": baseline,
        "best_novel_overall": df.iloc[0].to_dict(),
        "best_by_family": {
            family: grp.sort_values(["calmar_like", "annualized_return_pct"], ascending=[False, False]).iloc[0].to_dict()
            for family, grp in df.groupby("family")
        },
        "all_results": df.to_dict(orient="records"),
    }
    _atomic_write_json(OUTPUT_REPORT, report)

    print("\nSaved:")
    print(f"- {OUTPUT_CSV}")
    print(f"- {OUTPUT_JSON}")
    print(f"- {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
