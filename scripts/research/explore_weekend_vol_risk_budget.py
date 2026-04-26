from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd

from scripts.explore_weekend_vol_controls import run_variant

OUTPUT_DIR = Path("reports/optimizations/weekend_vol_risk_budget")
OUTPUT_JSON = OUTPUT_DIR / "risk_budget_experiments.json"
OUTPUT_CSV = OUTPUT_DIR / "risk_budget_experiments.csv"
OUTPUT_REPORT = OUTPUT_DIR / "risk_budget_report.json"

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


def _candidates() -> list[tuple[str, dict]]:
    return [
        ("best_4x", {**BASE_STACK, "leverage": 4.0}),
        ("best_3_5x", {**BASE_STACK, "leverage": 3.5}),
        ("best_3x", {**BASE_STACK, "leverage": 3.0}),
        ("best_2_5x", {**BASE_STACK, "leverage": 2.5}),
        ("best_2x", {**BASE_STACK, "leverage": 2.0}),
        ("best_1_5x", {**BASE_STACK, "leverage": 1.5}),
        ("best_3x_cd7", {**BASE_STACK, "leverage": 3.0, "loss_cooldown_days": 7}),
        ("best_2_5x_cd7", {**BASE_STACK, "leverage": 2.5, "loss_cooldown_days": 7}),
        ("best_2x_cd7", {**BASE_STACK, "leverage": 2.0, "loss_cooldown_days": 7}),
    ]


def _annotate(row: dict, overrides: dict) -> dict:
    row = dict(row)
    row["leverage"] = float(overrides.get("leverage", 0.0) or 0.0)
    row["loss_cooldown_days"] = int(overrides.get("loss_cooldown_days", 0) or 0)
    return row


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for name, overrides in _candidates():
        row = _annotate(run_variant(name, overrides), overrides)
        rows.append(row)
        print(
            f"{name:<14} Ann={row['annualized_return_pct']:>7.2f}% DD={row['max_drawdown_pct']:>6.2f}% "
            f"Sharpe={row['sharpe']:>4.2f} Calmar={row['calmar_like']:>5.2f} Trades={row['trades']:>4}"
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["max_drawdown_pct", "annualized_return_pct"], ascending=[True, False]).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    _atomic_write_json(OUTPUT_JSON, df.to_dict(orient="records"))

    target_10 = df[df["max_drawdown_pct"] <= 10.0].sort_values(
        ["annualized_return_pct", "sharpe"], ascending=[False, False]
    )
    report = {
        "base_4x": next(row for row in rows if row["name"] == "best_4x"),
        "best_under_10dd": None if target_10.empty else target_10.iloc[0].to_dict(),
        "target_10_candidates": target_10.to_dict(orient="records"),
        "frontier": df.to_dict(orient="records"),
    }
    _atomic_write_json(OUTPUT_REPORT, report)

    print("\nSaved:")
    print(f"- {OUTPUT_CSV}")
    print(f"- {OUTPUT_JSON}")
    print(f"- {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
