from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd

from scripts.explore_weekend_vol_controls import run_variant, select_best

OUTPUT_DIR = Path("reports/optimizations/weekend_vol_exits")
OUTPUT_JSON = OUTPUT_DIR / "exit_experiments.json"
OUTPUT_CSV = OUTPUT_DIR / "exit_experiments.csv"
OUTPUT_REPORT = OUTPUT_DIR / "exit_report.json"

BASE_STACK = {
    "entry_time_utc": "18:00",
    "entry_realized_vol_lookback_hours": 24,
    "entry_realized_vol_max": 1.2,
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
    rows: list[tuple[str, dict]] = [("base_t18_rv120", dict(BASE_STACK))]

    for tp in (30, 40, 50, 60, 70, 80):
        rows.append((f"tp_{tp}", {**BASE_STACK, "take_profit_pct": float(tp)}))

    for sl in (80, 100, 125, 150, 200):
        rows.append((f"sl_{sl}", {**BASE_STACK, "stop_loss_pct": float(sl)}))

    for eq_loss in (0.04, 0.05, 0.06, 0.08, 0.10):
        rows.append((
            f"eqstop_{int(eq_loss * 100):02d}",
            {**BASE_STACK, "max_loss_equity_pct": eq_loss},
        ))

    return rows


def build_combo_candidates(best_single: dict) -> list[tuple[str, dict]]:
    base = dict(BASE_STACK)
    if float(best_single.get("take_profit_pct", 0.0) or 0.0) > 0:
        base["take_profit_pct"] = float(best_single["take_profit_pct"])
    if float(best_single.get("stop_loss_pct", 0.0) or 0.0) > 0:
        base["stop_loss_pct"] = float(best_single["stop_loss_pct"])
    if float(best_single.get("max_loss_equity_pct", 0.0) or 0.0) > 0:
        base["max_loss_equity_pct"] = float(best_single["max_loss_equity_pct"])

    rows: list[tuple[str, dict]] = []
    for tp in (40, 50, 60, 70):
        rows.append((f"combo_{best_single['name']}_tp_{tp}", {**base, "take_profit_pct": float(tp)}))
    for sl in (80, 100, 125, 150):
        rows.append((f"combo_{best_single['name']}_sl_{sl}", {**base, "stop_loss_pct": float(sl)}))
    for eq_loss in (0.04, 0.05, 0.06, 0.08):
        rows.append((
            f"combo_{best_single['name']}_eq_{int(eq_loss * 100):02d}",
            {**base, "max_loss_equity_pct": eq_loss},
        ))
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    print("=== Phase 1: exit-control singles on top of t18 + rv120 ===")
    for name, overrides in _candidate_rows():
        row = run_variant(name, overrides)
        row["take_profit_pct"] = float(overrides.get("take_profit_pct", 0.0) or 0.0)
        row["stop_loss_pct"] = float(overrides.get("stop_loss_pct", 0.0) or 0.0)
        row["max_loss_equity_pct"] = float(overrides.get("max_loss_equity_pct", 0.0) or 0.0)
        all_rows.append(row)
        print(
            f"{name:<24} Ann={row['annualized_return_pct']:>7.2f}% DD={row['max_drawdown_pct']:>6.2f}% "
            f"Sharpe={row['sharpe']:>4.2f} Calmar={row['calmar_like']:>5.2f} Trades={row['trades']:>4}"
        )

    baseline = next(r for r in all_rows if r["name"] == "base_t18_rv120")
    best_single = select_best(all_rows, baseline)

    combo_rows: list[dict] = []
    if best_single is not None:
        print("\nBest single exit control:", best_single["name"])
        print("\n=== Phase 2: stack best exit control with another exit control ===")
        seen_names = {r["name"] for r in all_rows}
        for name, overrides in build_combo_candidates(best_single):
            if name in seen_names:
                continue
            row = run_variant(name, overrides)
            row["take_profit_pct"] = float(overrides.get("take_profit_pct", 0.0) or 0.0)
            row["stop_loss_pct"] = float(overrides.get("stop_loss_pct", 0.0) or 0.0)
            row["max_loss_equity_pct"] = float(overrides.get("max_loss_equity_pct", 0.0) or 0.0)
            combo_rows.append(row)
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
        "best_single_exit": best_single,
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
