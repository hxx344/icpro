from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd

from scripts.explore_weekend_vol_controls import run_variant, select_best

OUTPUT_DIR = Path("reports/optimizations/weekend_vol_advanced")
OUTPUT_JSON = OUTPUT_DIR / "advanced_experiments.json"
OUTPUT_CSV = OUTPUT_DIR / "advanced_experiments.csv"
OUTPUT_REPORT = OUTPUT_DIR / "advanced_report.json"

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
        ("asym_put40_call45", {**BASE_STACK, "target_put_delta": 0.40, "target_call_delta": 0.45}),
        ("asym_put35_call45", {**BASE_STACK, "target_put_delta": 0.35, "target_call_delta": 0.45}),
        ("asym_put40_call50", {**BASE_STACK, "target_put_delta": 0.40, "target_call_delta": 0.50}),
        ("asym_put35_call50", {**BASE_STACK, "target_put_delta": 0.35, "target_call_delta": 0.50}),
        ("asym_put45_call40", {**BASE_STACK, "target_put_delta": 0.45, "target_call_delta": 0.40}),
        ("sat08_tp60", {**BASE_STACK, "early_profit_close_day": "saturday", "early_profit_close_time_utc": "08:00", "early_profit_take_profit_pct": 60.0}),
        ("sat08_tp80", {**BASE_STACK, "early_profit_close_day": "saturday", "early_profit_close_time_utc": "08:00", "early_profit_take_profit_pct": 80.0}),
        ("sat20_tp60", {**BASE_STACK, "early_profit_close_day": "saturday", "early_profit_close_time_utc": "20:00", "early_profit_take_profit_pct": 60.0}),
        ("sat20_tp80", {**BASE_STACK, "early_profit_close_day": "saturday", "early_profit_close_time_utc": "20:00", "early_profit_take_profit_pct": 80.0}),
    ]


def build_combo_candidates(best_single: dict) -> list[tuple[str, dict]]:
    base = dict(BASE_STACK)
    if float(best_single.get("target_put_delta", 0.0) or 0.0) > 0:
        base["target_put_delta"] = float(best_single["target_put_delta"])
    if float(best_single.get("target_call_delta", 0.0) or 0.0) > 0:
        base["target_call_delta"] = float(best_single["target_call_delta"])
    if float(best_single.get("early_profit_take_profit_pct", 0.0) or 0.0) > 0:
        base["early_profit_close_day"] = best_single.get("early_profit_close_day") or "saturday"
        base["early_profit_close_time_utc"] = best_single.get("early_profit_close_time_utc") or "08:00"
        base["early_profit_take_profit_pct"] = float(best_single["early_profit_take_profit_pct"])

    return [
        ("combo_asym_put40_call50_sat08_tp80", {
            **BASE_STACK,
            "target_put_delta": 0.40,
            "target_call_delta": 0.50,
            "early_profit_close_day": "saturday",
            "early_profit_close_time_utc": "08:00",
            "early_profit_take_profit_pct": 80.0,
        }),
        ("combo_asym_put40_call50_sat20_tp80", {
            **BASE_STACK,
            "target_put_delta": 0.40,
            "target_call_delta": 0.50,
            "early_profit_close_day": "saturday",
            "early_profit_close_time_utc": "20:00",
            "early_profit_take_profit_pct": 80.0,
        }),
        ("combo_asym_put40_call45_sat08_tp80", {
            **BASE_STACK,
            "target_put_delta": 0.40,
            "target_call_delta": 0.45,
            "early_profit_close_day": "saturday",
            "early_profit_close_time_utc": "08:00",
            "early_profit_take_profit_pct": 80.0,
        }),
        ("combo_best_sat08_tp80", {
            **base,
            "early_profit_close_day": "saturday",
            "early_profit_close_time_utc": "08:00",
            "early_profit_take_profit_pct": 80.0,
        }),
        ("combo_best_sat20_tp80", {
            **base,
            "early_profit_close_day": "saturday",
            "early_profit_close_time_utc": "20:00",
            "early_profit_take_profit_pct": 80.0,
        }),
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    print("=== Phase 1: asymmetric structure + conditional Saturday profit harvest ===")
    for name, overrides in _candidate_rows():
        row = run_variant(name, overrides)
        row["target_put_delta"] = float(overrides.get("target_put_delta", overrides.get("target_delta", 0.45)) or 0.45)
        row["target_call_delta"] = float(overrides.get("target_call_delta", overrides.get("target_delta", 0.45)) or 0.45)
        row["early_profit_close_day"] = str(overrides.get("early_profit_close_day", "") or "")
        row["early_profit_close_time_utc"] = str(overrides.get("early_profit_close_time_utc", "") or "")
        row["early_profit_take_profit_pct"] = float(overrides.get("early_profit_take_profit_pct", 0.0) or 0.0)
        all_rows.append(row)
        print(
            f"{name:<28} Ann={row['annualized_return_pct']:>7.2f}% DD={row['max_drawdown_pct']:>6.2f}% "
            f"Sharpe={row['sharpe']:>4.2f} Calmar={row['calmar_like']:>5.2f} Trades={row['trades']:>4}"
        )

    baseline = next(r for r in all_rows if r["name"] == "base_best")
    best_single = select_best(all_rows, baseline)

    if best_single is not None:
        print("\nBest advanced single:", best_single["name"])
        print("\n=== Phase 2: stack advanced methods ===")
        seen_names = {r["name"] for r in all_rows}
        for name, overrides in build_combo_candidates(best_single):
            if name in seen_names:
                continue
            row = run_variant(name, overrides)
            row["target_put_delta"] = float(overrides.get("target_put_delta", overrides.get("target_delta", 0.45)) or 0.45)
            row["target_call_delta"] = float(overrides.get("target_call_delta", overrides.get("target_delta", 0.45)) or 0.45)
            row["early_profit_close_day"] = str(overrides.get("early_profit_close_day", "") or "")
            row["early_profit_close_time_utc"] = str(overrides.get("early_profit_close_time_utc", "") or "")
            row["early_profit_take_profit_pct"] = float(overrides.get("early_profit_take_profit_pct", 0.0) or 0.0)
            all_rows.append(row)
            print(
                f"{name:<28} Ann={row['annualized_return_pct']:>7.2f}% DD={row['max_drawdown_pct']:>6.2f}% "
                f"Sharpe={row['sharpe']:>4.2f} Calmar={row['calmar_like']:>5.2f} Trades={row['trades']:>4}"
            )

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["calmar_like", "sharpe", "annualized_return_pct"], ascending=[False, False, False]).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    _atomic_write_json(OUTPUT_JSON, df.to_dict(orient="records"))

    report = {
        "baseline": baseline,
        "best_single_advanced": best_single,
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
