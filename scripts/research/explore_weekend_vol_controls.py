from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import pandas as pd

from options_backtest.analytics.metrics import compute_metrics
from options_backtest.cli import _load_strategy
from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine

BASE_CONFIG = Path("configs/backtest/weekend_vol_btc_hourly.yaml")
OUTPUT_DIR = Path("reports/optimizations/weekend_vol_controls")
OUTPUT_JSON = OUTPUT_DIR / "control_experiments.json"
OUTPUT_CSV = OUTPUT_DIR / "control_experiments.csv"
OUTPUT_REPORT = OUTPUT_DIR / "control_report.json"

BASE_PARAMS = {
    "target_delta": 0.45,
    "wing_delta": 0.0,
    "leverage": 4.0,
    "max_delta_diff": 0.15,
    "entry_day": "friday",
    "entry_time_utc": "16:00",
    "close_day": "sunday",
    "close_time_utc": "08:00",
    "expire_day": "sunday",
    "expiry_selection": "first_after_close",
    "default_iv": 0.60,
    "quantity": 0.1,
    "quantity_step": 0.1,
    "max_positions": 1,
    "compound": True,
    "expiry_tolerance_hours": 12,
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


def run_variant(name: str, overrides: dict) -> dict:
    cfg = Config.from_yaml(BASE_CONFIG)
    cfg.report.generate_plots = False
    cfg.backtest.show_progress = False
    cfg.backtest.name = name
    params = dict(BASE_PARAMS)
    params.update(overrides)
    cfg.strategy.params.update(params)
    strategy = _load_strategy(cfg.strategy.name, cfg.strategy.params)
    engine = BacktestEngine(cfg, strategy)
    t0 = time.perf_counter()
    results = engine.run()
    elapsed = time.perf_counter() - t0
    metrics = compute_metrics(results)

    max_dd = abs(float(metrics.get("max_drawdown", 0.0))) * 100
    ann = float(metrics.get("annualized_return", 0.0)) * 100
    total_ret = float(metrics.get("total_return", 0.0)) * 100
    sharpe = float(metrics.get("sharpe_ratio", 0.0))
    calmar = ann / max_dd if max_dd > 0 else math.inf
    trades = int(metrics.get("total_trades", 0))
    win_rate = float(metrics.get("win_rate", 0.0)) * 100
    pf = float(metrics.get("profit_factor", 0.0))

    row = {
        "name": name,
        "entry_time_utc": params.get("entry_time_utc"),
        "entry_abs_move_lookback_hours": params.get("entry_abs_move_lookback_hours", 0),
        "entry_abs_move_max_pct": params.get("entry_abs_move_max_pct", 0.0),
        "entry_realized_vol_lookback_hours": params.get("entry_realized_vol_lookback_hours", 0),
        "entry_realized_vol_max": params.get("entry_realized_vol_max", 0.0),
        "total_return_pct": total_ret,
        "annualized_return_pct": ann,
        "max_drawdown_pct": max_dd,
        "sharpe": sharpe,
        "calmar_like": calmar,
        "profit_factor": pf,
        "win_rate_pct": win_rate,
        "trades": trades,
        "elapsed_sec": elapsed,
    }
    return row


def _candidate_rows() -> list[tuple[str, dict]]:
    rows: list[tuple[str, dict]] = []
    rows.append(("baseline", {}))

    for entry_time in ("17:00", "18:00", "19:00", "20:00"):
        rows.append((
            f"time_{entry_time[:2]}",
            {
                "entry_time_utc": entry_time,
            },
        ))

    for move_max in (0.06, 0.08, 0.10, 0.12):
        rows.append((
            f"move24_{int(move_max * 100):02d}",
            {
                "entry_abs_move_lookback_hours": 24,
                "entry_abs_move_max_pct": move_max,
            },
        ))

    for rv_max in (0.80, 1.00, 1.20, 1.40):
        rows.append((
            f"rv24_{int(rv_max * 100):03d}",
            {
                "entry_realized_vol_lookback_hours": 24,
                "entry_realized_vol_max": rv_max,
            },
        ))

    return rows


def select_best(rows: list[dict], baseline: dict) -> dict | None:
    candidates = [r for r in rows if r["name"] != "baseline"]
    if not candidates:
        return None

    baseline_ann = baseline["annualized_return_pct"]
    feasible = [r for r in candidates if r["annualized_return_pct"] >= baseline_ann * 0.85]
    if not feasible:
        feasible = candidates

    feasible.sort(
        key=lambda r: (
            r["calmar_like"],
            -r["max_drawdown_pct"],
            r["sharpe"],
            r["annualized_return_pct"],
        ),
        reverse=True,
    )
    return feasible[0]


def build_combo_candidates(best_single: dict) -> list[tuple[str, dict]]:
    base = {
        "entry_time_utc": best_single.get("entry_time_utc") or BASE_PARAMS["entry_time_utc"],
        "entry_abs_move_lookback_hours": int(best_single.get("entry_abs_move_lookback_hours", 0) or 0),
        "entry_abs_move_max_pct": float(best_single.get("entry_abs_move_max_pct", 0.0) or 0.0),
        "entry_realized_vol_lookback_hours": int(best_single.get("entry_realized_vol_lookback_hours", 0) or 0),
        "entry_realized_vol_max": float(best_single.get("entry_realized_vol_max", 0.0) or 0.0),
    }
    rows: list[tuple[str, dict]] = []
    for entry_time in ("16:00", "17:00", "18:00", "19:00", "20:00"):
        rows.append((f"combo_{best_single['name']}_t{entry_time[:2]}", {**base, "entry_time_utc": entry_time}))

    rows.append((
        f"combo_{best_single['name']}_move06",
        {
            **base,
            "entry_abs_move_lookback_hours": 24,
            "entry_abs_move_max_pct": 0.06,
        },
    ))
    rows.append((
        f"combo_{best_single['name']}_rv100",
        {
            **base,
            "entry_realized_vol_lookback_hours": 24,
            "entry_realized_vol_max": 1.00,
        },
    ))
    rows.append((
        f"combo_{best_single['name']}_rv120",
        {
            **base,
            "entry_realized_vol_lookback_hours": 24,
            "entry_realized_vol_max": 1.20,
        },
    ))
    rows.append((
        f"combo_{best_single['name']}_rv140",
        {
            **base,
            "entry_realized_vol_lookback_hours": 24,
            "entry_realized_vol_max": 1.40,
        },
    ))
    rows.append((
        f"combo_{best_single['name']}_move06_rv120",
        {
            **base,
            "entry_abs_move_lookback_hours": 24,
            "entry_abs_move_max_pct": 0.06,
            "entry_realized_vol_lookback_hours": 24,
            "entry_realized_vol_max": 1.20,
        },
    ))
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    print("=== Phase 1: baseline + single controls ===")
    for name, overrides in _candidate_rows():
        row = run_variant(name, overrides)
        all_rows.append(row)
        print(
            f"{name:<24} Ret={row['total_return_pct']:>8.2f}% "
            f"Ann={row['annualized_return_pct']:>7.2f}% DD={row['max_drawdown_pct']:>6.2f}% "
            f"Sharpe={row['sharpe']:>4.2f} Calmar={row['calmar_like']:>5.2f} Trades={row['trades']:>4}"
        )

    baseline = next(r for r in all_rows if r["name"] == "baseline")
    best_single = select_best(all_rows, baseline)

    if best_single is not None:
        print("\nBest single control candidate:", best_single["name"])

    combo_rows: list[dict] = []
    if best_single is not None:
        print("\n=== Phase 2: stack best control with timing / second control ===")
        seen_names = {r["name"] for r in all_rows}
        for name, overrides in build_combo_candidates(best_single):
            if name in seen_names:
                continue
            row = run_variant(name, overrides)
            combo_rows.append(row)
            all_rows.append(row)
            print(
                f"{name:<24} Ret={row['total_return_pct']:>8.2f}% "
                f"Ann={row['annualized_return_pct']:>7.2f}% DD={row['max_drawdown_pct']:>6.2f}% "
                f"Sharpe={row['sharpe']:>4.2f} Calmar={row['calmar_like']:>5.2f} Trades={row['trades']:>4}"
            )

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["calmar_like", "sharpe", "annualized_return_pct"], ascending=[False, False, False]).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    _atomic_write_json(OUTPUT_JSON, df.to_dict(orient="records"))

    best_overall = select_best(all_rows, baseline)
    report = {
        "baseline": baseline,
        "best_single_control": best_single,
        "best_overall": best_overall,
        "top10": df.head(10).to_dict(orient="records"),
    }
    _atomic_write_json(OUTPUT_REPORT, report)
    print("\nSaved:")
    print(f"- {OUTPUT_CSV}")
    print(f"- {OUTPUT_JSON}")
    print(f"- {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
