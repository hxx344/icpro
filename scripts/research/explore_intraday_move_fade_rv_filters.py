from __future__ import annotations

import argparse
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
from scripts.optimize_intraday_move_fade_drawdown import build_base_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore RV filters for IntradayMoveFade0DTE")
    parser.add_argument("--underlying", default="BTC")
    parser.add_argument("--x", type=float, default=0.005)
    parser.add_argument("--y", type=int, default=18)
    parser.add_argument("--z", type=float, default=0.03)
    parser.add_argument("--max-loss-equity", type=float, default=0.125)
    parser.add_argument("--initial-usd", type=float, default=10_000.0)
    return parser.parse_args()


def run_case(base_cfg, args: argparse.Namespace, lookback: int | None, rv_max: float | None) -> dict:
    cfg = base_cfg.model_copy(deep=True)
    cfg.strategy.name = "IntradayMoveFade0DTE"
    cfg.strategy.params = {
        "quantity": 1.0,
        "compound": False,
        "move_threshold_pct": args.x,
        "entry_hour": args.y,
        "otm_pct": args.z,
        "max_loss_equity_pct": args.max_loss_equity,
    }
    if lookback is not None and rv_max is not None:
        cfg.strategy.params["entry_realized_vol_lookback_hours"] = int(lookback)
        cfg.strategy.params["entry_realized_vol_max"] = float(rv_max)
        cfg.backtest.name = f"intraday_move_fade_rv{lookback}_{rv_max:.2f}"
    else:
        cfg.backtest.name = "intraday_move_fade_rv_baseline"

    strategy = _load_strategy(cfg.strategy.name, cfg.strategy.params)
    engine = BacktestEngine(cfg, strategy)
    t0 = time.perf_counter()
    results = engine.run()
    elapsed = time.perf_counter() - t0
    metrics = compute_metrics(results)
    return {
        "lookback": lookback,
        "rv_max": rv_max,
        "total_return_pct": float(metrics.get("total_return", 0.0)) * 100.0,
        "max_drawdown_pct": abs(float(metrics.get("max_drawdown", 0.0))) * 100.0,
        "sharpe": float(metrics.get("sharpe_ratio", 0.0)),
        "profit_factor": float(metrics.get("profit_factor", 0.0)),
        "win_rate_pct": float(metrics.get("win_rate", 0.0)) * 100.0,
        "trades": int(metrics.get("total_trades", 0)),
        "elapsed_sec": elapsed,
    }


def main() -> None:
    args = parse_args()
    base_cfg = build_base_config(args)

    out_dir = REPO_ROOT / "reports" / "optimizations" / "intraday_move_fade_0dte"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "strict_rv_filter_scan_y18_x0050_z0300.csv"
    json_path = out_dir / "strict_rv_filter_scan_y18_x0050_z0300.json"

    cases: list[tuple[int | None, float | None]] = [(None, None)]
    for lookback in (4, 6, 8, 12):
        for rv_max in (0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
            cases.append((lookback, rv_max))

    rows: list[dict] = []
    total = len(cases)
    for idx, (lookback, rv_max) in enumerate(cases, start=1):
        label = "baseline" if lookback is None else f"rv{lookback} <= {rv_max:.2f}"
        print(f"[{idx}/{total}] {label}", flush=True)
        row = run_case(base_cfg, args, lookback, rv_max)
        rows.append(row)
        print(
            f"    ret={row['total_return_pct']:.2f}% dd={row['max_drawdown_pct']:.2f}% "
            f"sh={row['sharpe']:.2f} trades={row['trades']} t={row['elapsed_sec']:.1f}s",
            flush=True,
        )

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)

    print("\n=== BEST DD PER LOOKBACK ===", flush=True)
    per_lb = df[df["lookback"].notna()].sort_values(
        ["lookback", "max_drawdown_pct", "total_return_pct"], ascending=[True, True, False]
    ).groupby("lookback", as_index=False).head(1)
    print(per_lb.to_string(index=False), flush=True)

    print("\n=== QUALIFIED DD <= 15% ===", flush=True)
    qualified = df[(df["lookback"].notna()) & (df["max_drawdown_pct"] <= 15.0)].sort_values(
        ["total_return_pct", "sharpe"], ascending=[False, False]
    )
    if qualified.empty:
        print("NONE", flush=True)
    else:
        print(qualified.to_string(index=False), flush=True)

    print(f"\nSaved: {csv_path}", flush=True)
    print(f"Saved: {json_path}", flush=True)


if __name__ == "__main__":
    main()