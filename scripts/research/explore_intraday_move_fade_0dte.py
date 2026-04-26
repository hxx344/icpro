from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import sys
import time
from itertools import product
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
from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from scripts.research.run_cached_hourly_backtest import detect_hourly_coverage


UNDERLYING = "BTC"
INITIAL_USD = 10_000.0
OUTPUT_DIR = Path("reports/optimizations/intraday_move_fade_0dte")
OUTPUT_CSV = OUTPUT_DIR / "grid_results.csv"
OUTPUT_JSON = OUTPUT_DIR / "grid_results.json"
OUTPUT_REPORT = OUTPUT_DIR / "grid_report.json"

X_VALUES = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030]
Y_VALUES = [16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7]
Z_VALUES = [0.005, 0.010, 0.020, 0.030]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid-search intraday 0DTE move-fade strategy on cached hourly data.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="并发 worker 数；1 表示串行运行。默认最多 4 个。",
    )
    return parser.parse_args()


def build_base_config() -> Config:
    coverage = detect_hourly_coverage(UNDERLYING)
    cfg = Config()
    cfg.backtest.name = "Intraday Move Fade 0DTE"
    cfg.backtest.start_date = coverage.start_date
    cfg.backtest.end_date = coverage.end_date
    cfg.backtest.time_step = "1h"
    cfg.backtest.underlying = UNDERLYING
    cfg.backtest.margin_mode = "USD"
    cfg.backtest.use_bs_only = False
    cfg.backtest.option_data_source = "options_hourly"
    cfg.backtest.option_snapshot_pick = "close"
    cfg.backtest.iv_mode = "fixed"
    cfg.backtest.fixed_iv = 0.60
    cfg.backtest.show_progress = False

    first_price = 1.0
    try:
        import pandas as pd  # local import for script start speed

        udf = pd.read_parquet(Path("data/underlying") / "btc_index_60.parquet", columns=["close"])
        if not udf.empty:
            first_price = float(udf["close"].iloc[0])
    except Exception:
        first_price = 1.0
    cfg.account.initial_balance = INITIAL_USD / max(first_price, 1.0)

    cfg.execution.slippage = 0.0001
    cfg.execution.taker_fee = 0.00024
    cfg.execution.maker_fee = 0.00024
    cfg.execution.min_fee = 0.00024
    cfg.execution.max_fee_pct = 0.10
    cfg.execution.delivery_fee = 0.00015
    cfg.execution.delivery_fee_max_pct = 0.10

    cfg.strategy.name = "IntradayMoveFade0DTE"
    cfg.strategy.params = {
        "quantity": 1.0,
        "compound": False,
    }
    cfg.report.output_dir = str(OUTPUT_DIR)
    cfg.report.generate_plots = False
    return cfg


def run_variant(base_cfg: Config, x: float, y: int, z: float) -> dict:
    cfg = base_cfg.model_copy(deep=True)
    cfg.backtest.name = f"movefade_x{int(x*10000):04d}_y{y:02d}_z{int(z*10000):04d}"
    cfg.strategy.params.update(
        {
            "move_threshold_pct": x,
            "entry_hour": y,
            "otm_pct": z,
        }
    )
    strategy = _load_strategy(cfg.strategy.name, cfg.strategy.params)
    engine = BacktestEngine(cfg, strategy)
    t0 = time.perf_counter()
    results = engine.run()
    elapsed = time.perf_counter() - t0
    metrics = compute_metrics(results)
    max_dd = abs(float(metrics.get("max_drawdown", 0.0))) * 100.0
    ann = float(metrics.get("annualized_return", 0.0)) * 100.0
    total_ret = float(metrics.get("total_return", 0.0)) * 100.0
    sharpe = float(metrics.get("sharpe_ratio", 0.0))
    calmar = ann / max_dd if max_dd > 0 else math.inf
    trades = int(metrics.get("total_trades", 0))
    win_rate = float(metrics.get("win_rate", 0.0)) * 100.0
    pf = float(metrics.get("profit_factor", 0.0))
    entries = ((results.get("strategy_diagnostics") or {}).get("intraday_move_fade_entries") or [])
    return {
        "move_threshold_pct": x,
        "entry_hour_utc": y,
        "otm_pct": z,
        "total_return_pct": total_ret,
        "annualized_return_pct": ann,
        "max_drawdown_pct": max_dd,
        "sharpe": sharpe,
        "calmar_like": calmar,
        "profit_factor": pf,
        "win_rate_pct": win_rate,
        "trades": trades,
        "signal_days": len(entries),
        "elapsed_sec": elapsed,
    }


def _run_variant_task(task: tuple[dict, float, int, float]) -> dict:
    base_cfg_data, x, y, z = task
    base_cfg = Config.model_validate(base_cfg_data)
    return run_variant(base_cfg, x, y, z)


def _iter_grid() -> list[tuple[float, int, float]]:
    return list(product(X_VALUES, Y_VALUES, Z_VALUES))


def _print_row(row: dict) -> None:
    print(
        f"x={row['move_threshold_pct']:.2%} y={int(row['entry_hour_utc']):02d}:00 z={row['otm_pct']:.2%} | "
        f"ann={row['annualized_return_pct']:>7.2f}% dd={row['max_drawdown_pct']:>6.2f}% "
        f"sharpe={row['sharpe']:>4.2f} calmar={row['calmar_like']:>5.2f} trades={row['trades']:>3}"
    )


def main() -> None:
    args = _parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_cfg = build_base_config()
    tasks = [(base_cfg.model_dump(mode="python"), x, y, z) for x, y, z in _iter_grid()]
    rows: list[dict] = []
    if args.workers <= 1:
        for task in tasks:
            row = _run_variant_task(task)
            rows.append(row)
            _print_row(row)
    else:
        print(f"Running grid with {args.workers} workers across {len(tasks)} combinations...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_run_variant_task, task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                rows.append(row)
                _print_row(row)

    rows_sorted = sorted(
        rows,
        key=lambda r: (r["calmar_like"], -r["max_drawdown_pct"], r["sharpe"], r["annualized_return_pct"]),
        reverse=True,
    )
    df = pd.DataFrame(rows_sorted)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(
            {
                "underlying": UNDERLYING,
                "coverage_start": base_cfg.backtest.start_date,
                "coverage_end": base_cfg.backtest.end_date,
                "quantity": base_cfg.strategy.params.get("quantity"),
                "grid": {
                    "x_values": X_VALUES,
                    "y_values": Y_VALUES,
                    "z_values": Z_VALUES,
                },
                "best": rows_sorted[0] if rows_sorted else {},
                "top10": rows_sorted[:10],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("\nTOP10")
    print(df.head(10).to_string(index=False))
    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_JSON}")
    print(f"Saved: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()