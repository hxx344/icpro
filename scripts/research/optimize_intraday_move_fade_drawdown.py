from __future__ import annotations

import argparse
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
from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from scripts.research.run_cached_hourly_backtest import detect_hourly_coverage

logger.remove()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize drawdown for IntradayMoveFade0DTE")
    parser.add_argument("--underlying", default="BTC")
    parser.add_argument("--x", type=float, default=0.005)
    parser.add_argument("--y", type=int, default=18)
    parser.add_argument("--z", type=float, default=0.03)
    parser.add_argument("--target-dd", type=float, default=15.0)
    parser.add_argument("--initial-usd", type=float, default=10_000.0)
    return parser.parse_args()


def build_base_config(args: argparse.Namespace) -> Config:
    coverage = detect_hourly_coverage(args.underlying)
    cfg = Config()
    cfg.backtest.name = "Intraday Move Fade DD Optimization"
    cfg.backtest.start_date = coverage.start_date
    cfg.backtest.end_date = coverage.end_date
    cfg.backtest.time_step = "1h"
    cfg.backtest.underlying = args.underlying.upper()
    cfg.backtest.margin_mode = "USD"
    cfg.backtest.use_bs_only = False
    cfg.backtest.option_data_source = "options_hourly"
    cfg.backtest.option_snapshot_pick = "close"
    cfg.backtest.iv_mode = "fixed"
    cfg.backtest.fixed_iv = 0.60
    cfg.backtest.show_progress = False

    first_price = 1.0
    try:
        udf = pd.read_parquet(REPO_ROOT / "data" / "underlying" / f"{args.underlying.lower()}_index_60.parquet", columns=["close"])
        if not udf.empty:
            first_price = float(udf["close"].iloc[0])
    except Exception:
        pass
    cfg.account.initial_balance = args.initial_usd / max(first_price, 1.0)

    cfg.execution.slippage = 0.0001
    cfg.execution.taker_fee = 0.00024
    cfg.execution.maker_fee = 0.00024
    cfg.execution.min_fee = 0.00024
    cfg.execution.max_fee_pct = 0.10
    cfg.execution.delivery_fee = 0.00015
    cfg.execution.delivery_fee_max_pct = 0.10
    cfg.execution.require_touch_quote = True
    cfg.execution.require_real_quote_source = True
    return cfg


def run_case(base_cfg: Config, args: argparse.Namespace, *, take_profit_pct: float = 0.0, stop_loss_pct: float = 0.0, underlying_move_stop_pct: float = 0.0, max_loss_equity_pct: float = 0.0) -> dict:
    cfg = base_cfg.model_copy(deep=True)
    cfg.backtest.name = (
        f"movefade_ddopt_x{int(args.x*10000):04d}_y{args.y:02d}_z{int(args.z*10000):04d}"
        f"_tp{take_profit_pct:g}_sl{stop_loss_pct:g}_um{underlying_move_stop_pct:g}_me{max_loss_equity_pct:g}"
    )
    cfg.strategy.name = "IntradayMoveFade0DTE"
    cfg.strategy.params = {
        "quantity": 1.0,
        "compound": False,
        "move_threshold_pct": args.x,
        "entry_hour": args.y,
        "otm_pct": args.z,
        "take_profit_pct": take_profit_pct,
        "stop_loss_pct": stop_loss_pct,
        "underlying_move_stop_pct": underlying_move_stop_pct,
        "max_loss_equity_pct": max_loss_equity_pct,
    }
    strategy = _load_strategy(cfg.strategy.name, cfg.strategy.params)
    engine = BacktestEngine(cfg, strategy)
    t0 = time.perf_counter()
    results = engine.run()
    elapsed = time.perf_counter() - t0
    metrics = compute_metrics(results)
    exits = ((results.get("strategy_diagnostics") or {}).get("intraday_move_fade_exits") or [])
    return {
        "take_profit_pct": float(take_profit_pct),
        "stop_loss_pct": float(stop_loss_pct),
        "underlying_move_stop_pct": float(underlying_move_stop_pct),
        "max_loss_equity_pct": float(max_loss_equity_pct),
        "annualized_return_pct": float(metrics.get("annualized_return", 0.0)) * 100.0,
        "total_return_pct": float(metrics.get("total_return", 0.0)) * 100.0,
        "max_drawdown_pct": abs(float(metrics.get("max_drawdown", 0.0))) * 100.0,
        "sharpe": float(metrics.get("sharpe_ratio", 0.0)),
        "profit_factor": float(metrics.get("profit_factor", 0.0)),
        "win_rate_pct": float(metrics.get("win_rate", 0.0)) * 100.0,
        "trades": int(metrics.get("total_trades", 0)),
        "exit_count": len(exits),
        "elapsed_sec": elapsed,
    }


def print_case(idx: int, total: int, row: dict) -> None:
    print(
        f"[{idx:>2d}/{total}] "
        f"tp={row['take_profit_pct']:>6.2f} sl={row['stop_loss_pct']:>6.2f} "
        f"um={row['underlying_move_stop_pct']*100:>5.2f}% me={row['max_loss_equity_pct']*100:>5.2f}% | "
        f"ret={row['total_return_pct']:>7.2f}% dd={row['max_drawdown_pct']:>6.2f}% "
        f"sh={row['sharpe']:>4.2f} trades={row['trades']:>3} exits={row['exit_count']:>3} t={row['elapsed_sec']:>5.1f}s",
        flush=True,
    )


def unique_rows(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates(
        subset=["take_profit_pct", "stop_loss_pct", "underlying_move_stop_pct", "max_loss_equity_pct"],
        keep="last",
    ).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    out_dir = REPO_ROOT / "reports" / "optimizations" / "intraday_move_fade_0dte"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"strict_drawdown_optimization_y{args.y:02d}_x{int(args.x*10000):04d}_z{int(args.z*10000):04d}.csv"
    json_path = out_dir / f"strict_drawdown_optimization_y{args.y:02d}_x{int(args.x*10000):04d}_z{int(args.z*10000):04d}.json"

    base_cfg = build_base_config(args)
    rows: list[dict] = []

    stage1 = [(tp, me) for me in [0.03, 0.05, 0.075, 0.10, 0.125, 0.15] for tp in [0.0, 25.0, 50.0, 75.0, 100.0]]
    print(f"Stage 1: {len(stage1)} cases", flush=True)
    for idx, (tp, me) in enumerate(stage1, start=1):
        row = run_case(base_cfg, args, take_profit_pct=tp, max_loss_equity_pct=me)
        row["stage"] = "stage1"
        rows.append(row)
        print_case(idx, len(stage1), row)

    df = unique_rows(rows)
    qualified = df[df["max_drawdown_pct"] <= args.target_dd].sort_values(["total_return_pct", "sharpe"], ascending=[False, False])
    seeds = qualified.head(5) if not qualified.empty else df.sort_values(["max_drawdown_pct", "total_return_pct"], ascending=[True, False]).head(5)

    stage2 = []
    for _, seed in seeds.iterrows():
        for um in [0.0, 0.005, 0.0075, 0.01, 0.015, 0.02]:
            stage2.append((float(seed["take_profit_pct"]), 0.0, um, float(seed["max_loss_equity_pct"])))
    print(f"Stage 2: {len(stage2)} cases", flush=True)
    for idx, (tp, sl, um, me) in enumerate(stage2, start=1):
        row = run_case(base_cfg, args, take_profit_pct=tp, stop_loss_pct=sl, underlying_move_stop_pct=um, max_loss_equity_pct=me)
        row["stage"] = "stage2"
        rows.append(row)
        print_case(idx, len(stage2), row)

    df = unique_rows(rows)
    qualified = df[df["max_drawdown_pct"] <= args.target_dd].sort_values(["total_return_pct", "sharpe"], ascending=[False, False])
    seeds = qualified.head(5) if not qualified.empty else df.sort_values(["max_drawdown_pct", "total_return_pct"], ascending=[True, False]).head(5)

    stage3 = []
    for _, seed in seeds.iterrows():
        for sl in [25.0, 50.0, 75.0, 100.0, 150.0]:
            stage3.append((float(seed["take_profit_pct"]), sl, float(seed["underlying_move_stop_pct"]), float(seed["max_loss_equity_pct"])))
    print(f"Stage 3: {len(stage3)} cases", flush=True)
    for idx, (tp, sl, um, me) in enumerate(stage3, start=1):
        row = run_case(base_cfg, args, take_profit_pct=tp, stop_loss_pct=sl, underlying_move_stop_pct=um, max_loss_equity_pct=me)
        row["stage"] = "stage3"
        rows.append(row)
        print_case(idx, len(stage3), row)

    final_df = unique_rows(rows).sort_values(["max_drawdown_pct", "total_return_pct", "sharpe"], ascending=[True, False, False]).reset_index(drop=True)
    qualified = final_df[final_df["max_drawdown_pct"] <= args.target_dd].sort_values(["total_return_pct", "sharpe"], ascending=[False, False]).reset_index(drop=True)

    final_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    final_df.to_json(json_path, orient="records", force_ascii=False, indent=2)

    print("\n=== QUALIFIED (DD <= target) ===", flush=True)
    if qualified.empty:
        print("NONE", flush=True)
    else:
        print(qualified.head(15).to_string(index=False), flush=True)

    print("\n=== LOWEST DD ===", flush=True)
    print(final_df.head(15).to_string(index=False), flush=True)
    print(f"\nSaved: {csv_path}", flush=True)
    print(f"Saved: {json_path}", flush=True)


if __name__ == "__main__":
    main()
