"""Grid search for weekly BTC Monday-entry Friday-expiry delta-selected short strangles.

Sweeps entry delta targets and per-leg stop-loss levels, prints a compact table,
and saves full results to reports/weekly_short_monfri_btc_delta_sl_grid.json.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="WARNING")

from options_backtest.analytics.metrics import compute_metrics
from options_backtest.cli import _load_strategy
from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine

DELTAS = [0.10, 0.15, 0.20, 0.25]
STOP_LOSSES = [50, 80, 100, 150, 200, 300, 9999]
BASE_CONFIGS = {
    0.10: "configs/backtest/weekly_short_monfri_btc_delta10.yaml",
    0.15: "configs/backtest/weekly_short_monfri_btc_delta15.yaml",
    0.20: "configs/backtest/weekly_short_monfri_btc_delta20.yaml",
    0.25: "configs/backtest/weekly_short_monfri_btc_delta25.yaml",
}


def run_case(target_delta: float, stop_loss_pct: int) -> dict:
    cfg = Config.from_yaml(BASE_CONFIGS[target_delta])
    cfg.strategy.params["selection_mode"] = "delta"
    cfg.strategy.params["target_delta"] = target_delta
    cfg.strategy.params["target_call_delta"] = target_delta
    cfg.strategy.params["target_put_delta"] = target_delta
    cfg.strategy.params["max_delta_diff"] = 0.05
    cfg.strategy.params["stop_loss_pct"] = stop_loss_pct
    cfg.strategy.params["take_profit_pct"] = 9999
    cfg.strategy.params["compound"] = True
    cfg.strategy.params["max_positions"] = 1
    cfg.strategy.params["hedge_otm_pct"] = 0.0
    label_sl = "NoSL" if stop_loss_pct >= 9999 else f"SL{stop_loss_pct}"
    cfg.backtest.name = f"MonFri Short Strangle Δ{target_delta:.2f} {label_sl}"

    strategy = _load_strategy(cfg.strategy.name, cfg.strategy.params)
    engine = BacktestEngine(cfg, strategy)

    t0 = time.perf_counter()
    results = engine.run()
    elapsed = time.perf_counter() - t0
    metrics = compute_metrics(results)

    return {
        "target_delta": target_delta,
        "max_delta_diff": cfg.strategy.params.get("max_delta_diff", 0.05),
        "stop_loss_pct": stop_loss_pct,
        "label": cfg.backtest.name,
        "total_return": metrics.get("total_return", 0.0),
        "annualized_return": metrics.get("annualized_return", 0.0),
        "max_drawdown": metrics.get("max_drawdown", 0.0),
        "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
        "win_rate": metrics.get("win_rate", 0.0),
        "profit_factor": metrics.get("profit_factor", 0.0),
        "total_trades": metrics.get("total_trades", 0),
        "total_fees": metrics.get("total_fees", 0.0),
        "final_equity": metrics.get("final_equity", 0.0),
        "elapsed_sec": elapsed,
    }


rows: list[dict] = []
total = len(DELTAS) * len(STOP_LOSSES)

print("BTC Weekly Short Strangle (Mon entry, Fri expiry) Delta x StopLoss Grid")
print(f"{'#':>2s} {'Delta':>7s} {'SL':>6s} {'Ret':>8s} {'MaxDD':>8s} {'Sharpe':>7s} {'WR':>6s} {'PF':>6s} {'Trades':>6s} {'Fees':>10s} {'Time':>6s}")
print("=" * 112)

for idx, (target_delta, stop_loss_pct) in enumerate(itertools.product(DELTAS, STOP_LOSSES), start=1):
    row = run_case(target_delta, stop_loss_pct)
    rows.append(row)
    sl_label = "NoSL" if stop_loss_pct >= 9999 else str(stop_loss_pct)
    print(
        f"{idx:>2d} {target_delta:>6.2f} {sl_label:>6s} "
        f"{row['total_return']*100:>+7.1f}% {row['max_drawdown']*100:>7.1f}% "
        f"{row['sharpe_ratio']:>7.2f} {row['win_rate']*100:>5.1f}% {row['profit_factor']:>6.2f} "
        f"{row['total_trades']:>6d} {row['total_fees']:>10.0f} {row['elapsed_sec']:>5.1f}s"
    )

print("\n" + "=" * 112)

valid = [r for r in rows if r["total_trades"] > 0]
if valid:
    best_return = max(valid, key=lambda r: r["total_return"])
    best_sharpe = max(valid, key=lambda r: r["sharpe_ratio"])
    best_dd = max(valid, key=lambda r: r["max_drawdown"])

    print("Best by return:")
    print(
        f"  Delta {best_return['target_delta']:.2f} / SL "
        f"{'NoSL' if best_return['stop_loss_pct'] >= 9999 else int(best_return['stop_loss_pct'])} -> "
        f"Ret={best_return['total_return']*100:+.2f}% DD={best_return['max_drawdown']*100:.2f}% "
        f"Sharpe={best_return['sharpe_ratio']:.2f} WR={best_return['win_rate']*100:.1f}%"
    )
    print("Best by Sharpe:")
    print(
        f"  Delta {best_sharpe['target_delta']:.2f} / SL "
        f"{'NoSL' if best_sharpe['stop_loss_pct'] >= 9999 else int(best_sharpe['stop_loss_pct'])} -> "
        f"Sharpe={best_sharpe['sharpe_ratio']:.2f} Ret={best_sharpe['total_return']*100:+.2f}% "
        f"DD={best_sharpe['max_drawdown']*100:.2f}% WR={best_sharpe['win_rate']*100:.1f}%"
    )
    print("Best by drawdown:")
    print(
        f"  Delta {best_dd['target_delta']:.2f} / SL "
        f"{'NoSL' if best_dd['stop_loss_pct'] >= 9999 else int(best_dd['stop_loss_pct'])} -> "
        f"DD={best_dd['max_drawdown']*100:.2f}% Ret={best_dd['total_return']*100:+.2f}% "
        f"Sharpe={best_dd['sharpe_ratio']:.2f} WR={best_dd['win_rate']*100:.1f}%"
    )

report_path = Path("reports/weekly_short_monfri_btc_delta_sl_grid.json")
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"\nSaved: {report_path.as_posix()}")
print(f"Total elapsed: {sum(r['elapsed_sec'] for r in rows):.1f}s ({sum(r['elapsed_sec'] for r in rows)/60:.1f}min)")
