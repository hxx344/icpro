"""Explore improvements around BTC weekly Mon->Fri 10% OTM short strangle with SL300.

Base anchor:
- otm_pct = 10%
- stop_loss_pct = 300

Tests a focused set of return/DD improvement ideas.
"""
from __future__ import annotations

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

BASE_CONFIG = "configs/backtest/weekly_short_monfri_btc_otm10.yaml"
BASE_STACK = {
    "otm_pct": 0.10,
    "stop_loss_pct": 300,
    "take_profit_pct": 9999,
    "compound": True,
    "max_positions": 1,
    "hedge_otm_pct": 0.0,
    "skip_weekends": True,
}

CASES = [
    ("base_sl300", {}),
    ("tp80_sl300", {"take_profit_pct": 80}),
    ("tp50_sl300", {"take_profit_pct": 50}),
    ("dd_mild", {"dd_start": 0.10, "dd_full": 0.30, "dd_min_scale": 0.50}),
    ("dd_strong", {"dd_start": 0.05, "dd_full": 0.20, "dd_min_scale": 0.20}),
    ("intraday_sl_5pct", {"intraday_sl_pct": 0.05}),
    ("intraday_sl_8pct", {"intraday_sl_pct": 0.08}),
    ("adaptive_otm", {"adaptive_otm": True, "vol_lookback": 24, "vol_otm_mult": 1.5}),
    ("vol_filter_100", {"vol_lookback": 24, "vol_threshold": 1.00}),
    ("vol_filter_80", {"vol_lookback": 24, "vol_threshold": 0.80}),
    ("tp80_dd_mild", {"take_profit_pct": 80, "dd_start": 0.10, "dd_full": 0.30, "dd_min_scale": 0.50}),
    ("adaptive_dd_mild", {"adaptive_otm": True, "vol_lookback": 24, "vol_otm_mult": 1.5, "dd_start": 0.10, "dd_full": 0.30, "dd_min_scale": 0.50}),
]


def run_case(name: str, overrides: dict) -> dict:
    cfg = Config.from_yaml(BASE_CONFIG)
    for k, v in BASE_STACK.items():
        cfg.strategy.params[k] = v
    for k, v in overrides.items():
        cfg.strategy.params[k] = v
    cfg.backtest.name = f"OTM10 SL300 {name}"

    strategy = _load_strategy(cfg.strategy.name, cfg.strategy.params)
    engine = BacktestEngine(cfg, strategy)

    t0 = time.perf_counter()
    results = engine.run()
    elapsed = time.perf_counter() - t0
    metrics = compute_metrics(results)

    return {
        "name": name,
        "overrides": overrides,
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
print("OTM10% + SL300 improvement exploration")
print(f"{'Case':<18s} {'Ret':>8s} {'MaxDD':>8s} {'Sharpe':>7s} {'WR':>6s} {'PF':>6s} {'Trades':>6s} {'Fees':>10s} {'Time':>6s}")
print("=" * 95)

for name, overrides in CASES:
    row = run_case(name, overrides)
    rows.append(row)
    print(
        f"{name:<18s} {row['total_return']*100:>+7.1f}% {row['max_drawdown']*100:>7.1f}% "
        f"{row['sharpe_ratio']:>7.2f} {row['win_rate']*100:>5.1f}% {row['profit_factor']:>6.2f} "
        f"{row['total_trades']:>6d} {row['total_fees']:>10.0f} {row['elapsed_sec']:>5.1f}s"
    )

print("\n" + "=" * 95)
valid = [r for r in rows if r['total_trades'] > 0]
if valid:
    best_return = max(valid, key=lambda r: r['total_return'])
    best_dd = max(valid, key=lambda r: r['max_drawdown'])
    best_sharpe = max(valid, key=lambda r: r['sharpe_ratio'])
    print(
        f"Best return: {best_return['name']} -> Ret={best_return['total_return']*100:+.2f}% "
        f"DD={best_return['max_drawdown']*100:.2f}% Sharpe={best_return['sharpe_ratio']:.2f}"
    )
    print(
        f"Best drawdown: {best_dd['name']} -> DD={best_dd['max_drawdown']*100:.2f}% "
        f"Ret={best_dd['total_return']*100:+.2f}% Sharpe={best_dd['sharpe_ratio']:.2f}"
    )
    print(
        f"Best sharpe: {best_sharpe['name']} -> Sharpe={best_sharpe['sharpe_ratio']:.2f} "
        f"Ret={best_sharpe['total_return']*100:+.2f}% DD={best_sharpe['max_drawdown']*100:.2f}%"
    )

report_path = Path("reports/weekly_short_monfri_btc_otm10_sl300_improvements.json")
report_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"\nSaved: {report_path.as_posix()}")
