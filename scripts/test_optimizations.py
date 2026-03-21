"""Batch-test multiple NakedCall optimization strategies.

Usage: python scripts/test_optimizations.py
"""
from __future__ import annotations

import json
import copy
from pathlib import Path

from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.analytics.metrics import compute_metrics
from options_backtest.cli import _load_strategy


# Each variant: (label, param overrides)
VARIANTS = [
    # Baseline
    ("Baseline ATM",          {"buy_protective_call": False, "strike_offset_pct": 0.0,
                               "stop_loss_pct": 0.0, "take_profit_pct": 0.0,
                               "vol_lookback_hours": 0, "max_vol_pct": 0.0,
                               "dynamic_otm": False, "max_loss_equity_pct": 0.0}),

    # --- A. OTM offset strategies ---
    ("OTM 1%",                {"strike_offset_pct": 0.01}),
    ("OTM 2%",                {"strike_offset_pct": 0.02}),
    ("OTM 3%",                {"strike_offset_pct": 0.03}),

    # --- B. Stop-loss strategies ---
    ("SL 150%",               {"stop_loss_pct": 1.5}),
    ("SL 200%",               {"stop_loss_pct": 2.0}),
    ("SL 300%",               {"stop_loss_pct": 3.0}),

    # --- C. OTM + Stop-loss combo ---
    ("OTM 2% + SL 200%",     {"strike_offset_pct": 0.02, "stop_loss_pct": 2.0}),
    ("OTM 2% + SL 300%",     {"strike_offset_pct": 0.02, "stop_loss_pct": 3.0}),

    # --- D. Volatility filter ---
    ("Vol Filter 80%",        {"vol_lookback_hours": 24, "max_vol_pct": 80}),
    ("Vol Filter 100%",       {"vol_lookback_hours": 24, "max_vol_pct": 100}),

    # --- E. Dynamic OTM (scaled by vol) ---
    ("DynOTM base=2%",        {"strike_offset_pct": 0.02, "dynamic_otm": True, "base_vol": 0.60,
                               "vol_lookback_hours": 24}),

    # --- F. Max loss equity stop ---
    ("EquityStop 3%",         {"max_loss_equity_pct": 0.03}),
    ("EquityStop 5%",         {"max_loss_equity_pct": 0.05}),

    # --- G. Combined best: OTM + SL + Vol filter ---
    ("Combined: OTM2+SL200+Vol100",
                              {"strike_offset_pct": 0.02, "stop_loss_pct": 2.0,
                               "vol_lookback_hours": 24, "max_vol_pct": 100}),
]

BASE_OVERRIDES = {
    "buy_protective_call": False,
    "stop_loss_pct": 0.0,
    "take_profit_pct": 0.0,
    "strike_offset_pct": 0.0,
    "vol_lookback_hours": 0,
    "max_vol_pct": 0.0,
    "dynamic_otm": False,
    "max_loss_equity_pct": 0.0,
}


def run_one(label: str, overrides: dict) -> dict:
    cfg = Config.from_yaml(Path("configs/backtest/naked_call_usd.yaml"))

    # Start from clean base, then apply variant overrides
    params = dict(cfg.strategy.params)
    params.update(BASE_OVERRIDES)
    params.update(overrides)
    cfg.strategy.params = params

    strategy = _load_strategy(cfg.strategy.name, params)
    engine = BacktestEngine(cfg, strategy)
    results = engine.run()
    metrics = compute_metrics(results)

    closed = results["closed_trades"]
    pnls = [t["pnl"] for t in closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    return {
        "label": label,
        "trades": len(pnls),
        "win_rate": len(wins) / len(pnls) * 100 if pnls else 0,
        "total_return": metrics["total_return"],
        "final_coin": metrics["final_equity"],
        "final_usd": metrics.get("final_usd", 0),
        "hedged_usd": metrics.get("final_hedged_usd", 0),
        "max_dd": metrics["max_drawdown"],
        "max_dd_hedged": metrics.get("max_drawdown_hedged", 0),
        "sharpe": metrics["sharpe_ratio"],
        "sharpe_hedged": metrics.get("sharpe_ratio_hedged", 0),
        "profit_factor": abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf"),
        "total_profit": sum(wins),
        "total_loss": sum(losses),
        "fees": metrics.get("total_fees", 0),
    }


def main():
    results = []
    for label, overrides in VARIANTS:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        r = run_one(label, overrides)
        results.append(r)
        print(f"  Return={r['total_return']:.2%}  MaxDD={r['max_dd']:.2%}  "
              f"Sharpe={r['sharpe']:.2f}  PF={r['profit_factor']:.2f}")

    # Summary table
    print(f"\n\n{'='*130}")
    print(f"  OPTIMIZATION COMPARISON   (ETH Naked Call 0-DTE, 90 days)")
    print(f"{'='*130}")
    hdr = (f"{'Strategy':<30s} {'#':>4s} {'Win%':>6s} {'Return':>8s} "
           f"{'FinalETH':>9s} {'USD':>8s} {'HgdUSD':>8s} "
           f"{'MaxDD':>7s} {'DDHgd':>7s} {'Sharpe':>7s} {'ShpHgd':>7s} {'PF':>6s}")
    print(hdr)
    print("-"*130)
    for r in results:
        line = (
            f"{r['label']:<30s} "
            f"{r['trades']:>4d} "
            f"{r['win_rate']:>5.1f}% "
            f"{r['total_return']:>7.2%} "
            f"{r['final_coin']:>8.4f} "
            f"${r['final_usd']:>7,.0f} "
            f"${r['hedged_usd']:>7,.0f} "
            f"{r['max_dd']:>6.2%} "
            f"{r['max_dd_hedged']:>6.2%} "
            f"{r['sharpe']:>7.2f} "
            f"{r['sharpe_hedged']:>7.2f} "
            f"{r['profit_factor']:>5.2f}"
        )
        print(line)

    # Highlight best by Sharpe (hedged)
    best_sharpe = max(results, key=lambda x: x["sharpe_hedged"])
    best_dd = min(results, key=lambda x: abs(x["max_dd_hedged"]))
    best_pf = max(results, key=lambda x: x["profit_factor"])
    best_ret = max(results, key=lambda x: x["total_return"])

    print(f"\n{'='*80}")
    print(f"  BEST BY CATEGORY:")
    print(f"  最高收�?    {best_ret['label']}  ({best_ret['total_return']:.2%})")
    print(f"  最�?Sharpe: {best_sharpe['label']}  ({best_sharpe['sharpe_hedged']:.2f})")
    print(f"  最小回�?    {best_dd['label']}  ({best_dd['max_dd_hedged']:.2%})")
    print(f"  最�?PF:     {best_pf['label']}  ({best_pf['profit_factor']:.2f})")
    print(f"{'='*80}")

    # Save
    out_dir = Path("reports/optimizations")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.json").open("w", encoding="utf8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved to {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
