"""Re-validate protective call strategy: run unhedged + hedged 3%/5%/10% and compare.

Usage: python scripts/research/validate_hedge.py
"""
from __future__ import annotations

import json
import copy
from pathlib import Path

from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.analytics.metrics import compute_metrics, print_metrics
from options_backtest.cli import _load_strategy


VARIANTS = [
    ("Unhedged",  False, 1.0),
    ("Hedge 3%",  True,  1.03),
    ("Hedge 5%",  True,  1.05),
    ("Hedge 10%", True,  1.10),
]


def run_variant(label: str, buy_protective: bool, hedge_pct: float) -> dict:
    cfg = Config.from_yaml(Path("configs/backtest/naked_call_usd.yaml"))
    cfg.strategy.params["buy_protective_call"] = buy_protective
    cfg.strategy.params["hedge_strike_pct"] = hedge_pct

    strategy = _load_strategy(cfg.strategy.name, cfg.strategy.params)
    engine = BacktestEngine(cfg, strategy)
    results = engine.run()
    metrics = compute_metrics(results)

    coin = cfg.backtest.underlying
    closed = results["closed_trades"]

    # Per-trade stats
    pnls = [t["pnl"] for t in closed]
    losses = [p for p in pnls if p < 0]
    wins = [p for p in pnls if p > 0]

    return {
        "label": label,
        "coin": coin,
        "total_trades": len(pnls),
        "winning": len(wins),
        "losing": len(losses),
        "win_rate": len(wins) / len(pnls) * 100 if pnls else 0,
        "total_return": metrics["total_return"],
        "final_equity_coin": metrics["final_equity"],
        "final_usd": metrics.get("final_usd", 0),
        "final_hedged_usd": metrics.get("final_hedged_usd", 0),
        "max_drawdown": metrics["max_drawdown"],
        "max_drawdown_hedged": metrics.get("max_drawdown_hedged", 0),
        "sharpe": metrics["sharpe_ratio"],
        "sharpe_hedged": metrics.get("sharpe_ratio_hedged", 0),
        "total_profit_coin": sum(wins),
        "total_loss_coin": sum(losses),
        "avg_win": sum(wins) / len(wins) if wins else 0,
        "avg_loss": sum(losses) / len(losses) if losses else 0,
        "profit_factor": abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf"),
        "total_fees": metrics.get("total_fees", 0),
    }


def main():
    all_results = []
    for label, buy_prot, hedge_pct in VARIANTS:
        print(f"\n{'='*60}")
        print(f"  Running: {label} (protective={buy_prot}, hedge_pct={hedge_pct})")
        print(f"{'='*60}")
        result = run_variant(label, buy_prot, hedge_pct)
        all_results.append(result)

    coin = all_results[0]["coin"]

    # Print comparison table
    print(f"\n\n{'='*100}")
    print(f"  COMPARISON SUMMARY  ({coin} Naked Call 0-DTE)")
    print(f"{'='*100}")

    header = f"{'Variant':<14s} {'Trades':>6s} {'WinRate':>8s} {'Return':>9s} {'Final':>8s} {'USD':>9s} {'HedgeUSD':>9s} {'MaxDD':>7s} {'Sharpe':>7s} {'PF':>6s} {'Fees':>8s}"
    print(header)
    print("-"*100)
    for r in all_results:
        line = (
            f"{r['label']:<14s} "
            f"{r['total_trades']:>6d} "
            f"{r['win_rate']:>7.1f}% "
            f"{r['total_return']:>8.2%} "
            f"{r['final_equity_coin']:>7.4f} "
            f"${r['final_usd']:>8,.0f} "
            f"${r['final_hedged_usd']:>8,.0f} "
            f"{r['max_drawdown']:>6.2%} "
            f"{r['sharpe']:>7.2f} "
            f"{r['profit_factor']:>5.2f} "
            f"{r['total_fees']:>7.5f}"
        )
        print(line)

    # Show profit/loss breakdown
    print(f"\n{'='*100}")
    print(f"  PROFIT / LOSS BREAKDOWN ({coin})")
    print(f"{'='*100}")
    header2 = f"{'Variant':<14s} {'Win#':>5s} {'Loss#':>5s} {'TotalProfit':>12s} {'TotalLoss':>12s} {'NetPnL':>12s} {'AvgWin':>10s} {'AvgLoss':>10s}"
    print(header2)
    print("-"*100)
    for r in all_results:
        net = r['total_profit_coin'] + r['total_loss_coin']
        line2 = (
            f"{r['label']:<14s} "
            f"{r['winning']:>5d} "
            f"{r['losing']:>5d} "
            f"{r['total_profit_coin']:>+11.6f} "
            f"{r['total_loss_coin']:>+11.6f} "
            f"{net:>+11.6f} "
            f"{r['avg_win']:>9.6f} "
            f"{r['avg_loss']:>9.6f}"
        )
        print(line2)

    # Save JSON
    out_dir = Path("reports/validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hedge_comparison.json"
    with out_path.open("w", encoding="utf8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved comparison to {out_path}")


if __name__ == "__main__":
    main()
