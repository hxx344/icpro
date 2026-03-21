"""Quick script to re-run the backtest and output detailed analysis."""

import sys
sys.path.insert(0, "src")

from pathlib import Path
from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.analytics.metrics import compute_metrics, print_metrics
import pandas as pd
import numpy as np


def main():
    cfg = Config.from_yaml(Path("configs/backtest/default.yaml"))

    # Resolve strategy
    from options_backtest.strategy.short_strangle import ShortStrangleStrategy
    strategy = ShortStrangleStrategy(params=cfg.strategy.params)

    engine = BacktestEngine(cfg, strategy)
    results = engine.run()

    # Print standard metrics
    metrics = compute_metrics(results)
    print_metrics(metrics)

    # --- Detailed drawdown analysis ---
    equity_history = results.get("equity_history", [])
    sample = equity_history[0]
    if len(sample) >= 5:
        eq_df = pd.DataFrame(equity_history, columns=["timestamp", "equity", "balance", "unrealized_pnl", "underlying_price"])
    else:
        eq_df = pd.DataFrame(equity_history, columns=["timestamp", "equity", "balance", "unrealized_pnl"])
    eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"])
    eq = eq_df["equity"].values.astype(float)
    ts = eq_df["timestamp"]

    # Drawdown series
    running_max = np.maximum.accumulate(eq)
    drawdown = (eq - running_max) / np.where(running_max > 0, running_max, 1)

    # Find worst drawdown period
    dd_end_idx = np.argmin(drawdown)
    dd_start_idx = np.argmax(eq[:dd_end_idx + 1]) if dd_end_idx > 0 else 0

    print("\n" + "=" * 60)
    print("  DRAWDOWN ANALYSIS")
    print("=" * 60)
    print(f"  Max DD Start:   {ts.iloc[dd_start_idx]} (equity={eq[dd_start_idx]:.4f} BTC)")
    print(f"  Max DD Bottom:  {ts.iloc[dd_end_idx]} (equity={eq[dd_end_idx]:.4f} BTC)")
    print(f"  Max Drawdown:   {drawdown[dd_end_idx]:.2%}")
    print(f"  Duration:       {(ts.iloc[dd_end_idx] - ts.iloc[dd_start_idx]).days} days")

    # Recovery
    post_dd = eq[dd_end_idx:]
    recovered = np.where(post_dd >= eq[dd_start_idx])[0]
    if len(recovered) > 0:
        recovery_idx = dd_end_idx + recovered[0]
        print(f"  Recovery Date:  {ts.iloc[recovery_idx]}")
        print(f"  Recovery Time:  {(ts.iloc[recovery_idx] - ts.iloc[dd_end_idx]).days} days")
    else:
        print(f"  Recovery:       NOT RECOVERED")

    # Yearly breakdown
    print("\n" + "=" * 60)
    print("  YEARLY BREAKDOWN")
    print("=" * 60)
    eq_df["year"] = eq_df["timestamp"].dt.year
    for year, group in eq_df.groupby("year"):
        if len(group) < 2:
            continue
        yr_eq = group["equity"].values
        yr_return = (yr_eq[-1] - yr_eq[0]) / yr_eq[0] if yr_eq[0] > 0 else 0
        yr_max = np.maximum.accumulate(yr_eq)
        yr_dd = (yr_eq - yr_max) / np.where(yr_max > 0, yr_max, 1)
        yr_max_dd = float(np.min(yr_dd))
        print(f"  {year}:  Return={yr_return:+.2%}  MaxDD={yr_max_dd:.2%}  "
              f"Start={yr_eq[0]:.4f}  End={yr_eq[-1]:.4f} BTC")

    # Trade summary by year
    closed_trades = results.get("closed_trades", [])
    if closed_trades:
        print("\n" + "=" * 60)
        print("  TRADE STATS BY YEAR")
        print("=" * 60)
        trade_df = pd.DataFrame(closed_trades)
        if "close_time" in trade_df.columns:
            trade_df["close_time"] = pd.to_datetime(trade_df["close_time"])
            trade_df["year"] = trade_df["close_time"].dt.year
            for year, tg in trade_df.groupby("year"):
                pnls = tg["pnl"].values
                wins = pnls[pnls > 0]
                losses = pnls[pnls < 0]
                print(f"  {year}: trades={len(pnls)}, wins={len(wins)}, losses={len(losses)}, "
                      f"win_rate={len(wins)/len(pnls):.1%}, "
                      f"total_pnl={pnls.sum():.4f} BTC, "
                      f"avg_pnl={pnls.mean():.6f}")

    # Top 10 worst trades
    if closed_trades:
        print("\n" + "=" * 60)
        print("  TOP 10 WORST TRADES")
        print("=" * 60)
        trade_df_sorted = trade_df.sort_values("pnl")
        for i, (_, row) in enumerate(trade_df_sorted.head(10).iterrows()):
            inst = row.get("instrument", "?")
            pnl = row["pnl"]
            ot = row.get("open_time", "?")
            ct = row.get("close_time", "?")
            print(f"  {i+1}. {inst}  PnL={pnl:.6f} BTC  open={ot}  close={ct}")

    print("=" * 60)


if __name__ == "__main__":
    main()
