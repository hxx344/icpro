"""Performance metrics computation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(results: dict) -> dict:
    """Compute performance metrics from backtest results.

    Parameters
    ----------
    results : dict returned by ``BacktestEngine.run()``

    Returns
    -------
    dict with computed metrics.
    """
    equity_history = results.get("equity_history", [])
    closed_trades = results.get("closed_trades", [])
    initial_balance = results.get("initial_balance", 1.0)

    if not equity_history:
        return {"error": "no equity data"}

    # Build equity series
    # Support both 4-column (legacy) and 5-column (with underlying_price) history
    sample = equity_history[0]
    if len(sample) >= 5:
        eq_df = pd.DataFrame(equity_history, columns=["timestamp", "equity", "balance", "unrealized_pnl", "underlying_price"])
    else:
        eq_df = pd.DataFrame(equity_history, columns=["timestamp", "equity", "balance", "unrealized_pnl"])
        eq_df["underlying_price"] = 0.0
    eq = eq_df["equity"].values.astype(float)
    ts = pd.to_datetime(eq_df["timestamp"], utc=True)
    underlying_prices = eq_df["underlying_price"].values.astype(float)

    # Basic return (coin-denominated)
    final_eq = float(eq[-1])
    total_return = (final_eq - initial_balance) / initial_balance
    underlying_name = results.get("underlying", "BTC")

    # USD equity = coin equity × underlying price
    has_underlying = underlying_prices[0] > 0
    if has_underlying:
        usd_eq: np.ndarray = eq * underlying_prices
        initial_usd = float(usd_eq[0])
        final_usd = float(usd_eq[-1])
        total_return_usd = (final_usd - initial_usd) / initial_usd if initial_usd > 0 else 0.0

        # Hedged USD (real-time delta-neutral):
        # Each step's coin PnL is locked in at that step's coin price.
        # hedged[t] = hedged[t-1] + (eq[t] - eq[t-1]) * price[t]
        # This models continuously adjusting the perp short to match coin equity.
        hedged_usd_eq: np.ndarray = np.empty_like(eq)
        hedged_usd_eq[0] = eq[0] * underlying_prices[0]
        delta_eq = np.diff(eq)  # coin PnL each step
        hedged_usd_eq[1:] = hedged_usd_eq[0] + np.cumsum(delta_eq * underlying_prices[1:])
        initial_hedged_usd = float(hedged_usd_eq[0])
        final_hedged_usd = float(hedged_usd_eq[-1])
        total_return_hedged = (final_hedged_usd - initial_hedged_usd) / initial_hedged_usd if initial_hedged_usd > 0 else 0.0
    else:
        usd_eq = np.zeros_like(eq)
        hedged_usd_eq = np.zeros_like(eq)
        initial_usd = 0.0
        final_usd = 0.0
        total_return_usd = 0.0
        initial_hedged_usd = 0.0
        final_hedged_usd = 0.0
        total_return_hedged = 0.0

    # Time span
    days = max((ts.max() - ts.min()).total_seconds() / 86400, 1)
    years = days / 365.25

    # Annualized return (coin)
    if total_return > -1:
        ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else total_return
    else:
        ann_return = -1.0

    # Annualized return (USD)
    if has_underlying and total_return_usd > -1:
        ann_return_usd = (1 + total_return_usd) ** (1 / years) - 1 if years > 0 else total_return_usd
    else:
        ann_return_usd = 0.0

    # Annualized return (Hedged USD)
    if has_underlying and total_return_hedged > -1:
        ann_return_hedged = (1 + total_return_hedged) ** (1 / years) - 1 if years > 0 else total_return_hedged
    else:
        ann_return_hedged = 0.0

    # Drawdown (coin)
    running_max = np.maximum.accumulate(eq)
    drawdown = (eq - running_max) / np.where(running_max > 0, running_max, 1)
    max_drawdown = float(np.min(drawdown))

    # Drawdown (USD)
    if has_underlying and usd_eq is not None:
        usd_running_max = np.maximum.accumulate(usd_eq)
        usd_drawdown = (usd_eq - usd_running_max) / np.where(usd_running_max > 0, usd_running_max, 1)
        max_drawdown_usd = float(np.min(usd_drawdown))
    else:
        max_drawdown_usd = 0.0

    # Drawdown (Hedged USD)
    if has_underlying and hedged_usd_eq is not None:
        hedged_running_max = np.maximum.accumulate(hedged_usd_eq)
        hedged_drawdown = (hedged_usd_eq - hedged_running_max) / np.where(hedged_running_max > 0, hedged_running_max, 1)
        max_drawdown_hedged = float(np.min(hedged_drawdown))
    else:
        max_drawdown_hedged = 0.0

    # Returns series – coin (hourly / per‑step)
    returns = np.diff(eq) / np.where(eq[:-1] != 0, eq[:-1], 1)

    # Sharpe (annualised assuming hourly steps → 8760 steps/year)
    steps_per_year = len(eq) / years if years > 0 else 8760
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(steps_per_year))
    else:
        sharpe = 0.0

    # Sharpe (USD)
    if has_underlying and usd_eq is not None and len(usd_eq) > 1:
        usd_returns = np.diff(usd_eq) / np.where(usd_eq[:-1] != 0, usd_eq[:-1], 1)
        if np.std(usd_returns) > 0:
            sharpe_usd = float(np.mean(usd_returns) / np.std(usd_returns) * np.sqrt(steps_per_year))
        else:
            sharpe_usd = 0.0
    else:
        sharpe_usd = 0.0

    # Sharpe (Hedged USD)
    if has_underlying and hedged_usd_eq is not None and len(hedged_usd_eq) > 1:
        hedged_returns = np.diff(hedged_usd_eq) / np.where(hedged_usd_eq[:-1] != 0, hedged_usd_eq[:-1], 1)
        if np.std(hedged_returns) > 0:
            sharpe_hedged = float(np.mean(hedged_returns) / np.std(hedged_returns) * np.sqrt(steps_per_year))
        else:
            sharpe_hedged = 0.0
    else:
        sharpe_hedged = 0.0

    # Trade statistics
    trade_pnls = [t.get("pnl", 0) for t in closed_trades]
    n_trades = len(trade_pnls)
    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p < 0]
    win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf")

    total_fees = results.get("total_fee_paid", 0.0)

    # Data source statistics
    data_source = results.get("data_source", {})

    metrics = {
        "initial_balance": initial_balance,
        "final_equity": final_eq,
        "total_return": total_return,
        "annualized_return": ann_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        # USD metrics (spot tracking)
        "initial_usd": initial_usd,
        "final_usd": final_usd,
        "total_return_usd": total_return_usd,
        "annualized_return_usd": ann_return_usd,
        "max_drawdown_usd": max_drawdown_usd,
        "sharpe_ratio_usd": sharpe_usd,
        # Hedged USD metrics (delta-neutral: short perp to offset spot)
        "initial_hedged_usd": initial_hedged_usd,
        "final_hedged_usd": final_hedged_usd,
        "total_return_hedged": total_return_hedged,
        "annualized_return_hedged": ann_return_hedged,
        "max_drawdown_hedged": max_drawdown_hedged,
        "sharpe_ratio_hedged": sharpe_hedged,
        # Trade stats
        "total_trades": n_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "total_fees": total_fees,
        "backtest_days": days,
        "underlying": underlying_name,
        "data_source": data_source,
    }

    return metrics


def print_metrics(metrics: dict) -> None:
    """Pretty‑print metrics to the console."""
    coin = metrics.get("underlying", "BTC")
    print("\n" + "=" * 50)
    print("  BACKTEST RESULTS")
    print("=" * 50)
    fmt = [
        ("Initial Balance",  f"{metrics.get('initial_balance', 0):.4f} {coin}"),
        ("Final Equity",     f"{metrics.get('final_equity', 0):.4f} {coin}"),
        ("Total Return",     f"{metrics.get('total_return', 0):.2%}"),
        ("Annualised Return", f"{metrics.get('annualized_return', 0):.2%}"),
        ("Max Drawdown",     f"{metrics.get('max_drawdown', 0):.2%}"),
        ("Sharpe Ratio",     f"{metrics.get('sharpe_ratio', 0):.2f}"),
        ("Total Trades",     f"{metrics.get('total_trades', 0)}"),
        ("Win Rate",         f"{metrics.get('win_rate', 0):.2%}"),
        ("Avg Win",          f"{metrics.get('avg_win', 0):.6f} {coin}"),
        ("Avg Loss",         f"{metrics.get('avg_loss', 0):.6f} {coin}"),
        ("Profit Factor",    f"{metrics.get('profit_factor', 0):.2f}"),
        ("Total Fees",       f"{metrics.get('total_fees', 0):.6f} {coin}"),
        ("Backtest Days",    f"{metrics.get('backtest_days', 0):.0f}"),
    ]
    for label, value in fmt:
        print(f"  {label:<22s} {value}")

    # USD metrics (spot tracking)
    initial_usd = metrics.get("initial_usd", 0)
    if initial_usd > 0:
        print("-" * 50)
        print("  USD METRICS (Spot Tracking)")
        print("-" * 50)
        usd_fmt = [
            ("Initial USD",      f"${metrics.get('initial_usd', 0):,.2f}"),
            ("Final USD",        f"${metrics.get('final_usd', 0):,.2f}"),
            ("Total Return USD", f"{metrics.get('total_return_usd', 0):.2%}"),
            ("Ann. Return USD",  f"{metrics.get('annualized_return_usd', 0):.2%}"),
            ("Max Drawdown USD", f"{metrics.get('max_drawdown_usd', 0):.2%}"),
            ("Sharpe Ratio USD", f"{metrics.get('sharpe_ratio_usd', 0):.2f}"),
        ]
        for label, value in usd_fmt:
            print(f"  {label:<22s} {value}")

    # Hedged USD metrics (delta-neutral)
    initial_hedged = metrics.get("initial_hedged_usd", 0)
    if initial_hedged > 0:
        print("-" * 50)
        print("  HEDGED USD (Short Perp Delta-Neutral)")
        print("-" * 50)
        hedged_fmt = [
            ("Initial Hedged USD", f"${metrics.get('initial_hedged_usd', 0):,.2f}"),
            ("Final Hedged USD",   f"${metrics.get('final_hedged_usd', 0):,.2f}"),
            ("Return Hedged",      f"{metrics.get('total_return_hedged', 0):.2%}"),
            ("Ann. Return Hedged", f"{metrics.get('annualized_return_hedged', 0):.2%}"),
            ("Max DD Hedged",      f"{metrics.get('max_drawdown_hedged', 0):.2%}"),
            ("Sharpe Hedged",      f"{metrics.get('sharpe_ratio_hedged', 0):.2f}"),
        ]
        for label, value in hedged_fmt:
            print(f"  {label:<22s} {value}")

    # Data source section
    ds = metrics.get("data_source", {})
    if ds:
        print("-" * 50)
        print("  DATA SOURCE")
        print("-" * 50)

        mark_total = ds.get("mark_total", 0)
        mark_mkt = ds.get("mark_market", 0)
        mark_syn = ds.get("mark_synth", 0)
        quote_total = ds.get("quote_total", 0)
        quote_mkt = ds.get("quote_market", 0)
        quote_syn = ds.get("quote_synth", 0)
        chain_total = ds.get("chain_total", 0)
        chain_mkt = ds.get("chain_market", 0)
        chain_syn = ds.get("chain_synth", 0)

        def _pct(n, total):
            return f"{n/total:.1%}" if total > 0 else "N/A"

        print(f"  {'Mark Pricing':<22s} Market: {mark_mkt}  B-S: {mark_syn}  "
              f"({_pct(mark_mkt, mark_total)} real)")
        print(f"  {'Order Quotes':<22s} Market: {quote_mkt}  B-S: {quote_syn}  "
              f"({_pct(quote_mkt, quote_total)} real)")
        print(f"  {'Chain Entries':<22s} Market: {chain_mkt}  B-S: {chain_syn}  "
              f"({_pct(chain_mkt, chain_total)} real)")

    print("=" * 50 + "\n")
