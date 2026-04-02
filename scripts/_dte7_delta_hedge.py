"""
BTC Weekly ATM Short Strangle + Hourly Delta Hedging
=====================================================
Open ATM strangle on Friday (DTE~7), hold to settlement.
Every hour, compute portfolio delta via Black-76 and maintain a
virtual BTC-PERPETUAL hedge position (delta ~ 0).

Approach: run the engine once for baseline, then re-run manually
with delta hedge overlay injected into the main loop.
"""
from __future__ import annotations

import sys, math

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

logger.remove()
logger.add(sys.stderr, level="WARNING")

from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.engine.settlement import check_and_settle
from options_backtest.pricing.black76 import delta as bs_delta
from options_backtest.analytics.metrics import compute_metrics, print_metrics
from options_backtest.analytics.plotting import generate_all_plots
from options_backtest.cli import _load_strategy

# -- Config ----------------------------------------------------------------
UNDERLYING = "BTC"
START = "2023-01-01"
END = "2026-01-31"
HEDGE_FEE_PCT = 0.0003  # perp taker fee per side

cfg = Config.from_yaml("configs/backtest/ic_0dte_8pct_direct.yaml")
cfg.backtest.underlying = UNDERLYING
cfg.backtest.start_date = START
cfg.backtest.end_date = END

# Weekly ATM naked strangle params
cfg.strategy.params["otm_pct"] = 0.0
cfg.strategy.params["hedge_otm_pct"] = 0.0
cfg.strategy.params["min_days_to_expiry"] = 6.0
cfg.strategy.params["max_days_to_expiry"] = 8.0
cfg.strategy.params["roll_daily"] = False
cfg.strategy.params["roll_days_before_expiry"] = 0
cfg.strategy.params["compound"] = True
cfg.strategy.params["take_profit_pct"] = 9999
cfg.strategy.params["stop_loss_pct"] = 9999
cfg.strategy.params["max_positions"] = 1
cfg.strategy.params["entry_hour"] = 8


# =========================================================================
#  1. Baseline: no delta hedge
# =========================================================================
print("=" * 60)
print("  1. BASELINE - Weekly ATM Strangle (no hedge)")
print("=" * 60)

cfg.backtest.name = "BTC DTE7 ATM Strangle (no hedge)"
s1 = _load_strategy(cfg.strategy.name, cfg.strategy.params)
e1 = BacktestEngine(cfg, s1)
r1 = e1.run()
m1 = compute_metrics(r1)
print_metrics(m1)


# =========================================================================
#  2. Delta-hedged version (manual loop)
# =========================================================================
print("\n" + "=" * 60)
print("  2. DELTA-HEDGED - Weekly ATM Strangle + hourly delta hedge")
print("=" * 60)

cfg.backtest.name = "BTC DTE7 ATM Strangle + Delta Hedge"
s2 = _load_strategy(cfg.strategy.name, cfg.strategy.params)
e2 = BacktestEngine(cfg, s2)

# Replicate engine.run() internals with delta hedge injected
bcfg = cfg.backtest
e2._load_data(bcfg.underlying, bcfg.start_date, bcfg.end_date, bcfg.time_step)

ts_values = e2._underlying_df["timestamp"].values
close_values = e2._underlying_df["close"].values.astype(np.float64)
n_steps = len(ts_values)

# USD margin init
if e2._margin_usd:
    first_price = float(close_values[0])
    if first_price > 0:
        usd_balance = e2.account.initial_balance * first_price
        e2.account.initial_balance = usd_balance
        e2.account.balance = usd_balance

# Init strategy
ts0 = pd.Timestamp(ts_values[0])
initial_ctx = e2._build_context(ts0, float(close_values[0]))
s2.initialize(initial_ctx)

# Local refs
position_mgr = e2.position_mgr
account = e2.account
matcher = e2.matcher
instrument_dict = e2._instrument_dict
settlements_df = e2._settlements_df
pending_orders = e2._pending_orders
ts_pd_all = pd.DatetimeIndex(ts_values).tz_localize("UTC")

# Delta hedge state
perp_qty = 0.0          # BTC qty of perp (+ = long)
prev_price = float(close_values[0])
hedge_pnl = 0.0
hedge_fees = 0.0
n_hedges = 0

# IV is resolved via engine's _resolve_proxy_iv (prepared during _load_data)


def _portfolio_delta(positions, price, ts_np):
    """Net portfolio delta (in coin units)."""
    net = 0.0
    for name, pos in positions.items():
        inst = instrument_dict.get(name)
        if inst is None:
            continue
        strike = inst.get("strike_price", inst.get("strike", 0))
        opt_raw = inst.get("option_type", "call")
        opt_type = "call" if str(opt_raw).lower().startswith("c") else "put"
        exp_ns = inst.get("_expiry_ns", 0)
        if exp_ns == 0:
            continue

        cur_ns = int(ts_np.astype("int64")) if hasattr(ts_np, "astype") else int(ts_np)
        dte_ns = exp_ns - cur_ns
        if dte_ns <= 0:
            if opt_type == "call":
                d = 1.0 if price > strike else 0.0
            else:
                d = -1.0 if price < strike else 0.0
        else:
            T = dte_ns / (365.25 * 86400 * 1e9)
            sigma = e2._resolve_proxy_iv(ts_np, price, strike, T)
            try:
                d = bs_delta(price, strike, T, sigma, option_type=opt_type, r=0.0)
            except Exception:
                d = 0.5 if opt_type == "call" else -0.5

        dir_sign = 1 if pos.direction.value == "long" else -1
        net += d * pos.quantity * dir_sign
    return net


# -- Main loop -----------------------------------------------------------
for i in tqdm(range(n_steps), desc="Delta-Hedged Backtest"):
    ts_np = ts_values[i]
    price = float(close_values[i])
    ts_pd = ts_pd_all[i]

    has_pos = bool(position_mgr.positions)

    # 1. Update marks + settlement
    if has_pos:
        mark_prices = e2._get_mark_prices_fast(ts_np, price)
        position_mgr.update_marks(mark_prices)
        check_and_settle(
            ts_np, position_mgr, account,
            matcher, instrument_dict, settlements_df,
            margin_usd=e2._margin_usd,
            settlement_index=e2._settlement_index,
        )

    # 2. Strategy step
    ctx = e2._build_context(ts_pd, price)
    pending_orders.clear()
    s2.on_step(ctx)

    # 3. Process orders
    if pending_orders:
        e2._process_orders(ts_np, price, ctx)

    # 4. Delta hedge
    dp = price - prev_price

    # Perp P&L from previous hedge position (USD margin: PnL in USD)
    if abs(perp_qty) > 1e-12:
        step_pnl = perp_qty * dp  # USD
        hedge_pnl += step_pnl
        account.balance += step_pnl

    # Rebalance hedge to neutralize portfolio delta
    if position_mgr.positions:
        port_delta = _portfolio_delta(position_mgr.positions, price, ts_np)
        target_perp = -port_delta  # neutralize
        trade_size = abs(target_perp - perp_qty)
        if trade_size > 1e-6:
            fee = trade_size * price * HEDGE_FEE_PCT
            account.balance -= fee
            hedge_fees += fee
            n_hedges += 1
            perp_qty = target_perp
    else:
        # No positions -> close perp
        if abs(perp_qty) > 1e-6:
            fee = abs(perp_qty) * price * HEDGE_FEE_PCT
            account.balance -= fee
            hedge_fees += fee
            n_hedges += 1
            perp_qty = 0.0

    prev_price = price

    # 5. Record equity
    account.record_equity(ts_pd, position_mgr.total_unrealized_pnl, price)

# Force close
if position_mgr.positions:
    last_ts = ts_pd_all[-1]
    last_price = float(close_values[-1])
    ctx = e2._build_context(last_ts, last_price)
    ctx.close_all()
    e2._process_orders(ts_values[-1], last_price, ctx)
    if abs(perp_qty) > 1e-6:
        fee = abs(perp_qty) * last_price * HEDGE_FEE_PCT
        account.balance -= fee
        hedge_fees += fee
        perp_qty = 0.0
    account.record_equity(last_ts, position_mgr.total_unrealized_pnl, last_price)

# Build results
r2 = e2._build_results()
m2 = compute_metrics(r2)
print_metrics(m2)

print(f"\n  Hedge Stats:")
print(f"    Rebalances:   {n_hedges:,}")
print(f"    Hedge PnL:    ${hedge_pnl:,.2f}")
print(f"    Hedge Fees:   ${hedge_fees:,.2f}")
print(f"    Net Impact:   ${hedge_pnl - hedge_fees:,.2f}")

# Generate charts
paths = generate_all_plots(r2, "reports")
for p in paths:
    print(f"Saved: {p}")

# =========================================================================
#  3. Side-by-side comparison
# =========================================================================
print("\n" + "=" * 60)
print("  COMPARISON")
print("=" * 60)
fmt = "  {:<20s} {:>9s} {:>9s} {:>8s} {:>7s} {:>7s}"
print(fmt.format("Strategy", "Return", "MaxDD", "Sharpe", "WR", "Trades"))
print("  " + "-" * 56)
for label, m in [("No Hedge", m1), ("Delta-Hedged", m2)]:
    ret = f"{m.get('total_return', 0) * 100:.1f}%"
    dd = f"{m.get('max_drawdown', 0) * 100:.1f}%"
    sr = f"{m.get('sharpe_ratio', 0):.2f}"
    wr = f"{m.get('win_rate', 0) * 100:.1f}%"
    tr = str(m.get("total_trades", 0))
    print(fmt.format(label, ret, dd, sr, wr, tr))
