"""生成 UTC 8:00 vs 15:00 开仓时间的权益曲线�?"""
import sys, os, gc
os.environ["LOGURU_LEVEL"] = "ERROR"
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.strategy.short_strangle import ShortStrangleStrategy
from options_backtest.engine.settlement import check_and_settle

BASE_CFG = "configs/backtest/short_strangle_0dte_12m.yaml"

# Load data once
cfg0 = Config.from_yaml(BASE_CFG)
strat0 = ShortStrangleStrategy(params=cfg0.strategy.params)
eng0 = BacktestEngine(cfg0, strat0)
bcfg = cfg0.backtest
eng0._load_data(bcfg.underlying, bcfg.start_date, bcfg.end_date, bcfg.time_step)
CACHED = {}
for attr in ("_instruments_df", "_underlying_df", "_option_data", "_settlements_df",
             "_ohlcv_index", "_instrument_dict",
             "_realized_vol_ts", "_realized_vol_values", "_dvol_ts", "_dvol_values",
             "_iv_observations"):
    CACHED[attr] = getattr(eng0, attr, None)
del eng0, strat0
gc.collect()


def run_scenario(overrides):
    cfg = Config.from_yaml(BASE_CFG)
    for k, v in overrides.items():
        cfg.strategy.params[k] = v
    strat = ShortStrangleStrategy(params=cfg.strategy.params)
    engine = BacktestEngine(cfg, strat)
    for attr, val in CACHED.items():
        if val is not None:
            setattr(engine, attr, val)
    ts_values = engine._underlying_df["timestamp"].values
    close_values = engine._underlying_df["close"].values.astype(np.float64)
    n_steps = len(ts_values)
    ts0 = pd.Timestamp(ts_values[0])
    initial_ctx = engine._build_context(ts0, float(close_values[0]))
    strat.initialize(initial_ctx)
    position_mgr = engine.position_mgr
    account = engine.account
    matcher = engine.matcher
    instrument_dict = engine._instrument_dict
    settlements_df = engine._settlements_df
    pending = engine._pending_orders
    for i in range(n_steps):
        ts_np = ts_values[i]
        price = float(close_values[i])
        if position_mgr.positions:
            mp = engine._get_mark_prices_fast(ts_np, price)
            position_mgr.update_marks(mp)
            check_and_settle(ts_np, position_mgr, account, matcher, instrument_dict, settlements_df)
        ts_pd = pd.Timestamp(ts_np, tz="UTC")
        ctx = engine._build_context(ts_pd, price)
        pending.clear()
        strat.on_step(ctx)
        if pending:
            engine._process_orders(ts_np, price, ctx)
        account.record_equity(ts_pd, position_mgr.total_unrealized_pnl, price)
    if position_mgr.positions:
        last_ts = pd.Timestamp(ts_values[-1], tz="UTC")
        last_price = float(close_values[-1])
        ctx = engine._build_context(last_ts, last_price)
        ctx.close_all()
        engine._process_orders(ts_values[-1], last_price, ctx)
        account.record_equity(last_ts, position_mgr.total_unrealized_pnl, last_price)
    return engine._build_results()


# Run three scenarios
print("Running 双卖 (baseline)...", file=sys.stderr, flush=True)
r_base = run_scenario({})
print("Running 双卖+SL6%...", file=sys.stderr, flush=True)
r_sl6 = run_scenario({"intraday_sl_pct": 0.06})
print("Running 铁鹰 保护8%OTM...", file=sys.stderr, flush=True)
r_ic8 = run_scenario({"hedge_otm_pct": 0.08})

# Extract equity curves
def extract_eq(r):
    eq = r["equity_history"]
    times = [e[0] for e in eq]
    vals = [e[1] for e in eq]
    return pd.Series(vals, index=pd.DatetimeIndex(times))

eq_base = extract_eq(r_base)
eq_sl6 = extract_eq(r_sl6)
eq_ic8 = extract_eq(r_ic8)

# Compute drawdowns
def calc_dd(eq_series):
    vals = eq_series.values
    rm = np.maximum.accumulate(vals)
    dd = (vals - rm) / rm * 100
    return pd.Series(dd, index=eq_series.index)

dd_base = calc_dd(eq_base)
dd_sl6 = calc_dd(eq_sl6)
dd_ic8 = calc_dd(eq_ic8)

# Downsample to daily
eq_base_d = eq_base.resample("1D").last().dropna()
eq_sl6_d = eq_sl6.resample("1D").last().dropna()
eq_ic8_d = eq_ic8.resample("1D").last().dropna()
dd_base_d = dd_base.resample("1D").min().dropna()
dd_sl6_d = dd_sl6.resample("1D").min().dropna()
dd_ic8_d = dd_ic8.resample("1D").min().dropna()

# Plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]})
fig.suptitle("1% OTM 0DTE: Strangle vs SL6% vs Iron Condor (8% hedge)", fontsize=14, fontweight="bold")

# Equity curve
ax1.plot(eq_base_d.index, eq_base_d.values, color="#9E9E9E", linewidth=1.2, alpha=0.7,
         label=f"Strangle (Final={eq_base.iloc[-1]:.1f}, DD={dd_base.min():.1f}%)")
ax1.plot(eq_sl6_d.index, eq_sl6_d.values, color="#2196F3", linewidth=1.5,
         label=f"Strangle+SL6% (Final={eq_sl6.iloc[-1]:.1f}, DD={dd_sl6.min():.1f}%)")
ax1.plot(eq_ic8_d.index, eq_ic8_d.values, color="#4CAF50", linewidth=1.5,
         label=f"Iron Condor 8% (Final={eq_ic8.iloc[-1]:.1f}, DD={dd_ic8.min():.1f}%)")
ax1.set_ylabel("Equity (ETH)", fontsize=11)
ax1.set_yscale("log")
ax1.legend(fontsize=10, loc="upper left")
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax1.xaxis.set_major_locator(mdates.MonthLocator())

# Drawdown
ax2.fill_between(dd_base_d.index, dd_base_d.values, 0, color="#9E9E9E", alpha=0.2, label=f"Strangle DD={dd_base.min():.1f}%")
ax2.fill_between(dd_sl6_d.index, dd_sl6_d.values, 0, color="#2196F3", alpha=0.3, label=f"SL6% DD={dd_sl6.min():.1f}%")
ax2.fill_between(dd_ic8_d.index, dd_ic8_d.values, 0, color="#4CAF50", alpha=0.3, label=f"IronCondor8% DD={dd_ic8.min():.1f}%")
ax2.set_ylabel("Drawdown %", fontsize=11)
ax2.set_xlabel("Date", fontsize=11)
ax2.legend(fontsize=9, loc="lower left")
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax2.xaxis.set_major_locator(mdates.MonthLocator())

plt.tight_layout()
out = "reports/equity_iron_condor.png"
plt.savefig(out, dpi=150)
print(f"Saved to {out}", file=sys.stderr, flush=True)
