"""SL6%双卖策略 - 不同开仓时间对�?"""
import sys, os, gc
os.environ["LOGURU_LEVEL"] = "ERROR"
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.strategy.short_strangle import ShortStrangleStrategy
from options_backtest.analytics.metrics import compute_metrics
from options_backtest.engine.settlement import check_and_settle

BASE_CFG = "configs/backtest/short_strangle_0dte_12m.yaml"

# Load data once
print("Loading data...", file=sys.stderr, flush=True)
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
print("Data loaded.", file=sys.stderr, flush=True)


def run_scenario(overrides: dict) -> dict:
    cfg = Config.from_yaml(BASE_CFG)
    for k, v in overrides.items():
        if k == "entry_hour":
            cfg.strategy.params[k] = v
        else:
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
        has_pos = bool(position_mgr.positions)
        if has_pos:
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


# Sweep entry_hour 0-23
results = []
for hour in range(24):
    try:
        r = run_scenario({"intraday_sl_pct": 0.06, "entry_hour": hour})
        m = compute_metrics(r)
        eq = r["equity_history"]
        final_eq = eq[-1][1]
        eqs = np.array([e[1] for e in eq])
        rm = np.maximum.accumulate(eqs)
        dd = (eqs - rm) / rm
        max_dd = float(np.min(dd)) * 100
        sharpe = m.get("sharpe_ratio", 0)
        wr = m.get("win_rate", 0) * 100
        trades = m.get("total_trades", 0)
        results.append({"hour": hour, "final": final_eq, "max_dd": max_dd,
                         "sharpe": sharpe, "wr": wr, "trades": trades})
        print(f"  Hour {hour:02d}:00 UTC  Final={final_eq:>8.1f}  DD={max_dd:>6.1f}%",
              file=sys.stderr, flush=True)
    except Exception as e:
        print(f"  Hour {hour:02d}: ERROR - {e}", file=sys.stderr, flush=True)
        results.append({"hour": hour, "final": 0, "max_dd": 0, "sharpe": 0, "wr": 0, "trades": 0})
    gc.collect()

# Print table
best = max(results, key=lambda x: x["final"])
lines = []
lines.append("")
lines.append("1% OTM 双卖 + SL6% �?不同开仓时间对�?(UTC)")
lines.append("=" * 90)
lines.append(f"  {'Hour':>6s} {'Final ETH':>10s} {'MaxDD%':>8s} {'Sharpe':>8s} {'WR%':>6s} {'Trades':>7s}  Note")
lines.append("=" * 90)
for r in results:
    note = ""
    if r["hour"] == best["hour"]:
        note = " �?BEST"
    if r["hour"] == 8:
        note += " (当前默认)"
    lines.append(
        f"  {r['hour']:02d}:00  {r['final']:>10.1f} {r['max_dd']:>7.1f}% {r['sharpe']:>8.2f} "
        f"{r['wr']:>5.1f}% {r['trades']:>7.0f}  {note}"
    )
lines.append("=" * 90)
lines.append(f"\n  最佳开仓时�? UTC {best['hour']:02d}:00  Final={best['final']:.1f} ETH  DD={best['max_dd']:.1f}%")

text = "\n".join(lines)
print(text, flush=True)

with open("entry_hour_sl6.txt", "w", encoding="utf-8") as f:
    f.write(text)
