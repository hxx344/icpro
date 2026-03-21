"""DD optimization �?load data once, rerun engine loop for each scenario."""
import sys, os, gc, copy
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

# ---- Step 1: load data exactly once via the first engine ----
print("Loading data...", file=sys.stderr, flush=True)
cfg0 = Config.from_yaml(BASE_CFG)
strat0 = ShortStrangleStrategy(params=cfg0.strategy.params)
eng0 = BacktestEngine(cfg0, strat0)
bcfg = cfg0.backtest
eng0._load_data(bcfg.underlying, bcfg.start_date, bcfg.end_date, bcfg.time_step)

# Store references to the heavy data (never mutated inside run-loop)
CACHED = {}
for attr in ("_instruments_df", "_underlying_df", "_option_data", "_settlements_df",
             "_ohlcv_index", "_instrument_dict",
             "_realized_vol_ts", "_realized_vol_values", "_dvol_ts", "_dvol_values",
             "_iv_observations"):
    CACHED[attr] = getattr(eng0, attr, None)
del eng0, strat0
gc.collect()
print("Data loaded & cached.", file=sys.stderr, flush=True)


def run_scenario(overrides: dict) -> dict:
    """Create a fresh engine, inject cached data, run the backtest loop."""
    cfg = Config.from_yaml(BASE_CFG)
    for k, v in overrides.items():
        cfg.strategy.params[k] = v
    strat = ShortStrangleStrategy(params=cfg.strategy.params)
    engine = BacktestEngine(cfg, strat)

    # Inject cached data (these are read-only in the run loop)
    for attr, val in CACHED.items():
        if val is not None:
            setattr(engine, attr, val)

    # Run backtest loop (mirrors BacktestEngine.run lines 203-272)
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


# ---- Scenarios: fine-tuning phase ----
scenarios = [
    ("Baseline",            {}),
    # Fine-grained SL sweep: 4% to 7%
    ("IntrSL_4%",           {"intraday_sl_pct": 0.04}),
    ("IntrSL_4.5%",         {"intraday_sl_pct": 0.045}),
    ("IntrSL_5%",           {"intraday_sl_pct": 0.05}),
    ("IntrSL_5.5%",         {"intraday_sl_pct": 0.055}),
    ("IntrSL_6%",           {"intraday_sl_pct": 0.06}),
    ("IntrSL_6.5%",         {"intraday_sl_pct": 0.065}),
    ("IntrSL_7%",           {"intraday_sl_pct": 0.07}),
    # Best single + ProgDD combos
    ("SL6+ProgDD_5-20",     {"intraday_sl_pct": 0.06, "dd_start": 0.05, "dd_full": 0.20, "dd_min_scale": 0.2}),
    ("SL6+ProgDD_8-20",     {"intraday_sl_pct": 0.06, "dd_start": 0.08, "dd_full": 0.20, "dd_min_scale": 0.3}),
    ("SL5+ProgDD_8-20",     {"intraday_sl_pct": 0.05, "dd_start": 0.08, "dd_full": 0.20, "dd_min_scale": 0.3}),
    ("SL5+ProgDD_5-20",     {"intraday_sl_pct": 0.05, "dd_start": 0.05, "dd_full": 0.20, "dd_min_scale": 0.2}),
    # SL + DD scaling (round 1 dd_reduce)
    ("SL6+DDScale15",       {"intraday_sl_pct": 0.06, "dd_reduce_threshold": 0.15, "dd_reduce_factor": 0.5}),
    ("SL5+DDScale15",       {"intraday_sl_pct": 0.05, "dd_reduce_threshold": 0.15, "dd_reduce_factor": 0.5}),
    ("SL6+DDScale10",       {"intraday_sl_pct": 0.06, "dd_reduce_threshold": 0.10, "dd_reduce_factor": 0.5}),
]

# ---- Run all ----
results = []
for i, (label, overrides) in enumerate(scenarios):
    try:
        r = run_scenario(overrides)
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
        results.append({"label": label, "final": final_eq, "max_dd": max_dd,
                         "sharpe": sharpe, "wr": wr, "trades": trades})
        print(f"[{i+1}/{len(scenarios)}] {label:22s} Final={final_eq:>8.1f} DD={max_dd:>6.1f}%",
              file=sys.stderr, flush=True)
    except Exception as e:
        import traceback
        print(f"[{i+1}/{len(scenarios)}] {label}: ERROR - {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        results.append({"label": label, "final": 0, "max_dd": 0, "sharpe": 0, "wr": 0, "trades": 0})
    gc.collect()

# ---- Print results table ----
base_dd = results[0]["max_dd"] if results else 0
base_ret = results[0]["final"] if results else 1

lines = []
lines.append("")
lines.append("=" * 110)
lines.append(f"  {'Strategy':<22s} {'Final':>8s} {'MaxDD%':>8s} {'DD改善':>8s} {'Ret变化':>8s} {'Sharpe':>7s} {'WR%':>6s} {'Trades':>7s}")
lines.append("=" * 110)
for r in results:
    dd_imp = base_dd - r["max_dd"]
    ret_vs = (r["final"] / base_ret - 1) * 100 if base_ret > 0 else 0
    lines.append(
        f"  {r['label']:<22s} {r['final']:>8.1f} {r['max_dd']:>7.1f}% {dd_imp:>+7.1f}pp "
        f"{ret_vs:>+7.0f}% {r['sharpe']:>7.2f} {r['wr']:>5.1f}% {r['trades']:>7.0f}"
    )
lines.append("=" * 110)

text = "\n".join(lines)
print(text, flush=True)

with open("dd_opt_results.txt", "w", encoding="utf-8") as f:
    f.write(text)
print("\nResults saved to dd_opt_results.txt", file=sys.stderr, flush=True)
