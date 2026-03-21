"""Comprehensive DD optimization: test all new mechanisms."""
import sys, os, traceback
os.environ["LOGURU_LEVEL"] = "ERROR"
sys.path.insert(0, "src")

import numpy as np
from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.strategy.short_strangle import ShortStrangleStrategy
from options_backtest.analytics.metrics import compute_metrics

BASE_CFG = "configs/backtest/short_strangle_0dte_12m.yaml"

scenarios = [
    # --- Baseline ---
    ("Baseline",            {}),

    # === A. Intraday stop-loss (close all if day loss > X%) ===
    ("IntradaySL_5%",       {"intraday_sl_pct": 0.05}),
    ("IntradaySL_8%",       {"intraday_sl_pct": 0.08}),
    ("IntradaySL_10%",      {"intraday_sl_pct": 0.10}),

    # === B. Equity curve MA filter ===
    ("EqMA_5d",             {"equity_ma_days": 5}),
    ("EqMA_10d",            {"equity_ma_days": 10}),
    ("EqMA_20d",            {"equity_ma_days": 20}),

    # === C. Progressive DD scaling (linear ramp) ===
    ("ProgDD_5-20%",        {"dd_start": 0.05, "dd_full": 0.20, "dd_min_scale": 0.2}),
    ("ProgDD_5-15%",        {"dd_start": 0.05, "dd_full": 0.15, "dd_min_scale": 0.3}),
    ("ProgDD_3-10%",        {"dd_start": 0.03, "dd_full": 0.10, "dd_min_scale": 0.3}),

    # === D. Trend filter (single-leg in trending market) ===
    ("Trend_24h_3%",        {"trend_filter": True, "trend_lookback": 24, "trend_threshold": 0.03}),
    ("Trend_24h_5%",        {"trend_filter": True, "trend_lookback": 24, "trend_threshold": 0.05}),
    ("Trend_12h_3%",        {"trend_filter": True, "trend_lookback": 12, "trend_threshold": 0.03}),

    # === E. Best combos ===
    ("IntrSL8+ProgDD",      {"intraday_sl_pct": 0.08, "dd_start": 0.05, "dd_full": 0.20, "dd_min_scale": 0.2}),
    ("IntrSL8+EqMA10",      {"intraday_sl_pct": 0.08, "equity_ma_days": 10}),
    ("IntrSL8+Trend5%",     {"intraday_sl_pct": 0.08, "trend_filter": True, "trend_lookback": 24, "trend_threshold": 0.05}),
    ("ProgDD+EqMA10",       {"dd_start": 0.05, "dd_full": 0.20, "dd_min_scale": 0.2, "equity_ma_days": 10}),
    ("ProgDD+Trend5%",      {"dd_start": 0.05, "dd_full": 0.20, "dd_min_scale": 0.2, "trend_filter": True, "trend_lookback": 24, "trend_threshold": 0.05}),
    ("AllThree",            {"intraday_sl_pct": 0.08, "dd_start": 0.05, "dd_full": 0.20, "dd_min_scale": 0.2, "trend_filter": True, "trend_lookback": 24, "trend_threshold": 0.05}),
]

results = []
for i, (label, overrides) in enumerate(scenarios):
    try:
        cfg = Config.from_yaml(BASE_CFG)
        for k, v in overrides.items():
            cfg.strategy.params[k] = v
        strat = ShortStrangleStrategy(params=cfg.strategy.params)
        engine = BacktestEngine(cfg, strat)
        r = engine.run()
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
        results.append({"label": label, "final": final_eq, "max_dd": max_dd, "sharpe": sharpe, "wr": wr, "trades": trades})
        sys.stderr.write(f"[{i+1}/{len(scenarios)}] {label:22s} Final={final_eq:>8.1f} DD={max_dd:>6.1f}%\n")
        sys.stderr.flush()
    except Exception as e:
        sys.stderr.write(f"[{i+1}/{len(scenarios)}] {label}: ERROR - {e}\n")
        traceback.print_exc(file=sys.stderr)
        results.append({"label": label, "final": 0, "max_dd": 0, "sharpe": 0, "wr": 0, "trades": 0})

# Build output
base_dd = results[0]["max_dd"] if results else 0
base_ret = results[0]["final"] if results else 1
lines = []
lines.append("=" * 105)
lines.append(f"  {'Strategy':<22s} {'Final':>8s} {'MaxDD%':>8s} {'DD改善':>7s} {'Ret%':>8s} {'Sharpe':>7s} {'WR%':>6s} {'Trades':>7s}")
lines.append("=" * 105)

for r in results:
    dd_imp = base_dd - r["max_dd"]
    ret_vs = (r["final"] / base_ret - 1) * 100 if base_ret > 0 else 0
    lines.append(
        f"  {r['label']:<22s} {r['final']:>8.1f} {r['max_dd']:>7.1f}% {dd_imp:>+6.1f}pp "
        f"{ret_vs:>+7.0f}% {r['sharpe']:>7.2f} {r['wr']:>5.1f}% {r['trades']:>7.0f}"
    )
lines.append("=" * 105)

text = "\n".join(lines)
print(text)
with open("dd_opt_full.txt", "w") as f:
    f.write(text)
