"""Diagnostic: DTE=5 vs DTE=7 hold-to-settlement — why such a huge gap?"""
import sys

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.analytics.metrics import compute_metrics
from options_backtest.cli import _load_strategy

def run_diagnostic(dte, otm, label):
    cfg = Config.from_yaml("configs/backtest/ic_0dte_8pct_direct.yaml")
    cfg.strategy.params["otm_pct"] = otm
    cfg.strategy.params["hedge_otm_pct"] = 0.0
    if dte == 0:
        cfg.strategy.params["min_days_to_expiry"] = 0.0
        cfg.strategy.params["max_days_to_expiry"] = 1.5
    else:
        cfg.strategy.params["min_days_to_expiry"] = max(dte - 0.5, 0)
        cfg.strategy.params["max_days_to_expiry"] = dte + 1.5
    cfg.strategy.params["roll_daily"] = False  # hold to settlement
    cfg.strategy.params["compound"] = True
    cfg.strategy.params["take_profit_pct"] = 9999
    cfg.strategy.params["stop_loss_pct"] = 9999
    cfg.strategy.params["max_positions"] = 1
    cfg.backtest.name = label

    s = _load_strategy(cfg.strategy.name, cfg.strategy.params)
    e = BacktestEngine(cfg, s)
    r = e.run()

    market_q = e._quote_source_market
    synth_q = e._quote_source_synth
    total_q = market_q + synth_q

    trades = e.position_mgr.closed_trades
    m = compute_metrics(r)

    return {
        "engine": e,
        "trades": trades,
        "metrics": m,
        "market_q": market_q,
        "synth_q": synth_q,
        "total_q": total_q,
    }

print("=" * 80)
print("  DTE=5 vs DTE=7 持有到结算 — 对比诊断 (OTM=2%)")
print("=" * 80)

for dte in [5, 7]:
    otm = 0.02
    label = f"DTE={dte} OTM={otm*100:.0f}%"
    d = run_diagnostic(dte, otm, label)
    trades = d["trades"]
    m = d["metrics"]

    print(f"\n{'='*60}")
    print(f"  {label} — hold-to-settlement")
    print(f"{'='*60}")

    # Quote sources
    print(f"\n  Quote Sources: Market={d['market_q']:,d} ({d['market_q']/d['total_q']*100:.1f}%), "
          f"Synthetic={d['synth_q']:,d} ({d['synth_q']/d['total_q']*100:.1f}%)")

    # Trade stats
    pnls = [t["pnl"] for t in trades]
    entries = [t["entry_price"] for t in trades]
    exits = [t["exit_price"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    close_types = {}
    for t in trades:
        ct = t.get("close_type", "unknown")
        close_types[ct] = close_types.get(ct, 0) + 1

    print(f"  Trades: {len(pnls)} legs,  Close types: {close_types}")
    print(f"  Winners: {len(wins)} ({len(wins)/len(pnls)*100:.1f}%),  Losers: {len(losses)} ({len(losses)/len(pnls)*100:.1f}%)")
    print(f"  Avg win: {sum(wins)/len(wins):.4f}" if wins else "  No wins")
    print(f"  Avg loss: {sum(losses)/len(losses):.4f}" if losses else "  No losses")
    print(f"  Avg entry premium: {sum(entries)/len(entries):.4f}")
    print(f"  Avg exit premium:  {sum(exits)/len(exits):.4f}")
    print(f"  Return: {m.get('total_return',0)*100:+.1f}%  MaxDD: {m.get('max_drawdown',0)*100:.1f}%  Sharpe: {m.get('sharpe_ratio',0):.2f}")

    # Expiry distribution — which expiries were actually used?
    expiry_counts = {}
    for t in trades:
        name = t["instrument_name"]
        # Parse expiry from instrument name like ETH-9MAY25-2000-C
        parts = name.split("-")
        if len(parts) >= 3:
            expiry = parts[1]
        else:
            expiry = "?"
        expiry_counts[expiry] = expiry_counts.get(expiry, 0) + 1

    print(f"\n  Expiries used: {len(expiry_counts)} unique")
    # Show all
    for exp, cnt in sorted(expiry_counts.items()):
        print(f"    {exp}: {cnt} legs")

    # Show all trades with details
    print(f"\n  All {len(trades)} trades:")
    print(f"  {'Entry Date':>12s} {'Exit Date':>12s} {'Instrument':>30s} {'Dir':>5s} {'Entry$':>10s} {'Exit$':>10s} {'PnL':>12s} {'Close':>10s}")
    for t in sorted(trades, key=lambda x: x["entry_time"]):
        entry_dt = t["entry_time"].strftime("%Y-%m-%d") if hasattr(t["entry_time"], "strftime") else str(t["entry_time"])[:10]
        exit_dt = t["exit_time"].strftime("%Y-%m-%d") if hasattr(t["exit_time"], "strftime") else str(t["exit_time"])[:10]
        print(f"  {entry_dt:>12s} {exit_dt:>12s} {t['instrument_name']:>30s} {t['direction']:>5s} "
              f"{t['entry_price']:>10.4f} {t['exit_price']:>10.4f} {t['pnl']:>+12.4f} {t['close_type']:>10s}")


# Now check the DTE filter: what instruments does each actually select?
print("\n\n" + "=" * 80)
print("  DTE Range Check — what instruments match each filter?")
print("=" * 80)

import pandas as pd
import numpy as np
from options_backtest.data.loader import load_instruments

cfg = Config.from_yaml("configs/backtest/ic_0dte_8pct_direct.yaml")

# Check a specific date to see which expiries have DTE in range
from options_backtest.data.loader import load_instruments
instruments = load_instruments(cfg)
print(f"\n  Total instruments loaded: {len(instruments)}")

# Count by expiry
from collections import Counter
exp_counter = Counter()
for inst in instruments:
    exp_counter[inst.get("expiration_date", "?")] += 1

# Show expiry frequency
print(f"  Unique expiries: {len(exp_counter)}")
# Sort by date and show first/last 10
sorted_exps = sorted(exp_counter.keys())
print(f"  First 5: {sorted_exps[:5]}")
print(f"  Last 5:  {sorted_exps[-5:]}")

# Check: on any given day, which expiries match DTE=5 vs DTE=7?
sample_dates = ["2025-06-01", "2025-09-01", "2025-12-01"]
for sd in sample_dates:
    sd_ts = pd.Timestamp(sd, tz="UTC")
    print(f"\n  On {sd}:")
    for exp_str in sorted_exps:
        exp_ts = pd.Timestamp(exp_str, tz="UTC") if not isinstance(exp_str, pd.Timestamp) else exp_str
        dte_days = (exp_ts - sd_ts).total_seconds() / 86400
        if 4.0 <= dte_days <= 9.0:
            cnt = exp_counter[exp_str]
            in5 = "✓ DTE5" if 4.5 <= dte_days <= 6.5 else ""
            in7 = "✓ DTE7" if 6.5 <= dte_days <= 8.5 else ""
            print(f"    {exp_str} → DTE={dte_days:.1f}d  ({cnt} instruments)  {in5}  {in7}")
