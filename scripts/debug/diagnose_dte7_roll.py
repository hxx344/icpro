"""Diagnostic: DTE=7 OTM=10% daily roll — analyze per-trade P&L and data sources."""
import sys

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.analytics.metrics import compute_metrics
from options_backtest.cli import _load_strategy

cfg = Config.from_yaml("configs/backtest/ic_0dte_8pct_direct.yaml")
cfg.strategy.params["otm_pct"] = 0.10
cfg.strategy.params["hedge_otm_pct"] = 0.0
cfg.strategy.params["min_days_to_expiry"] = 6.5
cfg.strategy.params["max_days_to_expiry"] = 8.5
cfg.strategy.params["roll_daily"] = True       # daily roll (old behavior)
cfg.strategy.params["compound"] = True
cfg.strategy.params["take_profit_pct"] = 9999
cfg.strategy.params["stop_loss_pct"] = 9999
cfg.strategy.params["max_positions"] = 1
cfg.backtest.name = "DTE=7 OTM=10% Daily Roll (Diagnostic)"

s = _load_strategy(cfg.strategy.name, cfg.strategy.params)
e = BacktestEngine(cfg, s)
r = e.run()

# Quote source stats
market_q = e._quote_source_market
synth_q = e._quote_source_synth
total_q = market_q + synth_q
print(f"=== Quote Sources ===")
print(f"  Market (OHLCV):  {market_q:,d}  ({market_q/total_q*100:.1f}%)" if total_q else "  No quotes")
print(f"  Synthetic (BS):  {synth_q:,d}  ({synth_q/total_q*100:.1f}%)" if total_q else "")
print(f"  Total:           {total_q:,d}")

# Per-trade analysis
trades = e.position_mgr.closed_trades
print(f"\n=== Trade Summary ({len(trades)} closed legs) ===")

pnls = [t["pnl"] for t in trades]
entries = [t["entry_price"] for t in trades]
exits = [t["exit_price"] for t in trades]

wins = [p for p in pnls if p > 0]
losses = [p for p in pnls if p < 0]

print(f"  Total trades:  {len(pnls)}")
print(f"  Winners:       {len(wins)} ({len(wins)/len(pnls)*100:.1f}%)")
print(f"  Losers:        {len(losses)} ({len(losses)/len(pnls)*100:.1f}%)")
print(f"  Avg win:       {sum(wins)/len(wins):.6f}" if wins else "  No wins")
print(f"  Avg loss:      {sum(losses)/len(losses):.6f}" if losses else "  No losses")
print(f"  Total PnL:     {sum(pnls):.6f}")

# Entry/exit price analysis (premium levels)
print(f"\n=== Premium Analysis ===")
print(f"  Avg entry premium:  {sum(entries)/len(entries):.6f}")
print(f"  Avg exit premium:   {sum(exits)/len(exits):.6f}")
print(f"  Avg daily decay:    {(sum(entries)-sum(exits))/len(entries):.6f}")
print(f"  Avg decay %:        {(sum(entries)-sum(exits))/sum(entries)*100:.2f}%")

# Show first 20 trades to check pattern
print(f"\n=== First 20 Trades ===")
print(f"{'Date':>12s} {'Instrument':>32s} {'Dir':>5s} {'Entry':>10s} {'Exit':>10s} {'PnL':>12s} {'Type':>8s}")
for t in trades[:20]:
    entry_dt = t["entry_time"].strftime("%Y-%m-%d") if hasattr(t["entry_time"], "strftime") else str(t["entry_time"])[:10]
    print(f"{entry_dt:>12s} {t['instrument_name']:>32s} {t['direction']:>5s} "
          f"{t['entry_price']:>10.6f} {t['exit_price']:>10.6f} {t['pnl']:>12.6f} {t['close_type']:>8s}")

# Show some losing trades
print(f"\n=== Top 10 Losing Trades ===")
sorted_trades = sorted(trades, key=lambda t: t["pnl"])
for t in sorted_trades[:10]:
    entry_dt = t["entry_time"].strftime("%Y-%m-%d") if hasattr(t["entry_time"], "strftime") else str(t["entry_time"])[:10]
    exit_dt = t["exit_time"].strftime("%Y-%m-%d") if hasattr(t["exit_time"], "strftime") else str(t["exit_time"])[:10]
    print(f"  {entry_dt} → {exit_dt}  {t['instrument_name']:>32s}  PnL={t['pnl']:>+12.6f}  "
          f"Entry={t['entry_price']:.6f}  Exit={t['exit_price']:.6f}")

# Metrics
m = compute_metrics(r)
print(f"\n=== Metrics ===")
print(f"  Return: {m.get('total_return', 0)*100:+.1f}%")
print(f"  MaxDD:  {m.get('max_drawdown', 0)*100:.1f}%")
print(f"  Sharpe: {m.get('sharpe_ratio', 0):.2f}")
print(f"  WinRate:{m.get('win_rate', 0)*100:.1f}%")

# Now compare: run the SAME config but with roll_daily=False (hold to settlement)
print("\n" + "="*60)
print("  COMPARISON: Same params but hold-to-settlement")
print("="*60)

cfg2 = Config.from_yaml("configs/backtest/ic_0dte_8pct_direct.yaml")
cfg2.strategy.params["otm_pct"] = 0.10
cfg2.strategy.params["hedge_otm_pct"] = 0.0
cfg2.strategy.params["min_days_to_expiry"] = 6.5
cfg2.strategy.params["max_days_to_expiry"] = 8.5
cfg2.strategy.params["roll_daily"] = False     # hold to settlement
cfg2.strategy.params["compound"] = True
cfg2.strategy.params["take_profit_pct"] = 9999
cfg2.strategy.params["stop_loss_pct"] = 9999
cfg2.strategy.params["max_positions"] = 1
cfg2.backtest.name = "DTE=7 OTM=10% Hold-to-Settlement (Diagnostic)"

s2 = _load_strategy(cfg2.strategy.name, cfg2.strategy.params)
e2 = BacktestEngine(cfg2, s2)
r2 = e2.run()

market_q2 = e2._quote_source_market
synth_q2 = e2._quote_source_synth
total_q2 = market_q2 + synth_q2
print(f"\n=== Quote Sources (Hold) ===")
print(f"  Market (OHLCV):  {market_q2:,d}  ({market_q2/total_q2*100:.1f}%)" if total_q2 else "")
print(f"  Synthetic (BS):  {synth_q2:,d}  ({synth_q2/total_q2*100:.1f}%)" if total_q2 else "")

trades2 = e2.position_mgr.closed_trades
print(f"\n=== Trade Summary (Hold) — {len(trades2)} legs ===")
pnls2 = [t["pnl"] for t in trades2]
wins2 = [p for p in pnls2 if p > 0]
losses2 = [p for p in pnls2 if p < 0]
print(f"  Winners: {len(wins2)} ({len(wins2)/len(pnls2)*100:.1f}%)" if pnls2 else "")
print(f"  Losers:  {len(losses2)} ({len(losses2)/len(pnls2)*100:.1f}%)" if pnls2 else "")
print(f"  Avg win:  {sum(wins2)/len(wins2):.6f}" if wins2 else "  No wins")
print(f"  Avg loss: {sum(losses2)/len(losses2):.6f}" if losses2 else "  No losses")

entries2 = [t["entry_price"] for t in trades2]
exits2 = [t["exit_price"] for t in trades2]
print(f"  Avg entry premium:  {sum(entries2)/len(entries2):.6f}" if entries2 else "")
print(f"  Avg exit premium:   {sum(exits2)/len(exits2):.6f}" if exits2 else "")

# Show close types 
close_types = {}
for t in trades2:
    ct = t.get("close_type", "unknown")
    close_types[ct] = close_types.get(ct, 0) + 1
print(f"  Close types: {close_types}")

print(f"\n=== Top 10 Losing Trades (Hold) ===")
sorted_trades2 = sorted(trades2, key=lambda t: t["pnl"])
for t in sorted_trades2[:10]:
    entry_dt = t["entry_time"].strftime("%Y-%m-%d") if hasattr(t["entry_time"], "strftime") else str(t["entry_time"])[:10]
    exit_dt = t["exit_time"].strftime("%Y-%m-%d") if hasattr(t["exit_time"], "strftime") else str(t["exit_time"])[:10]
    print(f"  {entry_dt} → {exit_dt}  {t['instrument_name']:>32s}  PnL={t['pnl']:>+12.6f}  "
          f"Entry={t['entry_price']:.6f}  Exit={t['exit_price']:.6f}")

m2 = compute_metrics(r2)
print(f"\n=== Metrics (Hold) ===")
print(f"  Return: {m2.get('total_return', 0)*100:+.1f}%")
print(f"  MaxDD:  {m2.get('max_drawdown', 0)*100:.1f}%")
print(f"  Sharpe: {m2.get('sharpe_ratio', 0):.2f}")
