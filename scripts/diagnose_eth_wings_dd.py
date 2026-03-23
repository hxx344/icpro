"""Diagnose ETH 0DTE ATM + 3% wings drawdown."""
import sys
sys.path.insert(0, "src")
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")
from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.analytics.metrics import compute_metrics
from options_backtest.cli import _load_strategy
import numpy as np

cfg = Config.from_yaml("configs/backtest/ic_0dte_8pct_direct.yaml")
cfg.strategy.params["otm_pct"] = 0.001
cfg.strategy.params["hedge_otm_pct"] = 0.03
cfg.strategy.params["min_days_to_expiry"] = 0.0
cfg.strategy.params["max_days_to_expiry"] = 1.5
cfg.strategy.params["roll_daily"] = True
cfg.strategy.params["compound"] = True
cfg.strategy.params["take_profit_pct"] = 9999
cfg.strategy.params["stop_loss_pct"] = 9999
cfg.strategy.params["max_positions"] = 1
cfg.backtest.name = "ETH 0DTE ATM Wing=3%"
s = _load_strategy(cfg.strategy.name, cfg.strategy.params)
e = BacktestEngine(cfg, s)
r = e.run()
trades = e.position_mgr.closed_trades

pnls = [t["pnl"] for t in trades]
wins = [p for p in pnls if p > 0]
losses = [p for p in pnls if p < 0]

print(f"Total trades: {len(trades)}")
print(f"Wins: {len(wins)} ({len(wins)/len(trades)*100:.1f}%)  Losses: {len(losses)} ({len(losses)/len(trades)*100:.1f}%)")
print(f"Avg win:  {np.mean(wins):.4f}")
print(f"Avg loss: {np.mean(losses):.4f}")
print(f"Max single loss: {min(pnls):.4f}")
print(f"Max single win:  {max(pnls):.4f}")

# PnL distribution
pnl_arr = np.array(pnls)
print(f"\nPnL percentiles:")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"  P{p:>2d}: {np.percentile(pnl_arr, p):.4f}")

# Find worst consecutive losses
print(f"\n--- Worst consecutive losing streaks ---")
streak = 0
max_streak = 0
streak_loss = 0
max_streak_loss = 0
streak_start = 0
max_streak_start = 0
for i, p in enumerate(pnls):
    if p < 0:
        if streak == 0:
            streak_start = i
        streak += 1
        streak_loss += p
        if streak > max_streak:
            max_streak = streak
            max_streak_loss = streak_loss
            max_streak_start = streak_start
    else:
        streak = 0
        streak_loss = 0
print(f"Max losing streak: {max_streak} trades, total loss: {max_streak_loss:.4f}")

# Equity curve & drawdown analysis 
equity = [1.0]
for p in pnls:
    equity.append(equity[-1] + p)
eq = np.array(equity)
running_max = np.maximum.accumulate(eq)
dd = (eq - running_max) / running_max
worst_dd_idx = np.argmin(dd)
print(f"\nWorst drawdown: {dd[worst_dd_idx]*100:.1f}% at trade #{worst_dd_idx}")
# Find the peak before the worst drawdown
peak_idx = np.argmax(eq[:worst_dd_idx+1])
print(f"Peak before DD: trade #{peak_idx}, equity={eq[peak_idx]:.4f}")
print(f"Trough at: trade #{worst_dd_idx}, equity={eq[worst_dd_idx]:.4f}")

# Show trades around the worst drawdown
print(f"\n--- Trades during worst drawdown (trade #{peak_idx} to #{worst_dd_idx}) ---")
dd_trades = trades[peak_idx:worst_dd_idx]
n_loss = sum(1 for t in dd_trades if t["pnl"] < 0)
n_win = sum(1 for t in dd_trades if t["pnl"] > 0)
total_loss = sum(t["pnl"] for t in dd_trades if t["pnl"] < 0)
total_win = sum(t["pnl"] for t in dd_trades if t["pnl"] > 0)
print(f"Trades in DD period: {len(dd_trades)} ({n_win} wins, {n_loss} losses)")
print(f"Total gain: {total_win:.4f}, Total loss: {total_loss:.4f}, Net: {total_win+total_loss:.4f}")

# Show the 10 worst individual trades
print(f"\n--- 10 Worst individual trades ---")
sorted_trades = sorted(trades, key=lambda t: t["pnl"])
for i, t in enumerate(sorted_trades[:10]):
    dt = t.get("open_time", t.get("entry_time", "?"))
    print(f"  #{i+1}: PnL={t['pnl']:.6f}  Date={dt}  Instruments: {[leg.get('instrument','?') for leg in t.get('legs',[])]}")

# Iron condor max loss calculation
print(f"\n--- Iron Condor max loss analysis ---")
print(f"With ATM short + 3% wings:")
print(f"  Wing width = 3% of underlying")
print(f"  Max loss per side = wing_width - premium_collected")
print(f"  But with compound=True, losses scale with equity")
# Check if losses are actually capped
abs_losses = sorted([abs(p) for p in pnls if p < 0], reverse=True)
print(f"  Top 10 absolute losses: {[f'{l:.6f}' for l in abs_losses[:10]]}")
print(f"  Median absolute loss: {np.median(abs_losses):.6f}")
print(f"  This is in BTC-denomination (USD margin sim)")
