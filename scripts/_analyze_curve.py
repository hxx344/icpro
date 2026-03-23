"""Analyze the 10% OTM short strangle equity curve anomaly."""
import sys; sys.path.insert(0, 'src')
from loguru import logger; logger.remove(); logger.add(sys.stderr, level='ERROR')
from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.cli import _load_strategy
import pandas as pd

cfg = Config.from_yaml('configs/backtest/ic_0dte_8pct_direct.yaml')
cfg.backtest.underlying = 'BTC'
cfg.backtest.start_date = '2023-01-01'
cfg.backtest.end_date = '2026-01-31'
cfg.strategy.params['otm_pct'] = 0.10
cfg.strategy.params['hedge_otm_pct'] = 0.0
cfg.strategy.params['min_days_to_expiry'] = 0.0
cfg.strategy.params['max_days_to_expiry'] = 1.5
cfg.strategy.params['roll_daily'] = True
cfg.strategy.params['compound'] = True
cfg.strategy.params['take_profit_pct'] = 9999
cfg.strategy.params['stop_loss_pct'] = 9999
cfg.strategy.params['max_positions'] = 1

s = _load_strategy(cfg.strategy.name, cfg.strategy.params)
e = BacktestEngine(cfg, s)
r = e.run()

trades = r['closed_trades']
df = pd.DataFrame(trades)
df['exit_time'] = pd.to_datetime(df['exit_time'])
df['quarter'] = df['exit_time'].dt.to_period('Q').astype(str)

print('=== Per-Quarter Summary ===')
for q, g in df.groupby('quarter'):
    wins = (g['pnl'] > 0).sum()
    losses = (g['pnl'] <= 0).sum()
    total_pnl = g['pnl'].sum()
    avg_ep = g['entry_price'].mean()
    print(f"{q}: {len(g):4d}t W={wins:3d} L={losses:2d} PnL={total_pnl:8.1f} avgEntry={avg_ep:.2f}")

print()
print('=== First 10 trades ===')
for _, t in df.head(10).iterrows():
    inst = t['instrument_name']
    ep = t['entry_price']
    xp = t['exit_price']
    pnl = t['pnl']
    ct = t['close_type']
    et = str(t['entry_time'])[:10]
    print(f"  {et} {inst}: entry={ep:.4f} exit={xp:.4f} pnl={pnl:.4f} type={ct}")

print()
print('=== Worst 10 trades ===')
for _, t in df.nsmallest(10, 'pnl').iterrows():
    inst = t['instrument_name']
    ep = t['entry_price']
    xp = t['exit_price']
    pnl = t['pnl']
    ct = t['close_type']
    et = str(t['entry_time'])[:10]
    print(f"  {et} {inst}: entry={ep:.4f} exit={xp:.4f} pnl={pnl:.4f} type={ct}")

# Equity history
eq = r['equity_history']
print(f'\n=== Equity History Structure ===')
print(f"Keys: {list(eq[0].keys()) if eq else 'empty'}")
print(f"Total points: {len(eq)}")

# Sample early equity points (every 2 days = 48h)
print('\n=== Equity Curve Sample ===')
for i in range(0, min(len(eq), 5000), 168):  # weekly
    pt = eq[i]
    ts = pt.get('timestamp', pt.get('time', ''))
    bal = pt.get('equity', pt.get('balance', 0))
    print(f"  [{i:5d}] {str(ts)[:16]}: equity={bal:.2f}")
