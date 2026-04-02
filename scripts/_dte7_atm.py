"""BTC DTE=7 ATM Short Strangle — open Friday, hold to settlement."""
import sys
from loguru import logger; logger.remove(); logger.add(sys.stderr, level='WARNING')

from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.analytics.metrics import compute_metrics, print_metrics
from options_backtest.analytics.plotting import generate_all_plots
from options_backtest.cli import _load_strategy

cfg = Config.from_yaml('configs/backtest/ic_0dte_8pct_direct.yaml')
cfg.backtest.underlying = 'BTC'
cfg.backtest.start_date = '2023-01-01'
cfg.backtest.end_date = '2026-01-31'
cfg.backtest.name = 'BTC DTE7 ATM ShortStrangle (Fri→Settlement)'

# --- Strategy params ---
cfg.strategy.params['otm_pct'] = 0.0       # ATM
cfg.strategy.params['hedge_otm_pct'] = 0.0  # naked (no wings)
cfg.strategy.params['min_days_to_expiry'] = 6.0   # DTE 6-8 → only matches on Fridays
cfg.strategy.params['max_days_to_expiry'] = 8.0
cfg.strategy.params['roll_daily'] = False   # NOT daily rolling — hold to expiry
cfg.strategy.params['roll_days_before_expiry'] = 0  # hold to settlement
cfg.strategy.params['compound'] = True
cfg.strategy.params['take_profit_pct'] = 9999  # no TP
cfg.strategy.params['stop_loss_pct'] = 9999    # no SL
cfg.strategy.params['max_positions'] = 1
cfg.strategy.params['quantity'] = 1.0

s = _load_strategy(cfg.strategy.name, cfg.strategy.params)
e = BacktestEngine(cfg, s)
r = e.run()
m = compute_metrics(r)
print_metrics(m)

# Analyze trades
import pandas as pd
trades = r['closed_trades']
df = pd.DataFrame(trades)
df['entry_time'] = pd.to_datetime(df['entry_time'])
df['exit_time'] = pd.to_datetime(df['exit_time'])

# Weekly stats
print("\n=== Trade Timing Analysis ===")
print(f"Total trades: {len(df)}")
if not df.empty:
    df['entry_dow'] = df['entry_time'].dt.day_name()
    print(f"Entry days distribution:\n{df['entry_dow'].value_counts().to_string()}")
    print(f"\nHolding duration (hours): mean={( (df['exit_time'] - df['entry_time']).dt.total_seconds() / 3600).mean():.1f}")
    
    # Per-year stats
    df['year'] = df['entry_time'].dt.year
    print("\n=== Per-Year Summary ===")
    for yr, g in df.groupby('year'):
        wins = (g['pnl'] > 0).sum()
        losses = (g['pnl'] <= 0).sum()
        total_pnl = g['pnl'].sum()
        avg_ep = g['entry_price'].mean()
        print(f"  {yr}: {len(g):3d} trades, W={wins:3d} L={losses:2d}, PnL={total_pnl:10.2f}, avgEntry={avg_ep:.4f}")

    # Top losses
    print("\n=== Worst 5 losses ===")
    for _, t in df.nsmallest(5, 'pnl').iterrows():
        print(f"  {str(t['entry_time'])[:10]} {t['instrument_name']}: pnl={t['pnl']:.4f}")

# Generate charts
paths = generate_all_plots(r, 'reports')
for p in paths:
    print(f"Saved: {p}")
