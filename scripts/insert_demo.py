"""Insert demo data for dashboard testing."""
import sys, random, json
from datetime import datetime, timezone, timedelta
from trader.config import load_config
from trader.storage import Storage

cfg = load_config("configs/trader/iron_condor_0dte.yaml")
s = Storage(cfg.storage.db_path)

base = datetime(2026, 3, 15, 8, 0, 0, tzinfo=timezone.utc)
equity = 10000.0

# Equity snapshots
for i in range(120):
    ts = base + timedelta(hours=i)
    random.seed(i)
    delta = random.uniform(-50, 70)
    equity += delta
    s.record_equity_snapshot(
        equity, equity * 0.8, delta,
        1 if i % 24 < 16 else 0,
        85000 + random.uniform(-2000, 2000),
    )

# Daily PnL
for d in range(5):
    day = (base + timedelta(days=d)).strftime("%Y-%m-%d")
    random.seed(d + 1000)
    start_eq = 10000 + d * 50
    end_eq = start_eq + random.uniform(-30, 80)
    s.record_daily_pnl(day, start_eq, end_eq, end_eq - start_eq, 0,
                        random.uniform(1, 5), random.randint(4, 8))

# Trades
roles = ["sell_put", "buy_put", "sell_call", "buy_call"]
opt_types = ["put", "put", "call", "call"]
strikes = [78000, 76000, 92000, 94000]
sides = ["SELL", "BUY", "SELL", "BUY"]
cps = ["P", "P", "C", "C"]

for i in range(8):
    idx = i % 4
    meta = {
        "leg_role": roles[idx],
        "option_type": opt_types[idx],
        "strike": strikes[idx],
        "underlying_price": 85000,
    }
    symbol = f"BTC-260318-{strikes[idx]}-{cps[idx]}"
    random.seed(i + 2000)
    trade_id = s.record_trade(
        "IC_20260318_080000_abc123", symbol, sides[idx],
        0.01, random.uniform(50, 200), random.uniform(0.5, 2),
        f"order_{i}", meta,
    )
    if i >= 4:
        s.close_trade(trade_id, random.uniform(20, 100), random.uniform(-20, 50))

s.close()
print("Demo data inserted OK")
