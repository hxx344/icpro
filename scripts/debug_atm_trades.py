"""Debug: trace individual trades for OTM=0% to verify PnL realism."""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ["LOGURU_LEVEL"] = "WARNING"
warnings.filterwarnings("ignore")

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.strategy.short_strangle import ShortStrangleStrategy

cfg = Config.from_yaml("configs/backtest/ic_0dte_8pct_direct.yaml")
cfg.strategy.params["otm_pct"] = 0.00       # ATM
cfg.strategy.params["hedge_otm_pct"] = 0.06  # 6% wing
cfg.strategy.params["compound"] = True

strategy = ShortStrangleStrategy(params=cfg.strategy.params)
engine = BacktestEngine(cfg, strategy)
results = engine.run()

# Analyze first N closed trades
trades = engine.position_mgr.closed_trades
print(f"\n{'='*100}")
print(f"Total closed trades: {len(trades)}")
print(f"Initial balance (USD): {engine.account.initial_balance:,.2f}")
print(f"Final balance (USD): {engine.account.balance:,.2f}")
print(f"Total fees: {engine.account.total_fee_paid:,.2f}")
print(f"{'='*100}")

# Group trades by exit_time (each condor = 4 legs settled together)
from collections import defaultdict
condor_groups = defaultdict(list)
for t in trades:
    key = str(t["exit_time"])[:13]  # group by hour
    condor_groups[key].append(t)

# Show first 10 condor groups
shown = 0
for key in sorted(condor_groups.keys()):
    if shown >= 15:
        break
    legs = condor_groups[key]
    total_pnl = sum(t["pnl"] for t in legs)
    total_fee = sum(t["fee"] for t in legs)
    print(f"\n--- [{key}] {len(legs)} legs, net PnL={total_pnl:+.4f}, fees={total_fee:.4f} ---")
    for t in legs:
        direction = t["direction"]
        name = t["instrument_name"]
        qty = t["quantity"]
        entry_p = t["entry_price"]
        exit_p = t["exit_price"]
        pnl = t["pnl"]
        close_type = t["close_type"]
        print(f"  {direction:5s} {name:35s} qty={qty:.4f} entry=${entry_p:>10.4f} exit=${exit_p:>10.4f} pnl=${pnl:>+12.4f} ({close_type})")
    shown += 1

# Equity curve samples
eq_hist = engine.account.equity_history
print(f"\n{'='*100}")
print("Equity curve (sampled):")
step = max(1, len(eq_hist) // 20)
for i in range(0, len(eq_hist), step):
    ts, eq, bal, upnl, price = eq_hist[i]
    print(f"  {str(ts)[:16]}  equity=${eq:>14,.2f}  balance=${bal:>14,.2f}  upnl=${upnl:>+10,.2f}  ETH=${price:,.2f}")
# last point
ts, eq, bal, upnl, price = eq_hist[-1]
print(f"  {str(ts)[:16]}  equity=${eq:>14,.2f}  balance=${bal:>14,.2f}  upnl=${upnl:>+10,.2f}  ETH=${price:,.2f}")
