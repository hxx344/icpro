"""ETH 0DTE ATM + Wings 2%-6% Grid."""
import sys, time
sys.path.insert(0, "src")
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")
from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.analytics.metrics import compute_metrics
from options_backtest.cli import _load_strategy

WINGS = [0.02, 0.03, 0.04, 0.05, 0.06]
BASE_YAML = "configs/backtest/ic_0dte_8pct_direct.yaml"

print("ETH 0DTE ATM Iron Condor - Wing Width Comparison")
print("Short strike: ATM (0.1% OTM)  |  Wings: 2%-6%")
print("=" * 120)

rows = []
for wing in WINGS:
    cfg = Config.from_yaml(BASE_YAML)
    cfg.strategy.params["otm_pct"] = 0.001
    cfg.strategy.params["hedge_otm_pct"] = wing
    cfg.strategy.params["min_days_to_expiry"] = 0.0
    cfg.strategy.params["max_days_to_expiry"] = 1.5
    cfg.strategy.params["roll_daily"] = True
    cfg.strategy.params["compound"] = True
    cfg.strategy.params["take_profit_pct"] = 9999
    cfg.strategy.params["stop_loss_pct"] = 9999
    cfg.strategy.params["max_positions"] = 1
    cfg.backtest.name = f"ETH 0DTE ATM Wing={wing*100:.0f}%"
    s = _load_strategy(cfg.strategy.name, cfg.strategy.params)
    e = BacktestEngine(cfg, s)
    t0 = time.perf_counter()
    r = e.run()
    elapsed = time.perf_counter() - t0
    m = compute_metrics(r)
    trades = e.position_mgr.closed_trades
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    mkt = e._quote_source_market
    syn = e._quote_source_synth
    tot = mkt + syn
    row = {"wing": wing*100, "return_pct": m.get("total_return",0)*100, "max_dd_pct": m.get("max_drawdown",0)*100, "sharpe": m.get("sharpe_ratio",0), "win_rate": m.get("win_rate",0)*100, "profit_factor": m.get("profit_factor",0), "trades": m.get("total_trades",0), "avg_win": sum(wins)/len(wins) if wins else 0, "avg_loss": sum(losses)/len(losses) if losses else 0, "mkt_pct": mkt/tot*100 if tot else 0, "elapsed": elapsed}
    rows.append(row)
    print(f"  Wing={wing*100:.0f}%  Return={row['return_pct']:>+7.1f}%  DD={row['max_dd_pct']:>6.1f}%  Sharpe={row['sharpe']:>6.2f}  WR={row['win_rate']:>5.1f}%  PF={row['profit_factor']:>5.2f}  Trades={row['trades']:>4.0f}  AvgW={row['avg_win']:>.6f}  AvgL={row['avg_loss']:>.6f}  Mkt={row['mkt_pct']:.0f}%  ({elapsed:.1f}s)")

print("\n--- Comparison with naked ATM (no wings) ---")
cfg = Config.from_yaml(BASE_YAML)
cfg.strategy.params["otm_pct"] = 0.001
cfg.strategy.params["hedge_otm_pct"] = 0.0
cfg.strategy.params["min_days_to_expiry"] = 0.0
cfg.strategy.params["max_days_to_expiry"] = 1.5
cfg.strategy.params["roll_daily"] = True
cfg.strategy.params["compound"] = True
cfg.strategy.params["take_profit_pct"] = 9999
cfg.strategy.params["stop_loss_pct"] = 9999
cfg.strategy.params["max_positions"] = 1
cfg.backtest.name = "ETH 0DTE ATM naked"
s = _load_strategy(cfg.strategy.name, cfg.strategy.params)
e = BacktestEngine(cfg, s)
t0 = time.perf_counter()
r = e.run()
elapsed = time.perf_counter() - t0
m = compute_metrics(r)
mkt = e._quote_source_market
syn = e._quote_source_synth
tot = mkt + syn
print(f"  Naked  Return={m.get('total_return',0)*100:>+7.1f}%  DD={m.get('max_drawdown',0)*100:>6.1f}%  Sharpe={m.get('sharpe_ratio',0):>6.2f}  WR={m.get('win_rate',0)*100:>5.1f}%  PF={m.get('profit_factor',0):>5.2f}  Trades={m.get('total_trades',0):>4.0f}  Mkt={mkt/tot*100 if tot else 0:.0f}%  ({elapsed:.1f}s)")
