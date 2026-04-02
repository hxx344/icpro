"""BTC ATM DTE=7 Backtest."""
import sys, time
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")
from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.analytics.metrics import compute_metrics
from options_backtest.cli import _load_strategy

BASE_YAML = "configs/backtest/ic_0dte_8pct_btc.yaml"
dte = 7
cfg = Config.from_yaml(BASE_YAML)
cfg.strategy.params["otm_pct"] = 0.001
cfg.strategy.params["hedge_otm_pct"] = 0.0
cfg.strategy.params["min_days_to_expiry"] = 6.5
cfg.strategy.params["max_days_to_expiry"] = 8.5
cfg.strategy.params["roll_daily"] = False
cfg.strategy.params["compound"] = True
cfg.strategy.params["take_profit_pct"] = 9999
cfg.strategy.params["stop_loss_pct"] = 9999
cfg.strategy.params["max_positions"] = 1
cfg.backtest.name = "BTC ATM DTE=7"
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
print(f"  DTE=7  (hold)  Return={m.get('total_return',0)*100:>+7.1f}%  DD={m.get('max_drawdown',0)*100:>6.1f}%  Sharpe={m.get('sharpe_ratio',0):>6.2f}  WR={m.get('win_rate',0)*100:>5.1f}%  PF={m.get('profit_factor',0):>5.2f}  Trades={m.get('total_trades',0):>4.0f}  AvgW={sum(wins)/len(wins) if wins else 0:>.4f}  AvgL={sum(losses)/len(losses) if losses else 0:>.4f}  Mkt={mkt/tot*100 if tot else 0:.0f}%  ({elapsed:.1f}s)")
