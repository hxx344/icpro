"""BTC ATM DTE 0-2 Grid."""
import sys, time
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")
from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.analytics.metrics import compute_metrics
from options_backtest.cli import _load_strategy

DTE_TARGETS = [0, 1, 2]
ATM_OTM = 0.001
BASE_YAML = "configs/backtest/ic_0dte_8pct_btc.yaml"
rows = []
for dte in DTE_TARGETS:
    cfg = Config.from_yaml(BASE_YAML)
    cfg.strategy.params["otm_pct"] = ATM_OTM
    cfg.strategy.params["hedge_otm_pct"] = 0.0
    if dte == 0:
        cfg.strategy.params["min_days_to_expiry"] = 0.0
        cfg.strategy.params["max_days_to_expiry"] = 1.5
    else:
        cfg.strategy.params["min_days_to_expiry"] = max(dte - 0.5, 0)
        cfg.strategy.params["max_days_to_expiry"] = dte + 1.5
    cfg.strategy.params["roll_daily"] = (dte <= 1)
    cfg.strategy.params["compound"] = True
    cfg.strategy.params["take_profit_pct"] = 9999
    cfg.strategy.params["stop_loss_pct"] = 9999
    cfg.strategy.params["max_positions"] = 1
    cfg.backtest.name = f"BTC ATM DTE={dte}"
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
    row = {"dte": dte, "return_pct": m.get("total_return",0)*100, "max_dd_pct": m.get("max_drawdown",0)*100, "sharpe": m.get("sharpe_ratio",0), "win_rate": m.get("win_rate",0)*100, "profit_factor": m.get("profit_factor",0), "trades": m.get("total_trades",0), "avg_win": sum(wins)/len(wins) if wins else 0, "avg_loss": sum(losses)/len(losses) if losses else 0, "mkt_pct": mkt/tot*100 if tot else 0, "elapsed": elapsed}
    rows.append(row)
    mode = "roll" if dte <= 1 else "hold"
    print(f"  DTE={dte}  ({mode})  Return={row['return_pct']:>+7.1f}%  DD={row['max_dd_pct']:>6.1f}%  Sharpe={row['sharpe']:>6.2f}  WR={row['win_rate']:>5.1f}%  PF={row['profit_factor']:>5.2f}  Trades={row['trades']:>4.0f}  AvgW={row['avg_win']:>.4f}  AvgL={row['avg_loss']:>.4f}  Mkt={row['mkt_pct']:.0f}%  ({elapsed:.1f}s)")
