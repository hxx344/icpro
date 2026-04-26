"""ETH 0DTE ATM: TP80 with SL50-SL100 + W4% fine grid."""
import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")
from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.analytics.metrics import compute_metrics
from options_backtest.cli import _load_strategy
import numpy as np

BASE_YAML = "configs/backtest/ic_0dte_8pct_direct.yaml"
BASE_PARAMS = {
    "otm_pct": 0.001, "hedge_otm_pct": 0.04,
    "min_days_to_expiry": 0.0, "max_days_to_expiry": 1.5,
    "roll_daily": True, "compound": True, "max_positions": 1,
}

sls = [50, 60, 70, 80, 90, 100, 120, 150]
print(f"ETH 0DTE ATM + Wing4% : TP80 + SL fine grid")
print(f"{'Strategy':<22s} {'Return':>8s} {'MaxDD':>8s} {'Sharpe':>7s} {'WR':>6s} {'PF':>6s} {'Trades':>6s} {'AvgW':>10s} {'AvgL':>10s}")
print("=" * 95)

for sl in sls:
    cfg = Config.from_yaml(BASE_YAML)
    for k, v in BASE_PARAMS.items():
        cfg.strategy.params[k] = v
    cfg.strategy.params["take_profit_pct"] = 80
    cfg.strategy.params["stop_loss_pct"] = sl
    cfg.backtest.name = f"TP80/SL{sl}+W4%"
    s = _load_strategy(cfg.strategy.name, cfg.strategy.params)
    e = BacktestEngine(cfg, s)
    r = e.run()
    m = compute_metrics(r)
    trades = e.position_mgr.closed_trades
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    aw = np.mean(wins) if wins else 0
    al = np.mean(losses) if losses else 0
    label = f"TP80/SL{sl}+W4%"
    print(f"{label:<22s} {m.get('total_return',0)*100:>+7.1f}% {m.get('max_drawdown',0)*100:>7.1f}% {m.get('sharpe_ratio',0):>7.2f} {m.get('win_rate',0)*100:>5.1f}% {m.get('profit_factor',0):>6.2f} {len(trades):>5d} {aw:>10.2f} {al:>10.2f}")
