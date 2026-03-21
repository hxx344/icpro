"""Compare OTM levels with full metrics."""
import sys, time
sys.path.insert(0, "src")

# Suppress loguru DEBUG logs (major perf drain)
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.analytics.metrics import compute_metrics
from options_backtest.cli import _load_strategy

otm_list = [0.0, 0.005, 0.01]

for otm in otm_list:
    cfg = Config.from_yaml("configs/backtest/ic_0dte_8pct_direct.yaml")
    cfg.strategy.params["otm_pct"] = otm
    s = _load_strategy(cfg.strategy.name, cfg.strategy.params)
    e = BacktestEngine(cfg, s)

    t0 = time.perf_counter()
    r = e.run()
    elapsed = time.perf_counter() - t0

    m = compute_metrics(r)
    mode = m.get("margin_mode", "?")
    cur = m.get("currency", "?")

    print(f"\n{'='*60}")
    print(f"  OTM = {otm*100:.1f}%  |  {mode} margin ({cur})  |  {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"  初始资金:           {m.get('initial_balance',0):>14,.2f} {cur}")
    print(f"  最终权益:           {m.get('final_equity',0):>14,.2f} {cur}")
    print(f"  总收益:             {m.get('total_return',0)*100:>14,.0f}%")
    print(f"  年化收益:           {m.get('annualized_return',0)*100:>14,.0f}%")
    print(f"  最大回撤:           {m.get('max_drawdown',0)*100:>14.2f}%")
    print(f"  Sharpe:             {m.get('sharpe_ratio',0):>14.2f}")
    print(f"  交易次数:           {m.get('total_trades',0):>14}")
    print(f"  胜率:               {m.get('win_rate',0)*100:>14.1f}%")
    print(f"  平均盈利:           {m.get('avg_win',0):>14.2f} {cur}")
    print(f"  平均亏损:           {m.get('avg_loss',0):>14.2f} {cur}")
    print(f"  盈亏比:             {m.get('profit_factor',0):>14.2f}")
    print(f"  总手续费:           {m.get('total_fees',0):>14.2f} {cur}")
