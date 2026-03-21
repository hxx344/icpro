"""Profile backtest execution to identify bottlenecks."""
import sys
import time
sys.path.insert(0, "src")

from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.analytics.metrics import compute_metrics
from options_backtest.cli import _load_strategy

cfg = Config.from_yaml("configs/backtest/ic_0dte_8pct_direct.yaml")
s = _load_strategy(cfg.strategy.name, cfg.strategy.params)
e = BacktestEngine(cfg, s)

# Phase 1: data loading
t0 = time.perf_counter()
e._load_data(cfg.backtest.underlying, cfg.backtest.start_date, cfg.backtest.end_date, cfg.backtest.time_step)
t1 = time.perf_counter()
print(f"[PROFILE] Data loading:    {t1-t0:.2f}s")
print(f"[PROFILE]   OHLCV series:  {len(e._ohlcv_index)}")
print(f"[PROFILE]   Instruments:   {len(e._instruments_df)}")

# Phase 2: full backtest run (separate engine to get total)
cfg2 = Config.from_yaml("configs/backtest/ic_0dte_8pct_direct.yaml")
s2 = _load_strategy(cfg2.strategy.name, cfg2.strategy.params)
e2 = BacktestEngine(cfg2, s2)

t2 = time.perf_counter()
results = e2.run()
t3 = time.perf_counter()
print(f"[PROFILE] Total run():     {t3-t2:.2f}s")
print(f"[PROFILE]   (Data load ~{t1-t0:.2f}s + Loop ~{t3-t2-(t1-t0):.2f}s)")
print(f"[PROFILE] Return: {results['total_return']*100:.0f}%")
print(f"[PROFILE] Trades: {results['total_trades']}")
