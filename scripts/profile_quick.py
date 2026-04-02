"""Quick profiling: where is time spent in a single backtest run?"""
import sys, time, cProfile, pstats, io

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.cli import _load_strategy

cfg = Config.from_yaml("configs/backtest/ic_0dte_8pct_direct.yaml")
cfg.strategy.params["otm_pct"] = 0.005
cfg.strategy.params["hedge_otm_pct"] = 0.025
cfg.strategy.params["compound"] = True

s = _load_strategy(cfg.strategy.name, cfg.strategy.params)
e = BacktestEngine(cfg, s)

# Warm up data cache
print("Loading data (first run)...")
t0 = time.perf_counter()
r = e.run()
print(f"First run: {time.perf_counter()-t0:.1f}s")

# Profile second run (data cached)
print("\nProfiling second run...")
s2 = _load_strategy(cfg.strategy.name, cfg.strategy.params)
e2 = BacktestEngine(cfg, s2)

pr = cProfile.Profile()
pr.enable()
t0 = time.perf_counter()
r2 = e2.run()
elapsed = time.perf_counter() - t0
pr.disable()

print(f"Second run: {elapsed:.1f}s")
print(f"\nTop 30 by cumulative time:")
stream = io.StringIO()
ps = pstats.Stats(pr, stream=stream)
ps.sort_stats("cumulative")
ps.print_stats(30)
print(stream.getvalue())

print(f"\nTop 30 by total (self) time:")
stream2 = io.StringIO()
ps2 = pstats.Stats(pr, stream=stream2)
ps2.sort_stats("tottime")
ps2.print_stats(30)
print(stream2.getvalue())
