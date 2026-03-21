import time, cProfile, pstats, io
from pathlib import Path
from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.strategy.short_strangle import ShortStrangleStrategy

cfg = Config.from_yaml(Path('configs/backtest/default.yaml'))
cfg.backtest.start_date = '2025-01-01'
cfg.backtest.end_date = '2025-02-28'

strategy = ShortStrangleStrategy(params=cfg.strategy.params)
engine = BacktestEngine(cfg, strategy)

pr = cProfile.Profile()
pr.enable()
t0 = time.perf_counter()
results = engine.run()
elapsed = time.perf_counter() - t0
pr.disable()

print(f'\nTotal: {elapsed:.2f}s')

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(40)
print(s.getvalue())
