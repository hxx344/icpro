"""Benchmark: run 1-month backtest and report timing."""
import time
from pathlib import Path

from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.strategy.short_strangle import ShortStrangleStrategy


def main():
    cfg = Config.from_yaml(Path("configs/backtest/default.yaml"))

    # Override to only 1 month
    cfg.backtest.start_date = "2025-01-01"
    cfg.backtest.end_date = "2025-01-31"

    strategy = ShortStrangleStrategy(params=cfg.strategy.params)
    engine = BacktestEngine(cfg, strategy)

    t0 = time.perf_counter()
    results = engine.run()
    elapsed = time.perf_counter() - t0

    print(f"\n{'='*50}")
    print(f"1-month backtest completed in {elapsed:.2f}s")
    print(f"Time steps: {len(results['equity_history'])}")
    print(f"Total trades: {results['total_trades']}")
    print(f"Total return: {results['total_return']:.4%}")
    print(f"Speed: {len(results['equity_history'])/elapsed:.0f} steps/sec")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
