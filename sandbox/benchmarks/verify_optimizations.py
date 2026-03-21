"""Verify optimizations work through the real CLI code path."""
import time
from pathlib import Path

from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.strategy.short_strangle import ShortStrangleStrategy


def main():
    cfg = Config.from_yaml(Path("configs/backtest/default.yaml"))

    # Test 1: one month
    cfg.backtest.start_date = "2025-01-01"
    cfg.backtest.end_date = "2025-01-31"

    strategy = ShortStrangleStrategy(params=cfg.strategy.params)
    engine = BacktestEngine(cfg, strategy)

    t0 = time.perf_counter()
    results = engine.run()
    elapsed = time.perf_counter() - t0

    steps = len(results["equity_history"])
    print(f"\n{'='*60}")
    print(f"[1-month] {elapsed:.2f}s | {steps} steps | {steps/elapsed:.0f} steps/sec")
    print(f"Return: {results['total_return']:.4%} | Trades: {results['total_trades']}")
    print(f"{'='*60}")

    # Test 2: three months to confirm scaling
    cfg2 = Config.from_yaml(Path("configs/backtest/default.yaml"))
    cfg2.backtest.start_date = "2025-01-01"
    cfg2.backtest.end_date = "2025-03-31"

    strategy2 = ShortStrangleStrategy(params=cfg2.strategy.params)
    engine2 = BacktestEngine(cfg2, strategy2)

    t1 = time.perf_counter()
    results2 = engine2.run()
    elapsed2 = time.perf_counter() - t1

    steps2 = len(results2["equity_history"])
    print(f"\n{'='*60}")
    print(f"[3-month] {elapsed2:.2f}s | {steps2} steps | {steps2/elapsed2:.0f} steps/sec")
    print(f"Return: {results2['total_return']:.4%} | Trades: {results2['total_trades']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
