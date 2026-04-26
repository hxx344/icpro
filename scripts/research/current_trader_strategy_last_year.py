"""Backtest the current live trader strategy over the latest available year.

Uses `configs/trader/weekend_vol_btc.yaml` as the source of truth and runs the
matching backtest logic in USD mode with the standard trader backtest starting
capital of 10,000 USD.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from options_backtest.analytics.metrics import compute_metrics
from options_backtest.analytics.plotting import plot_equity_curve
from options_backtest.cli import _load_strategy
from options_backtest.engine.backtest import BacktestEngine
from scripts.research.run_cached_hourly_backtest import (
    DEFAULT_TRADER_INITIAL_USD,
    _apply_initial_usd_balance,
    detect_hourly_coverage,
    load_run_config,
)

TRADER_CONFIG = Path("configs/trader/weekend_vol_btc.yaml")
OUTPUT_DIR = Path("reports/current_trader_strategy_last_year")


def main() -> None:
    loaded = load_run_config(TRADER_CONFIG)
    cfg = loaded.cfg

    coverage = detect_hourly_coverage(cfg.backtest.underlying)
    end_ts = coverage.end_ts.normalize()
    start_ts = end_ts - pd.Timedelta(days=364)

    cfg.backtest.name = "Current trader strategy (latest available year)"
    cfg.backtest.start_date = start_ts.strftime("%Y-%m-%d")
    cfg.backtest.end_date = end_ts.strftime("%Y-%m-%d")
    cfg.report.output_dir = str(OUTPUT_DIR)
    cfg.report.generate_plots = False

    initial_spot = _apply_initial_usd_balance(
        cfg,
        loaded.source_type,
        cfg.backtest.start_date,
        cfg.backtest.end_date,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    strategy = _load_strategy(cfg.strategy.name, cfg.strategy.params)
    engine = BacktestEngine(cfg, strategy)
    results = engine.run()
    metrics = compute_metrics(results)
    equity_path = plot_equity_curve(results, str(OUTPUT_DIR))

    payload = {
        "period": {
            "start_date": cfg.backtest.start_date,
            "end_date": cfg.backtest.end_date,
            "latest_available_data_end": coverage.end_date,
        },
        "source": {
            "config_type": loaded.source_type,
            "config_path": TRADER_CONFIG.as_posix(),
            "starting_capital_usd": DEFAULT_TRADER_INITIAL_USD,
            "initial_spot": initial_spot,
        },
        "strategy_params": cfg.strategy.params,
        "metrics": metrics,
        "artifacts": {
            "equity_curve_html": equity_path,
        },
    }

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 72)
    print("当前交易程序策略 - 最近一年回测")
    print(f"Period      : {cfg.backtest.start_date} -> {cfg.backtest.end_date}")
    print(f"Start USD   : {DEFAULT_TRADER_INITIAL_USD:,.2f}")
    print(f"Initial Spot: {0.0 if initial_spot is None else initial_spot:,.2f}")
    print(f"Total Return: {metrics['total_return'] * 100:.2f}%")
    print(f"Ann Return  : {metrics['annualized_return'] * 100:.2f}%")
    print(f"Max Drawdown: {metrics['max_drawdown'] * 100:.2f}%")
    print(f"Sharpe      : {metrics['sharpe_ratio']:.2f}")
    print(f"Trades      : {metrics['total_trades']}")
    print(f"Win Rate    : {metrics['win_rate'] * 100:.2f}%")
    print(f"Final Equity: {metrics['final_equity']:.2f}")
    print(f"Summary     : {summary_path.as_posix()}")
    print(f"Equity Curve: {Path(equity_path).as_posix()}")
    print("=" * 72)


if __name__ == "__main__":
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO")
    main()
