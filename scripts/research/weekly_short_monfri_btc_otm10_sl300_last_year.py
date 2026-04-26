"""Run trailing-one-year backtest for BTC weekly Mon->Fri 10% OTM short strangle with SL300.

Uses the latest available end date in the current dataset/config and builds a 1-year window.
Outputs metrics JSON and equity curve HTML.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from options_backtest.analytics.metrics import compute_metrics
from options_backtest.analytics.plotting import plot_equity_curve
from options_backtest.cli import _load_strategy
from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine

BASE_CONFIG = Path("configs/backtest/weekly_short_monfri_btc_otm10.yaml")
OUTPUT_DIR = Path("reports/weekly_short_monfri_btc_otm10_sl300_last_year")


def main() -> None:
    cfg = Config.from_yaml(BASE_CONFIG)

    end_ts = pd.Timestamp(cfg.backtest.end_date, tz="UTC")
    start_ts = end_ts - pd.Timedelta(days=364)

    cfg.backtest.name = "BTC Weekly Short Strangle (last available year, 10% OTM, SL300)"
    cfg.backtest.start_date = start_ts.strftime("%Y-%m-%d")
    cfg.backtest.end_date = end_ts.strftime("%Y-%m-%d")
    cfg.strategy.params["otm_pct"] = 0.10
    cfg.strategy.params["stop_loss_pct"] = 300
    cfg.strategy.params["take_profit_pct"] = 9999
    cfg.strategy.params["compound"] = True
    cfg.strategy.params["max_positions"] = 1
    cfg.strategy.params["hedge_otm_pct"] = 0.0
    cfg.strategy.params["skip_weekends"] = True
    cfg.report.output_dir = str(OUTPUT_DIR)
    cfg.report.generate_plots = False

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
            "source_note": "latest available year based on config/data end date",
        },
        "strategy": {
            "config": "weekly_short_monfri_btc_otm10.yaml",
            "otm_pct": 0.10,
            "stop_loss_pct": 300,
            "entry_weekdays": [0],
            "entry_hour": 8,
            "compound": True,
        },
        "metrics": metrics,
        "artifacts": {
            "equity_curve_html": equity_path,
        },
    }

    metrics_path = OUTPUT_DIR / "summary.json"
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 72)
    print("OTM10% + SL300 最近一年回测")
    print(f"Period      : {cfg.backtest.start_date} -> {cfg.backtest.end_date}")
    print(f"Total Return: {metrics['total_return'] * 100:.2f}%")
    print(f"Ann Return  : {metrics['annualized_return'] * 100:.2f}%")
    print(f"Max Drawdown: {metrics['max_drawdown'] * 100:.2f}%")
    print(f"Sharpe      : {metrics['sharpe_ratio']:.2f}")
    print(f"Trades      : {metrics['total_trades']}")
    print(f"Win Rate    : {metrics['win_rate'] * 100:.2f}%")
    print(f"Final Equity: {metrics['final_equity']:.2f}")
    print(f"Summary     : {metrics_path.as_posix()}")
    print(f"Equity Curve: {Path(equity_path).as_posix()}")
    print("=" * 72)


if __name__ == "__main__":
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO")
    main()
