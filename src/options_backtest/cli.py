"""CLI entry point for the options backtest system."""

from __future__ import annotations

from pathlib import Path

import click
from loguru import logger


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Deribit Options Strategy Backtesting System."""


# -----------------------------------------------------------------------
# Data fetching
# -----------------------------------------------------------------------

@cli.command()
@click.option("--underlying", "-u", default="BTC", help="Underlying asset (BTC/ETH)")
@click.option("--start", "-s", required=True, help="Start date YYYY-MM-DD")
@click.option("--end", "-e", required=True, help="End date YYYY-MM-DD")
@click.option("--resolution", "-r", default="1D", help="OHLCV resolution (1D, 60, 120, etc.)")
@click.option("--data-dir", "-d", default="data", help="Local data directory")
@click.option("--no-option-ohlcv", is_flag=True, help="Skip individual option OHLCV (faster)")
def fetch(underlying: str, start: str, end: str, resolution: str,
          data_dir: str, no_option_ohlcv: bool):
    """Fetch historical data from Deribit API."""
    from options_backtest.data.fetcher import run_fetch

    logger.info(f"Fetching {underlying} data: {start} �?{end}")
    run_fetch(
        underlying=underlying.upper(),
        start_date=start,
        end_date=end,
        resolution=resolution,
        data_dir=data_dir,
        fetch_option_ohlcv=not no_option_ohlcv,
    )
    logger.info("Data fetch complete.")


@cli.command("fetch-okx")
@click.option("--underlying", "-u", default="ETH", help="Underlying asset (BTC/ETH)")
@click.option("--start", "-s", required=True, help="Start date YYYY-MM-DD")
@click.option("--end", "-e", required=True, help="End date YYYY-MM-DD")
@click.option("--resolution", "-r", default="60", help="OHLCV resolution (60, 1D, etc.)")
@click.option("--data-dir", "-d", default="data", help="Local data directory")
@click.option("--no-option-ohlcv", is_flag=True, help="Skip individual option OHLCV (faster)")
def fetch_okx(underlying: str, start: str, end: str, resolution: str,
              data_dir: str, no_option_ohlcv: bool):
    """Fetch historical options data from OKX API (supports ~3 months of expired contracts)."""
    from options_backtest.data.okx_fetcher import run_okx_fetch

    logger.info(f"OKX: Fetching {underlying} data: {start} �?{end}")
    run_okx_fetch(
        underlying=underlying.upper(),
        start_date=start,
        end_date=end,
        resolution=resolution,
        data_dir=data_dir,
        fetch_option_ohlcv=not no_option_ohlcv,
    )
    logger.info("OKX data fetch complete.")


# -----------------------------------------------------------------------
# Run backtest
# -----------------------------------------------------------------------

@cli.command()
@click.option("--config", "-c", default="configs/backtest/default.yaml", help="Config YAML file")
def run(config: str):
    """Run a backtest with the given configuration."""
    from options_backtest.analytics.metrics import compute_metrics, print_metrics
    from options_backtest.analytics.plotting import generate_all_plots
    from options_backtest.config import Config
    from options_backtest.engine.backtest import BacktestEngine

    cfg_path = Path(config)
    if not cfg_path.exists():
        logger.error(f"Config file not found: {cfg_path}")
        raise SystemExit(1)

    cfg = Config.from_yaml(cfg_path)

    # Resolve strategy
    strategy = _load_strategy(cfg.strategy.name, cfg.strategy.params)

    # Run engine
    engine = BacktestEngine(cfg, strategy)
    results = engine.run()

    # Metrics
    metrics = compute_metrics(results)
    print_metrics(metrics)

    # Plots
    if cfg.report.generate_plots:
        output_dir = cfg.report.output_dir
        paths = generate_all_plots(results, output_dir)
        for p in paths:
            logger.info(f"Chart saved: {p}")


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

STRATEGY_MAP = {
    "longcall": "options_backtest.strategy.long_call.LongCallStrategy",
    "shortstrangle": "options_backtest.strategy.short_strangle.ShortStrangleStrategy",
    "ironcondor": "options_backtest.strategy.iron_condor.IronCondorStrategy",
    "coveredcall": "options_backtest.strategy.covered_call.CoveredCallStrategy",
    "callspreadcc": "options_backtest.strategy.call_spread_cc.CallSpreadCCStrategy",
    "dualinvest": "options_backtest.strategy.dual_invest.DualInvestStrategy",
    "shortput": "options_backtest.strategy.short_put.ShortPutStrategy",
}

# Register new strategies here
STRATEGY_MAP.update({
    "nakedcall": "options_backtest.strategy.naked_call.NakedCallStrategy",
    "intradaymovefade0dte": "options_backtest.strategy.intraday_move_fade_0dte.IntradayMoveFade0DTEStrategy",
    "weekendvol": "options_backtest.strategy.weekend_vol.WeekendVolStrategy",
    "weekendvolbacktest": "options_backtest.strategy.weekend_vol_backtest.WeekendVolBacktestStrategy",
    "longstrangle": "options_backtest.strategy.long_strangle.LongStrangleStrategy",
})


def _load_strategy(name: str, params: dict):
    """Instantiate a strategy by name."""
    key = name.lower().replace("_", "").replace("-", "")
    class_path = STRATEGY_MAP.get(key)
    if class_path is None:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {list(STRATEGY_MAP.keys())}"
        )

    module_path, class_name = class_path.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(params=params)


# -----------------------------------------------------------------------

if __name__ == "__main__":
    cli()
