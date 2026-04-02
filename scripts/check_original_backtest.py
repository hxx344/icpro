"""Quick standalone check of the existing ic_0dte_8pct_direct.yaml backtest."""
from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.analytics.metrics import compute_metrics
from options_backtest.cli import _load_strategy

cfg = Config.from_yaml("configs/backtest/ic_0dte_8pct_direct.yaml")

print("=" * 60)
print("Config:", cfg.backtest.name)
print("Strategy:", cfg.strategy.name)
print("Params:")
for k, v in sorted(cfg.strategy.params.items()):
    print(f"  {k}: {v}")
print("=" * 60)

strategy = _load_strategy(cfg.strategy.name, cfg.strategy.params)
print(f"Strategy class: {type(strategy).__name__}")
print(f"  compound: {getattr(strategy, 'compound', 'N/A')}")
print(f"  roll_daily: {getattr(strategy, 'roll_daily', 'N/A')}")
print(f"  entry_hour: {getattr(strategy, 'entry_hour', 'N/A')}")
print(f"  hedge_otm_pct: {getattr(strategy, 'hedge_otm_pct', 'N/A')}")
print(f"  otm_pct: {getattr(strategy, 'otm_pct', 'N/A')}")
print("=" * 60)

engine = BacktestEngine(cfg, strategy)
results = engine.run()
metrics = compute_metrics(results)

print()
print("=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Total return (coin): {metrics['total_return']*100:.2f}%")
print(f"Annualized (coin):   {metrics['annualized_return']*100:.2f}%")
print(f"Max drawdown (coin): {metrics['max_drawdown']*100:.2f}%")
print(f"Sharpe:              {metrics['sharpe_ratio']:.2f}")
print(f"Final equity:        {metrics['final_equity']:.4f} ETH")
print(f"Total trades:        {metrics.get('total_trades', 0)}")
print(f"Win rate:            {metrics.get('win_rate', 0)*100:.1f}%")
print(f"USD return:          {metrics.get('total_return_usd', 0)*100:.2f}%")
print(f"USD annual:          {metrics.get('annualized_return_usd', 0)*100:.2f}%")
print(f"USD max DD:          {metrics.get('max_drawdown_usd', 0)*100:.2f}%")
print("=" * 60)
