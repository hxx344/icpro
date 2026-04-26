"""Comprehensive strategy combo backtest: BTC & ETH 0DTE ATM."""
import sys, time, json
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")
from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.analytics.metrics import compute_metrics
from options_backtest.cli import _load_strategy

COMBOS = [
    # label, asset, yaml, overrides
    # --- Baselines ---
    ("BTC Naked",          "btc", {}),
    ("ETH Naked",          "eth", {}),
    # --- TP/SL only ---
    ("BTC TP50/SL100",     "btc", {"take_profit_pct": 50,  "stop_loss_pct": 100}),
    ("BTC TP50/SL150",     "btc", {"take_profit_pct": 50,  "stop_loss_pct": 150}),
    ("BTC TP50/SL200",     "btc", {"take_profit_pct": 50,  "stop_loss_pct": 200}),
    ("BTC TP80/SL150",     "btc", {"take_profit_pct": 80,  "stop_loss_pct": 150}),
    ("BTC TP80/SL200",     "btc", {"take_profit_pct": 80,  "stop_loss_pct": 200}),
    ("BTC TP30/SL100",     "btc", {"take_profit_pct": 30,  "stop_loss_pct": 100}),
    ("ETH TP50/SL100",     "eth", {"take_profit_pct": 50,  "stop_loss_pct": 100}),
    ("ETH TP50/SL150",     "eth", {"take_profit_pct": 50,  "stop_loss_pct": 150}),
    ("ETH TP50/SL200",     "eth", {"take_profit_pct": 50,  "stop_loss_pct": 200}),
    ("ETH TP80/SL150",     "eth", {"take_profit_pct": 80,  "stop_loss_pct": 150}),
    ("ETH TP80/SL200",     "eth", {"take_profit_pct": 80,  "stop_loss_pct": 200}),
    ("ETH TP30/SL100",     "eth", {"take_profit_pct": 30,  "stop_loss_pct": 100}),
    # --- Wings only (best: 4%) ---
    ("BTC Wing4%",         "btc", {"hedge_otm_pct": 0.04}),
    ("ETH Wing4%",         "eth", {"hedge_otm_pct": 0.04}),
    # --- TP/SL + Wings ---
    ("BTC TP50/SL150+W4%", "btc", {"take_profit_pct": 50, "stop_loss_pct": 150, "hedge_otm_pct": 0.04}),
    ("BTC TP80/SL200+W4%", "btc", {"take_profit_pct": 80, "stop_loss_pct": 200, "hedge_otm_pct": 0.04}),
    ("ETH TP50/SL150+W4%", "eth", {"take_profit_pct": 50, "stop_loss_pct": 150, "hedge_otm_pct": 0.04}),
    ("ETH TP80/SL200+W4%", "eth", {"take_profit_pct": 80, "stop_loss_pct": 200, "hedge_otm_pct": 0.04}),
    # --- DD scaling (progressive) ---
    ("BTC DD scale",       "btc", {"dd_start": 0.05, "dd_full": 0.20, "dd_min_scale": 0.2}),
    ("ETH DD scale",       "eth", {"dd_start": 0.05, "dd_full": 0.20, "dd_min_scale": 0.2}),
    # --- DD scale + TP/SL ---
    ("BTC TP50/SL150+DD",  "btc", {"take_profit_pct": 50, "stop_loss_pct": 150, "dd_start": 0.05, "dd_full": 0.20, "dd_min_scale": 0.2}),
    ("ETH TP50/SL150+DD",  "eth", {"take_profit_pct": 50, "stop_loss_pct": 150, "dd_start": 0.05, "dd_full": 0.20, "dd_min_scale": 0.2}),
    # --- Cooldown after loss ---
    ("BTC Cooldown1d",     "btc", {"cooldown_days": 1}),
    ("ETH Cooldown1d",     "eth", {"cooldown_days": 1}),
    # --- Vol filter ---
    ("BTC VolFilter",      "btc", {"vol_lookback": 24, "vol_threshold": 80.0}),
    ("ETH VolFilter",      "eth", {"vol_lookback": 24, "vol_threshold": 80.0}),
    # --- Full combo: TP/SL + Wings + DD scale ---
    ("BTC FULL",           "btc", {"take_profit_pct": 50, "stop_loss_pct": 150, "hedge_otm_pct": 0.04, "dd_start": 0.05, "dd_full": 0.20, "dd_min_scale": 0.2}),
    ("ETH FULL",           "eth", {"take_profit_pct": 50, "stop_loss_pct": 150, "hedge_otm_pct": 0.04, "dd_start": 0.05, "dd_full": 0.20, "dd_min_scale": 0.2}),
    # --- TP/SL + Vol filter ---
    ("BTC TP50/SL150+Vol", "btc", {"take_profit_pct": 50, "stop_loss_pct": 150, "vol_lookback": 24, "vol_threshold": 80.0}),
    ("ETH TP50/SL150+Vol", "eth", {"take_profit_pct": 50, "stop_loss_pct": 150, "vol_lookback": 24, "vol_threshold": 80.0}),
]

YAMLS = {
    "btc": "configs/backtest/ic_0dte_8pct_btc.yaml",
    "eth": "configs/backtest/ic_0dte_8pct_direct.yaml",
}

BASE_PARAMS = {
    "otm_pct": 0.001,
    "hedge_otm_pct": 0.0,
    "min_days_to_expiry": 0.0,
    "max_days_to_expiry": 1.5,
    "roll_daily": True,
    "compound": True,
    "take_profit_pct": 9999,
    "stop_loss_pct": 9999,
    "max_positions": 1,
}

print(f"{'Strategy':<24s} {'Return':>8s} {'MaxDD':>8s} {'Sharpe':>7s} {'WR':>6s} {'PF':>6s} {'Trades':>6s} {'Mkt%':>5s} {'Time':>5s}")
print("=" * 85)

results = []
for label, asset, overrides in COMBOS:
    yaml_path = YAMLS[asset]
    cfg = Config.from_yaml(yaml_path)
    for k, v in BASE_PARAMS.items():
        cfg.strategy.params[k] = v
    for k, v in overrides.items():
        cfg.strategy.params[k] = v
    cfg.backtest.name = label
    s = _load_strategy(cfg.strategy.name, cfg.strategy.params)
    e = BacktestEngine(cfg, s)
    t0 = time.perf_counter()
    r = e.run()
    elapsed = time.perf_counter() - t0
    m = compute_metrics(r)
    mkt = e._quote_source_market
    syn = e._quote_source_synth
    tot = mkt + syn
    row = {
        "label": label, "asset": asset,
        "return_pct": m.get("total_return",0)*100,
        "max_dd_pct": m.get("max_drawdown",0)*100,
        "sharpe": m.get("sharpe_ratio",0),
        "win_rate": m.get("win_rate",0)*100,
        "profit_factor": m.get("profit_factor",0),
        "trades": m.get("total_trades",0),
        "mkt_pct": mkt/tot*100 if tot else 0,
        "elapsed": elapsed,
    }
    results.append(row)
    print(f"{label:<24s} {row['return_pct']:>+7.1f}% {row['max_dd_pct']:>7.1f}% {row['sharpe']:>7.2f} {row['win_rate']:>5.1f}% {row['profit_factor']:>6.2f} {row['trades']:>5.0f} {row['mkt_pct']:>5.0f}% {elapsed:>4.1f}s")

# Save
with open("reports/strategy_combo_grid.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to reports/strategy_combo_grid.json")
