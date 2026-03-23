"""BTC DTE x OTM Grid - Short Strangle."""
import sys, time, itertools, json
sys.path.insert(0, "src")
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")
from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.analytics.metrics import compute_metrics
from options_backtest.cli import _load_strategy

DTE_TARGETS = [0, 1, 2, 3, 5, 7]
OTM_PCTS = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
BASE_YAML = "configs/backtest/ic_0dte_8pct_btc.yaml"
rows = []
total = len(DTE_TARGETS) * len(OTM_PCTS)
idx = 0
for dte, otm in itertools.product(DTE_TARGETS, OTM_PCTS):
    idx += 1
    cfg = Config.from_yaml(BASE_YAML)
    cfg.strategy.params["otm_pct"] = otm
    cfg.strategy.params["hedge_otm_pct"] = 0.0
    if dte == 0:
        cfg.strategy.params["min_days_to_expiry"] = 0.0
        cfg.strategy.params["max_days_to_expiry"] = 1.5
    else:
        cfg.strategy.params["min_days_to_expiry"] = max(dte - 0.5, 0)
        cfg.strategy.params["max_days_to_expiry"] = dte + 1.5
    cfg.strategy.params["roll_daily"] = (dte <= 1)
    cfg.strategy.params["compound"] = True
    cfg.strategy.params["take_profit_pct"] = 9999
    cfg.strategy.params["stop_loss_pct"] = 9999
    cfg.strategy.params["max_positions"] = 1
    label = f"DTE={dte} OTM={otm*100:.0f}%"
    cfg.backtest.name = f"BTC strangle {label}"
    s = _load_strategy(cfg.strategy.name, cfg.strategy.params)
    e = BacktestEngine(cfg, s)
    t0 = time.perf_counter()
    r = e.run()
    elapsed = time.perf_counter() - t0
    m = compute_metrics(r)
    mkt = e._quote_source_market
    syn = e._quote_source_synth
    tot = mkt + syn
    row = {"dte": dte, "otm": otm, "return_pct": m.get("total_return", 0)*100, "max_dd_pct": m.get("max_drawdown", 0)*100, "sharpe": m.get("sharpe_ratio", 0), "win_rate": m.get("win_rate", 0)*100, "profit_factor": m.get("profit_factor", 0), "trades": m.get("total_trades", 0), "mkt_pct": mkt/tot*100 if tot else 0, "elapsed": elapsed}
    rows.append(row)
    print(f"[{idx:2d}/{total}] {label:18s} -> Return={row['return_pct']:>+8.1f}%  DD={row['max_dd_pct']:.1f}%  Sharpe={row['sharpe']:.2f}  WR={row['win_rate']:.1f}%  Mkt={row['mkt_pct']:.0f}%  ({elapsed:.1f}s)")

def print_heatmap(name, key, fmt):
    print(f"\n  [{name}]")
    header = f"{'DTE':>8}"
    for otm in OTM_PCTS:
        header += f" | OTM{otm*100:3.0f}%"
    print(header)
    print("-" * (10 + 10 * len(OTM_PCTS)))
    for dte in DTE_TARGETS:
        line = f"  DTE={dte:2d}"
        for otm in OTM_PCTS:
            cell = next((r for r in rows if r["dte"] == dte and r["otm"] == otm), None)
            if cell:
                line += f" | {fmt.format(cell[key]):>7}"
            else:
                line += f" | {'N/A':>7}"
        print(line)

print("\n" + "=" * 100)
print("  BTC Short Strangle: DTE x OTM Grid")
print("=" * 100)
print_heatmap("Return %", "return_pct", "{:+.1f}")
print_heatmap("MaxDD %", "max_dd_pct", "{:.1f}")
print_heatmap("Sharpe", "sharpe", "{:.2f}")
print_heatmap("WinRate %", "win_rate", "{:.1f}")
print_heatmap("PF", "profit_factor", "{:.2f}")
print_heatmap("Trades", "trades", "{:.0f}")
print_heatmap("Market Data %", "mkt_pct", "{:.0f}")

print("\n" + "=" * 100)
valid = [r for r in rows if r["trades"] > 0]
if valid:
    best_ret = max(valid, key=lambda r: r["return_pct"])
    best_sh = max(valid, key=lambda r: r["sharpe"])
    print(f"  Best Return: DTE={best_ret['dte']} OTM={best_ret['otm']*100:.0f}% -> Return={best_ret['return_pct']:+.1f}%  DD={best_ret['max_dd_pct']:.1f}%  Sharpe={best_ret['sharpe']:.2f}")
    print(f"  Best Sharpe: DTE={best_sh['dte']} OTM={best_sh['otm']*100:.0f}% -> Sharpe={best_sh['sharpe']:.2f}  Return={best_sh['return_pct']:+.1f}%  DD={best_sh['max_dd_pct']:.1f}%")

import os
os.makedirs("reports", exist_ok=True)
with open("reports/btc_dte_otm_grid.json", "w") as f:
    json.dump(rows, f, indent=2, default=str)
print(f"\nSaved: reports/btc_dte_otm_grid.json")
total_time = sum(r["elapsed"] for r in rows)
print(f"Total: {total_time:.0f}s ({total_time/60:.1f}min)")
