"""Compare Short Strangle 0DTE: 1% vs 1.5% vs 2% OTM + Naked Call 12m."""
import sys, os
os.environ["LOGURU_LEVEL"] = "ERROR"
sys.path.insert(0, "src")

from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.strategy.short_strangle import ShortStrangleStrategy
from options_backtest.strategy.naked_call import NakedCallStrategy

configs = [
    ("1%OTM",  "configs/backtest/short_strangle_0dte_12m.yaml",       ShortStrangleStrategy),
    ("1.5%OTM","configs/backtest/short_strangle_0dte_12m_1.5pct.yaml",ShortStrangleStrategy),
    ("2%OTM",  "configs/backtest/short_strangle_0dte_12m_2pct.yaml",  ShortStrangleStrategy),
    ("NC_0DTE","configs/backtest/nc_eth_12m.yaml",                     NakedCallStrategy),
]

results = {}
for label, cfg_path, strat_cls in configs:
    cfg = Config.from_yaml(cfg_path)
    engine = BacktestEngine(cfg, strat_cls(params=cfg.strategy.params))
    r = engine.run()
    results[label] = r

# Build table
labels = list(results.keys())
eqs = {k: v["equity_history"] for k, v in results.items()}
ref = eqs[labels[0]]
step = max(1, len(ref) // 25)

out = []
hdr = "  %-12s" % "Date"
for l in labels:
    hdr += "  %9s" % l
hdr += "  %8s" % "ETH($)"
out.append("=" * len(hdr))
out.append(hdr)
out.append("=" * len(hdr))

for i in range(0, len(ref), step):
    t, _, _, _, ul = ref[i]
    line = "  %-12s" % str(t)[:10]
    for l in labels:
        eq = eqs[l]
        _, ev, _, _, _ = eq[min(i, len(eq)-1)]
        line += "  %9.2f" % ev
    line += "  %8.0f" % ul
    out.append(line)

# Final row
t, _, _, _, ul = ref[-1]
line = "  %-12s" % str(t)[:10]
for l in labels:
    eq = eqs[l]
    _, ev, _, _, _ = eq[-1]
    line += "  %9.2f" % ev
line += "  %8.0f  [END]" % ul
out.append(line)
out.append("=" * len(hdr))

# Metrics summary
out.append("")
out.append("METRICS SUMMARY")
out.append("-" * 70)
out.append("  %-12s %10s %10s %10s %10s %10s" % (
    "Metric", "1%OTM", "1.5%OTM", "2%OTM", "NC_0DTE", ""))
out.append("-" * 70)

from options_backtest.analytics.metrics import compute_metrics
for label in labels:
    m = compute_metrics(results[label])
    results[label]["_metrics"] = m

def fmt(v, pct=False):
    if pct:
        return "%.2f%%" % (v * 100 if abs(v) < 10 else v)
    return "%.2f" % v

metric_keys = [
    ("total_return",  "Return",       True,  100),
    ("max_drawdown",  "Max DD",       True,  100),
    ("sharpe_ratio",  "Sharpe",       False, 1),
    ("total_trades",  "Trades",       False, 1),
    ("win_rate",      "Win Rate",     True,  100),
    ("profit_factor", "PF",           False, 1),
    ("total_fees",    "Fees(coin)",   False, 1),
    ("final_equity",  "Final(coin)",  False, 1),
]

for key, name, is_pct, mult in metric_keys:
    line = "  %-12s" % name
    for l in labels:
        m = results[l]["_metrics"]
        v = m.get(key, 0) * mult
        if is_pct:
            line += " %9.1f%%" % v
        else:
            line += " %10.2f" % v
    out.append(line)

out.append("-" * 70)

with open("comparison_output.txt", "w") as f:
    f.write("\n".join(out))
print("DONE - saved to comparison_output.txt")
