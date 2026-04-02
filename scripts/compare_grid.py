"""Grid comparison: OTM width × Wing width — USD margin mode.

Runs a matrix of (otm_pct, wing_pct) combinations and prints a
compact summary table with key metrics for each cell.
"""
import sys, time, itertools

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.analytics.metrics import compute_metrics
from options_backtest.cli import _load_strategy

# ── Parameter grid ──────────────────────────────────────────
OTM_PCTS  = [0.0, 0.005, 0.01, 0.015, 0.02]       # short strike OTM %
WING_PCTS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.15]   # wing width (hedge distance from short)
BASE_YAML = "configs/backtest/ic_0dte_8pct_direct.yaml"

rows: list[dict] = []
total = len(OTM_PCTS) * len(WING_PCTS)
idx = 0

for otm, wing in itertools.product(OTM_PCTS, WING_PCTS):
    idx += 1
    hedge_otm = otm + wing  # hedge strike = short OTM + wing width

    cfg = Config.from_yaml(BASE_YAML)
    cfg.strategy.params["otm_pct"] = otm
    cfg.strategy.params["hedge_otm_pct"] = hedge_otm

    s = _load_strategy(cfg.strategy.name, cfg.strategy.params)
    e = BacktestEngine(cfg, s)

    t0 = time.perf_counter()
    r = e.run()
    elapsed = time.perf_counter() - t0

    m = compute_metrics(r)

    row = {
        "otm": otm,
        "wing": wing,
        "hedge_otm": hedge_otm,
        "return_pct": m.get("total_return", 0) * 100,
        "ann_return_pct": m.get("annualized_return", 0) * 100,
        "max_dd_pct": m.get("max_drawdown", 0) * 100,
        "sharpe": m.get("sharpe_ratio", 0),
        "win_rate": m.get("win_rate", 0) * 100,
        "profit_factor": m.get("profit_factor", 0),
        "trades": m.get("total_trades", 0),
        "fees": m.get("total_fees", 0),
        "elapsed": elapsed,
    }
    rows.append(row)

    print(f"[{idx:2d}/{total}] OTM={otm*100:4.1f}% Wing={wing*100:4.0f}% → "
          f"Return={row['return_pct']:>12,.0f}%  DD={row['max_dd_pct']:.1f}%  "
          f"Sharpe={row['sharpe']:.2f}  ({elapsed:.1f}s)")

# ── Summary table ───────────────────────────────────────────
print("\n" + "=" * 120)
print(f"{'':>10}", end="")
for wing in WING_PCTS:
    print(f" | Wing {wing*100:.0f}%{'':<6}", end="")
print()
print("-" * 120)

for metric_name, metric_key, fmt in [
    ("Return %", "return_pct", "{:>10,.0f}%"),
    ("MaxDD %",  "max_dd_pct", "{:>10.1f}%"),
    ("Sharpe",   "sharpe",     "{:>11.2f}"),
    ("WinRate%", "win_rate",   "{:>10.1f}%"),
    ("PF",       "profit_factor", "{:>11.2f}"),
]:
    for otm in OTM_PCTS:
        label = f"OTM {otm*100:.1f}%"
        line = f"{label:>10}"
        for wing in WING_PCTS:
            cell = next((r for r in rows if r["otm"] == otm and r["wing"] == wing), None)
            if cell:
                val = fmt.format(cell[metric_key])
                line += f" | {val:<12}"
            else:
                line += f" | {'N/A':<12}"
        if otm == OTM_PCTS[0]:
            print(f"\n  [{metric_name}]")
        print(line)

# ── Best combinations ──────────────────────────────────────
print("\n" + "=" * 120)
print("  Best by Return:  ", end="")
best_ret = max(rows, key=lambda r: r["return_pct"])
print(f"OTM={best_ret['otm']*100:.1f}% Wing={best_ret['wing']*100:.0f}% → {best_ret['return_pct']:,.0f}%")

print("  Best by Sharpe:  ", end="")
best_sh = max(rows, key=lambda r: r["sharpe"])
print(f"OTM={best_sh['otm']*100:.1f}% Wing={best_sh['wing']*100:.0f}% → Sharpe={best_sh['sharpe']:.2f}")

print("  Lowest MaxDD:    ", end="")
best_dd = max(rows, key=lambda r: r["max_dd_pct"])  # least negative
print(f"OTM={best_dd['otm']*100:.1f}% Wing={best_dd['wing']*100:.0f}% → DD={best_dd['max_dd_pct']:.1f}%")

print("  Best Return/DD:  ", end="")
best_ratio = max(rows, key=lambda r: r["return_pct"] / abs(r["max_dd_pct"]) if r["max_dd_pct"] != 0 else 0)
print(f"OTM={best_ratio['otm']*100:.1f}% Wing={best_ratio['wing']*100:.0f}% → "
      f"Return={best_ratio['return_pct']:,.0f}% / DD={best_ratio['max_dd_pct']:.1f}%")
