"""Compare pure Short Strangle (no wings) vs Iron Condor across OTM levels.

Runs each OTM level twice: once naked (hedge_otm=0) and once with the
best-performing wing (2%) from the grid search.
"""
import sys, time
sys.path.insert(0, "src")

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.analytics.metrics import compute_metrics
from options_backtest.cli import _load_strategy

OTM_PCTS = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
WING_FOR_IC = 0.02  # best wing from grid
BASE_YAML = "configs/backtest/ic_0dte_8pct_direct.yaml"

rows: list[dict] = []
total = len(OTM_PCTS) * 2  # naked + IC
idx = 0

for otm in OTM_PCTS:
    for mode, wing in [("Naked", 0.0), ("IC w2%", WING_FOR_IC)]:
        idx += 1
        hedge_otm = otm + wing if wing > 0 else 0.0

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
            "mode": mode,
            "return_pct": m.get("total_return", 0) * 100,
            "max_dd_pct": m.get("max_drawdown", 0) * 100,
            "sharpe": m.get("sharpe_ratio", 0),
            "win_rate": m.get("win_rate", 0) * 100,
            "profit_factor": m.get("profit_factor", 0),
            "trades": m.get("total_trades", 0),
            "elapsed": elapsed,
        }
        rows.append(row)

        print(f"[{idx:2d}/{total}] OTM={otm*100:4.1f}% {mode:7s} → "
              f"Return={row['return_pct']:>8,.1f}%  DD={row['max_dd_pct']:.1f}%  "
              f"Sharpe={row['sharpe']:.2f}  WR={row['win_rate']:.1f}%  PF={row['profit_factor']:.2f}  "
              f"({elapsed:.1f}s)")

# ── Summary table ───────────────────────────────────────────
print("\n" + "=" * 110)
header = f"{'OTM':>6} | {'Mode':^7} | {'Return':>10} | {'MaxDD':>8} | {'Sharpe':>7} | {'WinRate':>8} | {'PF':>6} | {'Trades':>7}"
print(header)
print("-" * 110)

for otm in OTM_PCTS:
    for mode in ["Naked", "IC w2%"]:
        r = next(x for x in rows if x["otm"] == otm and x["mode"] == mode)
        print(f"{otm*100:5.1f}% | {mode:^7s} | {r['return_pct']:>+9.1f}% | {r['max_dd_pct']:>7.1f}% | "
              f"{r['sharpe']:>7.2f} | {r['win_rate']:>7.1f}% | {r['profit_factor']:>5.2f} | {r['trades']:>7}")
    print("-" * 110)

# ── Comparison: naked vs IC ─────────────────────────────────
print("\n纯双卖 vs IC 对比:")
for otm in OTM_PCTS:
    naked = next(x for x in rows if x["otm"] == otm and x["mode"] == "Naked")
    ic = next(x for x in rows if x["otm"] == otm and x["mode"] == "IC w2%")
    ret_diff = naked["return_pct"] - ic["return_pct"]
    dd_diff = naked["max_dd_pct"] - ic["max_dd_pct"]
    print(f"  OTM={otm*100:.1f}%: Naked {naked['return_pct']:+.1f}% vs IC {ic['return_pct']:+.1f}% "
          f"(收益差={ret_diff:+.1f}%, 回撤差={dd_diff:+.1f}%)")
