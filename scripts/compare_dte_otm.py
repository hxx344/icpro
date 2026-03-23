"""Grid comparison: DTE (0-7) × OTM (2%-10%) — 纯双卖策略 (no wings).

Runs a matrix of (dte, otm_pct) combinations for naked short strangle
and prints a compact summary heatmap with key metrics.
"""
import sys, time, itertools, json
sys.path.insert(0, "src")

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.analytics.metrics import compute_metrics
from options_backtest.cli import _load_strategy

# ── Parameter grid ──────────────────────────────────────────
DTE_TARGETS = [0, 1, 2, 3, 5, 7]  # target DTE (days)
OTM_PCTS = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]  # OTM %
BASE_YAML = "configs/backtest/ic_0dte_8pct_direct.yaml"

rows: list[dict] = []
total = len(DTE_TARGETS) * len(OTM_PCTS)
idx = 0

for dte, otm in itertools.product(DTE_TARGETS, OTM_PCTS):
    idx += 1

    cfg = Config.from_yaml(BASE_YAML)
    cfg.strategy.params["otm_pct"] = otm
    cfg.strategy.params["hedge_otm_pct"] = 0.0  # 纯双卖，无对冲

    # DTE range: [dte, dte+1.5] to target specific expiry window
    if dte == 0:
        cfg.strategy.params["min_days_to_expiry"] = 0.0
        cfg.strategy.params["max_days_to_expiry"] = 1.5
    else:
        cfg.strategy.params["min_days_to_expiry"] = max(dte - 0.5, 0)
        cfg.strategy.params["max_days_to_expiry"] = dte + 1.5

    cfg.strategy.params["roll_daily"] = (dte <= 1)   # 0DTE/1DTE 每天换仓; DTE≥2 持有到结算
    cfg.strategy.params["compound"] = True
    cfg.strategy.params["take_profit_pct"] = 9999
    cfg.strategy.params["stop_loss_pct"] = 9999
    cfg.strategy.params["max_positions"] = 1

    label = f"DTE={dte} OTM={otm*100:.0f}%"
    cfg.backtest.name = f"ETH 双卖 {label}"

    s = _load_strategy(cfg.strategy.name, cfg.strategy.params)
    e = BacktestEngine(cfg, s)

    t0 = time.perf_counter()
    r = e.run()
    elapsed = time.perf_counter() - t0

    m = compute_metrics(r)

    row = {
        "dte": dte,
        "otm": otm,
        "return_pct": m.get("total_return", 0) * 100,
        "ann_return_pct": m.get("annualized_return", 0) * 100,
        "max_dd_pct": m.get("max_drawdown", 0) * 100,
        "sharpe": m.get("sharpe_ratio", 0),
        "win_rate": m.get("win_rate", 0) * 100,
        "profit_factor": m.get("profit_factor", 0),
        "trades": m.get("total_trades", 0),
        "elapsed": elapsed,
    }
    rows.append(row)

    print(f"[{idx:2d}/{total}] {label:18s} → "
          f"Return={row['return_pct']:>+8.1f}%  DD={row['max_dd_pct']:.1f}%  "
          f"Sharpe={row['sharpe']:.2f}  WR={row['win_rate']:.1f}%  "
          f"({elapsed:.1f}s)")

# ── Heatmap tables ──────────────────────────────────────────
def print_heatmap(metric_name: str, metric_key: str, fmt: str):
    print(f"\n  [{metric_name}]")
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
                val = fmt.format(cell[metric_key])
                line += f" | {val:>7}"
            else:
                line += f" | {'N/A':>7}"
        print(line)

print("\n" + "=" * 100)
print("  ETH 纯双卖策略: DTE × OTM Grid (无对冲)")
print("=" * 100)

print_heatmap("收益率 Return %", "return_pct", "{:+.1f}")
print_heatmap("最大回撤 MaxDD %", "max_dd_pct", "{:.1f}")
print_heatmap("夏普比率 Sharpe", "sharpe", "{:.2f}")
print_heatmap("胜率 WinRate %", "win_rate", "{:.1f}")
print_heatmap("盈亏比 PF", "profit_factor", "{:.2f}")
print_heatmap("交易次数", "trades", "{:.0f}")

# ── Best combinations ──────────────────────────────────────
print("\n" + "=" * 100)
valid = [r for r in rows if r["trades"] > 0]
if valid:
    best_ret = max(valid, key=lambda r: r["return_pct"])
    print(f"  最高收益: DTE={best_ret['dte']} OTM={best_ret['otm']*100:.0f}% → "
          f"Return={best_ret['return_pct']:+.1f}%  DD={best_ret['max_dd_pct']:.1f}%  Sharpe={best_ret['sharpe']:.2f}")

    best_sh = max(valid, key=lambda r: r["sharpe"])
    print(f"  最高Sharpe: DTE={best_sh['dte']} OTM={best_sh['otm']*100:.0f}% → "
          f"Sharpe={best_sh['sharpe']:.2f}  Return={best_sh['return_pct']:+.1f}%  DD={best_sh['max_dd_pct']:.1f}%")

    pos_rows = [r for r in valid if r["return_pct"] > 0 and r["max_dd_pct"] < 0]
    if pos_rows:
        best_ratio = max(pos_rows, key=lambda r: r["return_pct"] / abs(r["max_dd_pct"]))
        print(f"  最佳收益/回撤: DTE={best_ratio['dte']} OTM={best_ratio['otm']*100:.0f}% → "
              f"Return={best_ratio['return_pct']:+.1f}%  DD={best_ratio['max_dd_pct']:.1f}%  "
              f"Ratio={best_ratio['return_pct']/abs(best_ratio['max_dd_pct']):.2f}")

# ── Save JSON ──────────────────────────────────────────────
import os
os.makedirs("reports", exist_ok=True)
with open("reports/dte_otm_grid.json", "w") as f:
    json.dump(rows, f, indent=2, default=str)
print(f"\n结果已保存: reports/dte_otm_grid.json")

total_time = sum(r["elapsed"] for r in rows)
print(f"总耗时: {total_time:.0f}s ({total_time/60:.1f}min)")
