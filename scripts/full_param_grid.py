"""Comprehensive parameter grid: Underlying × DTE × OTM × Wings × StopLoss × TakeProfit.

Runs a full matrix of parameter combinations for BTC short strangle /
iron condor strategies across 3 years of data, and outputs a JSON + summary.

Dimensions:
  - Underlying: BTC
  - DTE: 0
  - OTM%: 0 (ATM)
  - Wings (hedge_otm_pct): 0 (naked), 2%, 3%, 4%
  - StopLoss%: None, 60%, 80%, 100%, 120%, 150%
  - TakeProfit%: None, 30%, 50%, 70%, 90%
"""
import sys, os, time, itertools, json
sys.path.insert(0, "src")

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.analytics.metrics import compute_metrics
from options_backtest.cli import _load_strategy

# ── Parameter grid ──────────────────────────────────────────
UNDERLYINGS = ["BTC"]
DTE_TARGETS = [0]
OTM_PCTS = [0.0]                                          # ATM only
WING_PCTS = [0.0, 0.02, 0.03, 0.04]                      # naked + 3 wing widths
SL_PCTS = [9999, 0.6, 0.8, 1.0, 1.2, 1.5]                # no SL + 5 SL levels
TP_PCTS = [9999, 30, 50, 70, 90]                           # no TP + 4 TP levels (% premium captured)

# Data date range (matching BTC data: 2023-01-01 ~ 2026-01-31)
START_DATE = "2023-01-01"
END_DATE = "2026-01-31"

BASE_YAML = "configs/backtest/ic_0dte_8pct_direct.yaml"

# ── Build all combos ─────────────────────────────────────────
combos = list(itertools.product(UNDERLYINGS, DTE_TARGETS, OTM_PCTS, WING_PCTS, SL_PCTS, TP_PCTS))
total = len(combos)
print(f"Total combinations: {total}")
print(f"Date range: {START_DATE} ~ {END_DATE}")
print(f"Underlyings: {UNDERLYINGS}")
print(f"DTE: {DTE_TARGETS}")
print(f"OTM%: {[f'{x*100:.0f}%' for x in OTM_PCTS]}")
print(f"Wings: {[f'{x*100:.0f}%' for x in WING_PCTS]}")
print(f"StopLoss: {[f'{x*100:.0f}%' if x < 9999 else 'None' for x in SL_PCTS]}")
print(f"TakeProfit: {[f'{x:.0f}%' if x < 9999 else 'None' for x in TP_PCTS]}")
print("=" * 120)

rows: list[dict] = []
idx = 0
t_global = time.perf_counter()

for underlying, dte, otm, wing, sl, tp in combos:
    idx += 1

    # Skip invalid: wings must be wider than OTM (otherwise long strike inside short strike)
    if wing > 0 and wing <= otm:
        continue

    # Skip invalid: ATM + 0 wing is fine (naked ATM strangle), but ATM + small wing is weird
    # We allow all valid combinations

    cfg = Config.from_yaml(BASE_YAML)

    # Override underlying & dates
    cfg.backtest.underlying = underlying
    cfg.backtest.start_date = START_DATE
    cfg.backtest.end_date = END_DATE

    # Strategy params
    cfg.strategy.params["otm_pct"] = otm
    cfg.strategy.params["hedge_otm_pct"] = wing  # 0 = no wings (strangle), >0 = iron condor

    # DTE range
    if dte == 0:
        cfg.strategy.params["min_days_to_expiry"] = 0.0
        cfg.strategy.params["max_days_to_expiry"] = 1.5
    else:
        cfg.strategy.params["min_days_to_expiry"] = max(dte - 0.5, 0)
        cfg.strategy.params["max_days_to_expiry"] = dte + 1.5

    cfg.strategy.params["roll_daily"] = (dte <= 1)  # 0DTE/1DTE daily roll; DTE>=2 hold to settlement
    cfg.strategy.params["compound"] = True
    cfg.strategy.params["take_profit_pct"] = tp
    cfg.strategy.params["stop_loss_pct"] = sl
    cfg.strategy.params["max_positions"] = 1

    # Label
    wing_label = f"W{wing*100:.0f}%" if wing > 0 else "naked"
    sl_label = f"SL{sl*100:.0f}%" if sl < 9999 else "noSL"
    tp_label = f"TP{tp:.0f}%" if tp < 9999 else "noTP"
    otm_label = f"ATM" if otm == 0 else f"OTM{otm*100:.0f}%"
    label = f"{underlying} DTE={dte} {otm_label} {wing_label} {sl_label} {tp_label}"
    cfg.backtest.name = label

    s = _load_strategy(cfg.strategy.name, cfg.strategy.params)
    e = BacktestEngine(cfg, s)

    t0 = time.perf_counter()
    try:
        r = e.run()
        m = compute_metrics(r)
        elapsed = time.perf_counter() - t0

        row = {
            "underlying": underlying,
            "dte": dte,
            "otm": otm,
            "wing": wing,
            "sl": sl,
            "tp": tp,
            "return_pct": m.get("total_return", 0) * 100,
            "ann_return_pct": m.get("annualized_return", 0) * 100,
            "max_dd_pct": m.get("max_drawdown", 0) * 100,
            "sharpe": m.get("sharpe_ratio", 0),
            "win_rate": m.get("win_rate", 0) * 100,
            "profit_factor": m.get("profit_factor", 0),
            "trades": m.get("total_trades", 0),
            "market_data_pct": 0,
            "elapsed": elapsed,
        }
        # Extract market data % if available
        if hasattr(e, '_quote_source_market') and hasattr(e, '_quote_source_synth'):
            total_quotes = e._quote_source_market + e._quote_source_synth
            if total_quotes > 0:
                row["market_data_pct"] = e._quote_source_market / total_quotes * 100
    except Exception as ex:
        elapsed = time.perf_counter() - t0
        row = {
            "underlying": underlying,
            "dte": dte,
            "otm": otm,
            "wing": wing,
            "sl": sl,
            "tp": tp,
            "return_pct": 0,
            "ann_return_pct": 0,
            "max_dd_pct": 0,
            "sharpe": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "trades": 0,
            "market_data_pct": 0,
            "elapsed": elapsed,
            "error": str(ex),
        }

    rows.append(row)

    # Progress
    eta_sec = (time.perf_counter() - t_global) / idx * (total - idx)
    eta_min = eta_sec / 60
    print(f"[{idx:4d}/{total}] {label:45s} → "
          f"Ret={row['return_pct']:>+8.1f}%  DD={row['max_dd_pct']:>6.1f}%  "
          f"Sharpe={row['sharpe']:>6.2f}  WR={row['win_rate']:>5.1f}%  "
          f"Trades={row['trades']:>4.0f}  "
          f"({elapsed:.1f}s)  ETA={eta_min:.0f}min")

    # Periodic save (every 50 combos)
    if idx % 50 == 0:
        os.makedirs("reports", exist_ok=True)
        with open("reports/full_param_grid_partial.json", "w") as f:
            json.dump(rows, f, indent=2, default=str)

# ── Save complete results ────────────────────────────────────
os.makedirs("reports", exist_ok=True)
with open("reports/full_param_grid.json", "w") as f:
    json.dump(rows, f, indent=2, default=str)

# ── Summary tables ───────────────────────────────────────────
total_time = time.perf_counter() - t_global
print(f"\n{'=' * 120}")
print(f"  FULL PARAMETER GRID COMPLETE — {total} combos in {total_time:.0f}s ({total_time/60:.1f}min)")
print(f"{'=' * 120}")

valid = [r for r in rows if r["trades"] > 0 and "error" not in r]


def _fmt_row(r):
    """Format a single result row for table printing."""
    otm_s = "ATM" if r["otm"] == 0 else f"{r['otm']*100:.0f}%"
    wing_s = "naked" if r["wing"] == 0 else f"{r['wing']*100:.0f}%"
    sl_s = "none" if r["sl"] >= 9999 else f"{r['sl']*100:.0f}%"
    tp_s = "none" if r["tp"] >= 9999 else f"{r['tp']:.0f}%"
    return otm_s, wing_s, sl_s, tp_s


TABLE_HDR = (f"  {'Rank':>4} {'Wing':>6} {'SL':>6} {'TP':>6} "
             f"{'Return%':>10} {'DD%':>7} {'Sharpe':>7} {'WR%':>6} {'PF':>6} {'Trades':>6}")


def _print_row(i, r):
    _, wing_s, sl_s, tp_s = _fmt_row(r)
    print(f"  {i:4d} {wing_s:>6} {sl_s:>6} {tp_s:>6} "
          f"{r['return_pct']:>+10.1f} {r['max_dd_pct']:>7.1f} {r['sharpe']:>7.2f} "
          f"{r['win_rate']:>6.1f} {r['profit_factor']:>6.2f} {r['trades']:>6.0f}")


# ── Top 20 by Sharpe ─────────────────────────────────────────
ul_rows = [r for r in valid if r["underlying"] == "BTC"]
if ul_rows:
    print(f"\n  {'─' * 60}")
    print(f"  BTC — Top 20 by Sharpe ratio")
    print(f"  {'─' * 60}")
    top = sorted(ul_rows, key=lambda r: r["sharpe"], reverse=True)[:20]
    print(TABLE_HDR)
    for i, r in enumerate(top, 1):
        _print_row(i, r)

# ── Top 20 by Return ─────────────────────────────────────────
if ul_rows:
    print(f"\n  {'─' * 60}")
    print(f"  BTC — Top 20 by Total Return")
    print(f"  {'─' * 60}")
    top = sorted(ul_rows, key=lambda r: r["return_pct"], reverse=True)[:20]
    print(TABLE_HDR)
    for i, r in enumerate(top, 1):
        _print_row(i, r)

# ── Best risk-adjusted (Sharpe > 1, lowest DD) ───────────────
risk_adj = [r for r in ul_rows if r["sharpe"] > 1 and r["return_pct"] > 0]
if risk_adj:
    print(f"\n  {'─' * 60}")
    print(f"  BTC — Best risk-adjusted (Sharpe>1, positive return, lowest DD)")
    print(f"  {'─' * 60}")
    top = sorted(risk_adj, key=lambda r: abs(r["max_dd_pct"]))[:15]
    print(TABLE_HDR)
    for i, r in enumerate(top, 1):
        _print_row(i, r)

# ── Wing × TP heatmap (no SL) ────────────────────────────────
print(f"\n  {'=' * 90}")
print(f"  BTC Wing × TakeProfit Comparison (DTE=0, no SL)")
print(f"  {'=' * 90}")

wing_tp_rows = [r for r in ul_rows if r["sl"] >= 9999]
for metric_name, metric_key, fmt in [
    ("Return %", "return_pct", "{:+.1f}"),
    ("MaxDD %", "max_dd_pct", "{:.1f}"),
    ("Sharpe", "sharpe", "{:.2f}"),
]:
    print(f"\n  [{metric_name}]")
    header = f"{'Wing':>8}"
    for tp in TP_PCTS:
        tp_label = "noTP" if tp >= 9999 else f"TP{tp:.0f}%"
        header += f" | {tp_label:>9}"
    print(f"  {header}")
    print("  " + "-" * (10 + 12 * len(TP_PCTS)))
    for wing in WING_PCTS:
        wing_label = "naked" if wing == 0 else f"W{wing*100:.0f}%"
        line = f"  {wing_label:>6}"
        for tp in TP_PCTS:
            cell = next((r for r in wing_tp_rows
                         if r["wing"] == wing and r["tp"] == tp), None)
            if cell and cell["trades"] > 0:
                val = fmt.format(cell[metric_key])
                line += f" | {val:>9}"
            else:
                line += f" | {'N/A':>9}"
        print(line)

# ── TP × SL heatmap (naked) ──────────────────────────────────
print(f"\n  {'=' * 90}")
print(f"  BTC Naked Strangle — SL × TP Comparison (DTE=0)")
print(f"  {'=' * 90}")

naked_rows = [r for r in ul_rows if r["wing"] == 0]
for metric_name, metric_key, fmt in [
    ("Return %", "return_pct", "{:+.1f}"),
    ("MaxDD %", "max_dd_pct", "{:.1f}"),
    ("Sharpe", "sharpe", "{:.2f}"),
]:
    print(f"\n  [{metric_name}]")
    header = f"{'SL':>8}"
    for tp in TP_PCTS:
        tp_label = "noTP" if tp >= 9999 else f"TP{tp:.0f}%"
        header += f" | {tp_label:>9}"
    print(f"  {header}")
    print("  " + "-" * (10 + 12 * len(TP_PCTS)))
    for sl in SL_PCTS:
        sl_label = "noSL" if sl >= 9999 else f"SL{sl*100:.0f}%"
        line = f"  {sl_label:>6}"
        for tp in TP_PCTS:
            cell = next((r for r in naked_rows
                         if r["sl"] == sl and r["tp"] == tp), None)
            if cell and cell["trades"] > 0:
                val = fmt.format(cell[metric_key])
                line += f" | {val:>9}"
            else:
                line += f" | {'N/A':>9}"
        print(line)

# ── Per-wing: SL × TP heatmap ────────────────────────────────
for wing in [w for w in WING_PCTS if w > 0]:
    wing_label = f"W{wing*100:.0f}%"
    print(f"\n  {'=' * 90}")
    print(f"  BTC Iron Condor {wing_label} — SL × TP Comparison (DTE=0)")
    print(f"  {'=' * 90}")

    w_rows = [r for r in ul_rows if r["wing"] == wing]
    for metric_name, metric_key, fmt in [
        ("Return %", "return_pct", "{:+.1f}"),
        ("MaxDD %", "max_dd_pct", "{:.1f}"),
        ("Sharpe", "sharpe", "{:.2f}"),
    ]:
        print(f"\n  [{metric_name}]")
        header = f"{'SL':>8}"
        for tp in TP_PCTS:
            tp_label = "noTP" if tp >= 9999 else f"TP{tp:.0f}%"
            header += f" | {tp_label:>9}"
        print(f"  {header}")
        print("  " + "-" * (10 + 12 * len(TP_PCTS)))
        for sl in SL_PCTS:
            sl_label = "noSL" if sl >= 9999 else f"SL{sl*100:.0f}%"
            line = f"  {sl_label:>6}"
            for tp in TP_PCTS:
                cell = next((r for r in w_rows
                             if r["sl"] == sl and r["tp"] == tp), None)
                if cell and cell["trades"] > 0:
                    val = fmt.format(cell[metric_key])
                    line += f" | {val:>9}"
                else:
                    line += f" | {'N/A':>9}"
            print(line)

print(f"\n完整结果已保存: reports/full_param_grid.json")
print(f"总耗时: {total_time:.0f}s ({total_time/60:.1f}min, {total_time/3600:.1f}h)")
