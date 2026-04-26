"""Compare weekend vol backtest: Saturday vs Sunday vs Monday settlement.

Runs the same strategy (delta=0.40, wing=0.05, 3x USD) with three
different settlement days and produces a comparison table + overlay chart.
"""

import sys
import json
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd

from scripts.backtest_weekend_vol import run_backtest, EXPIRE_DAY_MAP


# ──────────── Parameters (matching optimized config) ────────────
PARAMS = dict(
    data_dir="data",
    start_date="2022-01-01",
    end_date="2024-12-31",
    target_delta=0.40,
    max_delta_diff=0.10,
    slippage=0.05,
    iv_discount=0.25,
    leverage=3.0,
    margin_mode="usd",
    wing_delta=0.05,
    verbose=False,
)


def compute_stats(df: pd.DataFrame, start: str, end: str, initial_usd: float) -> dict:
    """Compute key performance stats from trade DataFrame."""
    if len(df) == 0:
        return {"trades": 0}

    n = len(df)
    wins = (df["pnl"] >= 0).sum()
    total_pnl = df["pnl"].sum()
    final_bal = df["balance"].iloc[-1]
    years = (date.fromisoformat(end) - date.fromisoformat(start)).days / 365.25
    pct_ret = total_pnl / initial_usd * 100
    apr = pct_ret / years if years > 0 else 0

    # Max drawdown from balance series
    balances = [initial_usd] + df["balance"].tolist()
    peak = balances[0]
    max_dd = 0.0
    for b in balances:
        peak = max(peak, b)
        dd = (peak - b) / peak
        max_dd = max(max_dd, dd)

    # Sharpe
    rets = df["pnl"] / initial_usd
    sharpe = rets.mean() / rets.std() * np.sqrt(52) if rets.std() > 0 else 0

    # Profit factor
    gw = df.loc[df["pnl"] >= 0, "pnl"].sum()
    gl = abs(df.loc[df["pnl"] < 0, "pnl"].sum())
    pf = gw / gl if gl > 0 else float("inf")

    calmar = (apr / 100) / max_dd if max_dd > 0 else float("inf")

    avg_premium = df["net_premium"].mean() if "net_premium" in df.columns else 0
    avg_qty = df["qty"].mean() if "qty" in df.columns else 0

    return {
        "trades": n,
        "wins": wins,
        "win_rate": wins / n * 100,
        "total_pnl": total_pnl,
        "final_balance": final_bal,
        "apr": apr,
        "max_dd": max_dd * 100,
        "sharpe": sharpe,
        "profit_factor": pf,
        "calmar": calmar,
        "avg_pnl": total_pnl / n,
        "avg_premium_btc": avg_premium,
        "avg_qty": avg_qty,
    }


def run_comparison():
    """Run backtest for each settlement day and print comparison."""

    expire_days = ["saturday", "sunday", "monday"]
    results = {}

    for day in expire_days:
        _, hours, _ = EXPIRE_DAY_MAP[day]
        print(f"\n{'─'*70}")
        print(f"  Running: {day.capitalize()} 08:00 UTC settlement ({hours}h hold)...")
        print(f"{'─'*70}")

        df = run_backtest(**PARAMS, expire_day=day)

        if len(df) == 0:
            print(f"  ⚠ No trades for {day}")
            results[day] = {"trades": 0, "df": pd.DataFrame(), "equity": []}
            continue

        initial_usd = df["spot_entry"].iloc[0]  # 1 BTC worth at first trade
        stats = compute_stats(df, PARAMS["start_date"], PARAMS["end_date"], initial_usd)
        stats["df"] = df
        stats["initial_usd"] = initial_usd

        # Build equity curve
        eq = [(date.fromisoformat(PARAMS["start_date"]), initial_usd)]
        for _, row in df.iterrows():
            eq.append((row["expiry"], row["balance"]))
        stats["equity"] = eq

        results[day] = stats

        print(f"  ✓ {stats['trades']} trades | APR {stats['apr']:.1f}% | "
              f"MaxDD {stats['max_dd']:.1f}% | Calmar {stats['calmar']:.2f} | "
              f"Sharpe {stats['sharpe']:.2f} | WR {stats['win_rate']:.1f}%")

    # ──────────── Comparison Table ────────────
    print(f"\n\n{'='*80}")
    print(f"  COMPARISON: Settlement Day Effect on Weekend Vol Strategy")
    print(f"  Delta=0.40  Wing=0.05  3x USD  IVd=25%  Period={PARAMS['start_date']}→{PARAMS['end_date']}")
    print(f"{'='*80}\n")

    header = f"{'Metric':<22} {'Saturday (16h)':>16} {'Sunday (40h)':>16} {'Monday (64h)':>16}"
    print(header)
    print("─" * len(header))

    metrics = [
        ("Trades", "trades", "{:.0f}"),
        ("Win Rate %", "win_rate", "{:.1f}"),
        ("APR %", "apr", "{:.1f}"),
        ("Max Drawdown %", "max_dd", "{:.1f}"),
        ("Sharpe (ann.)", "sharpe", "{:.2f}"),
        ("Calmar Ratio", "calmar", "{:.2f}"),
        ("Profit Factor", "profit_factor", "{:.2f}"),
        ("Total PnL ($)", "total_pnl", "{:,.0f}"),
        ("Final Balance ($)", "final_balance", "{:,.0f}"),
        ("Avg PnL/Trade ($)", "avg_pnl", "{:,.1f}"),
        ("Avg Premium (BTC)", "avg_premium_btc", "{:.6f}"),
    ]

    for label, key, fmt in metrics:
        vals = []
        for day in expire_days:
            s = results[day]
            v = s.get(key, 0)
            if isinstance(v, float) and (v == float("inf") or np.isinf(v)):
                vals.append("∞")
            else:
                vals.append(fmt.format(v))
        print(f"  {label:<20} {vals[0]:>16} {vals[1]:>16} {vals[2]:>16}")

    print(f"\n{'='*80}")

    # ──────────── Generate HTML comparison report ────────────
    _generate_comparison_report(results, expire_days)

    return results


def _generate_comparison_report(results: dict, expire_days: list):
    """Generate HTML with overlaid equity curves and comparison metrics."""

    colors = {
        "saturday": "#f9e2af",   # yellow
        "sunday":   "#89b4fa",   # blue
        "monday":   "#a6e3a1",   # green
    }
    hours_map = {"saturday": 16, "sunday": 40, "monday": 64}

    # Build chart data
    datasets_js = []
    for day in expire_days:
        eq = results[day].get("equity", [])
        if not eq:
            continue
        dates = [str(d) for d, _ in eq]
        values = [round(v, 2) for _, v in eq]
        color = colors[day]
        h = hours_map[day]
        datasets_js.append(f"""{{
            label: '{day.capitalize()} ({h}h)',
            data: {json.dumps(values)},
            borderColor: '{color}',
            backgroundColor: 'transparent',
            tension: 0.2,
            pointRadius: 1,
            borderWidth: 2
        }}""")

    # Use the longest date list as labels
    max_dates = []
    for day in expire_days:
        eq = results[day].get("equity", [])
        dates = [str(d) for d, _ in eq]
        if len(dates) > len(max_dates):
            max_dates = dates

    # Metrics table rows
    metric_rows = ""
    metrics_list = [
        ("Trades", "trades", "{:.0f}"),
        ("Win Rate", "win_rate", "{:.1f}%"),
        ("APR", "apr", "{:.1f}%"),
        ("Max Drawdown", "max_dd", "{:.1f}%"),
        ("Sharpe", "sharpe", "{:.2f}"),
        ("Calmar", "calmar", "{:.2f}"),
        ("Profit Factor", "profit_factor", "{:.2f}"),
        ("Total PnL", "total_pnl", "${:,.0f}"),
        ("Final Balance", "final_balance", "${:,.0f}"),
    ]

    for label, key, fmt in metrics_list:
        vals = []
        raw_vals = []
        for day in expire_days:
            s = results[day]
            v = s.get(key, 0)
            raw_vals.append(v if not (isinstance(v, float) and np.isinf(v)) else 0)
            if isinstance(v, float) and np.isinf(v):
                vals.append("∞")
            else:
                vals.append(fmt.format(v))

        # Highlight best value
        best_idx = -1
        if key in ("max_dd",):  # lower is better
            best_idx = int(np.argmin(raw_vals))
        elif key not in ("trades",):  # higher is better
            best_idx = int(np.argmax(raw_vals))

        cells = ""
        for i, v in enumerate(vals):
            style = ' style="color:#a6e3a1;font-weight:bold"' if i == best_idx else ''
            cells += f"<td{style}>{v}</td>"
        metric_rows += f"<tr><td>{label}</td>{cells}</tr>\n"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Weekend Vol: Settlement Day Comparison</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background: #1e1e2e; color: #cdd6f4; padding: 20px; }}
  .container {{ max-width: 1100px; margin: auto; }}
  h1 {{ color: #89b4fa; }}
  h2 {{ color: #cba6f7; margin-top: 30px; }}
  .chart-box {{ background: #313244; border-radius: 8px; padding: 20px; margin: 20px 0; }}
  table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
  th, td {{ padding: 10px 16px; text-align: right; border-bottom: 1px solid #45475a; }}
  th {{ color: #9399b2; font-weight: normal; }}
  td:first-child, th:first-child {{ text-align: left; }}
  tr:hover {{ background: #45475a33; }}
  .params {{ color: #9399b2; font-size: 14px; }}
  canvas {{ width: 100% !important; }}
</style></head>
<body><div class="container">
<h1>Weekend Vol: Settlement Day Comparison</h1>
<p class="params">
  Delta=0.40 | Wing=0.05 | 3x USD | IV discount=25% | Slippage=5%<br>
  Period: {PARAMS['start_date']} → {PARAMS['end_date']}
</p>

<div class="chart-box">
  <canvas id="equityChart" height="350"></canvas>
</div>

<h2>Performance Metrics</h2>
<table>
  <thead>
    <tr>
      <th>Metric</th>
      <th style="color:{colors['saturday']}">Saturday (16h)</th>
      <th style="color:{colors['sunday']}">Sunday (40h)</th>
      <th style="color:{colors['monday']}">Monday (64h)</th>
    </tr>
  </thead>
  <tbody>
    {metric_rows}
  </tbody>
</table>

<div class="chart-box">
  <canvas id="pnlCompare" height="250"></canvas>
</div>

<script>
const eqCtx = document.getElementById('equityChart').getContext('2d');
new Chart(eqCtx, {{
  type: 'line',
  data: {{
    labels: {json.dumps(max_dates)},
    datasets: [{','.join(datasets_js)}]
  }},
  options: {{
    responsive: true,
    plugins: {{
      title: {{ display: true, text: 'Equity Curves: Settlement Day Comparison', color: '#cdd6f4', font: {{size: 16}} }}
    }},
    scales: {{
      x: {{ ticks: {{ color: '#9399b2', maxTicksLimit: 20 }} }},
      y: {{ ticks: {{ color: '#9399b2', callback: function(v) {{ return '$' + v.toLocaleString(); }} }} }}
    }}
  }}
}});

// PnL distribution comparison
const pnlCtx = document.getElementById('pnlCompare').getContext('2d');
const pnlDatasets = [
{_build_pnl_datasets_js(results, expire_days, colors)}
];
new Chart(pnlCtx, {{
  type: 'bar',
  data: {{
    labels: {json.dumps(_get_pnl_labels(results, expire_days))},
    datasets: pnlDatasets
  }},
  options: {{
    responsive: true,
    plugins: {{
      title: {{ display: true, text: 'Weekly PnL Comparison', color: '#cdd6f4', font: {{size: 16}} }}
    }},
    scales: {{
      x: {{ ticks: {{ color: '#9399b2', maxTicksLimit: 25 }} }},
      y: {{ ticks: {{ color: '#9399b2' }} }}
    }}
  }}
}});
</script>
</div></body></html>"""

    out_dir = Path("reports") / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "settlement_day_comparison.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\n  Report saved: {out_path}")


def _build_pnl_datasets_js(results, expire_days, colors):
    """Build JavaScript datasets for PnL bar chart."""
    parts = []
    hours_map = {"saturday": 16, "sunday": 40, "monday": 64}
    for day in expire_days:
        df = results[day].get("df", pd.DataFrame())
        if len(df) == 0:
            continue
        pnl = [round(v, 2) for v in df["pnl"].tolist()]
        h = hours_map[day]
        parts.append(f"""{{
            label: '{day.capitalize()} ({h}h)',
            data: {json.dumps(pnl)},
            backgroundColor: '{colors[day]}88',
            borderColor: '{colors[day]}',
            borderWidth: 1
        }}""")
    return ",\n".join(parts)


def _get_pnl_labels(results, expire_days):
    """Get longest set of expiry date labels for PnL chart."""
    best = []
    for day in expire_days:
        df = results[day].get("df", pd.DataFrame())
        if len(df) > len(best):
            best = [str(d) for d in df["expiry"].tolist()]
    return best


if __name__ == "__main__":
    run_comparison()
