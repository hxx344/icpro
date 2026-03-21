"""Extract data from Plotly HTML reports and analyze backtest results."""
import json
import re
import sys
import base64
from pathlib import Path
from datetime import datetime
import numpy as np


def decode_bdata(obj):
    """Decode Plotly binary-encoded data (dtype+bdata dict)."""
    if isinstance(obj, list):
        return np.array(obj, dtype=float)
    if isinstance(obj, dict) and "bdata" in obj:
        raw = base64.b64decode(obj["bdata"])
        return np.frombuffer(raw, dtype=np.dtype(obj.get("dtype", "f8")))
    return obj


def extract_plotly_data(html_path: str) -> list[dict]:
    """Extract trace data from Plotly HTML file."""
    text = Path(html_path).read_text(encoding="utf-8")
    match = re.search(
        r'Plotly\.newPlot\(\s*"[^"]+"\s*,\s*(\[.*?\])\s*,\s*(\{.*?\})\s*\)',
        text, re.DOTALL,
    )
    if match:
        return json.loads(match.group(1))
    return []


def analyze_equity_curve():
    html_path = "reports/equity_curve.html"
    if not Path(html_path).exists():
        print("equity_curve.html not found")
        return

    traces = extract_plotly_data(html_path)
    if not traces:
        print("Could not extract Plotly data")
        return

    # Trace 0 = Equity, Trace 1 = Balance, Trace 2 = Drawdown
    equity_trace = traces[0]
    timestamps = equity_trace["x"]
    equity_values = decode_bdata(equity_trace["y"])

    n = len(timestamps)
    print(f"Total data points: {n}")
    print(f"Period: {timestamps[0]} to {timestamps[-1]}")
    print(f"Initial equity: {equity_values[0]:.6f} BTC")
    print(f"Final equity: {equity_values[-1]:.6f} BTC")
    print(f"Total return: {(equity_values[-1] - equity_values[0]) / equity_values[0] * 100:.2f}%")

    # Drawdown
    eq = equity_values.astype(float)
    running_max = np.maximum.accumulate(eq)
    drawdown = (eq - running_max) / np.where(running_max > 0, running_max, 1)
    max_dd = np.min(drawdown)
    max_dd_idx = np.argmin(drawdown)
    print(f"\nMax Drawdown: {max_dd * 100:.2f}%")
    print(f"Max DD time: {timestamps[max_dd_idx]}")
    print(f"Equity at max DD: {equity_values[max_dd_idx]:.6f} BTC")

    # Find drawdown period
    peak_idx = np.argmax(eq[:max_dd_idx + 1])
    print(f"Peak before DD: {timestamps[peak_idx]} (equity: {equity_values[peak_idx]:.6f})")

    # Recovery
    peak_val = eq[peak_idx]
    recovery_idx = None
    for i in range(max_dd_idx, len(eq)):
        if eq[i] >= peak_val:
            recovery_idx = i
            break
    if recovery_idx:
        print(f"Recovery time: {timestamps[recovery_idx]}")
    else:
        print("Never recovered from max drawdown")

    # Yearly breakdown
    print("\n" + "=" * 60)
    print("  YEARLY BREAKDOWN")
    print("=" * 60)
    dates = [datetime.fromisoformat(str(t).replace("Z", "+00:00").split(".")[0]) if isinstance(t, str) else t for t in timestamps]
    years = sorted(set(d.year for d in dates))

    for year in years:
        year_mask = [d.year == year for d in dates]
        year_indices = [i for i, m in enumerate(year_mask) if m]
        if not year_indices:
            continue

        start_eq = eq[year_indices[0]]
        end_eq = eq[year_indices[-1]]
        year_return = (end_eq - start_eq) / start_eq * 100

        year_eq = eq[year_indices]
        year_max = np.maximum.accumulate(year_eq)
        year_dd = (year_eq - year_max) / np.where(year_max > 0, year_max, 1)
        year_max_dd = np.min(year_dd) * 100

        print(f"  {year}: Return={year_return:+.2f}%, MaxDD={year_max_dd:.2f}%, "
              f"Start={start_eq:.4f}, End={end_eq:.4f}")

    # Top drawdown periods (find all drawdown valleys)
    print("\n" + "=" * 60)
    print("  TOP 5 DRAWDOWN PERIODS")
    print("=" * 60)

    # Find distinct drawdown periods
    in_drawdown = drawdown < -0.01  # more than 1% drawdown
    periods = []
    start = None
    for i in range(len(drawdown)):
        if in_drawdown[i] and start is None:
            start = i
        elif not in_drawdown[i] and start is not None:
            valley_idx = start + np.argmin(drawdown[start:i])
            periods.append({
                "start": start,
                "valley": valley_idx,
                "end": i,
                "depth": drawdown[valley_idx],
                "start_time": timestamps[start],
                "valley_time": timestamps[valley_idx],
                "end_time": timestamps[i],
            })
            start = None
    # Handle case where drawdown extends to end
    if start is not None:
        valley_idx = start + np.argmin(drawdown[start:])
        periods.append({
            "start": start,
            "valley": valley_idx,
            "end": len(drawdown) - 1,
            "depth": drawdown[valley_idx],
            "start_time": timestamps[start],
            "valley_time": timestamps[valley_idx],
            "end_time": timestamps[-1],
        })

    periods.sort(key=lambda p: p["depth"])
    for i, p in enumerate(periods[:5]):
        print(f"  #{i+1}: Depth={p['depth']*100:.2f}%, "
              f"Start={p['start_time']}, Bottom={p['valley_time']}, End={p['end_time']}")


def analyze_trades():
    html_path = "reports/trade_pnl.html"
    if not Path(html_path).exists():
        print("trade_pnl.html not found")
        return

    traces = extract_plotly_data(html_path)
    if not traces:
        print("Could not extract Plotly trade data")
        return

    pnl = decode_bdata(traces[0]["y"]).astype(float)

    print("\n" + "=" * 60)
    print("  TRADE ANALYSIS")
    print("=" * 60)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    print(f"  Total trades: {len(pnl)}")
    print(f"  Winning trades: {len(wins)} ({len(wins) / len(pnl) * 100:.1f}%)")
    print(f"  Losing trades: {len(losses)} ({len(losses) / len(pnl) * 100:.1f}%)")
    print(f"  Total PnL: {np.sum(pnl):.6f} BTC")
    print(f"  Avg Win: {np.mean(wins):.6f} BTC")
    print(f"  Avg Loss: {np.mean(losses):.6f} BTC")
    print(f"  Max Win: {np.max(pnl):.6f} BTC")
    print(f"  Max Loss: {np.min(pnl):.6f} BTC")
    if np.sum(losses) != 0:
        print(f"  Profit Factor: {abs(np.sum(wins) / np.sum(losses)):.2f}")

    print("\n  Top 10 Worst Trades:")
    sorted_idx = np.argsort(pnl)
    for i in range(min(10, len(sorted_idx))):
        idx = sorted_idx[i]
        print(f"    Trade #{idx}: PnL = {pnl[idx]:.6f} BTC")


if __name__ == "__main__":
    print("=" * 60)
    print("  BACKTEST RESULTS ANALYSIS")
    print("=" * 60)
    analyze_equity_curve()
    analyze_trades()
