"""Compare Iron Condor performance across different wing widths.

Runs multiple backtests with varying wing widths and generates a comparison
report with equity curves and key metrics.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import copy
import json

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger

from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.analytics.metrics import compute_metrics


# -----------------------------------------------------------------------
# Parameters to sweep
# -----------------------------------------------------------------------
BASE_CONFIG = "configs/backtest/ic_0dte_8pct_direct.yaml"

# short_otm_pct fixed at 1%, vary hedge_otm_pct (= otm + wing_width)
SHORT_OTM = 0.01  # 1% OTM (matching trader config)
# wing_width = hedge_otm - otm; we sweep hedge_otm_pct directly
WING_CONFIGS = [
    # (wing_label, hedge_otm_pct)  →  wing_width = hedge_otm - otm
    ("2%",  0.03),   # wing = 3% - 1% = 2%
    ("3%",  0.04),   # wing = 4% - 1% = 3%
    ("4%",  0.05),   # wing = 5% - 1% = 4%
    ("5%",  0.06),   # wing = 6% - 1% = 5%
    ("6%",  0.07),   # wing = 7% - 1% = 6%
    ("8%",  0.09),   # wing = 9% - 1% = 8%  (current)
    ("10%", 0.11),   # wing = 11% - 1% = 10%
]

# Override to ETH, 0DTE
UNDERLYING = "ETH"
START_DATE = "2025-03-14"
END_DATE = "2026-03-14"

OUTPUT_DIR = Path("reports/wing_width_comparison")


def run_single(wing_label: str, hedge_otm: float) -> dict:
    """Run one backtest with a specific wing width and return metrics."""
    cfg = Config.from_yaml(BASE_CONFIG)

    # Override only the hedge_otm_pct, keep everything else from the proven config
    cfg.strategy.params["hedge_otm_pct"] = hedge_otm

    wing_pct = hedge_otm - cfg.strategy.params.get("otm_pct", SHORT_OTM)
    label = f"wing_{wing_label}"

    logger.info(f"--- Running backtest: otm={cfg.strategy.params.get('otm_pct', SHORT_OTM)*100:.0f}%, "
                f"hedge_otm={hedge_otm*100:.0f}%, wing={wing_pct*100:.0f}% ---")

    from options_backtest.cli import _load_strategy
    strategy = _load_strategy(cfg.strategy.name, cfg.strategy.params)
    engine = BacktestEngine(cfg, strategy)
    results = engine.run()
    metrics = compute_metrics(results)

    return {
        "label": label,
        "wing_pct": wing_pct,
        "hedge_otm": hedge_otm,
        "short_otm": cfg.strategy.params.get("otm_pct", SHORT_OTM),
        "results": results,
        "metrics": metrics,
    }


def build_comparison_table(all_results: list[dict]) -> pd.DataFrame:
    """Build a DataFrame comparing key metrics across wing widths."""
    rows = []
    for r in all_results:
        m = r["metrics"]
        rows.append({
            "翼宽 (%)": f"{r['wing_pct']*100:.0f}%",
            "Short OTM": f"{r['short_otm']*100:.0f}%",
            "Hedge OTM": f"{r['hedge_otm']*100:.0f}%",
            "总收益率": f"{m.get('total_return', 0)*100:.2f}%",
            "年化收益率": f"{m.get('annualized_return', 0)*100:.2f}%",
            "最大回撤": f"{m.get('max_drawdown', 0)*100:.2f}%",
            "Sharpe": f"{m.get('sharpe_ratio', 0):.2f}",
            "Sortino": f"{m.get('sortino_ratio', 0):.2f}",
            "胜率": f"{m.get('win_rate', 0)*100:.1f}%",
            "总交易数": m.get("total_trades", 0),
            "收益率(USD)": f"{m.get('total_return_usd', 0)*100:.2f}%",
            "年化(USD)": f"{m.get('annualized_return_usd', 0)*100:.2f}%",
            "最大回撤(USD)": f"{m.get('max_drawdown_usd', 0)*100:.2f}%",
            "最终权益": f"{m.get('final_equity', 0):.4f}",
        })
    return pd.DataFrame(rows)


def plot_equity_curves(all_results: list[dict], output_path: Path) -> None:
    """Generate Plotly equity curve comparison chart."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("权益曲线 (ETH 本位)", "权益曲线 (USD)"),
        vertical_spacing=0.12,
    )

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
    ]

    for i, r in enumerate(all_results):
        eq_hist = r["results"].get("equity_history", [])
        if not eq_hist:
            continue

        sample = eq_hist[0]
        if len(sample) >= 5:
            df = pd.DataFrame(eq_hist, columns=["timestamp", "equity", "balance", "unrealized_pnl", "underlying_price"])
        else:
            df = pd.DataFrame(eq_hist, columns=["timestamp", "equity", "balance", "unrealized_pnl"])
            df["underlying_price"] = 0.0

        ts = pd.to_datetime(df["timestamp"], utc=True)
        eq = df["equity"].astype(float)
        color = colors[i % len(colors)]
        label = f"翼宽 {r['wing_pct']*100:.0f}% (hedge {r['hedge_otm']*100:.0f}%)"

        # Coin equity
        fig.add_trace(
            go.Scatter(x=ts, y=eq, mode="lines", name=label,
                       line=dict(color=color, width=1.5),
                       hovertemplate=f"{label}<br>%{{x}}<br>Equity: %{{y:.4f}} ETH<extra></extra>"),
            row=1, col=1,
        )

        # USD equity
        usd_eq = eq * df["underlying_price"].astype(float)
        if usd_eq.iloc[0] > 0:
            fig.add_trace(
                go.Scatter(x=ts, y=usd_eq, mode="lines", name=label,
                           line=dict(color=color, width=1.5),
                           showlegend=False,
                           hovertemplate=f"{label}<br>%{{x}}<br>$ %{{y:,.0f}}<extra></extra>"),
                row=2, col=1,
            )

    fig.update_layout(
        title=f"Iron Condor 翼宽对比  |  ETH  |  Short OTM={SHORT_OTM*100:.0f}%  |  "
              f"{START_DATE} ~ {END_DATE}",
        height=800,
        template="plotly_white",
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Equity (ETH)", row=1, col=1)
    fig.update_yaxes(title_text="Equity (USD)", row=2, col=1)

    fig.write_html(str(output_path), include_plotlyjs="cdn")
    logger.info(f"Equity comparison chart: {output_path}")


def plot_drawdown_comparison(all_results: list[dict], output_path: Path) -> None:
    """Generate drawdown comparison chart."""
    import numpy as np

    fig = go.Figure()
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
    ]

    for i, r in enumerate(all_results):
        eq_hist = r["results"].get("equity_history", [])
        if not eq_hist:
            continue
        sample = eq_hist[0]
        if len(sample) >= 5:
            df = pd.DataFrame(eq_hist, columns=["timestamp", "equity", "balance", "unrealized_pnl", "underlying_price"])
        else:
            df = pd.DataFrame(eq_hist, columns=["timestamp", "equity", "balance", "unrealized_pnl"])

        ts = pd.to_datetime(df["timestamp"], utc=True)
        eq = df["equity"].values.astype(float)
        running_max = np.maximum.accumulate(eq)
        dd = (eq - running_max) / np.where(running_max > 0, running_max, 1) * 100

        label = f"翼宽 {r['wing_pct']*100:.0f}%"
        fig.add_trace(go.Scatter(
            x=ts, y=dd, mode="lines", name=label,
            line=dict(color=colors[i % len(colors)], width=1.2),
            hovertemplate=f"{label}<br>%{{x}}<br>DD: %{{y:.2f}}%<extra></extra>",
        ))

    fig.update_layout(
        title="回撤对比",
        yaxis_title="Drawdown (%)",
        height=400,
        template="plotly_white",
        hovermode="x unified",
    )
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    logger.info(f"Drawdown chart: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    for wing_label, hedge_otm in WING_CONFIGS:
        try:
            result = run_single(wing_label, hedge_otm)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Failed wing_pct={wing_pct}: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        logger.error("No backtests completed!")
        return

    # Comparison table
    df = build_comparison_table(all_results)
    print("\n" + "=" * 80)
    print("Iron Condor Wing Width Comparison")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

    # Save table
    df.to_csv(OUTPUT_DIR / "wing_width_metrics.csv", index=False, encoding="utf-8-sig")

    # Save raw metrics as JSON
    metrics_json = []
    for r in all_results:
        m = {k: v for k, v in r["metrics"].items() if not isinstance(v, (pd.DataFrame, pd.Series))}
        m["wing_pct"] = r["wing_pct"]
        m["short_otm"] = r["short_otm"]
        m["hedge_otm"] = r["hedge_otm"]
        metrics_json.append(m)
    with open(OUTPUT_DIR / "wing_width_metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2, default=str)

    # Charts
    plot_equity_curves(all_results, OUTPUT_DIR / "equity_comparison.html")
    plot_drawdown_comparison(all_results, OUTPUT_DIR / "drawdown_comparison.html")

    logger.info(f"\nAll outputs saved to {OUTPUT_DIR}/")
    logger.info("Files: wing_width_metrics.csv, wing_width_metrics.json, "
                "equity_comparison.html, drawdown_comparison.html")


if __name__ == "__main__":
    main()
