"""Compare Iron Condor 0DTE ±8%: Direct entry vs Wait-for-Midpoint."""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.analytics.metrics import compute_metrics, print_metrics


def run_one(config_path: str) -> dict:
    cfg = Config.from_yaml(config_path)
    name = cfg.strategy.params.get("wait_for_midpoint", False)
    label = "Midpoint" if name else "Direct"
    print(f"\n{'='*60}")
    print(f"  Running: {cfg.backtest.name} ({label})")
    print(f"{'='*60}")

    from options_backtest.cli import _load_strategy
    strategy = _load_strategy(cfg.strategy.name, cfg.strategy.params)
    engine = BacktestEngine(cfg, strategy)
    results = engine.run()
    metrics = compute_metrics(results)
    print_metrics(metrics)
    return {"label": label, "config": cfg.backtest.name, "metrics": metrics, "results": results}


def main():
    configs = [
        "configs/backtest/ic_0dte_8pct_direct.yaml",
        "configs/backtest/ic_0dte_8pct_midpoint.yaml",
    ]

    all_results = []
    for c in configs:
        r = run_one(c)
        all_results.append(r)

    # --- Comparison summary ---
    print(f"\n{'='*70}")
    print("  COMPARISON SUMMARY: Direct vs Midpoint")
    print(f"{'='*70}")

    headers = ["Metric", "Direct (08:00)", "Midpoint (wait 2h)"]
    key_metrics = [
        ("total_return", "总收益率", True),
        ("annualized_return", "年化收益率", True),
        ("max_drawdown", "最大回撤", True),
        ("sharpe_ratio", "Sharpe Ratio", False),
        ("win_rate", "胜率", True),
        ("total_trades", "总交易次数", False),
        ("profit_factor", "Profit Factor", False),
        ("avg_win", "平均盈利 (ETH)", False),
        ("avg_loss", "平均亏损 (ETH)", False),
        ("total_fees", "总手续费 (ETH)", False),
        ("total_return_hedged", "对冲USD收益率", True),
        ("sharpe_ratio_hedged", "对冲USD Sharpe", False),
        ("max_drawdown_hedged", "对冲USD最大回撤", True),
    ]

    print(f"\n{'指标':<25} {'Direct (08:00)':>18} {'Midpoint (wait)':>18}")
    print("-" * 62)

    for key, label, is_pct in key_metrics:
        vals = []
        for r in all_results:
            m = r["metrics"]
            v = m.get(key, 0)
            if is_pct:
                vals.append(f"{v:.2%}")
            elif isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        print(f"{label:<25} {vals[0]:>18} {vals[1]:>18}")

    # --- Equity curve comparison ---
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import numpy as np

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.7, 0.3],
                            subplot_titles=("Equity Curve (ETH)", "Drawdown"),
                            vertical_spacing=0.08)

        colors = ["#2196F3", "#FF9800"]
        for i, r in enumerate(all_results):
            res = r["results"]
            eh = res.get("equity_history", [])
            if not eh:
                continue
            sample = eh[0]
            if len(sample) >= 5:
                df = pd.DataFrame(eh, columns=["timestamp", "equity", "balance", "unrealized_pnl", "underlying_price"])
            else:
                df = pd.DataFrame(eh, columns=["timestamp", "equity", "balance", "unrealized_pnl"])
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df["equity"],
                name=r["label"], line=dict(color=colors[i], width=1.5),
            ), row=1, col=1)

            # Drawdown
            eq = df["equity"].values.astype(float)
            running_max = np.maximum.accumulate(eq)
            dd = (eq - running_max) / np.where(running_max > 0, running_max, 1)
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=dd,
                name=f"{r['label']} DD", line=dict(color=colors[i], width=1),
                fill="tozeroy", fillcolor=f"rgba({','.join(str(int(colors[i][j:j+2], 16)) for j in (1,3,5))},0.1)",
            ), row=2, col=1)

        fig.update_layout(
            title="Iron Condor 0DTE ±8%: Direct vs Midpoint Entry",
            template="plotly_dark",
            height=700,
        )
        fig.update_yaxes(title_text="Equity (ETH)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown", tickformat=".1%", row=2, col=1)

        out_dir = Path("reports/ic_0dte_compare")
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_dir / "direct_vs_midpoint.html"))
        print(f"\nEquity chart saved to reports/ic_0dte_compare/direct_vs_midpoint.html")
    except Exception as e:
        print(f"\nChart generation failed: {e}")


if __name__ == "__main__":
    main()
