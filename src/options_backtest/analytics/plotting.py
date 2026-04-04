"""Plotly‑based interactive charts for backtest results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _build_weekend_vol_entry_df(results: dict) -> pd.DataFrame:
    diagnostics = (results.get("strategy_diagnostics") or {}).get("weekend_vol_entries") or []
    if not diagnostics:
        return pd.DataFrame()

    df = pd.DataFrame(diagnostics).copy()
    if df.empty or "entry_time" not in df.columns:
        return pd.DataFrame()

    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["entry_time"]).sort_values("entry_time").reset_index(drop=True)

    trades = pd.DataFrame(results.get("closed_trades", []) or [])
    if not trades.empty and {"entry_time", "direction", "entry_price"}.issubset(trades.columns):
        trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
        short_premium = (
            trades.loc[trades["direction"].astype(str).str.lower() == "short"]
            .groupby("entry_time", as_index=False)["entry_price"]
            .sum()
            .rename(columns={"entry_price": "combined_short_premium_actual"})
        )
        df = df.merge(short_premium, on="entry_time", how="left")
    else:
        df["combined_short_premium_actual"] = np.nan

    df["combined_short_premium"] = df["combined_short_premium_actual"].where(
        df["combined_short_premium_actual"].notna(),
        pd.to_numeric(df.get("combined_short_premium_per_btc"), errors="coerce"),
    )
    df["rv_24h"] = pd.to_numeric(df.get("rv_24h"), errors="coerce")
    df["avg_short_iv"] = pd.to_numeric(df.get("avg_short_iv"), errors="coerce")
    return df


def plot_equity_curve(results: dict, output_dir: str = "reports") -> str:
    """Plot equity curve and drawdown, save as HTML. Returns the file path."""
    equity_history = results.get("equity_history", [])
    if not equity_history:
        return ""

    margin_mode = str(results.get("margin_mode", "coin") or "coin").upper()
    underlying = str(results.get("underlying", "BTC") or "BTC")
    is_usd_margin = margin_mode == "USD"

    # Support both 4-column (legacy) and 5-column (with underlying_price) history
    sample = equity_history[0]
    if len(sample) >= 5:
        df = pd.DataFrame(equity_history, columns=["timestamp", "equity", "balance", "unrealized_pnl", "underlying_price"])
    else:
        df = pd.DataFrame(equity_history, columns=["timestamp", "equity", "balance", "unrealized_pnl"])
        df["underlying_price"] = 0.0
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    entry_df = _build_weekend_vol_entry_df(results)
    has_entry_diag = not entry_df.empty

    # Drawdown
    running_max = np.maximum.accumulate(df["equity"].values)
    drawdown = (df["equity"].values - running_max) / np.where(running_max > 0, running_max, 1)

    # For coin margin, show both coin equity and USD translations.
    # For USD margin, the primary equity series is already USD and should not
    # be labelled as BTC or duplicated in a second USD panel.
    has_usd_panel = (not is_usd_margin) and (df["underlying_price"].iloc[0] > 0)
    if has_usd_panel:
        n_rows = 4 if has_entry_diag else 3
        row_heights = [0.40, 0.22, 0.16, 0.22] if has_entry_diag else [0.45, 0.25, 0.30]
    else:
        n_rows = 3 if has_entry_diag else 2
        row_heights = [0.58, 0.17, 0.25] if has_entry_diag else [0.7, 0.3]

    if is_usd_margin:
        subtitles = ("Equity Curve (USD)", "Entry Diagnostics", "Drawdown") if has_entry_diag else ("Equity Curve (USD)", "Drawdown")
    else:
        subtitles = (
            ("Equity Curve ({})".format(underlying), "Equity Curve (USD)", "Entry Diagnostics", "Drawdown ({})".format(underlying))
            if has_usd_panel and has_entry_diag
            else ("Equity Curve ({})".format(underlying), "Equity Curve (USD)", "Drawdown ({})".format(underlying))
            if has_usd_panel
            else ("Equity Curve ({})".format(underlying), "Entry Diagnostics", "Drawdown")
            if has_entry_diag
            else ("Equity Curve ({})".format(underlying), "Drawdown")
        )

    specs = [[{}] for _ in range(n_rows)]
    if has_entry_diag:
        diag_row = 3 if has_usd_panel else 2
        specs[diag_row - 1] = [{"secondary_y": True}]
    else:
        diag_row = -1

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        specs=specs,
        row_heights=row_heights,
        subplot_titles=subtitles,
        vertical_spacing=0.06,
    )

    # Equity
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["equity"],
            name="Equity", line=dict(color="#2196F3", width=1.5),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["balance"],
            name="Balance", line=dict(color="#9E9E9E", width=1, dash="dot"),
        ),
        row=1, col=1,
    )

    # USD equity curve
    if has_usd_panel:
        usd_equity = df["equity"].values * df["underlying_price"].values
        eq_vals = df["equity"].values.astype(float)
        price_vals = df["underlying_price"].values.astype(float)

        # Real-time hedged USD: each step's BTC PnL locked at current price
        hedged_usd_equity = np.empty_like(eq_vals)
        hedged_usd_equity[0] = eq_vals[0] * price_vals[0]
        delta_eq = np.diff(eq_vals)
        hedged_usd_equity[1:] = hedged_usd_equity[0] + np.cumsum(delta_eq * price_vals[1:])

        fig.add_trace(
            go.Scatter(
                x=df["timestamp"], y=usd_equity,
                name="USD Equity (Unhedged)", line=dict(color="#FF9800", width=1.5),
            ),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"], y=hedged_usd_equity,
                name="USD Equity (Hedged)", line=dict(color="#4CAF50", width=1.5),
            ),
            row=2, col=1,
        )
        # Spot-only (buy & hold) baseline
        initial_price = price_vals[0]
        spot_only = price_vals / initial_price * usd_equity[0]
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"], y=spot_only,
                name="Spot Only", line=dict(color="#9E9E9E", width=1, dash="dash"),
            ),
            row=2, col=1,
        )
        # Cash baseline (no BTC exposure)
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"], y=[float(usd_equity[0])] * len(df),
                name="Cash (No Exposure)", line=dict(color="#BDBDBD", width=1, dash="dot"),
            ),
            row=2, col=1,
        )
        fig.update_yaxes(title_text="USD", row=2, col=1)

    if has_entry_diag:
        hover_text = np.column_stack(
            [
                entry_df.get("short_call_symbol", pd.Series([""] * len(entry_df))).fillna(""),
                pd.to_numeric(entry_df.get("short_call_delta"), errors="coerce").fillna(np.nan),
                pd.to_numeric(entry_df.get("short_call_iv"), errors="coerce").fillna(np.nan) * 100.0,
                entry_df.get("short_put_symbol", pd.Series([""] * len(entry_df))).fillna(""),
                pd.to_numeric(entry_df.get("short_put_delta"), errors="coerce").fillna(np.nan),
                pd.to_numeric(entry_df.get("short_put_iv"), errors="coerce").fillna(np.nan) * 100.0,
            ]
        )

        fig.add_trace(
            go.Bar(
                x=entry_df["entry_time"],
                y=entry_df["combined_short_premium"],
                name="Short Call+Put Premium (1 BTC)",
                marker_color="rgba(255, 152, 0, 0.55)",
                customdata=hover_text,
                hovertemplate=(
                    "Time: %{x}<br>"
                    "Combined premium: %{y:,.2f}<br>"
                    "Call: %{customdata[0]} Δ=%{customdata[1]:.3f} IV=%{customdata[2]:.1f}%<br>"
                    "Put: %{customdata[3]} Δ=%{customdata[4]:.3f} IV=%{customdata[5]:.1f}%<extra></extra>"
                ),
            ),
            row=diag_row,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=entry_df["entry_time"],
                y=entry_df["rv_24h"] * 100.0,
                mode="lines+markers",
                name="RV 24h",
                line=dict(color="#8E24AA", width=1.4),
                marker=dict(size=6),
                hovertemplate="Time: %{x}<br>RV 24h: %{y:.2f}%<extra></extra>",
            ),
            row=diag_row,
            col=1,
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=entry_df["entry_time"],
                y=entry_df["avg_short_iv"] * 100.0,
                mode="lines+markers",
                name="Avg Short IV",
                line=dict(color="#00897B", width=1.4),
                marker=dict(size=6),
                hovertemplate="Time: %{x}<br>Avg short IV: %{y:.2f}%<extra></extra>",
            ),
            row=diag_row,
            col=1,
            secondary_y=True,
        )
        fig.update_yaxes(title_text="Premium (USD)", row=diag_row, col=1, secondary_y=False)
        fig.update_yaxes(title_text="RV / IV %", row=diag_row, col=1, secondary_y=True)

    # Drawdown
    dd_row = n_rows
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=drawdown,
            name=f"Drawdown ({'USD' if is_usd_margin else underlying})", fill="tozeroy",
            line=dict(color="#F44336", width=1),
            fillcolor="rgba(244, 67, 54, 0.3)",
        ),
        row=dd_row, col=1,
    )

    fig.update_layout(
        title="Backtest: Equity & Drawdown",
        height=900 if has_usd_panel and has_entry_diag else 750 if has_usd_panel else 760 if has_entry_diag else 600,
        template="plotly_white",
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
    )
    fig.update_yaxes(title_text=("USD" if is_usd_margin else underlying), row=1, col=1)
    fig.update_yaxes(title_text="DD %", tickformat=".1%", row=dd_row, col=1)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "equity_curve.html"
    fig.write_html(str(path))
    return str(path)


def plot_trade_pnl(results: dict, output_dir: str = "reports") -> str:
    """Bar chart of per‑trade PnL."""
    trades = results.get("closed_trades", [])
    if not trades:
        return ""

    margin_mode = str(results.get("margin_mode", "coin") or "coin").upper()
    underlying = str(results.get("underlying", "BTC") or "BTC")
    pnl_unit = "USD" if margin_mode == "USD" else underlying

    df = pd.DataFrame(trades)
    colors = ["#4CAF50" if p > 0 else "#F44336" for p in df["pnl"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(len(df))),
        y=df["pnl"],
        marker_color=colors,
        name="Trade PnL",
    ))

    fig.update_layout(
        title=f"Per‑Trade PnL ({pnl_unit})",
        xaxis_title="Trade #",
        yaxis_title=f"PnL ({pnl_unit})",
        template="plotly_white",
        height=400,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "trade_pnl.html"
    fig.write_html(str(path))
    return str(path)


def plot_underlying_with_trades(
    results: dict,
    underlying_df: pd.DataFrame,
    output_dir: str = "reports",
) -> str:
    """Candlestick chart of the underlying with buy/sell markers."""
    trades = results.get("closed_trades", [])

    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=underlying_df["timestamp"],
        open=underlying_df["open"],
        high=underlying_df["high"],
        low=underlying_df["low"],
        close=underlying_df["close"],
        name="BTC/USD",
    ))

    # Trade markers
    if trades:
        df_t = pd.DataFrame(trades)
        # Entry markers
        entries = df_t[["entry_time", "instrument_name", "direction"]].copy()
        entries["entry_time"] = pd.to_datetime(entries["entry_time"])
        # We don't have exact underlying price at entry; skip price lookup for MVP
        # Just add time markers on the x‑axis

        for _, row in df_t.iterrows():
            color = "#4CAF50" if row["direction"] == "long" else "#F44336"
            marker_char = "▲" if row["direction"] == "long" else "▼"
            fig.add_vline(
                x=row["entry_time"], line_dash="dot", line_color=color, opacity=0.4,
            )

    fig.update_layout(
        title="Underlying (BTC/USD) with Trade Signals",
        yaxis_title="USD",
        template="plotly_white",
        height=500,
        xaxis_rangeslider_visible=False,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "underlying_trades.html"
    fig.write_html(str(path))
    return str(path)


def generate_all_plots(results: dict, output_dir: str = "reports") -> list[str]:
    """Generate all MVP charts. Returns list of saved file paths."""
    paths: list[str] = []
    p = plot_equity_curve(results, output_dir)
    if p:
        paths.append(p)
    p = plot_trade_pnl(results, output_dir)
    if p:
        paths.append(p)
    return paths
