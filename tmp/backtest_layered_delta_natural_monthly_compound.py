from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from options_backtest.data.loader import DataLoader
from scripts.optimize_btc_covered_call import (
    LongPutPosition,
    ShortCallPosition,
    _close_long_put,
    _close_short_call,
    _ensure_quote_index,
    _mark_long_put,
    _mark_short_call,
    _select_call_by_delta,
    _select_put_by_delta,
    _should_buy_protective_put,
    _to_utc,
)
from scripts.run_cached_hourly_backtest import _choose_date_window, detect_hourly_coverage

SRC = REPO / "tmp" / "scan_onchain_voting_bull_bear.py"
spec_obj = importlib.util.spec_from_file_location("onchain_vote", SRC)
if spec_obj is None or spec_obj.loader is None:
    raise RuntimeError(f"Cannot load module from {SRC}")
spec = spec_obj
mod = importlib.util.module_from_spec(spec)
loader = cast(Any, spec.loader)
loader.exec_module(mod)

OUT_DIR = REPO / "reports" / "optimizations" / "btc_covered_call_layered_delta_natural_monthly_compound"
VOTE_MEMBERS = [
    "rsi_14d_gt_50_lt_80",
    "price_gt_sma_30d",
    "roc_30d_gt_-5%",
    "onchain_sopr_proxy_155d_0p95_to_1p5",
    "fear_greed_25_to_80",
    "roc_730d_gt_+0%",
    "onchain_mvrv_1_to_3p5",
]
DELTA_7 = 0.01
DELTA_6 = 0.01
DELTA_45 = 0.30
DELTA_03 = 0.48
EXPIRY_DAYS = 7.0
SHORT_CALL_MULT = 2.0
PUT_MULT = 2.0
PUT_DELTA = 0.08
OPTION_FEE = 0.0003
DELIVERY_FEE = 0.0003
DRAWDOWN_TRIGGER = 0.10


def fmt_usd(v: float) -> str:
    return f"${v:,.0f}"


def fmt_pct(v: float) -> str:
    return f"{v * 100:,.2f}%"


def max_drawdown(s: pd.Series) -> float:
    return float((s / s.cummax() - 1.0).min())


def sharpe_hourly(equity: pd.Series) -> float:
    ret = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if ret.empty or float(ret.std()) == 0.0:
        return 0.0
    return float(ret.mean() / ret.std() * (24 * 365) ** 0.5)


def build_vote_count(daily: pd.DataFrame) -> pd.DataFrame:
    signals: dict[str, pd.Series] = mod.build_atomic_signals(daily)
    vote_mat = pd.concat([signals[name].astype(int).rename(name) for name in VOTE_MEMBERS], axis=1)
    out = pd.DataFrame({"date": daily["date"], "vote_count": vote_mat.sum(axis=1).astype(int)})
    vc = out["vote_count"].astype(int)
    out["call_delta_override"] = DELTA_03
    out.loc[vc.between(4, 5), "call_delta_override"] = DELTA_45
    out.loc[vc == 6, "call_delta_override"] = DELTA_6
    out.loc[vc == 7, "call_delta_override"] = DELTA_7
    out["signal"] = vc >= 6
    return out


def attach_vote_layers(underlying: pd.DataFrame, daily_votes: pd.DataFrame) -> pd.DataFrame:
    u = underlying.copy()
    u["date"] = pd.to_datetime(u["timestamp"], utc=True).dt.tz_localize(None).dt.normalize()
    u = u.merge(daily_votes[["date", "vote_count", "call_delta_override", "signal"]], on="date", how="left")
    u["vote_count"] = u["vote_count"].ffill().fillna(0).astype(int)
    u["call_delta_override"] = u["call_delta_override"].ffill().fillna(DELTA_03).astype(float)
    u["signal"] = u["signal"].ffill().fillna(False).astype(bool)
    close = u["close"].astype(float)
    u["trend_sma"] = close.where(~u["signal"], close * 0.999)
    u["trend_sma"] = u["trend_sma"].where(u["signal"], close * 1.001)
    return u


def current_equity(
    account_capital_usd: float,
    base_spot: float,
    long_btc_qty: float,
    option_cash_usd: float,
    spot: float,
    short_call_liability_usd: float,
    long_put_asset_usd: float,
) -> tuple[float, float]:
    long_btc_pnl_usd = (spot - base_spot) * long_btc_qty
    equity_usd = account_capital_usd + long_btc_pnl_usd + option_cash_usd - short_call_liability_usd + long_put_asset_usd
    return equity_usd, long_btc_pnl_usd


def run_natural_monthly_compound(underlying_df: pd.DataFrame, store: Any) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    account_capital_usd: float | None = None
    original_initial_usd: float | None = None
    base_spot: float | None = None
    long_btc_qty = 0.0
    option_cash_usd = 0.0
    last_rebalance_period: pd.Period | None = None
    call_pos: ShortCallPosition | None = None
    put_pos: LongPutPosition | None = None
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    rebalance_rows: list[dict[str, Any]] = []

    for row in underlying_df.itertuples(index=False):
        now = _to_utc(row.timestamp)
        spot = float(cast(Any, row.close))
        if spot <= 0:
            continue
        period = pd.Period(now.tz_convert(None), freq="M")
        if account_capital_usd is None:
            account_capital_usd = spot
            original_initial_usd = spot
            base_spot = spot
            long_btc_qty = 1.0
            last_rebalance_period = period
        assert account_capital_usd is not None and original_initial_usd is not None and base_spot is not None

        if call_pos is not None and now >= call_pos.expiry:
            cash_delta, trade = _close_short_call(store, call_pos, now, spot, OPTION_FEE, DELIVERY_FEE, close_type="settlement")
            option_cash_usd += cash_delta
            trade["leg"] = "short_call"
            trades.append(trade)
            call_pos = None

        if put_pos is not None and now >= put_pos.expiry:
            cash_delta, trade = _close_long_put(put_pos, now, spot, DELIVERY_FEE)
            option_cash_usd += cash_delta
            trade["leg"] = "long_put"
            trades.append(trade)
            put_pos = None

        short_call_liability_usd = _mark_short_call(store, call_pos, now, spot) if call_pos is not None else 0.0
        long_put_asset_usd = _mark_long_put(store, put_pos, now, spot) if put_pos is not None else 0.0
        equity_usd, long_btc_pnl_usd = current_equity(account_capital_usd, base_spot, long_btc_qty, option_cash_usd, spot, short_call_liability_usd, long_put_asset_usd)

        # Correct monthly compounding: do not interrupt open option structures.
        # If a new calendar month has started, wait until all options naturally expire, then rebalance before the next entry.
        if call_pos is None and put_pos is None and last_rebalance_period is not None and period != last_rebalance_period:
            prev_qty = long_btc_qty
            account_capital_usd = equity_usd
            base_spot = spot
            long_btc_qty = account_capital_usd / spot if spot > 0 else 0.0
            option_cash_usd = 0.0
            long_btc_pnl_usd = 0.0
            last_rebalance_period = period
            rebalance_rows.append(
                {
                    "timestamp": now.isoformat(),
                    "period": str(period),
                    "spot": spot,
                    "equity_usd": equity_usd,
                    "new_account_capital_usd": account_capital_usd,
                    "old_btc_qty": prev_qty,
                    "new_btc_qty": long_btc_qty,
                }
            )

        if call_pos is None and put_pos is None:
            snap = store.get_snapshot(now, pick="close")
            effective_delta = max(0.001, float(getattr(row, "call_delta_override", DELTA_03)))
            selected = _select_call_by_delta(snap, spot, now, EXPIRY_DAYS, effective_delta, "at_least_target")
            if selected is not None:
                selected_delta = float(selected.get("delta", np.nan))
                selected_strike = float(selected["strike_price"])
                actual_otm_value = selected_strike / spot - 1.0
                bid = float(selected.get("bid_price", np.nan))
                mark = float(selected.get("mark_price", np.nan))
                entry_price = bid if np.isfinite(bid) and bid > 0 else mark
                if np.isfinite(entry_price) and entry_price > 0:
                    call_qty = long_btc_qty * SHORT_CALL_MULT
                    put_qty = long_btc_qty * PUT_MULT
                    premium_usd = float(entry_price) * spot * call_qty
                    fee_usd = premium_usd * OPTION_FEE
                    option_cash_usd += premium_usd - fee_usd
                    call_pos = ShortCallPosition(
                        instrument_name=str(selected["instrument_name"]),
                        strike=selected_strike,
                        expiry=_to_utc(selected["expiration_date"]),
                        quantity=call_qty,
                        entry_price_btc=float(entry_price),
                        entry_spot=spot,
                        entry_premium_usd=premium_usd,
                        entry_fee_usd=fee_usd,
                        entry_time=now,
                        expiry_days=EXPIRY_DAYS,
                        otm_pct=actual_otm_value,
                        target_delta=effective_delta,
                        entry_delta=selected_delta if np.isfinite(selected_delta) else None,
                    )
                    buy_put = _should_buy_protective_put(row, spot, "below_sma_or_drawdown", DRAWDOWN_TRIGGER)
                    if buy_put:
                        put_selected = _select_put_by_delta(snap, call_pos.expiry, PUT_DELTA)
                        if put_selected is not None:
                            put_delta_value = float(put_selected.get("delta", np.nan))
                            put_strike = float(put_selected["strike_price"])
                            put_actual_otm = max(0.0, 1.0 - put_strike / spot)
                            ask = float(put_selected.get("ask_price", np.nan))
                            put_mark = float(put_selected.get("mark_price", np.nan))
                            put_entry_price = ask if np.isfinite(ask) and ask > 0 else put_mark
                            if np.isfinite(put_entry_price) and put_entry_price > 0:
                                put_premium_usd = float(put_entry_price) * spot * put_qty
                                put_fee_usd = put_premium_usd * OPTION_FEE
                                option_cash_usd -= put_premium_usd + put_fee_usd
                                put_pos = LongPutPosition(
                                    instrument_name=str(put_selected["instrument_name"]),
                                    strike=put_strike,
                                    expiry=_to_utc(put_selected["expiration_date"]),
                                    quantity=put_qty,
                                    entry_price_btc=float(put_entry_price),
                                    entry_spot=spot,
                                    entry_premium_usd=put_premium_usd,
                                    entry_fee_usd=put_fee_usd,
                                    entry_time=now,
                                    expiry_days=EXPIRY_DAYS,
                                    otm_pct=put_actual_otm,
                                    target_delta=PUT_DELTA,
                                    entry_delta=put_delta_value if np.isfinite(put_delta_value) else None,
                                )

        short_call_liability_usd = _mark_short_call(store, call_pos, now, spot) if call_pos is not None else 0.0
        long_put_asset_usd = _mark_long_put(store, put_pos, now, spot) if put_pos is not None else 0.0
        equity_usd, long_btc_pnl_usd = current_equity(account_capital_usd, base_spot, long_btc_qty, option_cash_usd, spot, short_call_liability_usd, long_put_asset_usd)
        equity_rows.append(
            {
                "timestamp": now.isoformat(),
                "spot": spot,
                "equity_usd": equity_usd,
                "original_initial_usd": original_initial_usd,
                "account_capital_usd": account_capital_usd,
                "base_spot": base_spot,
                "long_btc_qty": long_btc_qty,
                "long_btc_pnl_usd": long_btc_pnl_usd,
                "option_cash_usd": option_cash_usd,
                "short_call_liability_usd": short_call_liability_usd,
                "long_put_asset_usd": long_put_asset_usd,
                "open_call": None if call_pos is None else call_pos.instrument_name,
                "open_put": None if put_pos is None else put_pos.instrument_name,
                "vote_count": int(getattr(row, "vote_count", 0)),
                "call_delta_override": float(getattr(row, "call_delta_override", DELTA_03)),
                "rebalance_period": str(last_rebalance_period),
            }
        )

    eq = pd.DataFrame(equity_rows)
    if eq.empty:
        return {"error": "no equity rows"}, trades, eq, pd.DataFrame(rebalance_rows)
    eq["timestamp"] = pd.to_datetime(eq["timestamp"], utc=True)
    final_usd = float(eq["equity_usd"].iloc[-1])
    initial_usd = float(eq["original_initial_usd"].iloc[0])
    trades_df = pd.DataFrame(trades)
    metrics: dict[str, Any] = {
        "strategy": "7-factor vote-count layered delta, natural monthly compound only when flat",
        "note": "Month changes set a rebalance-due flag, but position size is reset only after existing option legs naturally expire and before the next new entry. No forced month-end close.",
        "initial_usd": initial_usd,
        "final_usd": final_usd,
        "total_return_usd": final_usd / initial_usd - 1.0,
        "max_drawdown_usd": max_drawdown(eq["equity_usd"]),
        "sharpe_usd": sharpe_hourly(eq["equity_usd"]),
        "rebalance_count": len(rebalance_rows),
        "delta_votes_7": DELTA_7,
        "delta_votes_6": DELTA_6,
        "delta_votes_4_5": DELTA_45,
        "delta_votes_0_3": DELTA_03,
    }
    if not trades_df.empty and "leg" in trades_df.columns:
        for leg in ["short_call", "long_put"]:
            part = trades_df[trades_df["leg"] == leg]
            metrics[f"{leg}_net_pnl_usd"] = float(part.get("pnl_usd", pd.Series(dtype=float)).sum())
            metrics[f"{leg}_premium_usd"] = float(part.get("entry_premium_usd", pd.Series(dtype=float)).sum())
            metrics[f"{leg}_payoff_usd"] = float(part.get("exit_value_usd", pd.Series(dtype=float)).sum())
    return metrics, trades, eq, pd.DataFrame(rebalance_rows)


def plot_equity(eq: pd.DataFrame, metrics: dict[str, Any]) -> None:
    first_spot = float(eq["spot"].iloc[0])
    initial = float(eq["original_initial_usd"].iloc[0])
    qty0 = initial / first_spot
    eq = eq.copy()
    eq["buy_hold_equity"] = initial + (eq["spot"].astype(float) - first_spot) * qty0
    eq["relative_to_btc"] = eq["equity_usd"] - eq["buy_hold_equity"]
    eq["strategy_drawdown"] = eq["equity_usd"] / eq["equity_usd"].cummax() - 1.0
    eq["buy_hold_drawdown"] = eq["buy_hold_equity"] / eq["buy_hold_equity"].cummax() - 1.0
    metrics["buy_hold_final_usd"] = float(eq["buy_hold_equity"].iloc[-1])
    metrics["buy_hold_return_usd"] = float(eq["buy_hold_equity"].iloc[-1] / initial - 1.0)
    metrics["buy_hold_max_drawdown_usd"] = max_drawdown(eq["buy_hold_equity"])
    metrics["relative_to_btc_final_usd"] = float(eq["relative_to_btc"].iloc[-1])

    fig: Any = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.52, 0.24, 0.24], subplot_titles=("自然月度复利策略权益 vs BTC买入持有", "相对BTC买入持有增益", "回撤曲线"))
    fig.add_trace(go.Scatter(x=eq["timestamp"], y=eq["equity_usd"], mode="lines", name="自然复利策略", line=dict(color="#22c55e", width=2.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=eq["timestamp"], y=eq["buy_hold_equity"], mode="lines", name="1x BTC持有", line=dict(color="#38bdf8", width=2.1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=eq["timestamp"], y=eq["relative_to_btc"], mode="lines", name="相对BTC增益", line=dict(color="#f59e0b", width=2.2), fill="tozeroy", fillcolor="rgba(245,158,11,.12)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=[eq["timestamp"].iloc[0], eq["timestamp"].iloc[-1]], y=[0, 0], mode="lines", name="零轴", line=dict(color="rgba(148,163,184,.65)", width=1, dash="dash"), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=eq["timestamp"], y=eq["strategy_drawdown"], mode="lines", name="策略回撤", line=dict(color="#22c55e", width=2.2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=eq["timestamp"], y=eq["buy_hold_drawdown"], mode="lines", name="BTC回撤", line=dict(color="#38bdf8", width=1.8, dash="dot")), row=3, col=1)
    summary_text = (
        f"最终权益：{fmt_usd(metrics['final_usd'])}<br>"
        f"总收益：{fmt_pct(metrics['total_return_usd'])}<br>"
        f"最大回撤：{fmt_pct(metrics['max_drawdown_usd'])}<br>"
        f"Sharpe：{metrics['sharpe_usd']:.3f}<br>"
        f"BTC持有：{fmt_usd(metrics['buy_hold_final_usd'])} / {fmt_pct(metrics['buy_hold_return_usd'])}<br>"
        f"相对BTC增益：{fmt_usd(metrics['relative_to_btc_final_usd'])}<br>"
        f"自然复利调仓：{metrics['rebalance_count']}次"
    )
    annotations = list(fig.layout["annotations"]) if fig.layout["annotations"] is not None else []
    fig.update_layout(title={"text": "BTC 7因子分层Delta策略｜自然到期后月度复利", "x": 0.5, "xanchor": "center"}, height=980, template="plotly_dark", paper_bgcolor="#0b1120", plot_bgcolor="#0f172a", font=dict(color="#e5e7eb", family="Microsoft YaHei, Arial"), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=70, r=285, t=90, b=45), annotations=annotations + [dict(text=summary_text, xref="paper", yref="paper", x=1.02, y=0.98, xanchor="left", yanchor="top", align="left", showarrow=False, bgcolor="rgba(15,23,42,0.92)", bordercolor="rgba(148,163,184,0.35)", borderwidth=1, font=dict(size=12, color="#e5e7eb"))])
    fig.update_yaxes(title_text="USD权益", row=1, col=1)
    fig.update_yaxes(title_text="USD", row=2, col=1)
    fig.update_yaxes(title_text="回撤", tickformat=".0%", row=3, col=1)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.16)")
    fig.write_html(OUT_DIR / "natural_monthly_compound_equity_curve.html", include_plotlyjs="cdn")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    daily = mod.load_daily_indicators()
    daily_votes = build_vote_count(daily)
    coverage = detect_hourly_coverage("BTC")
    start, end = _choose_date_window(coverage, "2023-04-25", "2026-04-25")
    loader = DataLoader("data")
    underlying = loader.load_underlying("BTC", resolution="60", start_date=start, end_date=end)
    store = loader.load_hourly_option_store("BTC", start_date=start, end_date=end)
    _ensure_quote_index(store)
    underlying = loader.align_underlying_to_hourly_store(underlying, store, pick="close").copy()
    underlying["timestamp"] = pd.to_datetime(underlying["timestamp"], utc=True)
    close = underlying["close"].astype(float)
    underlying["rolling_peak"] = close.rolling(24 * 30, min_periods=24).max()
    layered = attach_vote_layers(underlying, daily_votes)
    metrics, trades, eq, rebalances = run_natural_monthly_compound(layered, store)
    if eq.empty:
        raise RuntimeError("empty equity")
    plot_equity(eq, metrics)
    eq.to_csv(OUT_DIR / "natural_monthly_compound_equity.csv", index=False)
    pd.DataFrame(trades).to_csv(OUT_DIR / "natural_monthly_compound_trades.csv", index=False)
    rebalances.to_csv(OUT_DIR / "natural_monthly_compound_rebalances.csv", index=False)
    (OUT_DIR / "natural_monthly_compound_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved: {OUT_DIR.as_posix()}")


if __name__ == "__main__":
    main()
