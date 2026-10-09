"""Two BTC shorts with a persistent package stop; no live trading integration.

Hourly Deribit data approximate the supplied Bybit strategy. Thresholds are
fractions of entry premium AFTER entry fees; close fees are included in loss.
The training choice is frozen before validation/full-sample ranking.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from options_backtest.data.loader import DataLoader
from options_backtest.engine.settlement import _find_settlement_price
from scripts.research.backtest_btc_report_ic import daily_statistics


def fee(price, spot, quantity, rate, cap):
    return np.minimum(np.asarray(spot, dtype=float) * rate,
                      np.maximum(np.asarray(price, dtype=float), 0.) * cap) * quantity


def select_shorts(chain, now, target=.45, require_distinct=True):
    expiry = now.normalize() + pd.Timedelta(days=2, hours=8)
    frame = chain.loc[chain.expiration_date == expiry].copy()
    valid = np.isfinite(frame[["delta", "mark_price", "strike_price"]]).all(axis=1)
    frame = frame.loc[valid & frame.delta.abs().between(0., 1., inclusive="right")
                      & (frame.mark_price > 0) & (frame.strike_price > 0)]
    selected = []
    for kind in ("call", "put"):
        side = frame.loc[(frame.option_type == kind) & ((frame.delta > 0) if kind == "call" else (frame.delta < 0))].copy()
        if side.empty:
            return [], f"no_valid_{kind}"
        side["delta_error"] = (side.delta.abs() - target).abs()
        selected.append(side.sort_values(["delta_error", "strike_price", "instrument_name"]).iloc[0])
    if selected[1].strike_price > selected[0].strike_price or (
            require_distinct and selected[1].strike_price == selected[0].strike_price):
        return [], "overlapping_short_strikes"
    # Selection precedes liquidity checks; never switch to a more distant delta.
    for row in selected:
        if not all(np.isfinite(row[c]) for c in ("bid_price", "ask_price")) or not (
                0 < row.bid_price <= row.ask_price):
            return [], "missing_entry_quote"
    return selected, "candidate"


def choose_exit(net_credit, asks, close_fees, valid_quotes, enough_depth, threshold):
    """Index 0 is entry; final index is expiry. Never cap overshoot losses."""
    if not np.isfinite(net_credit) or net_credit <= 0:
        raise ValueError("Entry net credit must be positive")
    expiry_i = len(asks) - 1
    trigger_i = None
    missing, delayed = 0, 0
    fractions = (np.asarray(asks) + np.asarray(close_fees) - net_credit) / net_credit
    if threshold is not None:
        for i in range(1, expiry_i):
            if not valid_quotes[i]:
                missing += 1
                continue
            if trigger_i is None and fractions[i] >= threshold:
                trigger_i = i
            if trigger_i is not None:
                if enough_depth[i]:
                    return {"exit_i": i, "trigger_i": trigger_i, "stopped": True,
                            "missing_checks": missing, "insufficient_size_checks": delayed,
                            "exit_loss_fraction": float(fractions[i])}
                delayed += 1
    return {"exit_i": expiry_i, "trigger_i": trigger_i, "stopped": False,
            "missing_checks": missing, "insufficient_size_checks": delayed,
            "exit_loss_fraction": None}


@dataclass
class Week:
    entry: pd.Timestamp
    expiry: pd.Timestamp
    legs: list[dict]
    hours: pd.DatetimeIndex
    spots: np.ndarray
    bids: np.ndarray
    asks: np.ndarray
    marks: np.ndarray
    valid: np.ndarray
    depth: np.ndarray
    sizes: np.ndarray
    source_times: list[list]
    entry_prices: np.ndarray
    entry_fees: np.ndarray
    intrinsic: np.ndarray
    settlement_prices: np.ndarray
    settlement_sources: list[str]
    stale_marks: np.ndarray


def prepare(cfg, output):
    start, end = pd.to_datetime(cfg["start"], utc=True), pd.to_datetime(cfg["end"], utc=True)
    loader = DataLoader("data")
    store = loader.load_hourly_option_store("BTC", start, end)
    spot_frame = loader.load_underlying("BTC", "60", start, end)
    spot_frame.timestamp = pd.to_datetime(spot_frame.timestamp, utc=True)
    spots = spot_frame.set_index("timestamp").open
    timeline = pd.date_range(start, end, freq="h")
    if spots.reindex(timeline).isna().any():
        raise RuntimeError("Missing BTC index bars; do not invent stop prices")
    fridays = pd.date_range(start.normalize(), end.normalize(), freq="W-FRI") + pd.Timedelta(hours=cfg["entry_hour_utc"])
    fridays = fridays[(fridays >= start) & (fridays.normalize() + pd.Timedelta(days=2, hours=8) <= end)]
    cutoff = fridays[int(len(fridays) * cfg["train_fraction"])]
    candidates, eligibility, wanted = [], [], {}
    for now in fridays:
        legs, reason = select_shorts(store.get_snapshot(now, "open"), now,
                                     cfg["short_delta"], cfg["require_distinct_strikes"])
        eligibility.append({"entry_time": now, "reason": reason})
        if not legs:
            continue
        expiry = pd.Timestamp(legs[0].expiration_date)
        candidates.append((now, expiry, legs, len(eligibility) - 1))
        for hour in pd.date_range(now, expiry, freq="h", inclusive="left"):
            month = f"{hour:%Y-%m}"
            times, names = wanted.setdefault(month, (set(), set()))
            times.add(hour)
            names.update(row.instrument_name for row in legs)
    metadata = {}
    for month, (times, names) in wanted.items():
        raw = pd.read_parquet(Path("data/options_hourly/BTC") / f"{month}.parquet",
            columns=["hour", "symbol", "timestamp", "bid_amount", "ask_amount"],
            filters=[("hourly_pick", "==", "open"), ("hour", "in", sorted(times)), ("symbol", "in", sorted(names))])
        for row in raw.itertuples(index=False):
            key = (row.hour, row.symbol)
            assert key not in metadata, "Duplicate raw snapshot"
            metadata[key] = row
    settlements = loader.load_settlements("BTC")
    settlement_index = dict(zip(settlements.instrument_name, settlements.index_price))
    q = cfg["quantity"]
    weeks, entry_rows, source_rows = [], [], []
    for now, expiry, legs, reason_i in candidates:
        raw_entries = [metadata.get((now, leg.instrument_name)) for leg in legs]
        if any(r is None or not np.isfinite(r.bid_amount) or r.bid_amount < q for r in raw_entries):
            eligibility[reason_i]["reason"] = "insufficient_entry_size"
            continue
        if any(not 0 <= (pd.to_datetime(r.timestamp, unit="us", utc=True) - now).total_seconds()
               < cfg["snapshot_window_seconds"] for r in raw_entries):
            eligibility[reason_i]["reason"] = "outside_entry_window"
            continue
        hours = pd.date_range(now, expiry, freq="h")
        n = len(hours)
        bids, asks, marks = (np.full((n, 2), np.nan) for _ in range(3))
        valid, depth = np.zeros((n, 2), dtype=bool), np.zeros((n, 2), dtype=bool)
        sizes = np.full((n, 2), np.nan)
        source_times = [[None, None] for _ in hours]
        s = spots.reindex(hours).to_numpy(dtype=float)
        stale = np.zeros((n, 2), dtype=bool)
        leg_dicts = []
        for j, row in enumerate(legs):
            leg_dicts.append({"instrument_name": row.instrument_name, "kind": row.option_type,
                              "strike": float(row.strike_price), "delta": float(row.delta), "quantity": q})
            previous_mark = float(row.bid_price) * s[0]
            for i, hour in enumerate(hours[:-1]):
                bid, ask, mark = store.get_quote(row.instrument_name, hour, "open")
                bids[i, j] = bid * s[i] if bid is not None else np.nan
                asks[i, j] = ask * s[i] if ask is not None else np.nan
                if mark is not None and np.isfinite(mark) and mark > 0:
                    previous_mark = mark * s[i]
                else:
                    stale[i, j] = True
                marks[i, j] = previous_mark
                raw = metadata.get((hour, row.instrument_name))
                if raw is not None:
                    stamp = pd.to_datetime(raw.timestamp, unit="us", utc=True)
                    source_times[i][j] = stamp
                    sizes[i, j] = raw.ask_amount
                    delay = (stamp - hour).total_seconds()
                    valid[i, j] = (ask is not None and np.isfinite(ask) and ask > 0
                        and (bid is None or ask >= bid) and 0 <= delay < cfg["snapshot_window_seconds"])
                    depth[i, j] = np.isfinite(raw.ask_amount) and raw.ask_amount >= q
                    source_rows.append({"entry_time": now, "hour": hour, "instrument_name": row.instrument_name,
                        "source_time": stamp, "delay_seconds": delay, "ask_amount": raw.ask_amount,
                        "ask_usd": asks[i, j], "valid_quote": bool(valid[i, j])})
        entry_prices = bids[0].copy()
        entry_fees = fee(entry_prices, s[0], q, cfg["trading_fee_rate"], cfg["trading_fee_cap"])
        assert np.isfinite(entry_prices).all() and (entry_prices > 0).all()
        net_credit = float((entry_prices * q - entry_fees).sum())
        if net_credit <= 0:
            eligibility[reason_i]["reason"] = "nonpositive_net_credit"
            continue
        settlement_prices, sources, intrinsic = [], [], []
        for j, leg in enumerate(leg_dicts):
            price, source = settlement_index.get(leg["instrument_name"]), "instrument_record"
            if price is None or not np.isfinite(price) or price <= 0:
                price = _find_settlement_price(settlements, expiry, leg["instrument_name"])
                source = "date_record"
            if price is None or not np.isfinite(price) or price <= 0:
                price, source = s[-1], "expiry_hour_open"
            assert np.isfinite(price) and price > 0
            value = max(0., float(price) - leg["strike"]) if leg["kind"] == "call" else max(0., leg["strike"] - float(price))
            intrinsic.append(value)
            settlement_prices.append(float(price))
            sources.append(source)
            entry_rows.append({"entry_time": now, "expiry": expiry, **leg, "entry_price_usd": entry_prices[j],
                "entry_fee_usd": entry_fees[j], "spot_usd": s[0], "source_time": source_times[0][j],
                "bid_amount": raw_entries[j].bid_amount})
        eligibility[reason_i]["reason"] = "entered"
        eligibility[reason_i]["net_entry_credit_usd"] = net_credit
        weeks.append(Week(now, expiry, leg_dicts, hours, s, bids, asks, marks, valid, depth, sizes,
                          source_times, entry_prices, entry_fees, np.array(intrinsic),
                          np.array(settlement_prices), sources, stale))
    pd.DataFrame(eligibility).to_csv(output / "entry_eligibility.csv", index=False)
    pd.DataFrame(entry_rows).to_csv(output / "entry_legs.csv", index=False)
    pd.DataFrame(source_rows).to_csv(output / "quote_audit.csv", index=False)
    assert weeks and all(a.expiry < b.entry for a, b in zip(weeks, weeks[1:]))
    return weeks, timeline, cutoff, pd.DataFrame(eligibility)


def run_variant(weeks, timeline, cfg, threshold, delivery_rate=None):
    q = cfg["quantity"]
    rate = cfg["delivery_fee_rate"] if delivery_rate is None else delivery_rate
    changes = np.zeros(len(timeline))
    packages, legs_out = [], []
    for week in weeks:
        credit = float((week.entry_prices * q - week.entry_fees).sum())
        close_fees = fee(week.asks, week.spots[:, None], q, cfg["trading_fee_rate"], cfg["trading_fee_cap"])
        costs = week.asks.sum(axis=1) * q
        result = choose_exit(credit, costs, close_fees.sum(axis=1),
                             week.valid.all(axis=1), week.depth.all(axis=1), threshold)
        i, trigger_i = result["exit_i"], result["trigger_i"]
        if result["stopped"]:
            exit_prices, exit_fees = week.asks[i].copy(), close_fees[i].copy()
            close_type = "stop"
            assert week.valid[i].all() and week.depth[i].all()
        else:
            exit_prices = week.intrinsic.copy()
            exit_fees = fee(exit_prices, week.settlement_prices, q, rate, cfg["delivery_fee_cap"])
            close_type = "settlement"
        pnl = (week.entry_prices - exit_prices) * q - week.entry_fees - exit_fees
        net = float(pnl.sum())
        path = credit - week.marks[:i + 1].sum(axis=1) * q
        path[-1] = net
        indexes = timeline.get_indexer(week.hours[:i + 1])
        assert (indexes >= 0).all()
        changes[indexes] += np.diff(np.r_[0., path])
        stopped = bool(result["stopped"])
        row = {"entry_time": week.entry, "exit_time": week.hours[i], "net_pnl_usd": net,
            "net_entry_credit_usd": credit, "entry_fees_usd": float(week.entry_fees.sum()),
            "close_fees_usd": float(exit_fees.sum()) if stopped else 0.,
            "delivery_fees_usd": 0. if stopped else float(exit_fees.sum()), "close_type": close_type,
            "first_trigger_time": week.hours[trigger_i] if trigger_i is not None else pd.NaT,
            "trigger_delay_hours": i - trigger_i if trigger_i is not None else 0,
            "exit_loss_fraction": -net / credit,
            "overshoot_fraction": max(0., -net / credit - threshold) if stopped else 0.,
            "unfilled_stop_at_expiry": trigger_i is not None and not stopped,
            "missing_checks": result["missing_checks"], "insufficient_size_checks": result["insufficient_size_checks"],
            "stale_mark_observations": int(week.stale_marks[:i].sum())}
        packages.append(row)
        for j, leg in enumerate(week.legs):
            legs_out.append({"entry_time": week.entry, "exit_time": week.hours[i], **leg,
                "entry_price_usd": week.entry_prices[j], "exit_price_usd": exit_prices[j],
                "entry_fee_usd": week.entry_fees[j], "exit_fee_usd": exit_fees[j],
                "net_pnl_usd": pnl[j], "close_type": close_type,
                "exit_ask_amount": week.sizes[i, j] if stopped else np.nan,
                "exit_quote_time": week.source_times[i][j] if stopped else pd.NaT,
                "settlement_source": "" if stopped else week.settlement_sources[j]})
    packages, legs = pd.DataFrame(packages), pd.DataFrame(legs_out)
    curve = pd.Series(changes.cumsum(), index=timeline, name="cumulative_pnl_usd")
    assert len(legs) == len(packages) * 2
    assert abs(curve.iloc[-1] - packages.net_pnl_usd.sum()) < 1e-6
    assert abs(legs.net_pnl_usd.sum() - packages.net_pnl_usd.sum()) < 1e-6
    # Independent cashflow ledger: sell proceeds, entry fees, buy/settlement outflows.
    cash = (legs.entry_price_usd * q - legs.entry_fee_usd - legs.exit_price_usd * q - legs.exit_fee_usd).sum()
    assert abs(cash - curve.iloc[-1]) < 1e-6
    return packages, legs, curve


def describe(packages, curve):
    drawdown = curve - curve.cummax().clip(lower=0.)
    sessions = pd.date_range(curve.index.min().normalize() + pd.Timedelta(hours=21), curve.index.max(), freq="D")
    daily_equity = curve.reindex(curve.index.union(sessions)).sort_index().ffill().reindex(sessions)
    daily = daily_equity.diff()
    daily.iloc[0] = daily_equity.iloc[0]
    # The reporting end is flat after the final complete Sunday.
    assert abs(daily.sum() - packages.net_pnl_usd.sum()) < 1e-6
    stats = daily_statistics(daily)
    stop = packages.close_type.eq("stop")
    stats.update(packages=len(packages), stopped_packages=int(stop.sum()),
        hourly_max_drawdown_usd=float(drawdown.min()), worst_package_usd=float(packages.net_pnl_usd.min()),
        package_win_rate=float(packages.net_pnl_usd.gt(0).mean()),
        delayed_stops=int((stop & packages.trigger_delay_hours.gt(0)).sum()),
        unfilled_stop_at_expiry=int(packages.unfilled_stop_at_expiry.sum()),
        missing_checks=int(packages.missing_checks.sum()),
        insufficient_size_checks=int(packages.insufficient_size_checks.sum()),
        max_stop_loss_fraction=float(packages.loc[stop, "exit_loss_fraction"].max()) if stop.any() else None,
        max_overshoot_fraction=float(packages.loc[stop, "overshoot_fraction"].max()) if stop.any() else None)
    return stats


def rank(table):
    # Highest net profit, then smaller drawdown, then tighter threshold.
    return table.sort_values(["total_pnl_usd", "hourly_max_drawdown_usd", "threshold_pct"],
                             ascending=[False, False, True], na_position="last")


def write_json(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {k: clean(v) for k, v in item.items()}
        if isinstance(item, list):
            return [clean(v) for v in item]
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")


def render_report(output):
    """Render audited CSV results without repeating market loading or simulation."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    full = pd.read_csv(output / "full_sample_ranking.csv")
    train = pd.read_csv(output / "training_ranking.csv").set_index("variant")
    validation = pd.read_csv(output / "validation_results.csv").set_index("variant")
    cfg = json.loads((output / "effective_config.json").read_text(encoding="utf-8"))
    findings = json.loads((output / "findings.json").read_text(encoding="utf-8"))
    baseline = full.set_index("variant").loc["no_stop"]
    best = findings["full_sample_best_stop"]["variant"]
    grid = full.loc[full.variant != "no_stop"].sort_values("threshold_pct")
    eligibility = pd.read_csv(output / "entry_eligibility.csv")
    audit = pd.read_csv(output / "quote_audit.csv")
    baseline_legs = pd.read_csv(output / "no_stop" / "legs.csv")
    baseline_packages = pd.read_csv(output / "no_stop" / "packages.csv")
    selected = findings["training_selection"]
    tied = grid.loc[np.isclose(grid.total_pnl_usd, findings["full_sample_best_stop"]["total_pnl_usd"], rtol=0, atol=1e-7)]
    tied_text = "、".join(f"{x:g}%" for x in tied.threshold_pct)
    winner_text = "不止损" if full.iloc[0].variant == "no_stop" else full.iloc[0].variant.replace("stop_", "止损 ") + "%"
    train_text = "不止损" if selected["training_winner"]["variant"] == "no_stop" else selected["training_winner"]["variant"].replace("stop_", "止损 ") + "%"
    fallback_count = int(baseline_legs.settlement_source.eq("expiry_hour_open").sum())
    missing_count = int((~audit.valid_quote).sum())
    stale_count = int(baseline_packages.stale_mark_observations.sum())
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.12,
                        subplot_titles=("费后累计收益 · USD", "小时最大回撤幅度 · USD（越低越小）"))
    for row, column, multiplier, color in [(1, "total_pnl_usd", 1, "#166b88"),
                                          (2, "hourly_max_drawdown_usd", -1, "#a95724")]:
        fig.add_trace(go.Scatter(x=grid.threshold_pct.tolist(), y=(grid[column] * multiplier).tolist(),
            mode="lines+markers", name="止损组", showlegend=False, line=dict(color=color),
            hovertemplate="阈值 %{x}%<br>%{y:,.2f} USD<extra></extra>"), row=row, col=1)
        fig.add_hline(y=baseline[column] * multiplier, line_dash="dot", line_color="#58676d",
                      annotation_text="不止损", row=row, col=1)
    fig.update_xaxes(title_text="止损阈值：两腿费后亏损 / 开仓净收权利金（%）", row=2, col=1)
    fig.update_xaxes(range=[0, 400])
    fig.update_yaxes(tickformat=",.0f", zeroline=True, zerolinecolor="#bac2c4")
    fig.update_layout(height=660, template="plotly_white", margin=dict(l=70, r=35, t=70, b=80),
        font=dict(family="Segoe UI, Microsoft YaHei, sans-serif"),
        updatemenus=[dict(type="buttons", direction="right", x=0, y=1.13,
            buttons=[dict(label="0–400%", method="relayout", args=[{"xaxis.range": [0, 400], "xaxis2.range": [0, 400]}]),
                     dict(label="全部阈值", method="relayout", args=[{"xaxis.range": [0, 1000], "xaxis2.range": [0, 1000]}])])])

    tight = full.sort_values("hourly_max_drawdown_usd", ascending=False).iloc[0].variant
    chosen = list(dict.fromkeys(["no_stop", best, "stop_100", tight]))
    colors = ["#166b88", "#a95724", "#7855a1", "#518743"]
    equity = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.12,
                           subplot_titles=("累计费后损益 · USD", "小时回撤 · USD"))
    for label, color in zip(chosen, colors):
        data = pd.read_csv(output / label / "hourly_equity.csv")
        name = "不止损" if label == "no_stop" else label.replace("stop_", "止损 ") + "%"
        values = data[["cumulative_pnl_usd", "drawdown_usd"]]
        keep = values.ne(values.shift()).any(axis=1) | values.ne(values.shift(-1)).any(axis=1)
        shown = data.loc[keep]
        for row, column in [(1, "cumulative_pnl_usd"), (2, "drawdown_usd")]:
            equity.add_trace(go.Scatter(x=shown.timestamp.tolist(), y=shown[column].tolist(),
                mode="lines", name=name, legendgroup=label, showlegend=row == 1,
                line=dict(color=color, width=1.6), connectgaps=False,
                hovertemplate="%{x}<br>%{y:,.2f} USD<extra>" + name + "</extra>"), row=row, col=1)
    equity.add_vline(x=findings["training_selection"]["cutoff"])
    equity.update_xaxes(title_text="时间（UTC）；竖线后为验证期", row=2, col=1)
    equity.update_yaxes(tickformat=",.0f")
    equity.update_layout(height=680, template="plotly_white", margin=dict(l=70, r=30, t=65, b=70),
                          legend=dict(orientation="h", y=1.11), font=dict(family="Segoe UI, Microsoft YaHei, sans-serif"))
    view = full[["variant", "threshold_pct", "total_pnl_usd", "hourly_max_drawdown_usd",
                 "daily_pnl_sharpe_365", "stopped_packages", "delayed_stops"]].copy()
    view["train_pnl"] = view.variant.map(train.total_pnl_usd)
    view["validation_pnl"] = view.variant.map(validation.total_pnl_usd)
    view["threshold_pct"] = view.threshold_pct.map(lambda x: "不止损" if pd.isna(x) else f"{x:g}%")
    view = view.drop(columns="variant").rename(columns={"threshold_pct": "阈值", "total_pnl_usd": "全期净收益 USD",
        "hourly_max_drawdown_usd": "小时最大回撤 USD", "daily_pnl_sharpe_365": "日损益夏普",
        "stopped_packages": "止损次数", "delayed_stops": "延迟成交次数", "train_pnl": "训练期净收益 USD",
        "validation_pnl": "验证期净收益 USD"})
    table = view.to_html(index=False, float_format=lambda x: f"{x:,.2f}", border=0)
    config = dict(responsive=True, displaylogo=False, scrollZoom=False)
    html = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>BTC 双卖：止损阈值回测</title>
<style>body{{font:16px/1.65 "Segoe UI","Microsoft YaHei",sans-serif;color:#25373d;background:#f1f4f3;margin:0}}
main{{max-width:1160px;margin:auto;padding:32px 20px}}h1{{font-size:30px;margin:0 0 10px}}h2{{font-size:21px}}
.card{{background:white;border:1px solid #dce3e2;border-radius:10px;padding:20px;margin:22px 0}}
.lead{{font-size:20px}}.muted{{color:#54676f}}.table{{overflow:auto}}table{{border-collapse:collapse;white-space:nowrap;width:100%}}
th,td{{padding:8px 12px;text-align:right;border-bottom:1px solid #e1e7e5}}th{{background:#e9efed;position:sticky;top:0}}a{{color:#166b88}}
@media(max-width:600px){{main{{padding:18px 8px}}.card{{padding:12px}}h1{{font-size:24px}}}}</style></head><body><main>
<p class="muted">BTC · 两腿裸卖 · 固定每腿 {cfg['quantity']:g} BTC · 所有金额为 USD · 时间为 UTC</p>
<h1>BTC 双卖：止损阈值与收益</h1>
<p class="lead">不止损净赚 {baseline.total_pnl_usd:,.0f} 美元；收益最高的止损组净赚 {findings['full_sample_best_stop']['total_pnl_usd']:,.0f} 美元。</p>
<p>数据期 {cfg['start'][:10]} 至 {cfg['end'][:10]}，{len(eligibility)} 个计划周末中 {len(baseline_packages)} 组共同开仓。按全期净收益，最佳方案为“{winner_text}”。收益最高的止损阈值为 {tied_text}；这是测试网格内的样本结果。</p>
<div class="card"><h2>阈值与收益、回撤</h2><p>虚线为不止损对照。默认显示 0–400%，可切换全部阈值；悬停查看数值，拖动缩放。</p>
{fig.to_html(full_html=False, include_plotlyjs=True, config=config, div_id='threshold-chart')}</div>
<div class="card"><h2>收益路径与时间顺序验证</h2>
<p>前 {cfg['train_fraction']:.0%} 计划周末用于选参，共 {selected['train_packages']} 组；{selected['cutoff'][:10]} 起验证，共 {selected['validation_packages']} 组。训练期选出“{train_text}”，验证期净收益 {findings['training_choice_validation']['total_pnl_usd']:,.0f} 美元。</p>
{equity.to_html(full_html=False, include_plotlyjs=False, config=config, div_id='equity-chart')}</div>
<div class="card"><h2>成交口径与数据限制</h2><p>净收权利金 C 已扣开仓费；若两腿 ask 回购成本加平仓费 ≥ C × (1 + 阈值)，触发整组止损。从周五 22:00 至周日 07:00 每小时检查。触发后锁定，直到两腿均有足够 ask 数量才成交；不能成交则继续持仓至到期。</p>
<p>使用 Deribit 小时盘口近似给定 Bybit 策略；不复刻秒级 BBO 挂单。入场按 bid、平仓按 ask，每腿盘口数量至少 {cfg['quantity']:g} BTC；固定数量、不复利、不对冲，不模拟保证金或强平。</p>
<p>本结果对极宽价差、盘口数量及触发后的止损锁定规则敏感。盘口不足时记录触发，后续报价恢复后也会平仓；详细触发时间与延迟保存在各阈值的 packages.csv。</p>
<p><a href="https://www.bybit.com/en/help-center/article/Bybit-Option-Fees-Explained">Bybit 费率假设</a>：taker {cfg['trading_fee_rate']:.3%}，每腿不超过权利金 {cfg['trading_fee_cap']:.0%}；实值交割费 {cfg['delivery_fee_rate']:.3%}，不超过内在价值 {cfg['delivery_fee_cap']:.1%}。按当前费率统一估算，非历史费率复原；另存免交割费敏感性分析。</p>
<p>{fallback_count} 条到期腿缺少交割记录，使用到期小时现货开盘价；{missing_count} 次腿报价不可用，无法在该小时判定止损。沿用旧 mark 的观测数为 {stale_count}。小时检查无法观察小时内穿越；阈值不等于实际最大亏损。</p></div>
<div class="card"><h2>全部 {len(full)} 组结果</h2><p>按全期净收益排序。日损益夏普按每日 21:00 UTC、365 日年化，保留空仓日；它不是保证金收益率。</p>
<div class="table">{table}</div></div></main></body></html>'''
    (output / "comparison.html").write_text(html, encoding="utf-8")
    write_json(output / "chart_data.json", {"thresholds": json.loads(grid.to_json(orient="records")),
        "curve_variants": chosen, "currency": "USD", "timezone": "UTC", "sampling": "hourly_open"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/backtest/btc_naked_strangle_stop_sweep.yaml")
    parser.add_argument("--output-dir", default="reports/btc_naked_strangle_stop_sweep")
    parser.add_argument("--report-only", action="store_true", help="Reuse completed CSVs and regenerate HTML only")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        for filename in ("findings.json", "frozen_selection.json"):
            path = output / filename
            write_json(path, json.loads(path.read_text(encoding="utf-8")))
        render_report(output)
        return
    write_json(output / "effective_config.json", cfg)
    logger.remove()
    logger.add(str(output / "run.log"), level="INFO")
    started = perf_counter()
    weeks, timeline, cutoff, eligibility = prepare(cfg, output)
    print(f"Prepared {len(weeks)} packages; holdout starts {cutoff}; "
          f"entry reasons={eligibility.reason.value_counts().to_dict()}", flush=True)
    thresholds = [None, *(x / 100 for x in cfg["thresholds_pct"])]
    labels = ["no_stop", *(f"stop_{x:g}" for x in cfg["thresholds_pct"])]
    training_weeks = [w for w in weeks if w.entry < cutoff]
    training_times = timeline[timeline < cutoff.normalize()]
    training = []
    for value, label in zip(thresholds, labels):
        packs, _, curve = run_variant(training_weeks, training_times, cfg, value)
        training.append({"variant": label, "threshold_pct": None if value is None else value * 100,
                         **describe(packs, curve)})
    train = rank(pd.DataFrame(training))
    train.to_csv(output / "training_ranking.csv", index=False)
    selected = train.iloc[0].to_dict()
    selection = {"rule": "training_net_pnl_then_drawdown_then_lower_threshold", "cutoff": str(cutoff),
        "train_packages": len(training_weeks), "validation_packages": len(weeks) - len(training_weeks),
        "training_winner": selected, "frozen_before_validation": True}
    write_json(output / "frozen_selection.json", selection)
    print(f"Frozen training selection: {selected['variant']}", flush=True)
    all_rows, validation, yearly, sensitivity = [], [], [], []
    for value, label in zip(thresholds, labels):
        packs, legs, curve = run_variant(weeks, timeline, cfg, value)
        folder = output / label
        folder.mkdir(exist_ok=True)
        packs.to_csv(folder / "packages.csv", index=False)
        legs.to_csv(folder / "legs.csv", index=False)
        pd.DataFrame({"cumulative_pnl_usd": curve, "drawdown_usd": curve - curve.cummax().clip(lower=0.)}).to_csv(folder / "hourly_equity.csv", index_label="timestamp")
        base = {"variant": label, "threshold_pct": None if value is None else value * 100}
        all_rows.append({**base, **describe(packs, curve)})
        for year in sorted(set(timeline.year)):
            sub = packs.loc[packs.entry_time.dt.year == year]
            yearly.append({**base, "year": int(year), "packages": len(sub), "total_pnl_usd": float(sub.net_pnl_usd.sum())})
        validation_weeks = [w for w in weeks if w.entry >= cutoff]
        vp, _, vc = run_variant(validation_weeks, timeline[timeline >= cutoff.normalize()], cfg, value)
        validation.append({**base, **describe(vp, vc)})
        # Daily-option delivery-fee exemption sensitivity; same trigger/entry policy.
        free_packs, _, free_curve = run_variant(weeks, timeline, cfg, value, delivery_rate=0.)
        sensitivity.append({**base, **describe(free_packs, free_curve)})
    full = rank(pd.DataFrame(all_rows))
    full.to_csv(output / "full_sample_ranking.csv", index=False)
    pd.DataFrame(validation).to_csv(output / "validation_results.csv", index=False)
    pd.DataFrame(yearly).to_csv(output / "yearly_results.csv", index=False)
    rank(pd.DataFrame(sensitivity)).to_csv(output / "no_delivery_fee_ranking.csv", index=False)
    findings = {"training_selection": selection, "full_sample_winner": full.iloc[0].to_dict(),
        "full_sample_best_stop": full.loc[full.variant != "no_stop"].iloc[0].to_dict(),
        "training_choice_validation": next(r for r in validation if r["variant"] == selected["variant"]),
        "no_stop_validation": next(r for r in validation if r["variant"] == "no_stop"),
        "elapsed_seconds": perf_counter() - started}
    write_json(output / "findings.json", findings)
    render_report(output)
    print(json.dumps(findings, indent=2), flush=True)


if __name__ == "__main__":
    main()
