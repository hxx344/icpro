"""Replace BTC protection with equal entry-notional ETH options on matched weeks."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from loguru import logger
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from options_backtest.data.loader import DataLoader
from options_backtest.data.models import Direction, OrderRequest
from options_backtest.engine.matcher import Matcher
from options_backtest.engine.position import PositionManager
from options_backtest.engine.settlement import _find_settlement_price
from scripts.research.backtest_btc_report_ic import Config, CashMarketValueAccount, daily_statistics
from scripts.research.sweep_sunday_premium_take_profit import evaluate_premium_target


def asset(name):
    return name.split("-", 1)[0]


def equal_notional_quantity(btc_price, eth_price, btc_quantity=1.):
    if not all(np.isfinite(v) and v > 0 for v in (btc_price, eth_price, btc_quantity)):
        raise ValueError("Positive contemporaneous BTC/ETH prices and quantity required")
    return float(btc_quantity) * float(btc_price) / float(eth_price)


def select_eth_wings(chain, expiry, spot, delta=.10):
    if chain.empty:
        return [], "missing_eth_chain"
    frame = chain.loc[chain.expiration_date == expiry].copy()
    cols = ["delta", "strike_price", "bid_price", "ask_price", "mark_price"]
    valid = np.isfinite(frame[cols]).all(axis=1)
    frame = frame.loc[valid & (frame.bid_price > 0) & (frame.ask_price >= frame.bid_price)
                      & (frame.mark_price > 0)]
    wings = []
    for kind in ("call", "put"):
        side = frame.loc[frame.option_type.str.startswith(kind[0])]
        side = side.loc[((side.strike_price > spot) & (side.delta > 0)) if kind == "call"
                        else ((side.strike_price < spot) & (side.delta < 0))].copy()
        if side.empty:
            return [], f"missing_eth_{kind}_wing"
        side["delta_error"] = (side.delta.abs() - delta).abs()
        wings.append(side.sort_values(["delta_error", "instrument_name"]).iloc[0])
    return wings, "eligible"


class Market:
    def __init__(self, cfg, entry_times):
        loader = DataLoader("data")
        start, end = cfg.backtest.start_date, cfg.backtest.end_date
        self.stores, self.spots, self.settlements, self.settlement_index = {}, {}, {}, {}
        for coin in ("BTC", "ETH"):
            store = loader.load_hourly_option_store(coin, start, end)
            self.stores[coin] = store
            spot = loader.load_underlying(coin, "60", start, end)
            spot.timestamp = pd.to_datetime(spot.timestamp, utc=True)
            self.spots[coin] = spot.set_index("timestamp").open.to_dict()
            if coin == "BTC":
                aligned = loader.align_underlying_to_hourly_store(spot, store, "open")
                self.timeline = pd.DatetimeIndex(aligned.timestamp)
            frame = loader.load_settlements(coin)
            self.settlements[coin] = frame
            self.settlement_index[coin] = dict(zip(frame.instrument_name, frame.index_price))
        hours = set(entry_times)
        for entry in entry_times:
            sunday = entry.normalize() + pd.Timedelta(days=2)
            hours.update(pd.date_range(sunday, sunday + pd.Timedelta(hours=7), freq="h"))
        self.spot_audit = []
        for coin in ("BTC", "ETH"):
            missing = [now for now in self.timeline
                       if not np.isfinite(self.spot(coin, now)) or self.spot(coin, now) <= 0]
            for month in sorted({f"{now:%Y-%m}" for now in missing}):
                times = [now for now in missing if f"{now:%Y-%m}" == month]
                path = Path("data/options_hourly") / coin / f"{month}.parquet"
                raw = pd.read_parquet(path, columns=["hour", "underlying_price"],
                    filters=[("hourly_pick", "==", "open"), ("hour", "in", times)])
                raw = raw.loc[np.isfinite(raw.underlying_price) & (raw.underlying_price > 0)]
                for now, price in raw.groupby("hour").underlying_price.median().items():
                    self.spots[coin][now] = float(price)
                    self.spot_audit.append({"asset": coin, "timestamp": now, "price": float(price),
                                           "source": "raw_same_hour_open_index_median"})
        self.metadata = {}
        by_month = {}
        for now in sorted(hours):
            by_month.setdefault(f"{now:%Y-%m}", []).append(now)
        for coin in ("BTC", "ETH"):
            for month, times in by_month.items():
                path = Path("data/options_hourly") / coin / f"{month}.parquet"
                frame = pd.read_parquet(path, columns=["hour", "symbol", "timestamp", "bid_amount", "ask_amount"],
                    filters=[("hourly_pick", "==", "open"), ("hour", "in", times)])
                for now, group in frame.groupby("hour", sort=False):
                    group = group.set_index("symbol")
                    assert group.index.is_unique, f"Duplicate {coin} snapshot {now}"
                    self.metadata[(coin, now)] = group

    def spot(self, coin, now):
        return self.spots[coin].get(now, np.nan)

    @lru_cache(maxsize=200000)
    def quote(self, name, now):
        coin = asset(name)
        spot = self.spot(coin, now)
        if not np.isfinite(spot) or spot <= 0:
            return None, None, None
        quote = self.stores[coin].get_quote(name, now, "open")
        return tuple(v * spot if v is not None else None for v in quote)

    def depth(self, name, now, side, quantity):
        frame = self.metadata.get((asset(name), now))
        if frame is None or name not in frame.index:
            return None
        row = frame.loc[name]
        size = row[f"{side}_amount"]
        if not np.isfinite(size) or size < quantity:
            return None
        timestamp = pd.to_datetime(row.timestamp, unit="us", utc=True)
        delay = (timestamp - now).total_seconds()
        assert 0 <= delay < 3600, "Quote outside selected hour"
        return {"source_time": timestamp, "visible_size": size, "snapshot_delay_seconds": delay}

    def settlement(self, name, expiry, now):
        coin = asset(name)
        price = self.settlement_index[coin].get(name)
        source = "instrument_record"
        if price is None:
            price = _find_settlement_price(self.settlements[coin], expiry, name)
            source = "date_record"
        if price is None:
            price, source = self.spot(coin, now), "same_asset_hour_open"
        assert np.isfinite(price) and price > 0, f"Missing {coin} settlement"
        return float(price), source


def make_plans(baseline, market):
    plans, hybrids, eligibility = {}, {}, []
    for entry, group in baseline.groupby("entry_time", sort=True):
        group = group.set_index("leg").loc[["SC", "SP", "LC", "LP"]].reset_index()
        plan = group[["instrument_name", "leg", "quantity", "strike", "expiry", "delta"]].to_dict("records")
        for leg in plan:
            leg["entry_time"] = entry
        plans[entry] = plan
        btc, eth = market.spot("BTC", entry), market.spot("ETH", entry)
        if not np.isfinite(eth) or eth <= 0:
            eligibility.append({"entry_time": entry, "reason": "missing_eth_spot"})
            continue
        quantity = equal_notional_quantity(btc, eth)
        wings, reason = select_eth_wings(market.stores["ETH"].get_snapshot(entry, "open"),
                                         plan[0]["expiry"], eth)
        if wings:
            for row in wings:
                if market.depth(row.instrument_name, entry, "ask", quantity) is None:
                    reason = "insufficient_eth_wing_size"
                    break
        if reason == "eligible":
            hybrid = copy.deepcopy(plan[:2])
            for row, label in zip(wings, ["LC", "LP"]):
                hybrid.append({"instrument_name": row.instrument_name, "leg": label,
                    "quantity": quantity, "strike": float(row.strike_price), "expiry": row.expiration_date,
                    "delta": float(row.delta), "entry_time": entry})
            hybrids[entry] = hybrid
        eligibility.append({"entry_time": entry, "reason": reason,
            "btc_price": btc, "eth_price": eth, "eth_quantity_per_wing": quantity,
            "btc_notional_per_leg": btc, "eth_notional_per_wing": quantity * eth})
    return plans, hybrids, pd.DataFrame(eligibility)


def execute(legs, now, market, pm, account, matcher, closing=False):
    prepared = []
    for leg in legs:
        name, qty = leg["instrument_name"], leg["quantity"]
        is_buy = leg["leg"].startswith("L") != closing
        side = "ask" if is_buy else "bid"
        quote = market.quote(name, now)
        assert quote[2] is not None and quote[2] > 0
        depth = market.depth(name, now, side, qty)
        assert depth is not None, "Insufficient executable package size"
        order = OrderRequest(instrument_name=name, quantity=qty,
                             direction=Direction.LONG if is_buy else Direction.SHORT)
        fill = matcher.execute(order, now, *quote, market.spot(asset(name), now))
        assert fill is not None
        prepared.append((leg, fill, side, depth))
    audit = []
    for leg, fill, side, depth in prepared:
        pm.apply_fill(fill)
        account.pay_fee(fill.fee)
        flow = fill.fill_price * fill.quantity
        account.withdraw(flow) if side == "ask" else account.deposit(flow)
        audit.append({**leg, "fill_time": now, "fill_price": fill.fill_price,
                      "fill_fee": fill.fee, "side": side, "closing": closing,
                      "underlying_price": fill.underlying_price, **depth})
    return audit


def simulate(plans, market, cfg, target, output):
    output.mkdir(parents=True, exist_ok=True)
    pm = PositionManager()
    account = CashMarketValueAccount(10000., pm.positions)
    matcher = Matcher(cfg.execution, margin_usd=True)
    fills, checks, exits, settlements, stale = [], [], [], [], []
    current = None
    for now in market.timeline:
        if pm.positions:
            if now >= current[0]["expiry"]:
                for leg in current:
                    name = leg["instrument_name"]
                    price, source = market.settlement(name, leg["expiry"], now)
                    # Parquet may hold float32 strikes; do all USD arithmetic in float64.
                    pnl = pm.settle_expired(name, float(price), float(leg["strike"]),
                        "call" if leg["leg"].endswith("C") else "put", now,
                        delivery_fee_per_qty=cfg.execution.delivery_fee * price,
                        delivery_fee_max_pct=cfg.execution.delivery_fee_max_pct, margin_usd=True)
                    account.balance += pnl
                    settlements.append({"instrument_name": name, "entry_time": leg["entry_time"],
                        "exit_time": now, "asset": asset(name), "settlement_price": price, "source": source})
            else:
                for name, position in pm.positions.items():
                    mark = market.quote(name, now)[2]
                    if mark is not None and np.isfinite(mark) and mark > 0:
                        position.update_mark(mark)
                    else:
                        stale.append({"timestamp": now, "instrument_name": name})
                if target is not None and now.weekday() == 6:
                    result = evaluate_premium_target(pm.positions,
                        {name: market.quote(name, now) for name in pm.positions}, target)
                    if result["reason"] == "target_reached":
                        if any(market.depth(name, now, "ask" if pos.direction_sign < 0 else "bid", pos.quantity)
                               is None for name, pos in pm.positions.items()):
                            result["reason"] = "insufficient_close_size"
                        else:
                            fills.extend(execute(current, now, market, pm, account, matcher, closing=True))
                            exits.append({"entry_time": current[0]["entry_time"], "exit_time": now, **result})
                    checks.append({"timestamp": now, "entry_time": current[0]["entry_time"], **result})
        if now in plans:
            assert not pm.positions, "Overlapping packages"
            current = plans[now]
            fills.extend(execute(current, now, market, pm, account, matcher))
            for name, pos in pm.positions.items():
                pos.update_mark(market.quote(name, now)[2])
        account.record_equity(now, underlying_price=market.spot("BTC", now))
    assert not pm.positions, "Incomplete last package"
    equity = pd.DataFrame(account._equity_history,
        columns=["timestamp", "equity_usd", "cash_usd", "position_value_usd", "btc_spot"]).set_index("timestamp")
    equity["cumulative_pnl_usd"] = equity.equity_usd - 10000.
    equity["drawdown_usd"] = equity.cumulative_pnl_usd - equity.cumulative_pnl_usd.cummax().clip(lower=0)
    trades = pd.DataFrame(pm.closed_trades)
    assert not trades.empty, "No executable matched packages"
    audit = pd.DataFrame(fills)
    entries = audit.loc[~audit.closing, ["entry_time", "instrument_name", "fill_fee", "leg", "delta"]]
    trades = trades.merge(entries.rename(columns={"fill_fee": "entry_fee"}),
                          on=["entry_time", "instrument_name"], validate="one_to_one")
    trades["close_fee"] = trades.fee.where(trades.close_type == "trade", 0.)
    trades["delivery_fee"] = trades.fee.where(trades.close_type == "settlement", 0.)
    trades["net_pnl_usd"] = trades.pnl - trades.entry_fee - trades.close_fee
    trades["gross_pnl_usd"] = trades.net_pnl_usd + trades.entry_fee + trades.fee
    packages = trades.groupby("entry_time").agg(exit_time=("exit_time", "max"), legs=("leg", "size"),
        net_pnl_usd=("net_pnl_usd", "sum"), entry_fees_usd=("entry_fee", "sum"),
        close_fees_usd=("close_fee", "sum"), settlement_fees_usd=("delivery_fee", "sum"),
        gross_pnl_usd=("gross_pnl_usd", "sum"))
    assert packages.legs.eq(4).all() and len(packages) == len(plans)
    assert trades.groupby("entry_time").exit_time.nunique().eq(1).all()
    error = packages.net_pnl_usd.sum() - equity.cumulative_pnl_usd.iloc[-1]
    assert abs(error) < 1e-6, f"Cash/ledger mismatch: {error}"
    sessions = pd.date_range(equity.index.min().normalize() + pd.Timedelta(hours=21), equity.index.max(), freq="D")
    daily_equity = equity.equity_usd.reindex(equity.index.union(sessions)).sort_index().ffill().reindex(sessions)
    daily = daily_equity.diff()
    daily.iloc[0] = daily_equity.iloc[0] - 10000.
    assert abs(daily.sum() - packages.net_pnl_usd.sum()) < 1e-6
    stats = {**daily_statistics(daily), "packages": len(packages), "early_close_packages": len(exits),
        "hourly_max_drawdown_usd": equity.drawdown_usd.min(),
        "package_win_rate": packages.net_pnl_usd.gt(0).mean(),
        "worst_package_usd": packages.net_pnl_usd.min(), "best_package_usd": packages.net_pnl_usd.max(),
        "entry_fees_usd": packages.entry_fees_usd.sum(), "close_fees_usd": packages.close_fees_usd.sum(),
        "settlement_fees_usd": packages.settlement_fees_usd.sum(), "stale_mark_observations": len(stale),
        "ledger_error_usd": error,
        "exit_check_reasons": pd.Series([e["reason"] for e in checks]).value_counts().to_dict()}
    yearly = [{"year": int(year), **daily_statistics(values)} for year, values in daily.groupby(daily.index.year)]
    equity.to_csv(output / "hourly_equity.csv")
    trades.to_csv(output / "legs.csv", index=False)
    packages.to_csv(output / "packages.csv")
    audit.to_csv(output / "fill_audit.csv", index=False)
    daily.to_csv(output / "daily_pnl.csv")
    for name, rows in [("exit_checks", checks), ("exit_events", exits), ("settlement_audit", settlements),
                       ("stale_marks", stale), ("yearly", yearly)]:
        pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False)
    (output / "summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats, equity, trades, yearly


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", default="reports/btc_sunday_take_profit_45_10")
    parser.add_argument("--output-dir", default="reports/btc_eth_wings_45_10")
    args = parser.parse_args()
    started = perf_counter()
    baseline_dir, output = Path(args.baseline_dir), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(str(output / "run.log"), level="INFO")
    cfg = Config.from_yaml("configs/backtest/btc_report_ic_45_10.yaml")
    baseline = pd.read_csv(baseline_dir / "hold_to_expiry/legs.csv")
    for col in ("entry_time", "expiry"):
        baseline[col] = pd.to_datetime(baseline[col], utc=True)
    print("Loading BTC/ETH cached history and executable quote sizes", flush=True)
    market = Market(cfg, baseline.entry_time.unique())
    pd.DataFrame(market.spot_audit).to_csv(output / "spot_price_audit.csv", index=False)
    plans, hybrids, eligibility = make_plans(baseline, market)
    eligibility.to_csv(output / "eligibility.csv", index=False)
    pd.DataFrame([leg for legs in hybrids.values() for leg in legs]).to_csv(output / "hybrid_entry_plan.csv", index=False)
    matched = {time: plans[time] for time in hybrids}
    print(f"Original weeks={len(plans)}; matched weeks={len(matched)}; "
          f"reasons={eligibility.reason.value_counts().to_dict()}", flush=True)
    rows, years = [], []
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Cumulative net PnL USD", "Hourly drawdown USD"))
    for threshold in [None, 25, 50, 60, 70, 80, 90, 95]:
        name = "hold_to_expiry" if threshold is None else f"tp_{threshold}"
        for family, selected in [("btc_full", plans), ("btc_matched", matched), ("eth_wings", hybrids)]:
            stats, equity, trades, yearly = simulate(selected, market, cfg,
                None if threshold is None else threshold / 100, output / family / name)
            if family == "btc_full":
                original = pd.read_csv(baseline_dir / name / "hourly_equity.csv")
                np.testing.assert_allclose(equity.equity_usd, original.equity_usd, rtol=0, atol=1e-7)
                original_trades = pd.read_csv(baseline_dir / name / "legs.csv")
                keys = ["entry_time", "instrument_name"]
                original_trades.entry_time = pd.to_datetime(original_trades.entry_time, utc=True)
                a, b = trades.sort_values(keys), original_trades.sort_values(keys)
                np.testing.assert_allclose(a.net_pnl_usd, b.net_pnl_usd, rtol=0, atol=1e-7)
                assert a.exit_time.astype(str).tolist() == pd.to_datetime(b.exit_time, utc=True).astype(str).tolist()
            row = {"family": family, "threshold_pct": threshold, "variant": name, **stats}
            rows.append(row)
            years.extend({"family": family, "variant": name, **y} for y in yearly)
            print(json.dumps({k: row[k] for k in ["family", "variant", "packages", "total_pnl_usd",
                "hourly_max_drawdown_usd", "early_close_packages"]}), flush=True)
            if family != "btc_full":
                label = f"{family} / {name}"
                fig.add_trace(go.Scatter(x=equity.index, y=equity.cumulative_pnl_usd, name=label), row=1, col=1)
                fig.add_trace(go.Scatter(x=equity.index, y=equity.drawdown_usd, name=label, showlegend=False), row=2, col=1)
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(years).to_csv(output / "yearly_comparison.csv", index=False)
    (output / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fig.update_layout(template="plotly_white", height=850, title="BTC shorts / equal-notional ETH wings | matched entry weeks")
    fig.write_html(output / "comparison.html", include_plotlyjs=True)
    print(f"Done in {perf_counter() - started:.2f}s: {output}", flush=True)


if __name__ == "__main__":
    main()
