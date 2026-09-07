"""Audited listed-option approximation of the user's 45/15 BTC SABR report.

Uses cached hourly opening snapshots, Friday 21 UTC entries, listed Sunday
08 UTC expiries, fixed size, no stops/hedges. This does NOT construct a SABR
surface or a synthetic 48-hour expiry. Run from the repository root.
"""
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
sys.path.insert(0, str(ROOT / "src"))
from options_backtest.config import Config
from options_backtest.engine.account import Account
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.strategy.base import BaseStrategy


_ENTRY_METADATA_PREFETCH: dict[tuple[str, int, int], dict[pd.Timestamp, pd.DataFrame]] = {}


def _metadata_source_key(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


def _index_entry_metadata(df: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    df = df.set_index("symbol")
    if not df.index.is_unique:
        raise RuntimeError(f"Duplicate opening snapshots at {now}")
    return df


@lru_cache(maxsize=512)
def _read_entry_metadata(source: tuple[str, int, int], now: pd.Timestamp) -> pd.DataFrame:
    prepared = _ENTRY_METADATA_PREFETCH.get(source, {}).get(now)
    if prepared is not None:
        if not prepared.index.is_unique:
            raise RuntimeError(f"Duplicate opening snapshots at {now}")
        return prepared
    df = pd.read_parquet(source[0], columns=["symbol", "timestamp", "bid_amount", "ask_amount"],
                         filters=[("hourly_pick", "==", "open"), ("hour", "==", now)])
    return _index_entry_metadata(df, now)


def entry_metadata(now: pd.Timestamp) -> pd.DataFrame:
    """Return original sizes/timestamps, invalidating cache on source changes."""
    path = ROOT / "data" / "options_hourly" / "BTC" / f"{now:%Y-%m}.parquet"
    return _read_entry_metadata(_metadata_source_key(path), now)


def prefetch_entry_metadata(cfg: Config, *, additional_hours=()) -> None:
    """Read candidate entry hours together, once per month instead of per hour.

    Only immutable raw quote metadata is prefetched. Trading still evaluates
    the current hour in order; future snapshots are not exposed to selection.
    """
    start = pd.to_datetime(cfg.backtest.start_date, utc=True)
    end = pd.to_datetime(cfg.backtest.end_date, utc=True)
    days = pd.date_range(start.normalize(), end.normalize(), freq="D")
    entry_hour = int(cfg.strategy.params["entry_hour_utc"])
    last_hour = min(23, entry_hour + int(cfg.strategy.params["entry_retry_hours"]))
    by_month: dict[str, list[pd.Timestamp]] = {}
    for day in days[days.weekday == 4]:
        for hour in range(entry_hour, last_hour + 1):
            now = day + pd.Timedelta(hours=hour)
            if start <= now <= end:
                by_month.setdefault(f"{now:%Y-%m}", []).append(now)
    for timestamp in additional_hours:
        now = pd.to_datetime(timestamp, utc=True)
        if start <= now <= end:
            by_month.setdefault(f"{now:%Y-%m}", []).append(now)
    _ENTRY_METADATA_PREFETCH.clear()
    _read_entry_metadata.cache_clear()
    for month, hours in by_month.items():
        hours = sorted(set(hours))
        path = ROOT / "data" / "options_hourly" / "BTC" / f"{month}.parquet"
        if not path.exists():
            continue  # No engine snapshots can be loaded from this month either.
        source = _metadata_source_key(path)
        raw = pd.read_parquet(
            path, columns=["hour", "symbol", "timestamp", "bid_amount", "ask_amount"],
            filters=[("hourly_pick", "==", "open"), ("hour", "in", hours)],
        )
        grouped = {
            now: group.drop(columns="hour").set_index("symbol")
            for now, group in raw.groupby("hour", sort=False)
        }
        empty = _index_entry_metadata(raw.iloc[:0].drop(columns="hour"), None)
        _ENTRY_METADATA_PREFETCH[source] = {now: grouped.get(now, empty) for now in hours}


def has_package_depth(legs, metadata, quantity):
    for i, leg in enumerate(legs):
        if leg.instrument_name not in metadata.index:
            return False
        size = metadata.loc[leg.instrument_name, "bid_amount" if i < 2 else "ask_amount"]
        if not np.isfinite(size) or size < quantity:
            return False
    return True


def select_package(chain: pd.DataFrame, now: pd.Timestamp, short_delta: float,
                   wing_delta: float) -> tuple[list[pd.Series], str]:
    """Select a complete executable package; never submit unprotected shorts."""
    if chain.empty:
        return [], "no_chain"
    expiry = now.normalize() + pd.Timedelta(days=2, hours=8)
    df = chain.loc[pd.to_datetime(chain.expiration_date, utc=True) == expiry].copy()
    if df.empty:
        return [], "no_sunday_expiry"
    numeric = ["delta", "strike_price", "mark_price", "bid_price", "ask_price"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    valid = np.isfinite(df[numeric]).all(axis=1)
    valid &= (df.mark_price > 0) & (df.bid_price > 0) & (df.ask_price >= df.bid_price)
    df = df.loc[valid].copy()
    legs = []
    for kind in ("call", "put"):
        side = df.loc[df.option_type.str.lower().str.startswith(kind[0])].copy()
        side = side.loc[(side.delta > 0) if kind == "call" else (side.delta < 0)]
        if side.empty:
            return [], f"no_executable_{kind}"
        side["error"] = (side.delta.abs() - short_delta).abs()
        short = side.sort_values(["error", "instrument_name"]).iloc[0]
        wings = side.loc[(side.strike_price > short.strike_price) if kind == "call"
                         else (side.strike_price < short.strike_price)].copy()
        if wings.empty:
            return [], f"no_executable_{kind}_wing"
        wings["error"] = (wings.delta.abs() - wing_delta).abs()
        wing = wings.sort_values(["error", "instrument_name"]).iloc[0]
        legs.extend([short, wing])
    return [legs[0], legs[2], legs[1], legs[3]], "entered"


class ReportStrategy(BaseStrategy):
    name = "BTCReportListedIC"

    def __init__(self, params, require_depth=False):
        super().__init__(params)
        self.entries = []
        self.attempts = []
        self.fills = []
        self.expiries = {}
        self.last_entry_day = None
        self.require_depth = require_depth

    def on_step(self, context):
        now = pd.Timestamp(context.current_time)
        if context.positions or now.weekday() != 4 or now.minute != 0:
            return
        hour = int(self.params["entry_hour_utc"])
        if not hour <= now.hour <= hour + int(self.params["entry_retry_hours"]):
            return
        if self.last_entry_day == now.date():
            return
        chain = context.option_chain
        frame = chain._to_dataframe() if hasattr(chain, "_to_dataframe") else chain
        legs, reason = select_package(frame, now, self.params["short_delta"],
                                      self.params["wing_delta"])
        if legs and self.require_depth and not has_package_depth(
                legs, entry_metadata(now), self.params["quantity"]):
            legs, reason = [], "insufficient_touch_size"
        self.attempts.append({"timestamp": now, "reason": reason})
        if not legs:
            return
        for i, leg in enumerate(legs):
            self.expiries[leg.instrument_name] = pd.Timestamp(leg.expiration_date)
            target = self.params["short_delta"] if i < 2 else self.params["wing_delta"]
            self.entries.append({
                "entry_time": now, "instrument_name": leg.instrument_name,
                "leg": ("SC", "SP", "LC", "LP")[i], "delta": float(leg.delta),
                "target_abs_delta": target, "delta_error": abs(abs(leg.delta) - target),
                "strike": float(leg.strike_price), "expiry": leg.expiration_date,
                "dte_hours": (pd.Timestamp(leg.expiration_date) - now).total_seconds() / 3600,
                "spot": context.underlying_price,
            })
            action = context.sell if i < 2 else context.buy
            action(leg.instrument_name, self.params["quantity"])
        self.last_entry_day = now.date()

    def on_fill(self, context, fill):
        self.fills.append({"entry_time": pd.Timestamp(fill.timestamp),
                           "instrument_name": fill.instrument_name,
                           "entry_fee": float(fill.fee)})


class CashMarketValueAccount(Account):
    """Premiums already affect cash: equity must add signed market VALUE."""

    def __init__(self, initial_balance, positions):
        super().__init__(initial_balance=initial_balance)
        self.positions = positions

    def equity(self, unrealized_pnl=0.0):
        return self.balance + sum(p.current_mark_price * p.quantity * p.direction_sign
                                  for p in self.positions.values())

    def record_equity(self, timestamp, unrealized_pnl=0.0, underlying_price=0.0):
        equity = self.equity()
        self._equity_history.append((timestamp, equity, self.balance,
                                     equity - self.balance, underlying_price))


class ReportEngine(BacktestEngine):
    """Isolate research accounting and opening-snapshot timing from core code."""

    def __init__(self, cfg, strategy, mark_only=False):
        super().__init__(cfg, strategy)
        self.account = CashMarketValueAccount(cfg.account.initial_balance,
                                              self.position_mgr.positions)
        self.mark_only = mark_only
        self.stale_marks = []
        self.market_marks = 0

    def _load_data(self, underlying, start, end, step):
        started = perf_counter()
        super()._load_data(underlying, start, end, step)
        self._underlying_df = self._underlying_df.copy()
        # Both spot and options use the beginning of the hour, not hour-end data.
        self._underlying_df["close"] = self._underlying_df["open"]
        spot = float(self._underlying_df["close"].iloc[0])
        self.account.initial_balance = self.strategy.params["initial_usd"] / spot
        self.load_seconds = perf_counter() - started

    def _get_quotes_fast(self, instrument_name, ts_np, underlying_price=0.0,
                         price_field="close", *, for_execution=False):
        bid, ask, mark = self._options_hourly_store.get_quote(instrument_name, ts_np, "open")
        self._quote_source_market += 1
        if mark is None:
            return None, None, None
        if self.mark_only:
            bid = ask = mark
        return tuple(v * underlying_price if v is not None else None for v in (bid, ask, mark))

    def _get_mark_prices_fast(self, ts_np, underlying_price):
        marks = {}
        now = pd.Timestamp(ts_np).tz_localize("UTC") if pd.Timestamp(ts_np).tzinfo is None else pd.Timestamp(ts_np)
        for name, pos in self.position_mgr.positions.items():
            # The hourly chain can contain newer contracts absent from the catalogue.
            expiry = self.strategy.expiries[name]
            if now >= expiry:
                continue  # Core settlement runs immediately afterward.
            _, _, mark = self._options_hourly_store.get_quote(name, ts_np, "open")
            if mark is not None and np.isfinite(mark) and mark > 0:
                marks[name] = mark * underlying_price
                self.market_marks += 1
            else:
                # Carry prior USD value; never invent a volatility surface.
                marks[name] = pos.current_mark_price
                self.stale_marks.append({"timestamp": now, "instrument_name": name})
        return marks

    def _process_orders(self, ts_np, underlying_price, ctx=None, price_field="close"):
        expected = len(self._pending_orders)
        before = len(self.strategy.fills) + len(getattr(self.strategy, "exit_fills", ()))
        super()._process_orders(ts_np, underlying_price, ctx, price_field)
        after = len(self.strategy.fills) + len(getattr(self.strategy, "exit_fills", ()))
        if after - before != expected:
            raise RuntimeError("Incomplete package execution; discard this run")
        self.position_mgr.update_marks(self._get_mark_prices_fast(ts_np, underlying_price))


def daily_statistics(pnl: pd.Series) -> dict:
    std = float(pnl.std(ddof=1))
    cumulative = pnl.cumsum()
    drawdown = cumulative - cumulative.cummax().clip(lower=0)
    max_dd = float(drawdown.min())
    if abs(max_dd) < 1e-8:
        max_dd = 0.0
    nonzero = pnl.loc[pnl.abs() > 1e-8]
    return {
        "days": len(pnl), "total_pnl_usd": float(pnl.sum()),
        "daily_pnl_sharpe_365": float(pnl.mean() / std * np.sqrt(365)) if std > 0 else None,
        "daily_max_drawdown_usd": max_dd,
        "pnl_calmar_365": float(pnl.mean() * 365 / abs(max_dd)) if max_dd < 0 else None,
        "nonzero_daily_win_rate": float((nonzero > 0).mean()) if len(nonzero) else None,
        "nonzero_days": len(nonzero), "winning_days": int((nonzero > 0).sum()),
    }


def audit_raw_entries(trades):
    parts = []
    for timestamp, group in trades.groupby("entry_time"):
        raw = entry_metadata(timestamp).rename_axis("instrument_name").reset_index()
        subset = group[["entry_time", "instrument_name", "leg", "quantity"]].merge(
            raw, on="instrument_name", validate="one_to_one")
        parts.append(subset)
    audit = pd.concat(parts, ignore_index=True)
    if len(audit) != len(trades):
        raise RuntimeError("Selected contracts cannot be reconciled to raw snapshots")
    audit["snapshot_time"] = pd.to_datetime(audit.timestamp, unit="us", utc=True)
    audit["snapshot_delay_seconds"] = (audit.snapshot_time - audit.entry_time).dt.total_seconds()
    audit["touch_size"] = audit.bid_amount.where(audit.leg.isin(["SC", "SP"]), audit.ask_amount)
    if not audit.snapshot_delay_seconds.between(0, 3599.999999).all():
        raise RuntimeError("Raw timestamp outside the selected opening-snapshot bucket")
    return audit


def summarize(engine, results, output, *, allow_early_close=False):
    output.mkdir(parents=True, exist_ok=True)
    strategy = engine.strategy
    equity = pd.DataFrame(results["equity_history"], columns=[
        "timestamp", "equity_usd", "cash_usd", "position_value_usd", "spot"])
    equity["timestamp"] = pd.to_datetime(equity.timestamp, utc=True)
    equity = equity.drop_duplicates("timestamp", keep="last").set_index("timestamp")
    equity["cumulative_pnl_usd"] = equity.equity_usd - results["initial_balance"]
    equity["drawdown_usd"] = equity.cumulative_pnl_usd - equity.cumulative_pnl_usd.cummax().clip(lower=0)
    # Sample every calendar day at the report's 21:00 UTC session, keeping flat days.
    sessions = pd.date_range(equity.index.min().normalize() + pd.Timedelta(hours=21),
                             equity.index.max(), freq="D")
    session_equity = equity.equity_usd.reindex(equity.index.union(sessions)).sort_index().ffill().reindex(sessions)
    daily = session_equity.diff()
    daily.iloc[0] = session_equity.iloc[0] - results["initial_balance"]
    # A partial final day is unnecessary here: final Tuesday is flat after Sunday expiry.
    if not np.isclose(daily.sum(), equity.cumulative_pnl_usd.iloc[-1], atol=1e-6):
        raise RuntimeError("Daily/session PnL does not reconcile with final equity")
    trades = pd.DataFrame(results["closed_trades"])
    if trades.empty:
        raise RuntimeError("No complete packages traded")
    trades["entry_time"] = pd.to_datetime(trades.entry_time, utc=True)
    trades["exit_time"] = pd.to_datetime(trades.exit_time, utc=True)
    fees = pd.DataFrame(strategy.fills)
    fees["entry_time"] = pd.to_datetime(fees.entry_time, utc=True)
    if allow_early_close:
        assert trades.close_type.isin(["settlement", "trade"]).all()
        expected_exits = {pd.to_datetime(e["entry_time"], utc=True): e["exit_time"]
                          for e in strategy.exits}
        closed = trades.loc[trades.close_type == "trade"]
        assert closed.exit_time.eq(closed.entry_time.map(expected_exits)).all(), "Unscheduled exit"
        assert trades.groupby("entry_time").exit_time.nunique().eq(1).all(), "Split package exit"
        assert trades.groupby("entry_time").close_type.nunique().eq(1).all(), "Mixed exit types"
    elif not trades.close_type.eq("settlement").all():
        raise RuntimeError("Unexpected early or forced close in hold-to-expiry run")
    trades = trades.merge(fees, on=["entry_time", "instrument_name"], validate="one_to_one")
    trades["net_pnl_usd"] = trades.pnl - trades.entry_fee
    if allow_early_close:
        # Trade-close PnL excludes its fill fee; settlement PnL includes delivery fee.
        trades["close_fee"] = trades.fee.where(trades.close_type == "trade", 0.0)
        trades["delivery_fee"] = trades.fee.where(trades.close_type == "settlement", 0.0)
        trades["net_pnl_usd"] -= trades.close_fee
    trades["gross_pnl_usd"] = trades.net_pnl_usd + trades.entry_fee + trades.fee
    entries = pd.DataFrame(strategy.entries)
    trades = trades.merge(entries, on=["entry_time", "instrument_name"], validate="one_to_one")
    audit = audit_raw_entries(trades)
    if strategy.require_depth:
        assert (audit.touch_size >= audit.quantity).all(), "Insufficient quoted size"
    packages = trades.groupby("entry_time").agg(
        exit_time=("exit_time", "max"), legs=("instrument_name", "size"),
        gross_pnl_usd=("gross_pnl_usd", "sum"), entry_fees_usd=("entry_fee", "sum"),
        settlement_fees_usd=("fee", "sum"), net_pnl_usd=("net_pnl_usd", "sum"),
        min_dte_hours=("dte_hours", "min"), max_dte_hours=("dte_hours", "max"))
    if allow_early_close:
        packages["settlement_fees_usd"] = trades.groupby("entry_time").delivery_fee.sum()
        packages["close_fees_usd"] = trades.groupby("entry_time").close_fee.sum()
    assert packages.legs.eq(4).all(), "Partial packages"
    assert trades.quantity.eq(strategy.params["quantity"]).all(), "Unexpected position sizing"
    assert len(packages) == len(strategy.entries) // 4
    assert (packages.index[1:] > pd.DatetimeIndex(packages.exit_time.iloc[:-1])).all(), "Overlapping packages"
    error = float(packages.net_pnl_usd.sum() - equity.cumulative_pnl_usd.iloc[-1])
    assert abs(error) < 1e-6, f"Cash/ledger reconciliation error: {error}"
    expected = pd.date_range(equity.index.min().normalize(), equity.index.max().normalize(), freq="W-FRI")
    attempts = pd.DataFrame(strategy.attempts)
    stats = daily_statistics(daily)
    stats.update({
        "start": str(equity.index.min()), "end": str(equity.index.max()),
        "initial_usd_for_display": results["initial_balance"],
        "final_equity_usd": float(equity.equity_usd.iloc[-1]),
        "hourly_max_drawdown_usd": float(equity.drawdown_usd.min()),
        "packages": len(packages), "legs": len(trades),
        "package_win_rate": float((packages.net_pnl_usd > 0).mean()),
        "best_package_usd": float(packages.net_pnl_usd.max()),
        "worst_package_usd": float(packages.net_pnl_usd.min()),
        "gross_pnl_usd": float(packages.gross_pnl_usd.sum()),
        "entry_fees_usd": float(packages.entry_fees_usd.sum()),
        "settlement_fees_usd": float(packages.settlement_fees_usd.sum()),
        "eligible_fridays": len(expected), "skipped_fridays": len(expected) - len(packages),
        "entry_attempt_reasons": attempts.reason.value_counts().to_dict(),
        "entry_hours_utc": pd.Series(packages.index.hour).value_counts().sort_index().to_dict(),
        "delta_error_mean": float(trades.delta_error.mean()),
        "delta_error_max": float(trades.delta_error.max()),
        "dte_hours": sorted(trades.dte_hours.unique().tolist()),
        "stale_mark_observations": len(engine.stale_marks),
        "market_mark_observations": engine.market_marks,
        "ledger_reconciliation_error_usd": error,
        "min_snapshot_delay_seconds": float(audit.snapshot_delay_seconds.min()),
        "max_snapshot_delay_seconds": float(audit.snapshot_delay_seconds.max()),
        "legs_touch_size_below_quantity": int((audit.touch_size < audit.quantity).sum()),
    })
    if allow_early_close:
        stats["close_fees_usd"] = float(packages.close_fees_usd.sum())
        stats["early_close_packages"] = len(strategy.exits)
    equity.to_csv(output / "hourly_equity.csv")
    daily.rename("pnl_usd").to_csv(output / "daily_pnl_21utc.csv", index_label="timestamp")
    trades.to_csv(output / "legs.csv", index=False)
    packages.to_csv(output / "packages.csv")
    attempts.to_csv(output / "entry_attempts.csv", index=False)
    audit.to_csv(output / "source_snapshot_audit.csv", index=False)
    pd.DataFrame(engine.stale_marks, columns=["timestamp", "instrument_name"]).to_csv(output / "stale_marks.csv", index=False)
    years = []
    for year, pnl in daily.groupby(daily.index.year):
        item = {"year": int(year), **daily_statistics(pnl)}
        group = packages.loc[packages.exit_time.dt.year == year]
        item.update(packages=len(group), package_win_rate=float((group.net_pnl_usd > 0).mean()))
        years.append(item)
    pd.DataFrame(years).to_csv(output / "yearly.csv", index=False)
    (output / "summary.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats, equity, years


def write_report(output, summaries, cfg, config_path):
    params = cfg.strategy.params
    short_delta, wing_delta = params["short_delta"], params["wing_delta"]
    delta_label = f"{short_delta * 100:g}/{wing_delta * 100:g}"
    labels = {"mark_no_fees": "标记价、无费用基准", "touch_with_fees": "盘口价格、含费用（不校验数量）",
              "touch_depth_checked": "盘口价格、含费用、校验四腿数量"}
    lines = [
        f"# BTC {delta_label} 四腿策略本地回测", "",
        "参考文件：`f27123cd490f461f_report.txt`。这是本地挂牌期权近似回测，不是原报告的 SABR/WV4 曲面复现。", "",
        "## 结果", "",
        "| 口径 | 净盈亏 USD | 日度 PnL Sharpe | 日度最大回撤 USD | 小时最大回撤 USD | 组合数 | 组合胜率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in summaries.items():
        s = item["summary"]
        lines.append(f"| {labels[name]} | {s['total_pnl_usd']:,.2f} | {s['daily_pnl_sharpe_365']:.3f} | "
                     f"{s['daily_max_drawdown_usd']:,.2f} | {s['hourly_max_drawdown_usd']:,.2f} | "
                     f"{s['packages']} | {s['package_win_rate']:.2%} |")
    key = "touch_depth_checked" if "touch_depth_checked" in summaries else next(iter(summaries))
    s = summaries[key]["summary"]
    lines += ["", f"统计区间：{s['start']} 至 {s['end']}，共 {s['days']} 个日度观测。", "",
              "## 数量校验版本分年结果", "",
              "| 年份 | 净盈亏 USD | 日度 PnL Sharpe | 组合数 | 组合胜率 |",
              "|---|---:|---:|---:|---:|"]
    for row in summaries[key]["yearly"]:
        lines.append(f"| {int(row['year'])} | {row['total_pnl_usd']:,.2f} | "
                     f"{row['daily_pnl_sharpe_365']:.3f} | {int(row['packages'])} | {row['package_win_rate']:.2%} |")
    lines += ["", "2023、2026 为不完整年度，不将上述盈亏当作全年收益。", "",
              "## 交易规则与数据差异", "",
              f"- BTC，USD 线性盈亏。卖 Call/Put 的绝对 Delta 目标各 {short_delta:g}，买 Call/Put 各 {wing_delta:g}，每腿固定 {params['quantity']:g} BTC，不复利、不做 Delta 对冲，不设额外止盈止损。",
              "- 周五 21:00 UTC 开仓，最多持有一组，至周日挂牌合约 08:00 UTC 结算。若整组无法成交，22:00、23:00 重试，之后跳过该周。数量校验版本也对盘口数量不足作同样处理。",
              "- 标记价与不校验数量的盘口版本使用相同选腿，便于拆分价差与费用影响；数量校验版本的交易时间和样本会变化，不能把它与前两版的差额全部解释为费用。",
              "- 使用小时 `open` 快照及标的小时开盘价。原始快照在整点后数秒到达，不是严格同一毫秒的同时成交；不能复现原报告逐分钟向后搜索。",
              "- 本地期权只覆盖 2023-03-20 至 2026-03-24，不能复现原报告 2019-06-15 至 2026-09-03 的区间。",
              "- 没有 SABR/WV4 参数与原永续 ticker：不构造合成 48 小时期权；实际持有 33–35 小时，使用盘口给出的 Delta 选择最近可报价执行价。执行价离散，部分组合的两条卖腿同执行价，形成铁蝶。",
              "- 四腿同一到期日；买入 Call 执行价高于卖 Call，买入 Put 执行价低于卖 Put。未用未来盈亏筛选组合；保护翼不齐全时整组跳过。",
              "- 费用采用项目假设：每腿开仓费 min(0.00024 × BTC 指数价 × 数量，权利金 × 10%)；沿用引擎交割费用，对收到正内在价值的多头收取 min(0.00015 × 结算价 × 数量，内在价值 × 10%)。这是模型参数，不是原报告披露的费用。",
              "- 优先用本地到期结算记录，无记录时用当时标的价格计算内在价值。缺失持仓标记价会延续上一个 USD 市值并记录；本次未发生。",
              "- 数量校验仅核对最优档显示数量，不模拟排队、跨腿成交时差或冲击成本。10,000 USD 初始资金仅用于展示权益；仓位始终固定，未实现保证金和强平模型。", "",
              "## 指标与核查", "",
              "- 每日 21:00 UTC 采样权益，保留空仓日。Sharpe = mean(日度美元 PnL) / sample_std(日度美元 PnL) × sqrt(365)，不是按账户百分比收益计算的 Sharpe。原报告未披露年化常数，不能保证统计公式完全相同。",
              "- 组合胜率按一整组四腿的费后净盈亏计算。非零日胜率按每日非零 PnL 计算；两者不可混用，也不使用单腿胜率替代组合胜率。",
              "- 现金已收付权利金，持仓权益按现金＋有符号的期权市值计算，避免重复计入开仓权利金。相关修正只在本次研究脚本中生效。",
              f"- 数量校验版本：{s['eligible_fridays']} 个候选周五，成交 {s['packages']} 组，跳过 {s['skipped_fridays']} 周；开仓小时分布 {s['entry_hours_utc']}。",
              f"- 四腿平均绝对 Delta 误差 {s['delta_error_mean']:.4f}，最大 {s['delta_error_max']:.4f}；这些结果不能当作精确 {short_delta:g}/{wing_delta:g} Delta 曲面组合。",
              f"- 数量不足的已成交腿 {s['legs_touch_size_below_quantity']}；缺失持仓报价记录 {s['stale_mark_observations']}；原始快照较整点延迟 {s['min_snapshot_delay_seconds']:.3f}–{s['max_snapshot_delay_seconds']:.3f} 秒。",
              f"- 逐腿现金流、整组净 PnL 与最终权益已核对；整组账本差额 {s['ledger_reconciliation_error_usd']:.2e} USD。所有组合均四腿完整、无重叠、到期结算。",
              f"- 数量校验版本非零日胜率 {s['nonzero_daily_win_rate']:.2%}（{s['winning_days']}/{s['nonzero_days']}）；日度美元 PnL Calmar {s['pnl_calmar_365']:.3f}。", "",
              "## 复现与交付", "",
              "在项目根目录运行：", "", "```powershell",
              rf'.\.venv-1\Scripts\python.exe scripts/research/backtest_btc_report_ic.py --config "{config_path}"', "```", "",
              f"配置：`{config_path}`。可用 `--mode` 单独运行一种口径；更改区间时请用新的 `--output-dir`。结束时间应落在组合结算之后，否则完整性检查会拒绝输出。", "",
              f"报告目录：`{output.as_posix()}`。`comparison.html` 为交互曲线，`comparison.json` 为汇总；各口径子目录含 `legs.csv`、`packages.csv`、`daily_pnl_21utc.csv`、`hourly_equity.csv`、`yearly.csv`、`entry_attempts.csv`、`source_snapshot_audit.csv`。", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    started = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/backtest/btc_report_ic_45_15.yaml")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--output-dir")
    parser.add_argument("--mode", choices=["all", "mark_no_fees", "touch_with_fees",
                                           "touch_depth_checked"], default="all")
    args = parser.parse_args()
    cfg = Config.from_yaml(args.config)
    if args.start:
        cfg.backtest.start_date = args.start
    if args.end:
        cfg.backtest.end_date = args.end
    output = ROOT / (args.output_dir or cfg.report.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(output / "effective_config.yaml")
    logger.remove()
    logger.add(str(output / "run.log"), level="INFO")
    logger.add(sys.stderr, level="WARNING")
    prefetch_started = perf_counter()
    prefetch_entry_metadata(cfg)
    timings = {"metadata_prefetch_seconds": perf_counter() - prefetch_started, "modes": {}}
    summaries = {}
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Cumulative PnL (USD)", "Hourly drawdown (USD)"))
    modes = (("mark_no_fees", True), ("touch_with_fees", False), ("touch_depth_checked", False))
    for name, mark_only in modes:
        if args.mode != "all" and name != args.mode:
            continue
        run_cfg = copy.deepcopy(cfg)
        if mark_only:
            for field in ("taker_fee", "maker_fee", "min_fee", "delivery_fee"):
                setattr(run_cfg.execution, field, 0.0)
        strategy = ReportStrategy(run_cfg.strategy.params, require_depth=name == "touch_depth_checked")
        engine = ReportEngine(run_cfg, strategy, mark_only)
        (output / name).mkdir(parents=True, exist_ok=True)
        run_cfg.to_yaml(output / name / "effective_config.yaml")
        print(f"Running {name}", flush=True)
        run_started = perf_counter()
        results = engine.run()
        run_seconds = perf_counter() - run_started
        export_started = perf_counter()
        stats, equity, years = summarize(engine, results, output / name)
        timings["modes"][name] = {
            "data_load_seconds": engine.load_seconds,
            "simulation_seconds": run_seconds - engine.load_seconds,
            "export_seconds": perf_counter() - export_started,
        }
        summaries[name] = {"summary": stats, "yearly": years}
        print(json.dumps(stats, ensure_ascii=False), flush=True)
    charts_started = perf_counter()
    for name, _ in modes:
        summary_path = output / name / "summary.json"
        if not summary_path.exists():
            continue
        if name not in summaries:
            summaries[name] = {
                "summary": json.loads(summary_path.read_text(encoding="utf-8")),
                "yearly": pd.read_csv(output / name / "yearly.csv").to_dict("records"),
            }
        windows = {(item["summary"]["start"], item["summary"]["end"]) for item in summaries.values()}
        if len(windows) > 1:
            raise RuntimeError("Different windows in output directory; use a separate --output-dir")
        equity = pd.read_csv(output / name / "hourly_equity.csv", index_col="timestamp", parse_dates=True)
        fig.add_trace(go.Scatter(x=equity.index, y=equity.cumulative_pnl_usd, name=name), row=1, col=1)
        fig.add_trace(go.Scatter(x=equity.index, y=equity.drawdown_usd, name=name, showlegend=False), row=2, col=1)
    fig.update_layout(template="plotly_white", height=850,
                      title=f"BTC listed IC {cfg.strategy.params['short_delta'] * 100:g}/{cfg.strategy.params['wing_delta'] * 100:g} | Fixed {cfg.strategy.params['quantity']:g} BTC/leg | Fri 21 UTC to Sun 08 UTC")
    fig.write_html(output / "comparison.html", include_plotlyjs=True)
    (output / "comparison.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(output, summaries, cfg, args.config)
    timings["comparison_export_seconds"] = perf_counter() - charts_started
    timings["total_seconds_excluding_imports"] = perf_counter() - started
    (output / "performance.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
    print(f"Timing: {timings['total_seconds_excluding_imports']:.2f}s (excluding imports)", flush=True)
    print(f"Reports: {output}", flush=True)


if __name__ == "__main__":
    main()
