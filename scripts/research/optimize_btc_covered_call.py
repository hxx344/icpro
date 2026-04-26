from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from options_backtest.data.loader import DataLoader
from scripts.research.run_cached_hourly_backtest import detect_hourly_coverage, _choose_date_window

logger.remove()
logger.add(sys.stderr, level="WARNING")


@dataclass
class ShortCallPosition:
    instrument_name: str
    strike: float
    expiry: pd.Timestamp
    quantity: float
    entry_price_btc: float
    entry_spot: float
    entry_premium_usd: float
    entry_fee_usd: float
    entry_time: pd.Timestamp
    expiry_days: float
    otm_pct: float
    target_delta: float | None = None
    entry_delta: float | None = None


@dataclass
class LongPutPosition:
    instrument_name: str
    strike: float
    expiry: pd.Timestamp
    quantity: float
    entry_price_btc: float
    entry_spot: float
    entry_premium_usd: float
    entry_fee_usd: float
    entry_time: pd.Timestamp
    expiry_days: float
    otm_pct: float
    target_delta: float | None = None
    entry_delta: float | None = None


@dataclass
class LongCallPosition:
    instrument_name: str
    strike: float
    expiry: pd.Timestamp
    quantity: float
    entry_price_btc: float
    entry_spot: float
    entry_premium_usd: float
    entry_fee_usd: float
    entry_time: pd.Timestamp
    expiry_days: float
    otm_pct: float
    wing_otm_add: float
    entry_delta: float | None = None


def _parse_float_grid(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _parse_optional_float_grid(value: str) -> list[float | None]:
    out: list[float | None] = []
    for raw in value.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token in {"none", "off", "no", "nan", "null", "-"}:
            out.append(None)
        else:
            out.append(float(token))
    return out


def _to_utc(ts: Any) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        return out.tz_localize("UTC")
    return out.tz_convert("UTC")


def _safe_row_float(row: Any, name: str, default: float = float("nan")) -> float:
    try:
        value = getattr(row, name)
    except AttributeError:
        return default
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return default


def _is_bear_regime(row: Any, spot: float, drawdown_trigger_pct: float) -> bool:
    sma = _safe_row_float(row, "trend_sma")
    peak = _safe_row_float(row, "rolling_peak")
    below_sma = np.isfinite(sma) and spot < sma
    drawdown = np.isfinite(peak) and peak > 0 and spot / peak - 1.0 <= -abs(float(drawdown_trigger_pct))
    return bool(below_sma or drawdown)


def _should_buy_protective_put(row: Any, spot: float, put_rule: str, drawdown_trigger_pct: float) -> bool:
    rule = str(put_rule).lower()
    if rule in {"always", "all"}:
        return True
    sma = _safe_row_float(row, "trend_sma")
    peak = _safe_row_float(row, "rolling_peak")
    below_sma = np.isfinite(sma) and spot < sma
    drawdown = np.isfinite(peak) and peak > 0 and spot / peak - 1.0 <= -abs(float(drawdown_trigger_pct))
    if rule in {"below_sma", "bear_sma", "trend_bear"}:
        return bool(below_sma)
    if rule in {"drawdown", "drawdown_trigger"}:
        return bool(drawdown)
    if rule in {"below_sma_or_drawdown", "bear_or_drawdown"}:
        return bool(below_sma or drawdown)
    return False


def _effective_call_otm(row: Any, spot: float, base_otm_pct: float, call_rule: str, call_otm_adjust: float, drawdown_trigger_pct: float) -> float | None:
    rule = str(call_rule).lower()
    if rule == "static":
        return float(base_otm_pct)
    sma = _safe_row_float(row, "trend_sma")
    above_sma = np.isfinite(sma) and spot > sma
    bear = _is_bear_regime(row, spot, drawdown_trigger_pct)
    if rule == "wide_above_sma":
        return float(base_otm_pct) + float(call_otm_adjust) if above_sma else float(base_otm_pct)
    if rule == "close_bear":
        return max(0.0, float(base_otm_pct) - float(call_otm_adjust)) if bear else float(base_otm_pct)
    if rule == "regime_adjust":
        if above_sma and not bear:
            return float(base_otm_pct) + float(call_otm_adjust)
        if bear:
            return max(0.0, float(base_otm_pct) - float(call_otm_adjust))
    if rule in {"skip_above_sma", "sell_only_bear"}:
        return max(0.0, float(base_otm_pct) - float(call_otm_adjust)) if bear else None
    return float(base_otm_pct)


def _effective_call_delta(row: Any, spot: float, base_delta: float, call_rule: str, delta_adjust: float, drawdown_trigger_pct: float) -> float | None:
    rule = str(call_rule).lower()
    base = max(0.001, float(base_delta))
    adjust = max(0.0, float(delta_adjust))
    if rule == "static":
        return base
    sma = _safe_row_float(row, "trend_sma")
    above_sma = np.isfinite(sma) and spot > sma
    bear = _is_bear_regime(row, spot, drawdown_trigger_pct)
    if rule == "wide_above_sma":
        return max(0.001, base - adjust) if above_sma else base
    if rule == "close_bear":
        return base + adjust if bear else base
    if rule == "regime_adjust":
        if above_sma and not bear:
            return max(0.001, base - adjust)
        if bear:
            return base + adjust
    if rule in {"skip_above_sma", "sell_only_bear"}:
        return base + adjust if bear else None
    return base


def _max_drawdown(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    peak = np.maximum.accumulate(values)
    dd = values / np.where(peak > 0, peak, 1.0) - 1.0
    return float(np.min(dd))


def _annualized_return(total_return: float, days: float) -> float:
    years = max(days / 365.25, 1e-9)
    if total_return <= -1:
        return -1.0
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def _sharpe(values: np.ndarray, days: float) -> float:
    if len(values) < 3:
        return 0.0
    rets = np.diff(values) / np.where(values[:-1] != 0, values[:-1], 1.0)
    if len(rets) < 2 or float(np.std(rets)) <= 0:
        return 0.0
    steps_per_year = len(values) / max(days / 365.25, 1e-9)
    return float(np.mean(rets) / np.std(rets) * math.sqrt(steps_per_year))


def _ensure_quote_index(store) -> None:
    # 磁盘版 HourlyOptionStore 为了节省缓存体积，quote_index 可能为空。
    # 这里一次性构建按 instrument 的报价索引，之后 get_quote 是二分查找，避免每个小时重新扫整张 snapshot。
    if getattr(store, "quote_index", None) and any(store.quote_index.get(pick) for pick in ("open", "close")):
        return
    frame = store.frame
    quote_index: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {"open": {}, "close": {}}
    available_ts_ns: dict[str, np.ndarray] = {"open": np.array([], dtype=np.int64), "close": np.array([], dtype=np.int64)}

    for pick in ("open", "close"):
        sub = frame[frame["hourly_pick"] == pick]
        if sub.empty:
            continue
        ts_ns_all = sub["hour"].to_numpy(dtype="datetime64[ns]").astype("int64")
        available_ts_ns[pick] = np.sort(np.unique(ts_ns_all))
        grouped = sub.sort_values(["instrument_name", "hour"])
        for name, grp in grouped.groupby("instrument_name", sort=False):
            quote_index[pick][str(name)] = (
                grp["hour"].to_numpy(dtype="datetime64[ns]").astype("int64"),
                grp["bid_price"].astype(float).to_numpy(),
                grp["ask_price"].astype(float).to_numpy(),
                grp["mark_price"].astype(float).to_numpy(),
            )
    store.quote_index = quote_index
    store.available_ts_ns = available_ts_ns


def _select_call(snapshot: pd.DataFrame, spot: float, now: pd.Timestamp, expiry_days: float, otm_pct: float) -> pd.Series | None:
    if snapshot.empty or spot <= 0:
        return None

    df = snapshot.copy()
    df["expiration_date"] = pd.to_datetime(df["expiration_date"], utc=True)
    df["strike_price"] = pd.to_numeric(df["strike_price"], errors="coerce")
    df["bid_price"] = pd.to_numeric(df["bid_price"], errors="coerce")
    df["mark_price"] = pd.to_numeric(df["mark_price"], errors="coerce")
    df["dte_days"] = (df["expiration_date"] - now).dt.total_seconds() / 86400.0

    calls = df[
        df["option_type"].astype(str).str.lower().str.startswith("c")
        & (df["dte_days"] > 0.25)
        & (df["dte_days"] <= max(float(expiry_days) * 2.0, float(expiry_days) + 3.0))
        & (df["strike_price"] > 0)
    ].copy()
    if calls.empty:
        return None

    # 先选最接近目标到期天数的到期日，再在该到期日内选目标 ATM/OTM strike。
    expiry_dte = calls.groupby("expiration_date")["dte_days"].median()
    best_expiry = (expiry_dte - float(expiry_days)).abs().idxmin()
    calls = calls[calls["expiration_date"] == best_expiry].copy()
    if calls.empty:
        return None

    target_strike = spot * (1.0 + float(otm_pct))
    above = calls[calls["strike_price"] >= target_strike].copy()
    if above.empty:
        above = calls.copy()
    above["strike_dist"] = (above["strike_price"] - target_strike).abs()
    return above.sort_values(["strike_dist", "strike_price"]).iloc[0]


def _select_call_by_delta(snapshot: pd.DataFrame, spot: float, now: pd.Timestamp, expiry_days: float, target_delta: float, pick_mode: str = "closest") -> pd.Series | None:
    if snapshot.empty or spot <= 0:
        return None

    df = snapshot.copy()
    df["expiration_date"] = pd.to_datetime(df["expiration_date"], utc=True)
    df["strike_price"] = pd.to_numeric(df["strike_price"], errors="coerce")
    df["bid_price"] = pd.to_numeric(df["bid_price"], errors="coerce")
    df["mark_price"] = pd.to_numeric(df["mark_price"], errors="coerce")
    if "delta" not in df.columns:
        return None
    df["delta"] = pd.to_numeric(df["delta"], errors="coerce")
    df["dte_days"] = (df["expiration_date"] - now).dt.total_seconds() / 86400.0

    calls = df[
        df["option_type"].astype(str).str.lower().str.startswith("c")
        & (df["dte_days"] > 0.25)
        & (df["dte_days"] <= max(float(expiry_days) * 2.0, float(expiry_days) + 3.0))
        & (df["strike_price"] > 0)
        & np.isfinite(df["delta"])
        & (df["delta"] > 0)
    ].copy()
    if calls.empty:
        return None

    expiry_dte = calls.groupby("expiration_date")["dte_days"].median()
    best_expiry = (expiry_dte - float(expiry_days)).abs().idxmin()
    calls = calls[calls["expiration_date"] == best_expiry].copy()
    if calls.empty:
        return None

    if str(pick_mode).lower() in {"at_least_target", "gte", "not_below", "floor"}:
        above = calls[calls["delta"] >= float(target_delta)].copy()
        if not above.empty:
            above["delta_dist"] = above["delta"] - float(target_delta)
            return above.sort_values(["delta_dist", "strike_price"]).iloc[0]

    calls["delta_dist"] = (calls["delta"] - float(target_delta)).abs()
    return calls.sort_values(["delta_dist", "strike_price"]).iloc[0]


def _select_put(snapshot: pd.DataFrame, spot: float, expiry: pd.Timestamp, otm_pct: float) -> pd.Series | None:
    if snapshot.empty or spot <= 0:
        return None

    df = snapshot.copy()
    df["expiration_date"] = pd.to_datetime(df["expiration_date"], utc=True)
    df["strike_price"] = pd.to_numeric(df["strike_price"], errors="coerce")
    df["ask_price"] = pd.to_numeric(df["ask_price"], errors="coerce")
    df["mark_price"] = pd.to_numeric(df["mark_price"], errors="coerce")

    puts = df[
        df["option_type"].astype(str).str.lower().str.startswith("p")
        & (df["expiration_date"] == expiry)
        & (df["strike_price"] > 0)
    ].copy()
    if puts.empty:
        return None

    target_strike = spot * (1.0 - float(otm_pct))
    below = puts[puts["strike_price"] <= target_strike].copy()
    if below.empty:
        below = puts.copy()
    below["strike_dist"] = (below["strike_price"] - target_strike).abs()
    return below.sort_values(["strike_dist", "strike_price"], ascending=[True, False]).iloc[0]


def _select_put_by_delta(snapshot: pd.DataFrame, expiry: pd.Timestamp, target_abs_delta: float) -> pd.Series | None:
    if snapshot.empty:
        return None

    df = snapshot.copy()
    df["expiration_date"] = pd.to_datetime(df["expiration_date"], utc=True)
    df["strike_price"] = pd.to_numeric(df["strike_price"], errors="coerce")
    df["ask_price"] = pd.to_numeric(df["ask_price"], errors="coerce")
    df["mark_price"] = pd.to_numeric(df["mark_price"], errors="coerce")
    if "delta" not in df.columns:
        return None
    df["delta"] = pd.to_numeric(df["delta"], errors="coerce")
    puts = df[
        df["option_type"].astype(str).str.lower().str.startswith("p")
        & (df["expiration_date"] == expiry)
        & (df["strike_price"] > 0)
        & np.isfinite(df["delta"])
        & (df["delta"] < 0)
    ].copy()
    if puts.empty:
        return None
    puts["abs_delta"] = puts["delta"].abs()
    puts["delta_dist"] = (puts["abs_delta"] - float(target_abs_delta)).abs()
    return puts.sort_values(["delta_dist", "strike_price"], ascending=[True, False]).iloc[0]


def _select_long_call_wing(snapshot: pd.DataFrame, spot: float, expiry: pd.Timestamp, short_call_strike: float, wing_otm_add: float) -> pd.Series | None:
    if snapshot.empty or spot <= 0:
        return None

    df = snapshot.copy()
    df["expiration_date"] = pd.to_datetime(df["expiration_date"], utc=True)
    df["strike_price"] = pd.to_numeric(df["strike_price"], errors="coerce")
    df["ask_price"] = pd.to_numeric(df["ask_price"], errors="coerce")
    df["mark_price"] = pd.to_numeric(df["mark_price"], errors="coerce")

    target_strike = max(float(short_call_strike), spot) * (1.0 + max(float(wing_otm_add), 0.0))
    calls = df[
        df["option_type"].astype(str).str.lower().str.startswith("c")
        & (df["expiration_date"] == expiry)
        & (df["strike_price"] > float(short_call_strike))
    ].copy()
    if calls.empty:
        return None
    above = calls[calls["strike_price"] >= target_strike].copy()
    if above.empty:
        above = calls.copy()
    above["strike_dist"] = (above["strike_price"] - target_strike).abs()
    return above.sort_values(["strike_dist", "strike_price"]).iloc[0]


def _close_short_call(
    store,
    pos: ShortCallPosition,
    now: pd.Timestamp,
    spot: float,
    option_fee_pct: float,
    delivery_fee_pct: float,
    close_type: str,
) -> tuple[float, dict[str, Any]]:
    if now >= pos.expiry:
        exit_price_btc = max(0.0, (spot - pos.strike) / spot)
        exit_value_usd = max(0.0, spot - pos.strike) * pos.quantity
        fee_usd = exit_value_usd * delivery_fee_pct if exit_value_usd > 0 else 0.0
        cash_delta_usd = -(exit_value_usd + fee_usd)
        actual_close_type = "settlement"
    else:
        _, ask, mark = store.get_quote(pos.instrument_name, now, pick="close")
        exit_price_btc = ask if ask is not None and ask > 0 else mark
        if exit_price_btc is None or not np.isfinite(exit_price_btc) or exit_price_btc <= 0:
            exit_price_btc = max(0.0, (spot - pos.strike) / spot)
        exit_value_usd = float(exit_price_btc) * spot * pos.quantity
        fee_usd = exit_value_usd * option_fee_pct
        cash_delta_usd = -(exit_value_usd + fee_usd)
        actual_close_type = close_type

    pnl_usd = pos.entry_premium_usd - pos.entry_fee_usd + cash_delta_usd
    trade = {
        "entry_time": pos.entry_time.isoformat(),
        "exit_time": now.isoformat(),
        "instrument_name": pos.instrument_name,
        "strike": pos.strike,
        "expiry": pos.expiry.isoformat(),
        "quantity": pos.quantity,
        "expiry_days": pos.expiry_days,
        "otm_pct": pos.otm_pct,
        "target_delta": pos.target_delta,
        "entry_delta": pos.entry_delta,
        "entry_price_btc": pos.entry_price_btc,
        "entry_spot": pos.entry_spot,
        "entry_premium_usd": pos.entry_premium_usd,
        "entry_fee_usd": pos.entry_fee_usd,
        "exit_price_btc": float(exit_price_btc),
        "exit_spot": spot,
        "exit_value_usd": float(exit_value_usd),
        "exit_fee_usd": float(fee_usd),
        "pnl_usd": float(pnl_usd),
        "close_type": actual_close_type,
        "spot_exit": spot,
    }
    return cash_delta_usd, trade


def _mark_short_call(store, pos: ShortCallPosition, now: pd.Timestamp, spot: float) -> float:
    _, _, mark = store.get_quote(pos.instrument_name, now, pick="close")
    if mark is not None and mark > 0:
        return float(mark) * spot * pos.quantity
    return max(0.0, spot - pos.strike) * pos.quantity


def _close_long_put(
    pos: LongPutPosition,
    now: pd.Timestamp,
    spot: float,
    delivery_fee_pct: float,
) -> tuple[float, dict[str, Any]]:
    exit_price_btc = max(0.0, (pos.strike - spot) / spot)
    exit_value_usd = max(0.0, pos.strike - spot) * pos.quantity
    fee_usd = exit_value_usd * delivery_fee_pct if exit_value_usd > 0 else 0.0
    cash_delta_usd = exit_value_usd - fee_usd
    pnl_usd = cash_delta_usd - pos.entry_premium_usd - pos.entry_fee_usd
    trade = {
        "entry_time": pos.entry_time.isoformat(),
        "exit_time": now.isoformat(),
        "instrument_name": pos.instrument_name,
        "leg": "long_put",
        "strike": pos.strike,
        "expiry": pos.expiry.isoformat(),
        "quantity": pos.quantity,
        "expiry_days": pos.expiry_days,
        "otm_pct": pos.otm_pct,
        "target_delta": pos.target_delta,
        "entry_delta": pos.entry_delta,
        "entry_price_btc": pos.entry_price_btc,
        "entry_spot": pos.entry_spot,
        "entry_premium_usd": pos.entry_premium_usd,
        "entry_fee_usd": pos.entry_fee_usd,
        "exit_price_btc": float(exit_price_btc),
        "exit_spot": spot,
        "exit_value_usd": float(exit_value_usd),
        "exit_fee_usd": float(fee_usd),
        "pnl_usd": float(pnl_usd),
        "close_type": "settlement",
        "spot_exit": spot,
    }
    return cash_delta_usd, trade


def _mark_long_put(store, pos: LongPutPosition, now: pd.Timestamp, spot: float) -> float:
    _, _, mark = store.get_quote(pos.instrument_name, now, pick="close")
    if mark is not None and mark > 0:
        return float(mark) * spot * pos.quantity
    return max(0.0, pos.strike - spot) * pos.quantity


def _close_long_call(
    pos: LongCallPosition,
    now: pd.Timestamp,
    spot: float,
    delivery_fee_pct: float,
) -> tuple[float, dict[str, Any]]:
    exit_price_btc = max(0.0, (spot - pos.strike) / spot)
    exit_value_usd = max(0.0, spot - pos.strike) * pos.quantity
    fee_usd = exit_value_usd * delivery_fee_pct if exit_value_usd > 0 else 0.0
    cash_delta_usd = exit_value_usd - fee_usd
    pnl_usd = cash_delta_usd - pos.entry_premium_usd - pos.entry_fee_usd
    trade = {
        "entry_time": pos.entry_time.isoformat(),
        "exit_time": now.isoformat(),
        "instrument_name": pos.instrument_name,
        "leg": "long_call",
        "strike": pos.strike,
        "expiry": pos.expiry.isoformat(),
        "quantity": pos.quantity,
        "expiry_days": pos.expiry_days,
        "otm_pct": pos.otm_pct,
        "wing_otm_add": pos.wing_otm_add,
        "entry_delta": pos.entry_delta,
        "entry_price_btc": pos.entry_price_btc,
        "entry_spot": pos.entry_spot,
        "entry_premium_usd": pos.entry_premium_usd,
        "entry_fee_usd": pos.entry_fee_usd,
        "exit_price_btc": float(exit_price_btc),
        "exit_spot": spot,
        "exit_value_usd": float(exit_value_usd),
        "exit_fee_usd": float(fee_usd),
        "pnl_usd": float(pnl_usd),
        "close_type": "settlement",
        "spot_exit": spot,
    }
    return cash_delta_usd, trade


def _mark_long_call(store, pos: LongCallPosition, now: pd.Timestamp, spot: float) -> float:
    _, _, mark = store.get_quote(pos.instrument_name, now, pick="close")
    if mark is not None and mark > 0:
        return float(mark) * spot * pos.quantity
    return max(0.0, spot - pos.strike) * pos.quantity


def run_covered_call(
    underlying_df: pd.DataFrame,
    store,
    *,
    expiry_days: float,
    selection_mode: str,
    otm_pct: float,
    call_delta: float | None,
    call_delta_pick_mode: str,
    protective_put_otm_pct: float | None,
    protective_put_delta: float | None,
    protective_put_rule: str,
    call_rule: str,
    call_otm_adjust: float,
    drawdown_trigger_pct: float,
    initial_btc: float,
    initial_capital_usd: float | None,
    short_call_multiplier: float,
    protective_call_multiplier: float,
    protective_call_otm_add: float | None,
    protective_put_multiplier: float,
    option_fee_pct: float,
    delivery_fee_pct: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    # USD 保证金账户：固定 1x BTC 多头合约作为底仓，不持有现货 BTC。
    # 多头底仓贡献 USD 浮盈浮亏；卖 call 权利金、结算和手续费全部按 USD 计。
    long_btc_qty = float(initial_btc)
    option_cash_usd = 0.0
    entry_spot: float | None = None
    account_initial_usd: float | None = None
    call_pos: ShortCallPosition | None = None
    long_call_pos: LongCallPosition | None = None
    put_pos: LongPutPosition | None = None
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []

    for row in underlying_df.itertuples(index=False):
        now = _to_utc(row.timestamp)
        spot = float(cast(Any, row.close))
        if spot <= 0:
            continue
        if entry_spot is None:
            entry_spot = spot
            account_initial_usd = float(initial_capital_usd) if initial_capital_usd is not None else long_btc_qty * spot
        assert account_initial_usd is not None
        account_initial_usd_value = float(account_initial_usd)

        if call_pos is not None and now >= call_pos.expiry:
            cash_delta, trade = _close_short_call(
                store,
            call_pos,
                now,
                spot,
                option_fee_pct,
                delivery_fee_pct,
                close_type="settlement",
            )
            option_cash_usd += cash_delta
            trade["leg"] = "short_call"
            trades.append(trade)
            call_pos = None

        if long_call_pos is not None and now >= long_call_pos.expiry:
            cash_delta, trade = _close_long_call(long_call_pos, now, spot, delivery_fee_pct)
            option_cash_usd += cash_delta
            trades.append(trade)
            long_call_pos = None

        if put_pos is not None and now >= put_pos.expiry:
            cash_delta, trade = _close_long_put(put_pos, now, spot, delivery_fee_pct)
            option_cash_usd += cash_delta
            trades.append(trade)
            put_pos = None

        # 只在没有未到期 call 时卖下一张：卖 1/2/7 天后到期的 call，持有到期结算，再卖下一张。
        if call_pos is None and long_call_pos is None and put_pos is None:
            snap = store.get_snapshot(now, pick="close")
            mode = str(selection_mode).lower()
            effective_otm_pct: float | None = None
            effective_delta: float | None = None
            if mode == "delta":
                if call_delta is not None:
                    effective_delta = _effective_call_delta(row, spot, call_delta, call_rule, call_otm_adjust, drawdown_trigger_pct)
                delta_override = _safe_row_float(row, "call_delta_override")
                if np.isfinite(delta_override) and delta_override > 0:
                    effective_delta = max(0.001, float(delta_override))
                selected = None if effective_delta is None else _select_call_by_delta(snap, spot, now, expiry_days, effective_delta, call_delta_pick_mode)
            else:
                effective_otm_pct = _effective_call_otm(row, spot, otm_pct, call_rule, call_otm_adjust, drawdown_trigger_pct)
                selected = None if effective_otm_pct is None else _select_call(snap, spot, now, expiry_days, effective_otm_pct)
            if selected is not None:
                selected_delta = float(selected.get("delta", np.nan))
                selected_strike = float(selected["strike_price"])
                actual_otm_value = selected_strike / spot - 1.0
                bid = float(selected.get("bid_price", np.nan))
                mark = float(selected.get("mark_price", np.nan))
                entry_price = bid if np.isfinite(bid) and bid > 0 else mark
                if np.isfinite(entry_price) and entry_price > 0:
                    call_qty = long_btc_qty * max(float(short_call_multiplier), 0.0)
                    put_qty = long_btc_qty * max(float(protective_put_multiplier), 0.0)
                    premium_usd = float(entry_price) * spot * call_qty
                    fee_usd = premium_usd * option_fee_pct
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
                        expiry_days=float(expiry_days),
                        otm_pct=actual_otm_value,
                        target_delta=effective_delta,
                        entry_delta=selected_delta if np.isfinite(selected_delta) else None,
                    )
                    buy_call_wing = protective_call_otm_add is not None and protective_call_multiplier > 0
                    if buy_call_wing:
                        wing_add_value = float(cast(float, protective_call_otm_add))
                        long_call_selected = _select_long_call_wing(snap, spot, call_pos.expiry, call_pos.strike, wing_add_value)
                        if long_call_selected is not None:
                            long_call_delta_value = float(long_call_selected.get("delta", np.nan))
                            long_call_strike = float(long_call_selected["strike_price"])
                            long_call_actual_otm = long_call_strike / spot - 1.0
                            ask = float(long_call_selected.get("ask_price", np.nan))
                            long_call_mark = float(long_call_selected.get("mark_price", np.nan))
                            long_call_entry_price = ask if np.isfinite(ask) and ask > 0 else long_call_mark
                            if np.isfinite(long_call_entry_price) and long_call_entry_price > 0:
                                long_call_qty = long_btc_qty * max(float(protective_call_multiplier), 0.0)
                                long_call_premium_usd = float(long_call_entry_price) * spot * long_call_qty
                                long_call_fee_usd = long_call_premium_usd * option_fee_pct
                                option_cash_usd -= long_call_premium_usd + long_call_fee_usd
                                long_call_pos = LongCallPosition(
                                    instrument_name=str(long_call_selected["instrument_name"]),
                                    strike=long_call_strike,
                                    expiry=_to_utc(long_call_selected["expiration_date"]),
                                    quantity=long_call_qty,
                                    entry_price_btc=float(long_call_entry_price),
                                    entry_spot=spot,
                                    entry_premium_usd=long_call_premium_usd,
                                    entry_fee_usd=long_call_fee_usd,
                                    entry_time=now,
                                    expiry_days=float(expiry_days),
                                    otm_pct=long_call_actual_otm,
                                    wing_otm_add=wing_add_value,
                                    entry_delta=long_call_delta_value if np.isfinite(long_call_delta_value) else None,
                                )
                    buy_put = (protective_put_otm_pct is not None or protective_put_delta is not None) and _should_buy_protective_put(row, spot, protective_put_rule, drawdown_trigger_pct)
                    if buy_put:
                        if mode == "delta" and protective_put_delta is not None:
                            put_selected = _select_put_by_delta(snap, call_pos.expiry, protective_put_delta)
                        else:
                            put_otm_value = float(cast(float, protective_put_otm_pct))
                            put_selected = _select_put(snap, spot, call_pos.expiry, put_otm_value)
                        if put_selected is not None:
                            put_delta_value = float(put_selected.get("delta", np.nan))
                            put_strike = float(put_selected["strike_price"])
                            put_actual_otm = max(0.0, 1.0 - put_strike / spot)
                            ask = float(put_selected.get("ask_price", np.nan))
                            put_mark = float(put_selected.get("mark_price", np.nan))
                            put_entry_price = ask if np.isfinite(ask) and ask > 0 else put_mark
                            if np.isfinite(put_entry_price) and put_entry_price > 0:
                                put_premium_usd = float(put_entry_price) * spot * put_qty
                                put_fee_usd = put_premium_usd * option_fee_pct
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
                                    expiry_days=float(expiry_days),
                                    otm_pct=put_actual_otm,
                                    target_delta=protective_put_delta,
                                    entry_delta=put_delta_value if np.isfinite(put_delta_value) else None,
                                )

        short_call_liability_usd = _mark_short_call(store, call_pos, now, spot) if call_pos is not None else 0.0
        long_call_asset_usd = _mark_long_call(store, long_call_pos, now, spot) if long_call_pos is not None else 0.0
        long_put_asset_usd = _mark_long_put(store, put_pos, now, spot) if put_pos is not None else 0.0
        long_btc_pnl_usd = (spot - float(entry_spot)) * long_btc_qty
        equity_usd = account_initial_usd_value + long_btc_pnl_usd + option_cash_usd - short_call_liability_usd + long_call_asset_usd + long_put_asset_usd
        equity_rows.append({
            "timestamp": now.isoformat(),
            "spot": spot,
            "equity_usd": equity_usd,
            "initial_capital_usd": account_initial_usd_value,
            "long_btc_qty": long_btc_qty,
            "long_btc_pnl_usd": long_btc_pnl_usd,
            "option_cash_usd": option_cash_usd,
            "short_call_liability_usd": short_call_liability_usd,
            "long_call_asset_usd": long_call_asset_usd,
            "long_put_asset_usd": long_put_asset_usd,
            "open_call": None if call_pos is None else call_pos.instrument_name,
            "open_long_call": None if long_call_pos is None else long_call_pos.instrument_name,
            "open_put": None if put_pos is None else put_pos.instrument_name,
        })

    eq = pd.DataFrame(equity_rows)
    if eq.empty:
        return {"error": "no equity rows"}, trades, eq

    usd_values = eq["equity_usd"].to_numpy(dtype=float)
    timestamps = pd.to_datetime(eq["timestamp"], utc=True)
    days = max((timestamps.max() - timestamps.min()).total_seconds() / 86400.0, 1.0)
    initial_usd = float(eq["initial_capital_usd"].iloc[0])
    final_usd = float(usd_values[-1])
    final_btc_equivalent = final_usd / float(eq["spot"].iloc[-1]) if float(eq["spot"].iloc[-1]) > 0 else 0.0
    risk_values_usd = np.concatenate([[initial_usd], usd_values])
    trade_pnls = np.array([float(t["pnl_usd"]) for t in trades], dtype=float)
    wins = trade_pnls[trade_pnls > 0]
    losses = trade_pnls[trade_pnls < 0]
    short_call_trades = [t for t in trades if t.get("leg") == "short_call"]
    long_put_trades = [t for t in trades if t.get("leg") == "long_put"]
    long_call_trades = [t for t in trades if t.get("leg") == "long_call"]

    def _sum_trade_field(items: list[dict[str, Any]], field: str) -> float:
        return float(sum(float(t.get(field, 0.0) or 0.0) for t in items))

    def _min_trade_pnl(items: list[dict[str, Any]]) -> float:
        return float(min((float(t.get("pnl_usd", 0.0) or 0.0) for t in items), default=0.0))

    def _max_trade_pnl(items: list[dict[str, Any]]) -> float:
        return float(max((float(t.get("pnl_usd", 0.0) or 0.0) for t in items), default=0.0))

    metrics = {
        "expiry_days": float(expiry_days),
        "selection_mode": str(selection_mode),
        "otm_pct": float(otm_pct),
        "call_delta": call_delta,
        "call_delta_pick_mode": str(call_delta_pick_mode),
        "protective_put_otm_pct": protective_put_otm_pct,
        "protective_put_delta": protective_put_delta,
        "protective_put_rule": str(protective_put_rule),
        "call_rule": str(call_rule),
        "call_otm_adjust": float(call_otm_adjust),
        "drawdown_trigger_pct": float(drawdown_trigger_pct),
        "long_btc_qty": float(initial_btc),
        "short_call_multiplier": float(short_call_multiplier),
        "protective_call_multiplier": float(protective_call_multiplier),
        "protective_call_otm_add": protective_call_otm_add,
        "protective_put_multiplier": float(protective_put_multiplier),
        "initial_usd": initial_usd,
        "final_usd": final_usd,
        "final_btc_equivalent": final_btc_equivalent,
        "total_return_usd": final_usd / initial_usd - 1.0 if initial_usd > 0 else 0.0,
        "annualized_return_usd": _annualized_return(final_usd / initial_usd - 1.0 if initial_usd > 0 else 0.0, days),
        "max_drawdown_usd": _max_drawdown(risk_values_usd),
        "sharpe_usd": _sharpe(risk_values_usd, days),
        "trade_count": int(len(trades)),
        "win_rate": float(len(wins) / len(trade_pnls)) if len(trade_pnls) else 0.0,
        "profit_factor": float(abs(wins.sum() / losses.sum())) if len(losses) and losses.sum() != 0 else float("inf"),
        "days": days,
        "short_call_count": int(len(short_call_trades)),
        "short_call_premium_usd": _sum_trade_field(short_call_trades, "entry_premium_usd"),
        "short_call_payoff_usd": _sum_trade_field(short_call_trades, "exit_value_usd"),
        "short_call_net_pnl_usd": _sum_trade_field(short_call_trades, "pnl_usd"),
        "short_call_worst_trade_usd": _min_trade_pnl(short_call_trades),
        "short_call_best_trade_usd": _max_trade_pnl(short_call_trades),
        "long_put_count": int(len(long_put_trades)),
        "long_put_premium_usd": _sum_trade_field(long_put_trades, "entry_premium_usd"),
        "long_put_payoff_usd": _sum_trade_field(long_put_trades, "exit_value_usd"),
        "long_put_net_pnl_usd": _sum_trade_field(long_put_trades, "pnl_usd"),
        "long_put_worst_trade_usd": _min_trade_pnl(long_put_trades),
        "long_put_best_trade_usd": _max_trade_pnl(long_put_trades),
        "long_call_count": int(len(long_call_trades)),
        "long_call_premium_usd": _sum_trade_field(long_call_trades, "entry_premium_usd"),
        "long_call_payoff_usd": _sum_trade_field(long_call_trades, "exit_value_usd"),
        "long_call_net_pnl_usd": _sum_trade_field(long_call_trades, "pnl_usd"),
    }
    return metrics, trades, eq


def main() -> None:
    parser = argparse.ArgumentParser(description="BTC covered call USD-margin hold-to-expiry grid backtest")
    parser.add_argument("--start", default="2023-04-25")
    parser.add_argument("--end", default="2026-04-25")
    parser.add_argument("--selection-mode", choices=["otm", "delta"], default="otm", help="Select option legs by fixed OTM percent or by market delta")
    parser.add_argument("--call-delta-pick-mode", choices=["closest", "at_least_target"], default="closest", help="When selecting calls by delta, choose absolute closest or nearest delta not below target")
    parser.add_argument("--expiry-days-grid", default="1,2,7", help="Sell calls expiring in these many days, then hold to expiry")
    parser.add_argument("--otm-grid", default="0,0.01,0.02,0.05,0.10,0.15,0.20")
    parser.add_argument("--call-delta-grid", default="0.05,0.08,0.10,0.12,0.15", help="Target call delta grid when --selection-mode delta")
    parser.add_argument("--protective-put-otm-grid", default="none", help="Optional long put OTM grid, e.g. none,0.05,0.10,0.15")
    parser.add_argument("--protective-put-delta-grid", default="none", help="Target absolute put delta grid when --selection-mode delta, e.g. none,0.03,0.05,0.10")
    parser.add_argument("--protective-put-rule-grid", default="always", help="Put rules: always, below_sma, drawdown, below_sma_or_drawdown")
    parser.add_argument("--call-rule-grid", default="static", help="Call rules: static, wide_above_sma, close_bear, regime_adjust, skip_above_sma, sell_only_bear")
    parser.add_argument("--call-otm-adjust-grid", default="0", help="OTM adjustment for non-static call rules")
    parser.add_argument("--trend-sma-hours", type=int, default=24 * 30)
    parser.add_argument("--drawdown-lookback-hours", type=int, default=24 * 30)
    parser.add_argument("--drawdown-trigger-pct", type=float, default=0.10)
    parser.add_argument("--initial-btc", type=float, default=1.0, help="Long BTC contract notional and short call quantity")
    parser.add_argument("--initial-capital-usd", type=float, default=None, help="USD margin capital; default is initial_btc * first spot, i.e. 1x notional")
    parser.add_argument("--short-call-multiplier", type=float, default=1.0, help="Short call quantity as a multiple of initial_btc; 2.0 means 1x long BTC plus 2x short calls")
    parser.add_argument("--protective-call-multiplier", type=float, default=0.0, help="Long farther call quantity as a multiple of initial_btc, used to cap extra naked-call risk")
    parser.add_argument("--protective-call-otm-add-grid", default="none", help="Buy long call this additional OTM above the short call strike, e.g. none,0.03,0.05,0.10")
    parser.add_argument("--protective-put-multiplier", type=float, default=1.0, help="Long protective put quantity as a multiple of initial_btc")
    parser.add_argument("--option-fee-pct", type=float, default=0.0003, help="Option trade fee as fraction of USD premium")
    parser.add_argument("--delivery-fee-pct", type=float, default=0.0003, help="Delivery fee as fraction of USD settlement payoff when ITM")
    parser.add_argument("--output-dir", default="reports/optimizations/btc_covered_call")
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    coverage = detect_hourly_coverage("BTC")
    start_date, end_date = _choose_date_window(coverage, args.start, args.end)

    loader = DataLoader("data")
    underlying = loader.load_underlying("BTC", resolution="60", start_date=start_date, end_date=end_date)
    store = loader.load_hourly_option_store("BTC", start_date=start_date, end_date=end_date)
    _ensure_quote_index(store)
    underlying = loader.align_underlying_to_hourly_store(underlying, store, pick="close")
    if underlying.empty:
        raise RuntimeError("No aligned hourly underlying/options data")
    underlying = underlying.copy()
    underlying["trend_sma"] = underlying["close"].rolling(args.trend_sma_hours, min_periods=24).mean()
    underlying["rolling_peak"] = underlying["close"].rolling(args.drawdown_lookback_hours, min_periods=24).max()

    expiry_days_grid = _parse_float_grid(args.expiry_days_grid)
    otm_grid = _parse_float_grid(args.otm_grid)
    call_delta_grid = _parse_float_grid(args.call_delta_grid)
    protective_put_otm_grid = _parse_optional_float_grid(args.protective_put_otm_grid)
    protective_put_delta_grid = _parse_optional_float_grid(args.protective_put_delta_grid)
    protective_call_otm_add_grid = _parse_optional_float_grid(args.protective_call_otm_add_grid)
    protective_put_rule_grid = [x.strip() for x in args.protective_put_rule_grid.split(",") if x.strip()]
    call_rule_grid = [x.strip() for x in args.call_rule_grid.split(",") if x.strip()]
    call_otm_adjust_grid = _parse_float_grid(args.call_otm_adjust_grid)

    put_specs: list[tuple[float | None, str]] = []
    put_source_grid = protective_put_delta_grid if args.selection_mode == "delta" else protective_put_otm_grid
    for put_value in put_source_grid:
        if put_value is None:
            put_specs.append((None, "none"))
        else:
            for put_rule in protective_put_rule_grid:
                put_specs.append((put_value, put_rule))

    call_specs: list[tuple[str, float]] = []
    for call_rule in call_rule_grid:
        if str(call_rule).lower() == "static":
            call_specs.append((call_rule, 0.0))
        else:
            for adjust in call_otm_adjust_grid:
                call_specs.append((call_rule, adjust))

    rows: list[dict[str, Any]] = []
    best_key = None
    best_payload = None
    t0 = time.perf_counter()
    if args.selection_mode == "delta":
        combos = [
            (expiry_days, 0.0, call_delta, None, put_value, put_rule, call_rule, call_adjust, call_wing_add)
            for expiry_days in expiry_days_grid
            for call_delta in call_delta_grid
            for put_value, put_rule in put_specs
            for call_rule, call_adjust in call_specs
            for call_wing_add in protective_call_otm_add_grid
        ]
    else:
        combos = [
            (expiry_days, otm, None, put_value, None, put_rule, call_rule, call_adjust, call_wing_add)
            for expiry_days in expiry_days_grid
            for otm in otm_grid
            for put_value, put_rule in put_specs
            for call_rule, call_adjust in call_specs
            for call_wing_add in protective_call_otm_add_grid
        ]
    for expiry_days, otm, call_delta, put_otm, put_delta, put_rule, call_rule, call_adjust, call_wing_add in tqdm(combos, desc="Covered-call grid"):
        metrics, trades, equity = run_covered_call(
            underlying,
            store,
            expiry_days=expiry_days,
            selection_mode=args.selection_mode,
            otm_pct=otm,
            call_delta=call_delta,
            call_delta_pick_mode=args.call_delta_pick_mode,
            protective_put_otm_pct=put_otm,
            protective_put_delta=put_delta,
            protective_put_rule=put_rule,
            call_rule=call_rule,
            call_otm_adjust=call_adjust,
            drawdown_trigger_pct=args.drawdown_trigger_pct,
            initial_btc=args.initial_btc,
            initial_capital_usd=args.initial_capital_usd,
            short_call_multiplier=args.short_call_multiplier,
            protective_call_multiplier=args.protective_call_multiplier,
            protective_call_otm_add=call_wing_add,
            protective_put_multiplier=args.protective_put_multiplier,
            option_fee_pct=args.option_fee_pct,
            delivery_fee_pct=args.delivery_fee_pct,
        )
        rows.append(metrics)
        if best_key is None or metrics.get("final_usd", -1e9) > best_key:
            best_key = metrics.get("final_usd", -1e9)
            best_payload = (metrics, trades, equity)

    df = pd.DataFrame(rows).sort_values(["final_usd", "max_drawdown_usd"], ascending=[False, False]).reset_index(drop=True)
    sharpe_df = pd.DataFrame(rows).sort_values(["sharpe_usd", "total_return_usd"], ascending=[False, False]).reset_index(drop=True)
    drawdown_df = pd.DataFrame(rows).sort_values(["max_drawdown_usd", "total_return_usd"], ascending=[False, False]).reset_index(drop=True)
    csv_path = out_dir / "covered_call_grid.csv"
    sharpe_csv_path = out_dir / "covered_call_grid_by_sharpe.csv"
    drawdown_csv_path = out_dir / "covered_call_grid_by_drawdown.csv"
    json_path = out_dir / "covered_call_grid.json"
    top_path = out_dir / "covered_call_grid_top10.json"
    meta_path = out_dir / "covered_call_grid_meta.json"
    df.to_csv(csv_path, index=False)
    sharpe_df.to_csv(sharpe_csv_path, index=False)
    drawdown_df.to_csv(drawdown_csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    top_path.write_text(df.head(10).to_json(orient="records", indent=2), encoding="utf-8")

    if best_payload is not None:
        best_metrics, best_trades, best_equity = best_payload
        pd.DataFrame(best_trades).to_csv(out_dir / "best_trades.csv", index=False)
        best_equity.to_csv(out_dir / "best_equity.csv", index=False)
    else:
        best_metrics = {}

    meta = {
        "strategy": "USD margin account: fixed 1x long BTC contract base + sell same-quantity BTC calls and optionally buy same-expiry protective puts",
        "accounting": "USD-denominated: long BTC contract PnL = (spot - entry_spot) * qty; option premium, payoff, liability and fees are converted to USD at the event/mark spot",
        "data_resolution": "hourly option cache and hourly BTC index; strategy trades only at initial sale and after each option expiry",
        "roll_rule": "no early roll; after a short call expires/settles, sell the next call at the next available hourly close",
        "coverage": {"requested_start": args.start, "requested_end": args.end, "actual_start": start_date, "actual_end": end_date},
        "long_btc_qty": args.initial_btc,
        "initial_capital_usd": args.initial_capital_usd,
        "short_call_multiplier": args.short_call_multiplier,
        "protective_call_multiplier": args.protective_call_multiplier,
        "protective_call_otm_add_grid": protective_call_otm_add_grid,
        "protective_put_multiplier": args.protective_put_multiplier,
        "option_fee_pct": args.option_fee_pct,
        "delivery_fee_pct": args.delivery_fee_pct,
        "expiry_days_grid": expiry_days_grid,
        "selection_mode": args.selection_mode,
        "call_delta_pick_mode": args.call_delta_pick_mode,
        "otm_grid": otm_grid,
        "call_delta_grid": call_delta_grid,
        "protective_put_otm_grid": protective_put_otm_grid,
        "protective_put_delta_grid": protective_put_delta_grid,
        "protective_put_rule_grid": protective_put_rule_grid,
        "call_rule_grid": call_rule_grid,
        "call_otm_adjust_grid": call_otm_adjust_grid,
        "trend_sma_hours": args.trend_sma_hours,
        "drawdown_lookback_hours": args.drawdown_lookback_hours,
        "drawdown_trigger_pct": args.drawdown_trigger_pct,
        "combo_count": len(combos),
        "elapsed_sec": time.perf_counter() - t0,
        "best_by_final_usd": best_metrics,
        "best_by_sharpe_usd": sharpe_df.iloc[0].to_dict() if not sharpe_df.empty else {},
        "best_by_lowest_drawdown_usd": drawdown_df.iloc[0].to_dict() if not drawdown_df.empty else {},
        "outputs": {
            "csv": csv_path.as_posix(),
            "by_sharpe_csv": sharpe_csv_path.as_posix(),
            "by_drawdown_csv": drawdown_csv_path.as_posix(),
            "json": json_path.as_posix(),
            "top10": top_path.as_posix(),
            "best_trades": (out_dir / "best_trades.csv").as_posix(),
            "best_equity": (out_dir / "best_equity.csv").as_posix(),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("=" * 72)
    print("BTC covered call USD-margin hold-to-expiry grid complete")
    print(f"Window: {start_date} -> {end_date}")
    print(f"Combos: {len(combos)}")
    print("Top 10 by final USD:")
    cols = ["expiry_days", "selection_mode", "otm_pct", "call_delta", "call_delta_pick_mode", "call_rule", "call_otm_adjust", "short_call_multiplier", "protective_call_multiplier", "protective_call_otm_add", "protective_put_otm_pct", "protective_put_delta", "protective_put_rule", "initial_usd", "final_usd", "total_return_usd", "max_drawdown_usd", "sharpe_usd", "final_btc_equivalent", "trade_count"]
    print(df[cols].head(10).to_string(index=False))
    print("Top 10 by Sharpe USD:")
    print(sharpe_df[cols].head(10).to_string(index=False))
    print(f"Saved: {csv_path.as_posix()}")


if __name__ == "__main__":
    main()