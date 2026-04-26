from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time as time_mod
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from options_backtest.pricing.black76 import delta as bs_delta
from trader.config import TraderConfig, load_config

DERIBIT_API_BASE = "https://www.deribit.com/api/v2"
SYMBOL_RE = re.compile(
    r"^(?P<ul>[A-Z]+)-(?P<day>\d{1,2})(?P<mon>[A-Z]{3})(?P<yy>\d{2})-(?P<strike>\d+(?:\.\d+)*)-(?P<cp>[CP])$"
)
MONTH_MAP = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


@dataclass
class InstrumentMeta:
    symbol: str
    strike: float
    option_type: str
    expiry: datetime
    min_trade_amount: float


@dataclass
class TradeSnapshot:
    symbol: str
    strike: float
    option_type: str
    trade_ts: str
    entry_price_btc: float
    mark_price_btc: float
    index_price: float
    iv: float
    abs_delta: float
    delta_diff: float
    contracts: float


class DeribitPublicClient:
    def __init__(self) -> None:
        self.session = requests.Session()

    def _get(self, method: str, params: dict[str, Any]) -> Any:
        response = self.session.get(f"{DERIBIT_API_BASE}/{method}", params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            raise RuntimeError(str(payload["error"]))
        return payload.get("result")

    def get_instruments(self, currency: str, expired: bool = True) -> list[dict[str, Any]]:
        result = self._get(
            "public/get_instruments",
            {"currency": currency.upper(), "kind": "option", "expired": str(expired).lower()},
        )
        return list(result or [])

    def get_chart_data(
        self,
        instrument_name: str,
        start_ts: datetime,
        end_ts: datetime,
        resolution: str = "60",
    ) -> dict[str, Any]:
        return self._get(
            "public/get_tradingview_chart_data",
            {
                "instrument_name": instrument_name,
                "start_timestamp": int(start_ts.timestamp() * 1000),
                "end_timestamp": int(end_ts.timestamp() * 1000),
                "resolution": resolution,
            },
        )

    def get_trades_by_instrument(
        self,
        instrument_name: str,
        start_ts: datetime,
        end_ts: datetime,
        page_size: int = 1000,
        sleep_sec: float = 0.05,
    ) -> list[dict[str, Any]]:
        all_trades: list[dict[str, Any]] = []
        seen_trade_ids: set[str] = set()
        cursor_ts_ms = int(start_ts.timestamp() * 1000)
        end_ts_ms = int(end_ts.timestamp() * 1000)

        while cursor_ts_ms <= end_ts_ms:
            result = self._get(
                "public/get_last_trades_by_instrument_and_time",
                {
                    "instrument_name": instrument_name,
                    "start_timestamp": cursor_ts_ms,
                    "end_timestamp": end_ts_ms,
                    "count": page_size,
                    "sorting": "asc",
                },
            ) or {}
            page = list(result.get("trades") or [])
            if not page:
                break

            new_rows = 0
            max_ts = cursor_ts_ms
            for trade in page:
                trade_id = str(trade.get("trade_id") or "")
                if trade_id and trade_id in seen_trade_ids:
                    continue
                if trade_id:
                    seen_trade_ids.add(trade_id)
                all_trades.append(trade)
                new_rows += 1
                max_ts = max(max_ts, int(trade.get("timestamp") or cursor_ts_ms))

            has_more = bool(result.get("has_more"))
            if not has_more or new_rows == 0:
                break
            cursor_ts_ms = max_ts
            time_mod.sleep(sleep_sec)

        all_trades.sort(key=lambda item: (int(item.get("timestamp") or 0), int(item.get("trade_seq") or 0)))
        return all_trades


def parse_symbol(symbol: str) -> dict[str, Any] | None:
    match = SYMBOL_RE.match(symbol)
    if not match:
        return None
    expiry = datetime(
        2000 + int(match.group("yy")),
        MONTH_MAP[match.group("mon")],
        int(match.group("day")),
        8,
        0,
        tzinfo=timezone.utc,
    )
    return {
        "underlying": match.group("ul"),
        "expiry": expiry,
        "strike": float(match.group("strike")),
        "option_type": "call" if match.group("cp") == "C" else "put",
    }


def next_sunday_0800(reference: datetime) -> datetime:
    days_until = (6 - reference.weekday()) % 7
    sunday = (reference + timedelta(days=days_until)).date()
    return datetime.combine(sunday, time(8, 0), tzinfo=timezone.utc)


def normalize_instruments(raw_instruments: list[dict[str, Any]], underlying: str) -> list[InstrumentMeta]:
    instruments: list[InstrumentMeta] = []
    for row in raw_instruments:
        symbol = str(row.get("instrument_name") or "")
        parsed = parse_symbol(symbol)
        if parsed is None or parsed["underlying"] != underlying.upper():
            continue
        instruments.append(
            InstrumentMeta(
                symbol=symbol,
                strike=float(parsed["strike"]),
                option_type=str(parsed["option_type"]),
                expiry=parsed["expiry"],
                min_trade_amount=float(row.get("min_trade_amount") or 0.0),
            )
        )
    return instruments


def choose_target_expiry(instruments: list[InstrumentMeta], entry_ts: datetime, preferred_expiry: datetime) -> datetime:
    expiries = sorted({inst.expiry for inst in instruments if inst.expiry > entry_ts})
    if not expiries:
        raise ValueError("No future expiries returned by Deribit")
    return min(expiries, key=lambda expiry: (abs((expiry - preferred_expiry).total_seconds()), expiry))


def build_spot_series(chart_data: dict[str, Any]) -> list[tuple[datetime, float]]:
    ticks = chart_data.get("ticks") or []
    closes = chart_data.get("close") or []
    out: list[tuple[datetime, float]] = []
    for ts_ms, close in zip(ticks, closes, strict=False):
        if close is None:
            continue
        out.append((datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc), float(close)))
    return out


def nearest_spot_at_or_before(series: list[tuple[datetime, float]], ts: datetime) -> float:
    eligible = [price for dt_value, price in series if dt_value <= ts]
    if not eligible:
        raise ValueError(f"No spot data available at or before {ts.isoformat()}")
    return float(eligible[-1])


def compute_realized_vol(series: list[tuple[datetime, float]], end_ts: datetime, lookback_hours: int) -> float | None:
    if lookback_hours <= 1:
        return None
    window = [price for ts, price in series if ts <= end_ts][-lookback_hours - 1 :]
    if len(window) < 3:
        return None
    log_returns = []
    for prev, cur in zip(window[:-1], window[1:], strict=False):
        if prev <= 0 or cur <= 0:
            continue
        log_returns.append(math.log(cur / prev))
    if len(log_returns) < 2:
        return None
    mean = sum(log_returns) / len(log_returns)
    variance = sum((x - mean) ** 2 for x in log_returns) / (len(log_returns) - 1)
    return math.sqrt(variance) * math.sqrt(24 * 365.25)


def latest_trade_snapshot_before(
    trades: list[dict[str, Any]],
    entry_ts: datetime,
    strike: float,
    option_type: str,
    target_delta: float,
    max_delta_diff: float,
) -> TradeSnapshot | None:
    eligible = [trade for trade in trades if datetime.fromtimestamp(int(trade.get("timestamp") or 0) / 1000, tz=timezone.utc) <= entry_ts]
    if not eligible:
        return None
    trade = eligible[-1]
    iv_pct = float(trade.get("iv") or 0.0)
    index_price = float(trade.get("index_price") or 0.0)
    mark_price = float(trade.get("mark_price") or 0.0)
    trade_price = float(trade.get("price") or mark_price or 0.0)
    if iv_pct <= 0 or index_price <= 0 or trade_price <= 0:
        return None
    parsed = parse_symbol(str(trade.get("instrument_name") or ""))
    if parsed is None:
        return None
    T_years = max((parsed["expiry"] - entry_ts).total_seconds() / (365.25 * 86400), 1e-9)
    abs_delta = abs(bs_delta(index_price, strike, T_years, iv_pct / 100.0, option_type=option_type, r=0.0))
    delta_diff = abs(abs_delta - target_delta)
    if delta_diff > max_delta_diff:
        return None
    return TradeSnapshot(
        symbol=str(trade.get("instrument_name") or ""),
        strike=strike,
        option_type=option_type,
        trade_ts=datetime.fromtimestamp(int(trade.get("timestamp") or 0) / 1000, tz=timezone.utc).isoformat(),
        entry_price_btc=trade_price,
        mark_price_btc=mark_price if mark_price > 0 else trade_price,
        index_price=index_price,
        iv=iv_pct / 100.0,
        abs_delta=abs_delta,
        delta_diff=delta_diff,
        contracts=float(trade.get("contracts") or trade.get("amount") or 0.0),
    )


def select_short_leg(
    snapshots: list[TradeSnapshot],
    option_type: str,
    spot: float,
    target_delta: float,
) -> TradeSnapshot:
    candidates = []
    for snap in snapshots:
        if snap.option_type != option_type:
            continue
        if option_type == "call" and snap.strike <= spot:
            continue
        if option_type == "put" and snap.strike >= spot:
            continue
        candidates.append(snap)
    if not candidates:
        raise ValueError(f"No eligible {option_type} candidates found")
    return min(candidates, key=lambda snap: (abs(snap.abs_delta - target_delta), snap.delta_diff, abs(snap.strike - spot)))


def build_hourly_mark_series(
    trades: list[dict[str, Any]],
    start_ts: datetime,
    end_ts: datetime,
) -> list[dict[str, Any]]:
    hour_map: dict[datetime, dict[str, Any]] = {}
    for trade in trades:
        ts = datetime.fromtimestamp(int(trade.get("timestamp") or 0) / 1000, tz=timezone.utc)
        if ts < start_ts or ts > end_ts:
            continue
        bucket = ts.replace(minute=0, second=0, microsecond=0)
        hour_map[bucket] = {
            "timestamp": bucket,
            "mark_price_btc": float(trade.get("mark_price") or trade.get("price") or 0.0),
            "trade_price_btc": float(trade.get("price") or trade.get("mark_price") or 0.0),
            "index_price": float(trade.get("index_price") or 0.0),
            "iv": float(trade.get("iv") or 0.0) / 100.0 if float(trade.get("iv") or 0.0) > 0 else None,
            "trade_count": int(hour_map.get(bucket, {}).get("trade_count", 0)) + 1,
        }

    series: list[dict[str, Any]] = []
    cursor = start_ts.replace(minute=0, second=0, microsecond=0)
    if cursor < start_ts:
        cursor += timedelta(hours=1)
    last_mark: float | None = None
    last_trade: float | None = None
    last_index: float | None = None
    last_iv: float | None = None
    while cursor <= end_ts:
        item = hour_map.get(cursor)
        if item is not None:
            last_mark = float(item.get("mark_price_btc") or last_mark or 0.0)
            last_trade = float(item.get("trade_price_btc") or last_trade or 0.0)
            last_index = float(item.get("index_price") or last_index or 0.0)
            last_iv = item.get("iv") if item.get("iv") is not None else last_iv
            series.append(
                {
                    "timestamp": cursor.isoformat(),
                    "mark_price_btc": last_mark,
                    "trade_price_btc": last_trade,
                    "index_price": last_index,
                    "iv": last_iv,
                    "trade_count": int(item.get("trade_count") or 0),
                    "carried_forward": False,
                }
            )
        elif last_mark is not None:
            series.append(
                {
                    "timestamp": cursor.isoformat(),
                    "mark_price_btc": last_mark,
                    "trade_price_btc": last_trade,
                    "index_price": last_index,
                    "iv": last_iv,
                    "trade_count": 0,
                    "carried_forward": True,
                }
            )
        cursor += timedelta(hours=1)
    return series


def evaluate_position_hourly(
    cfg: TraderConfig,
    spot_series: list[tuple[datetime, float]],
    sell_call: TradeSnapshot,
    sell_put: TradeSnapshot,
    call_hourly: list[dict[str, Any]],
    put_hourly: list[dict[str, Any]],
    entry_ts: datetime,
    expiry: datetime,
) -> dict[str, Any]:
    entry_credit_btc_per_combo = sell_call.entry_price_btc + sell_put.entry_price_btc
    quantity = float(cfg.strategy.quantity)
    hourly_points: list[dict[str, Any]] = []
    exit_reason = "settlement"
    exit_ts = expiry
    exit_cost_btc_per_combo = 0.0
    exit_spot = nearest_spot_at_or_before(spot_series, expiry)

    by_ts_call = {item["timestamp"]: item for item in call_hourly}
    by_ts_put = {item["timestamp"]: item for item in put_hourly}
    move_filter_pct = float(cfg.strategy.stop_loss_underlying_move_pct or 0.0)
    stop_loss_pct = float(cfg.strategy.stop_loss_pct or 0.0)
    entry_spot = nearest_spot_at_or_before(spot_series, entry_ts)

    cursor = (entry_ts + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    while cursor <= expiry:
        ts_iso = cursor.isoformat()
        call_state = by_ts_call.get(ts_iso)
        put_state = by_ts_put.get(ts_iso)
        if call_state and put_state:
            basket_close_cost = float(call_state.get("mark_price_btc") or 0.0) + float(put_state.get("mark_price_btc") or 0.0)
            basket_pnl_pct = (entry_credit_btc_per_combo - basket_close_cost) / entry_credit_btc_per_combo * 100.0
            spot_now = nearest_spot_at_or_before(spot_series, cursor)
            max_move_pct = abs(spot_now / entry_spot - 1.0) * 100.0
            point = {
                "timestamp": ts_iso,
                "call_mark_btc": float(call_state.get("mark_price_btc") or 0.0),
                "put_mark_btc": float(put_state.get("mark_price_btc") or 0.0),
                "basket_close_cost_btc_per_combo": basket_close_cost,
                "basket_pnl_pct": basket_pnl_pct,
                "spot": spot_now,
                "abs_move_pct": max_move_pct,
                "call_trade_count": int(call_state.get("trade_count") or 0),
                "put_trade_count": int(put_state.get("trade_count") or 0),
                "call_carried_forward": bool(call_state.get("carried_forward")),
                "put_carried_forward": bool(put_state.get("carried_forward")),
            }
            hourly_points.append(point)
            if stop_loss_pct > 0 and basket_pnl_pct <= -stop_loss_pct and (move_filter_pct <= 0 or max_move_pct >= move_filter_pct):
                exit_reason = "stop_loss"
                exit_ts = cursor
                exit_cost_btc_per_combo = basket_close_cost
                exit_spot = spot_now
                break
        cursor += timedelta(hours=1)

    if exit_reason == "settlement":
        final_ts = max((entry for entry in hourly_points if entry["timestamp"] <= expiry.isoformat()), key=lambda x: x["timestamp"], default=None)
        if final_ts is not None:
            exit_cost_btc_per_combo = float(final_ts["basket_close_cost_btc_per_combo"])
            exit_spot = float(final_ts["spot"])
        else:
            exit_cost_btc_per_combo = 0.0

    pnl_btc_per_combo = entry_credit_btc_per_combo - exit_cost_btc_per_combo
    pnl_btc_total = pnl_btc_per_combo * quantity
    pnl_usd_approx_total = pnl_btc_total * exit_spot
    return {
        "entry_credit_btc_per_combo": entry_credit_btc_per_combo,
        "entry_credit_usd_approx_total": entry_credit_btc_per_combo * quantity * entry_spot,
        "exit_reason": exit_reason,
        "exit_timestamp": exit_ts.isoformat(),
        "exit_spot": exit_spot,
        "exit_cost_btc_per_combo": exit_cost_btc_per_combo,
        "pnl_btc_per_combo": pnl_btc_per_combo,
        "pnl_btc_total": pnl_btc_total,
        "pnl_usd_approx_total": pnl_usd_approx_total,
        "return_on_credit_pct": (pnl_btc_per_combo / entry_credit_btc_per_combo * 100.0) if entry_credit_btc_per_combo > 0 else None,
        "hourly_monitor_points": hourly_points,
    }


def run_analysis(as_of_date: date, config_path: str) -> dict[str, Any]:
    cfg = load_config(config_path)
    underlying = cfg.strategy.underlying.upper()
    entry_day = as_of_date - timedelta(days=(as_of_date.weekday() - 4) % 7)
    entry_ts = datetime.combine(entry_day, time.fromisoformat(cfg.strategy.entry_time_utc), tzinfo=timezone.utc)
    preferred_expiry = next_sunday_0800(entry_ts)

    client = DeribitPublicClient()
    instruments = normalize_instruments(client.get_instruments(underlying, expired=True), underlying)
    expiry = choose_target_expiry(instruments, entry_ts, preferred_expiry)
    target_instruments = [inst for inst in instruments if inst.expiry == expiry]

    spot_series = build_spot_series(
        client.get_chart_data(
            f"{underlying}-PERPETUAL",
            entry_ts - timedelta(hours=max(int(cfg.strategy.entry_realized_vol_lookback_hours or 0), 24) + 6),
            expiry + timedelta(hours=2),
            resolution="60",
        )
    )
    entry_spot = nearest_spot_at_or_before(spot_series, entry_ts)
    rv_value = compute_realized_vol(spot_series, entry_ts, int(cfg.strategy.entry_realized_vol_lookback_hours or 0))

    start_fetch_ts = entry_ts - timedelta(hours=36)
    snapshots: list[TradeSnapshot] = []
    trade_cache: dict[str, list[dict[str, Any]]] = {}
    instruments_with_pre_entry_trades = 0
    for inst in target_instruments:
        trades = client.get_trades_by_instrument(inst.symbol, start_fetch_ts, expiry)
        trade_cache[inst.symbol] = trades
        if any(datetime.fromtimestamp(int(trade.get("timestamp") or 0) / 1000, tz=timezone.utc) <= entry_ts for trade in trades):
            instruments_with_pre_entry_trades += 1
        snap = latest_trade_snapshot_before(
            trades,
            entry_ts,
            inst.strike,
            inst.option_type,
            float(cfg.strategy.target_delta),
            float(cfg.strategy.max_delta_diff),
        )
        if snap is not None:
            snapshots.append(snap)

    if not snapshots:
        return {
            "as_of_date": as_of_date.isoformat(),
            "config_path": config_path,
            "analysis_mode": "Deribit public trade API aggregated to hourly option marks",
            "notes": [
                "期权数据来自 Deribit public/get_last_trades_by_instrument_and_time，并按小时聚合。",
                "本周目标到期日合约在入场时点之前没有足够的小时级真实成交，无法按纯小时级真实成交重建两条短腿。",
                "按严格的可交易数据口径，本次应视为无交易。",
            ],
            "strategy": {
                "underlying": underlying,
                "target_delta": float(cfg.strategy.target_delta),
                "wing_delta": float(cfg.strategy.wing_delta),
                "max_delta_diff": float(cfg.strategy.max_delta_diff),
                "entry_time_utc": cfg.strategy.entry_time_utc,
                "quantity": float(cfg.strategy.quantity),
                "stop_loss_pct": float(cfg.strategy.stop_loss_pct),
                "stop_loss_underlying_move_pct": float(cfg.strategy.stop_loss_underlying_move_pct),
                "entry_realized_vol_lookback_hours": int(cfg.strategy.entry_realized_vol_lookback_hours),
                "entry_realized_vol_max": float(cfg.strategy.entry_realized_vol_max),
            },
            "entry": {
                "entry_timestamp": entry_ts.isoformat(),
                "entry_spot": entry_spot,
                "rv_filter_value": rv_value,
                "rv_filter_passed": rv_value is None or rv_value <= float(cfg.strategy.entry_realized_vol_max),
            },
            "expiry": {
                "preferred_expiry": preferred_expiry.isoformat(),
                "selected_expiry": expiry.isoformat(),
                "selected_vs_preferred_diff_hours": (expiry - preferred_expiry).total_seconds() / 3600.0,
            },
            "candidate_count": 0,
            "target_expiry_instrument_count": len(target_instruments),
            "target_expiry_instruments_with_pre_entry_trades": instruments_with_pre_entry_trades,
            "selected_legs": None,
            "result": {
                "trade_executed": False,
                "reason": "no_hourly_tradeable_option_candidates_before_entry",
                "pnl_btc_total": 0.0,
                "pnl_usd_approx_total": 0.0,
            },
        }

    sell_call = select_short_leg(snapshots, "call", entry_spot, float(cfg.strategy.target_delta))
    sell_put = select_short_leg(snapshots, "put", entry_spot, float(cfg.strategy.target_delta))

    call_hourly = build_hourly_mark_series(trade_cache[sell_call.symbol], entry_ts, expiry)
    put_hourly = build_hourly_mark_series(trade_cache[sell_put.symbol], entry_ts, expiry)
    result = evaluate_position_hourly(cfg, spot_series, sell_call, sell_put, call_hourly, put_hourly, entry_ts, expiry)

    return {
        "as_of_date": as_of_date.isoformat(),
        "config_path": config_path,
        "analysis_mode": "Deribit public trade API aggregated to hourly option marks",
        "notes": [
            "期权数据来自 Deribit public/get_last_trades_by_instrument_and_time，并按小时聚合。",
            "入场价使用入场时点之前最近一笔真实成交价。",
            "持仓监控使用每小时最后一笔成交附带的 mark_price，并在无成交小时向前沿用最近值。",
        ],
        "strategy": {
            "underlying": underlying,
            "target_delta": float(cfg.strategy.target_delta),
            "wing_delta": float(cfg.strategy.wing_delta),
            "max_delta_diff": float(cfg.strategy.max_delta_diff),
            "entry_time_utc": cfg.strategy.entry_time_utc,
            "quantity": float(cfg.strategy.quantity),
            "stop_loss_pct": float(cfg.strategy.stop_loss_pct),
            "stop_loss_underlying_move_pct": float(cfg.strategy.stop_loss_underlying_move_pct),
            "entry_realized_vol_lookback_hours": int(cfg.strategy.entry_realized_vol_lookback_hours),
            "entry_realized_vol_max": float(cfg.strategy.entry_realized_vol_max),
        },
        "entry": {
            "entry_timestamp": entry_ts.isoformat(),
            "entry_spot": entry_spot,
            "rv_filter_value": rv_value,
            "rv_filter_passed": rv_value is None or rv_value <= float(cfg.strategy.entry_realized_vol_max),
        },
        "expiry": {
            "preferred_expiry": preferred_expiry.isoformat(),
            "selected_expiry": expiry.isoformat(),
            "selected_vs_preferred_diff_hours": (expiry - preferred_expiry).total_seconds() / 3600.0,
        },
        "candidate_count": len(snapshots),
        "target_expiry_instrument_count": len(target_instruments),
        "target_expiry_instruments_with_pre_entry_trades": instruments_with_pre_entry_trades,
        "selected_legs": {
            "sell_call": asdict(sell_call),
            "sell_put": asdict(sell_put),
        },
        "result": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay current week WeekendVol using Deribit hourly option data")
    parser.add_argument("--as-of-date", default=datetime.now(timezone.utc).date().isoformat())
    parser.add_argument("--config", default="configs/trader/weekend_vol_btc.yaml")
    parser.add_argument(
        "--output",
        default="reports/validation/current_week_deribit_weekend_vol_hourly.json",
    )
    args = parser.parse_args()

    result = run_analysis(date.fromisoformat(args.as_of_date), args.config)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nSaved report to {output_path}")


if __name__ == "__main__":
    main()
