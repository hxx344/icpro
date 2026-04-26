from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from options_backtest.pricing.black76 import delta as bs_delta
from options_backtest.pricing.iv_solver import implied_volatility_btc
from scripts.download_cdd_data import API_TOKEN
from trader.config import TraderConfig, load_config

DERIBIT_API_BASE = "https://www.deribit.com/api/v2"
CDD_API_BASE = "https://api.cryptodatadownload.com/v1"
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
class LegSnapshot:
    symbol: str
    strike: float
    option_type: str
    entry_price_btc: float
    implied_vol: float
    abs_delta: float
    delta_diff: float


class DeribitPublicClient:
    def __init__(self) -> None:
        self.session = requests.Session()

    def get_chart_data(
        self,
        instrument_name: str,
        start_ts: datetime,
        end_ts: datetime,
        resolution: str = "60",
    ) -> dict[str, Any]:
        response = self.session.get(
            f"{DERIBIT_API_BASE}/public/get_tradingview_chart_data",
            params={
                "instrument_name": instrument_name,
                "start_timestamp": int(start_ts.timestamp() * 1000),
                "end_timestamp": int(end_ts.timestamp() * 1000),
                "resolution": resolution,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        result = payload.get("result") or {}
        if result.get("status") == "no_data":
            return {"ticks": [], "close": [], "open": [], "high": [], "low": []}
        return result


class CDDDailyClient:
    def __init__(self, token: str) -> None:
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Token {token}"})

    def get_date_snapshot(self, currency: str, snapshot_date: date) -> list[dict[str, Any]]:
        response = self.session.get(
            f"{CDD_API_BASE}/data/ohlc/deribit/options/",
            params={
                "currency": currency,
                "date": snapshot_date.isoformat(),
                "limit": 5000,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        return list(payload.get("result") or [])

    def get_symbol_history(self, currency: str, symbol: str, limit: int = 20) -> list[dict[str, Any]]:
        response = self.session.get(
            f"{CDD_API_BASE}/data/ohlc/deribit/options/",
            params={
                "currency": currency,
                "symbol": symbol,
                "limit": limit,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        return list(payload.get("result") or [])


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


def build_hourly_series(chart: dict[str, Any]) -> list[tuple[datetime, float]]:
    ticks = chart.get("ticks") or []
    closes = chart.get("close") or []
    series: list[tuple[datetime, float]] = []
    for ts_ms, close in zip(ticks, closes, strict=False):
        if close is None:
            continue
        series.append((datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc), float(close)))
    return series


def nearest_close_at_or_before(series: list[tuple[datetime, float]], ts: datetime) -> float:
    eligible = [value for dt_value, value in series if dt_value <= ts]
    if not eligible:
        raise ValueError(f"No chart data available at or before {ts.isoformat()}")
    return float(eligible[-1])


def compute_realized_vol_from_hourly(series: list[tuple[datetime, float]], end_ts: datetime, lookback_hours: int) -> float | None:
    window = [(ts, px) for ts, px in series if ts <= end_ts and ts >= end_ts - timedelta(hours=lookback_hours + 2)]
    closes = [px for _, px in window][-lookback_hours - 1 :]
    if len(closes) < 3:
        return None
    arr = np.asarray(closes, dtype=float)
    rets = np.diff(np.log(arr))
    if len(rets) == 0:
        return None
    return float(np.std(rets, ddof=1) * math.sqrt(24 * 365.25))


def choose_expiry(snapshot_rows: list[dict[str, Any]], entry_ts: datetime, preferred_expiry: datetime) -> datetime:
    expiries = sorted(
        {
            parsed["expiry"]
            for row in snapshot_rows
            if (parsed := parse_symbol(str(row.get("symbol") or ""))) is not None and parsed["expiry"] > entry_ts
        }
    )
    if not expiries:
        raise ValueError("No future expiries found in snapshot")
    return min(expiries, key=lambda expiry: (abs((expiry - preferred_expiry).total_seconds()), expiry))


def select_leg(
    rows: list[dict[str, Any]],
    option_type: str,
    spot: float,
    target_delta: float,
    expiry: datetime,
    entry_ts: datetime,
) -> LegSnapshot:
    best: LegSnapshot | None = None
    best_diff = float("inf")
    T_years = max((expiry - entry_ts).total_seconds() / (365.25 * 86400), 1e-9)
    for row in rows:
        parsed = parse_symbol(str(row.get("symbol") or ""))
        if parsed is None or parsed["expiry"] != expiry or parsed["option_type"] != option_type:
            continue
        strike = float(parsed["strike"])
        if option_type == "call" and strike <= spot:
            continue
        if option_type == "put" and strike >= spot:
            continue
        price_btc = float(row.get("close") or 0.0)
        if price_btc <= 0:
            continue
        implied_vol = implied_volatility_btc(price_btc, spot, strike, T_years, option_type=option_type)
        if not np.isfinite(implied_vol) or implied_vol <= 0:
            continue
        abs_delta = abs(bs_delta(spot, strike, T_years, implied_vol, option_type=option_type, r=0.0))
        diff = abs(abs_delta - target_delta)
        if diff < best_diff:
            best_diff = diff
            best = LegSnapshot(
                symbol=str(row["symbol"]),
                strike=strike,
                option_type=option_type,
                entry_price_btc=price_btc,
                implied_vol=float(implied_vol),
                abs_delta=float(abs_delta),
                delta_diff=float(diff),
            )
    if best is None:
        raise ValueError(f"No {option_type} leg could be selected")
    return best


def intrinsic_btc(option_type: str, strike: float, spot: float) -> float:
    if option_type == "call":
        return max(spot - strike, 0.0) / spot if spot > 0 else 0.0
    return max(strike - spot, 0.0) / spot if spot > 0 else 0.0


def max_abs_move_pct(series: list[tuple[datetime, float]], entry_ts: datetime, end_ts: datetime) -> float:
    segment = [price for ts, price in series if entry_ts <= ts <= end_ts]
    if not segment:
        return 0.0
    entry_price = float(segment[0])
    return max(abs(price / entry_price - 1.0) * 100.0 for price in segment)


def evaluate_position(
    cfg: TraderConfig,
    spot_series: list[tuple[datetime, float]],
    expiry: datetime,
    entry_ts: datetime,
    sell_call: LegSnapshot,
    sell_put: LegSnapshot,
    cdd_client: CDDDailyClient,
) -> dict[str, Any]:
    entry_credit_btc_per_combo = sell_call.entry_price_btc + sell_put.entry_price_btc
    entry_spot = nearest_close_at_or_before(spot_series, entry_ts)
    quantity = float(cfg.strategy.quantity)
    history_points: list[dict[str, Any]] = []
    exit_reason = "settlement"
    exit_ts = expiry
    exit_spot = nearest_close_at_or_before(spot_series, expiry)
    exit_cost_btc_per_combo = (
        intrinsic_btc("call", sell_call.strike, exit_spot)
        + intrinsic_btc("put", sell_put.strike, exit_spot)
    )

    symbol_histories = {
        sell_call.symbol: {row["date"]: row for row in cdd_client.get_symbol_history(cfg.strategy.underlying.upper(), sell_call.symbol)},
        sell_put.symbol: {row["date"]: row for row in cdd_client.get_symbol_history(cfg.strategy.underlying.upper(), sell_put.symbol)},
    }

    current_day = entry_ts.date() + timedelta(days=1)
    while current_day < expiry.date():
        day_key = current_day.isoformat()
        call_row = symbol_histories[sell_call.symbol].get(day_key)
        put_row = symbol_histories[sell_put.symbol].get(day_key)
        if call_row and put_row:
            basket_close_cost = float(call_row.get("close") or 0.0) + float(put_row.get("close") or 0.0)
            pnl_pct = (entry_credit_btc_per_combo - basket_close_cost) / entry_credit_btc_per_combo * 100.0
            day_end = datetime.combine(current_day, time(23, 59), tzinfo=timezone.utc)
            move_pct = max_abs_move_pct(spot_series, entry_ts, min(day_end, expiry))
            history_points.append(
                {
                    "date": day_key,
                    "call_close_btc": float(call_row.get("close") or 0.0),
                    "put_close_btc": float(put_row.get("close") or 0.0),
                    "basket_close_cost_btc_per_combo": basket_close_cost,
                    "basket_pnl_pct": pnl_pct,
                    "max_abs_move_pct": move_pct,
                }
            )
            stop_loss_pct = float(cfg.strategy.stop_loss_pct or 0.0)
            move_filter_pct = float(cfg.strategy.stop_loss_underlying_move_pct or 0.0)
            if (
                stop_loss_pct > 0
                and pnl_pct <= -stop_loss_pct
                and (move_filter_pct <= 0 or move_pct >= move_filter_pct)
            ):
                exit_reason = "stop_loss"
                exit_ts = min(day_end, expiry)
                exit_spot = nearest_close_at_or_before(spot_series, exit_ts)
                exit_cost_btc_per_combo = basket_close_cost
                break
        current_day += timedelta(days=1)

    pnl_btc_per_combo = entry_credit_btc_per_combo - exit_cost_btc_per_combo
    pnl_btc_total = pnl_btc_per_combo * quantity
    pnl_usd_approx = pnl_btc_total * exit_spot
    entry_credit_usd_approx = entry_credit_btc_per_combo * quantity * entry_spot
    return {
        "entry_spot": entry_spot,
        "entry_credit_btc_per_combo": entry_credit_btc_per_combo,
        "entry_credit_usd_approx_total": entry_credit_usd_approx,
        "exit_reason": exit_reason,
        "exit_timestamp": exit_ts.isoformat(),
        "exit_spot": exit_spot,
        "exit_cost_btc_per_combo": exit_cost_btc_per_combo,
        "pnl_btc_per_combo": pnl_btc_per_combo,
        "pnl_btc_total": pnl_btc_total,
        "pnl_usd_approx_total": pnl_usd_approx,
        "return_on_credit_pct": (pnl_btc_per_combo / entry_credit_btc_per_combo * 100.0) if entry_credit_btc_per_combo > 0 else None,
        "daily_monitor_points": history_points,
    }


def run_analysis(as_of_date: date, config_path: str) -> dict[str, Any]:
    cfg = load_config(config_path)
    underlying = cfg.strategy.underlying.upper()
    entry_day = as_of_date - timedelta(days=(as_of_date.weekday() - 4) % 7)
    entry_ts = datetime.combine(entry_day, time.fromisoformat(cfg.strategy.entry_time_utc), tzinfo=timezone.utc)
    preferred_expiry = next_sunday_0800(entry_ts)

    deribit = DeribitPublicClient()
    cdd = CDDDailyClient(API_TOKEN)

    spot_history = build_hourly_series(
        deribit.get_chart_data(
            f"{underlying}-PERPETUAL",
            entry_ts - timedelta(hours=max(int(cfg.strategy.entry_realized_vol_lookback_hours), 24) + 4),
            preferred_expiry + timedelta(hours=2),
            resolution="60",
        )
    )
    if not spot_history:
        raise ValueError("No Deribit spot history returned")

    rv24 = None
    if int(cfg.strategy.entry_realized_vol_lookback_hours or 0) > 1:
        rv24 = compute_realized_vol_from_hourly(spot_history, entry_ts, int(cfg.strategy.entry_realized_vol_lookback_hours))

    snapshot_rows = cdd.get_date_snapshot(underlying, entry_day)
    expiry = choose_expiry(snapshot_rows, entry_ts, preferred_expiry)
    snapshot_spot = nearest_close_at_or_before(spot_history, entry_ts)

    sell_call = select_leg(snapshot_rows, "call", snapshot_spot, float(cfg.strategy.target_delta), expiry, entry_ts)
    sell_put = select_leg(snapshot_rows, "put", snapshot_spot, float(cfg.strategy.target_delta), expiry, entry_ts)

    evaluation = evaluate_position(cfg, spot_history, expiry, entry_ts, sell_call, sell_put, cdd)
    target_diff_hours = (expiry - preferred_expiry).total_seconds() / 3600.0

    return {
        "as_of_date": as_of_date.isoformat(),
        "config_path": config_path,
        "analysis_mode": "Deribit public hourly spot + CDD daily option OHLC proxy",
        "notes": [
            "期权入场价使用 CDD 的日线 close 近似，不是精确到 18:00 的盘口成交价。",
            "止损检查使用日线 close 近似，不是实盘逐小时 mark。",
            "到期价值按 Deribit BTC-PERPETUAL 小时线在到期时点附近的价格计算内在价值。",
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
            "entry_spot": snapshot_spot,
            "rv_filter_value": rv24,
            "rv_filter_passed": rv24 is None or rv24 <= float(cfg.strategy.entry_realized_vol_max),
        },
        "expiry": {
            "preferred_expiry": preferred_expiry.isoformat(),
            "selected_expiry": expiry.isoformat(),
            "selected_vs_preferred_diff_hours": target_diff_hours,
        },
        "selected_legs": {
            "sell_call": asdict(sell_call),
            "sell_put": asdict(sell_put),
        },
        "result": evaluation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Approximate this week's WeekendVol PnL using Deribit public APIs")
    parser.add_argument("--as-of-date", default=datetime.now(timezone.utc).date().isoformat(), help="Reference date in UTC, e.g. 2026-04-12")
    parser.add_argument("--config", default="configs/trader/weekend_vol_btc.yaml", help="Trader config path")
    parser.add_argument("--output", default="reports/validation/current_week_deribit_weekend_vol.json", help="Output JSON path")
    args = parser.parse_args()

    as_of_date = date.fromisoformat(args.as_of_date)
    result = run_analysis(as_of_date, args.config)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved report to {output_path}")


if __name__ == "__main__":
    main()
