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

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from options_backtest.pricing.black76 import delta as bs_delta
from options_backtest.pricing.iv_solver import implied_volatility
from trader.config import TraderConfig, load_config

BINANCE_EAPI_BASE = "https://eapi.binance.com"
BINANCE_SPOT_BASE = "https://api.binance.com"
SYMBOL_RE = re.compile(r"^(?P<ul>[A-Z]+)-(?P<yymmdd>\d{6})-(?P<strike>\d+(?:\.\d+)*)-(?P<cp>[CP])$")


@dataclass
class BinanceInstrument:
    symbol: str
    strike: float
    option_type: str
    expiry: datetime
    min_qty: float


@dataclass
class LegBar:
    symbol: str
    strike: float
    option_type: str
    bar_close_time: str
    close_price: float
    implied_vol: float
    abs_delta: float
    delta_diff: float


class BinanceOptionsClient:
    def __init__(self) -> None:
        self.session = requests.Session()

    def get_exchange_info(self) -> dict[str, Any]:
        response = self.session.get(f"{BINANCE_EAPI_BASE}/eapi/v1/exchangeInfo", timeout=30)
        response.raise_for_status()
        return response.json()

    def get_option_klines(
        self,
        symbol: str,
        start_ts: datetime,
        end_ts: datetime,
        interval: str = "1h",
        limit: int = 1000,
    ) -> list[list[Any]]:
        response = self.session.get(
            f"{BINANCE_EAPI_BASE}/eapi/v1/klines",
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": int(start_ts.timestamp() * 1000),
                "endTime": int(end_ts.timestamp() * 1000),
                "limit": limit,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and payload.get("code"):
            raise RuntimeError(str(payload))
        return list(payload)

    def get_spot_klines(
        self,
        symbol: str,
        start_ts: datetime,
        end_ts: datetime,
        interval: str = "1h",
        limit: int = 1000,
    ) -> list[list[Any]]:
        response = self.session.get(
            f"{BINANCE_SPOT_BASE}/api/v3/klines",
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": int(start_ts.timestamp() * 1000),
                "endTime": int(end_ts.timestamp() * 1000),
                "limit": limit,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and payload.get("code"):
            raise RuntimeError(str(payload))
        return list(payload)


def parse_binance_symbol(symbol: str) -> dict[str, Any] | None:
    match = SYMBOL_RE.match(symbol)
    if not match:
        return None
    yymmdd = match.group("yymmdd")
    expiry = datetime(
        2000 + int(yymmdd[:2]),
        int(yymmdd[2:4]),
        int(yymmdd[4:6]),
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


def load_instruments(exchange_info: dict[str, Any], underlying: str) -> list[BinanceInstrument]:
    instruments: list[BinanceInstrument] = []
    for row in exchange_info.get("optionSymbols") or []:
        symbol = str(row.get("symbol") or "")
        parsed = parse_binance_symbol(symbol)
        if parsed is None:
            continue
        if parsed["underlying"] != underlying.upper():
            continue
        if str(row.get("status") or "") != "TRADING":
            continue
        instruments.append(
            BinanceInstrument(
                symbol=symbol,
                strike=float(parsed["strike"]),
                option_type=str(parsed["option_type"]),
                expiry=parsed["expiry"],
                min_qty=float(row.get("minQty") or 0.0),
            )
        )
    return instruments


def choose_expiry(instruments: list[BinanceInstrument], entry_ts: datetime, preferred_expiry: datetime) -> datetime:
    expiries = sorted({inst.expiry for inst in instruments if inst.expiry > entry_ts})
    if not expiries:
        raise ValueError("No future Binance option expiries available")
    return min(expiries, key=lambda expiry: (abs((expiry - preferred_expiry).total_seconds()), expiry))


def build_spot_series(klines: list[list[Any]]) -> list[tuple[datetime, float]]:
    return [
        (
            datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc) + timedelta(hours=1),
            float(row[4]),
        )
        for row in klines
    ]


def nearest_spot_at_or_before(series: list[tuple[datetime, float]], ts: datetime) -> float:
    eligible = [price for dt_value, price in series if dt_value <= ts]
    if not eligible:
        raise ValueError(f"No spot data at or before {ts.isoformat()}")
    return float(eligible[-1])


def compute_realized_vol(series: list[tuple[datetime, float]], end_ts: datetime, lookback_hours: int) -> float | None:
    if lookback_hours <= 1:
        return None
    closes = [price for ts, price in series if ts <= end_ts][-lookback_hours - 1 :]
    if len(closes) < 3:
        return None
    log_returns = [math.log(cur / prev) for prev, cur in zip(closes[:-1], closes[1:], strict=False) if prev > 0 and cur > 0]
    if len(log_returns) < 2:
        return None
    mean = sum(log_returns) / len(log_returns)
    variance = sum((x - mean) ** 2 for x in log_returns) / (len(log_returns) - 1)
    return math.sqrt(variance) * math.sqrt(24 * 365.25)


def last_bar_at_or_before(klines: list[list[Any]], ts: datetime) -> list[Any] | None:
    eligible = [
        row
        for row in klines
        if (datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc) + timedelta(hours=1)) <= ts
    ]
    if not eligible:
        return None
    return eligible[-1]


def build_leg_from_bar(
    instrument: BinanceInstrument,
    bar: list[Any],
    spot: float,
    entry_ts: datetime,
    target_delta: float,
) -> LegBar | None:
    close_price = float(bar[4])
    if close_price <= 0 or spot <= 0:
        return None
    T_years = max((instrument.expiry - entry_ts).total_seconds() / (365.25 * 86400), 1e-9)
    iv = implied_volatility(close_price, spot, instrument.strike, T_years, option_type=instrument.option_type, r=0.0)
    if not math.isfinite(iv) or iv <= 0:
        return None
    abs_delta = abs(bs_delta(spot, instrument.strike, T_years, iv, option_type=instrument.option_type, r=0.0))
    return LegBar(
        symbol=instrument.symbol,
        strike=instrument.strike,
        option_type=instrument.option_type,
        bar_close_time=(datetime.fromtimestamp(int(bar[0]) / 1000, tz=timezone.utc) + timedelta(hours=1)).isoformat(),
        close_price=close_price,
        implied_vol=float(iv),
        abs_delta=float(abs_delta),
        delta_diff=float(abs(abs_delta - target_delta)),
    )


def select_leg(candidates: list[LegBar], option_type: str, spot: float, target_delta: float, max_delta_diff: float) -> LegBar:
    filtered: list[LegBar] = []
    for leg in candidates:
        if leg.option_type != option_type:
            continue
        if option_type == "call" and leg.strike <= spot:
            continue
        if option_type == "put" and leg.strike >= spot:
            continue
        if leg.delta_diff > max_delta_diff:
            continue
        filtered.append(leg)
    if not filtered:
        raise ValueError(f"No eligible Binance {option_type} leg found")
    return min(filtered, key=lambda leg: (leg.delta_diff, abs(leg.strike - spot), leg.strike))


def build_hourly_close_series(klines: list[list[Any]], start_ts: datetime, end_ts: datetime) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in klines:
        ts = datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc) + timedelta(hours=1)
        if ts < start_ts or ts > end_ts:
            continue
        out.append(
            {
                "timestamp": ts.isoformat(),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "trade_count": int(row[8]),
            }
        )
    return out


def intrinsic_value(option_type: str, strike: float, spot: float) -> float:
    if option_type == "call":
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


def evaluate_position(
    cfg: TraderConfig,
    spot_series: list[tuple[datetime, float]],
    sell_call: LegBar,
    sell_put: LegBar,
    call_series: list[dict[str, Any]],
    put_series: list[dict[str, Any]],
    entry_ts: datetime,
    expiry: datetime,
) -> dict[str, Any]:
    entry_credit_per_combo = sell_call.close_price + sell_put.close_price
    quantity = float(cfg.strategy.quantity)
    entry_spot = nearest_spot_at_or_before(spot_series, entry_ts)
    by_ts_call = {row["timestamp"]: row for row in call_series}
    by_ts_put = {row["timestamp"]: row for row in put_series}
    hourly_monitor_points: list[dict[str, Any]] = []
    exit_reason = "settlement"
    exit_ts = expiry
    exit_cost_per_combo = intrinsic_value("call", sell_call.strike, nearest_spot_at_or_before(spot_series, expiry)) + intrinsic_value("put", sell_put.strike, nearest_spot_at_or_before(spot_series, expiry))
    exit_spot = nearest_spot_at_or_before(spot_series, expiry)

    cursor = (entry_ts + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    stop_loss_pct = float(cfg.strategy.stop_loss_pct or 0.0)
    move_filter_pct = float(cfg.strategy.stop_loss_underlying_move_pct or 0.0)
    while cursor < expiry:
        ts_iso = cursor.isoformat()
        call_bar = by_ts_call.get(ts_iso)
        put_bar = by_ts_put.get(ts_iso)
        if call_bar and put_bar:
            basket_cost = float(call_bar["close"]) + float(put_bar["close"])
            basket_pnl_pct = (entry_credit_per_combo - basket_cost) / entry_credit_per_combo * 100.0
            spot_now = nearest_spot_at_or_before(spot_series, cursor)
            move_pct = abs(spot_now / entry_spot - 1.0) * 100.0
            point = {
                "timestamp": ts_iso,
                "call_close": float(call_bar["close"]),
                "put_close": float(put_bar["close"]),
                "basket_close_cost_per_combo": basket_cost,
                "basket_pnl_pct": basket_pnl_pct,
                "spot": spot_now,
                "abs_move_pct": move_pct,
                "call_trade_count": int(call_bar["trade_count"]),
                "put_trade_count": int(put_bar["trade_count"]),
            }
            hourly_monitor_points.append(point)
            if stop_loss_pct > 0 and basket_pnl_pct <= -stop_loss_pct and (move_filter_pct <= 0 or move_pct >= move_filter_pct):
                exit_reason = "stop_loss"
                exit_ts = cursor
                exit_spot = spot_now
                exit_cost_per_combo = basket_cost
                break
        cursor += timedelta(hours=1)

    pnl_per_combo = entry_credit_per_combo - exit_cost_per_combo
    pnl_total = pnl_per_combo * quantity
    pnl_usd = pnl_total
    return {
        "entry_credit_per_combo": entry_credit_per_combo,
        "entry_credit_total": entry_credit_per_combo * quantity,
        "exit_reason": exit_reason,
        "exit_timestamp": exit_ts.isoformat(),
        "exit_spot": exit_spot,
        "exit_cost_per_combo": exit_cost_per_combo,
        "pnl_per_combo": pnl_per_combo,
        "pnl_total": pnl_total,
        "pnl_usd_approx_total": pnl_usd,
        "return_on_credit_pct": (pnl_per_combo / entry_credit_per_combo * 100.0) if entry_credit_per_combo > 0 else None,
        "hourly_monitor_points": hourly_monitor_points,
    }


def run_analysis(as_of_date: date, config_path: str) -> dict[str, Any]:
    cfg = load_config(config_path)
    underlying = cfg.strategy.underlying.upper()
    entry_day = as_of_date - timedelta(days=(as_of_date.weekday() - 4) % 7)
    entry_ts = datetime.combine(entry_day, time.fromisoformat(cfg.strategy.entry_time_utc), tzinfo=timezone.utc)
    preferred_expiry = next_sunday_0800(entry_ts)

    client = BinanceOptionsClient()
    exchange_info = client.get_exchange_info()
    instruments = load_instruments(exchange_info, underlying)
    expiry = choose_expiry(instruments, entry_ts, preferred_expiry)
    target_instruments = [inst for inst in instruments if inst.expiry == expiry]

    spot_klines = client.get_spot_klines(
        f"{underlying}USDT",
        entry_ts - timedelta(hours=max(int(cfg.strategy.entry_realized_vol_lookback_hours or 0), 24) + 6),
        expiry + timedelta(hours=2),
        interval="1h",
    )
    spot_series = build_spot_series(spot_klines)
    entry_spot = nearest_spot_at_or_before(spot_series, entry_ts)
    rv_value = compute_realized_vol(spot_series, entry_ts, int(cfg.strategy.entry_realized_vol_lookback_hours or 0))

    leg_candidates: list[LegBar] = []
    kline_cache: dict[str, list[list[Any]]] = {}
    start_ts = entry_ts - timedelta(hours=12)
    for inst in target_instruments:
        klines = client.get_option_klines(inst.symbol, start_ts, expiry + timedelta(hours=1), interval="1h")
        kline_cache[inst.symbol] = klines
        bar = last_bar_at_or_before(klines, entry_ts)
        if bar is None:
            continue
        leg = build_leg_from_bar(inst, bar, entry_spot, entry_ts, float(cfg.strategy.target_delta))
        if leg is not None:
            leg_candidates.append(leg)

    sell_call = select_leg(leg_candidates, "call", entry_spot, float(cfg.strategy.target_delta), float(cfg.strategy.max_delta_diff))
    sell_put = select_leg(leg_candidates, "put", entry_spot, float(cfg.strategy.target_delta), float(cfg.strategy.max_delta_diff))

    call_series = build_hourly_close_series(kline_cache[sell_call.symbol], entry_ts, expiry)
    put_series = build_hourly_close_series(kline_cache[sell_put.symbol], entry_ts, expiry)
    result = evaluate_position(cfg, spot_series, sell_call, sell_put, call_series, put_series, entry_ts, expiry)

    return {
        "as_of_date": as_of_date.isoformat(),
        "config_path": config_path,
        "analysis_mode": "Binance Options hourly kline replay",
        "notes": [
            "期权价格使用 Binance Options 1h kline close。",
            "Binance 本周没有周日到期合约，因此选取入场后最接近周日 08:00 的可交易到期日。",
            "delta 由期权小时 close + BTCUSDT 小时 close 反推 IV 后计算。",
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
        "candidate_count": len(leg_candidates),
        "target_expiry_instrument_count": len(target_instruments),
        "selected_legs": {
            "sell_call": asdict(sell_call),
            "sell_put": asdict(sell_put),
        },
        "result": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay current week using Binance Options hourly klines")
    parser.add_argument("--as-of-date", default=datetime.now(timezone.utc).date().isoformat())
    parser.add_argument("--config", default="configs/trader/weekend_vol_btc.yaml")
    parser.add_argument("--output", default="reports/validation/current_week_binance_options_hourly.json")
    args = parser.parse_args()

    result = run_analysis(date.fromisoformat(args.as_of_date), args.config)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nSaved report to {output_path}")


if __name__ == "__main__":
    main()
