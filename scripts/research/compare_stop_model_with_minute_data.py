#!/usr/bin/env python3
from __future__ import annotations

"""Compare estimated stop prices vs minute-level real quotes for sample days.

Defaults target the previously identified top-risk dates for the best
`IntradayMoveFade0DTE` configuration.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from options_backtest.pricing.black76 import option_price, option_price_btc

LOG = logging.getLogger("compare_stop_model_with_minute_data")
DEFAULT_MINUTE_DIR = Path("data/options_minute_daily")
DEFAULT_RANKING = Path("reports/optimizations/intraday_move_fade_0dte/best_strategy_adverse_move_ranking.csv")
DEFAULT_OUTPUT = Path("reports/optimizations/intraday_move_fade_0dte_stop_grid/minute_stop_price_comparison.csv")
YEAR_MINUTES = 365.0 * 24.0 * 60.0
DEFAULT_SAMPLE_DATES = [
    "2025-11-21",
    "2024-08-05",
    "2025-01-20",
    "2024-03-19",
    "2024-11-06",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare estimated stop prices with minute-level real quotes.")
    parser.add_argument("--dates", type=str, default=",".join(DEFAULT_SAMPLE_DATES), help="Comma-separated entry dates YYYY-MM-DD")
    parser.add_argument("--stop-pct", type=float, default=1.0, help="Underlying adverse move stop threshold, percent")
    parser.add_argument("--sigma", type=float, default=0.6, help="Black-76 sigma used by the model")
    parser.add_argument("--minute-dir", type=str, default=str(DEFAULT_MINUTE_DIR), help="Minute parquet root")
    parser.add_argument("--ranking-csv", type=str, default=str(DEFAULT_RANKING), help="Entry ranking CSV")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output CSV path")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return parser.parse_args()


def _parse_dates(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _pick_actual_stop_price(row: pd.Series) -> tuple[float | None, str]:
    for col in ("ask_price", "mark_price", "last_price", "bid_price"):
        val = row.get(col)
        if pd.notna(val):
            return float(val), col
    return None, "missing"


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    dates = _parse_dates(args.dates)
    stop_frac = float(args.stop_pct) / 100.0
    minute_dir = Path(args.minute_dir)
    ranking_path = Path(args.ranking_csv)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ranking = pd.read_csv(ranking_path)
    ranking["entry_date"] = pd.to_datetime(ranking["entry_time"], utc=True).dt.strftime("%Y-%m-%d")

    rows: list[dict] = []
    for entry_date in dates:
        match = ranking[ranking["entry_date"] == entry_date]
        if match.empty:
            LOG.warning("No ranking row for %s", entry_date)
            continue
        case = match.iloc[0]
        underlying = str(case["instrument_name"]).split("-")[0].upper()
        minute_path = minute_dir / underlying / f"{entry_date}.parquet"
        if not minute_path.exists():
            LOG.warning("Minute parquet missing for %s: %s", entry_date, minute_path)
            continue

        minute_df = pd.read_parquet(minute_path)
        minute_df["minute"] = pd.to_datetime(minute_df["minute"], utc=True)
        minute_df["expiration"] = pd.to_datetime(minute_df["expiration"], utc=True, errors="coerce")
        symbol = str(case["instrument_name"])
        symbol_df = minute_df[minute_df["symbol"] == symbol].copy().sort_values("minute")
        if symbol_df.empty:
            LOG.warning("No symbol rows for %s in %s", symbol, minute_path)
            continue

        minute_ohlc = minute_df[["minute", "underlying_open", "underlying_high", "underlying_low", "underlying_close"]].drop_duplicates("minute").sort_values("minute")
        entry_time = pd.Timestamp(case["entry_time"])
        if entry_time.tzinfo is None:
            entry_time = entry_time.tz_localize("UTC")
        expiry_time = pd.Timestamp(symbol_df["expiration"].dropna().iloc[0]) if symbol_df["expiration"].notna().any() else (entry_time + pd.Timedelta(hours=1))
        option_type = str(case["option_type"]).lower()
        entry_spot = float(case["spot_at_entry"])
        strike = float(case["strike_price"])
        stop_spot = entry_spot * (1.0 + stop_frac) if option_type == "call" else entry_spot * (1.0 - stop_frac)

        minute_ohlc = minute_ohlc[minute_ohlc["minute"] >= entry_time.floor("min")]
        if option_type == "call":
            trigger = minute_ohlc[minute_ohlc["underlying_high"] >= stop_spot]
        else:
            trigger = minute_ohlc[minute_ohlc["underlying_low"] <= stop_spot]
        triggered = not trigger.empty
        if not triggered:
            rows.append({
                "entry_date": entry_date,
                "instrument_name": symbol,
                "stop_pct": args.stop_pct,
                "triggered": False,
                "entry_spot": entry_spot,
                "stop_spot": stop_spot,
            })
            continue

        trigger_row = trigger.iloc[0]
        trigger_minute = pd.Timestamp(trigger_row["minute"])
        quote_row = symbol_df[symbol_df["minute"] == trigger_minute]
        quote_source = "same_minute"
        if quote_row.empty:
            quote_row = symbol_df[symbol_df["minute"] > trigger_minute].head(1)
            quote_source = "next_available_minute"
        if quote_row.empty:
            actual_price = None
            actual_price_field = "missing"
            actual_quote_minute = None
            actual_underlying_close = None
        else:
            qr = quote_row.iloc[0]
            actual_price, actual_price_field = _pick_actual_stop_price(qr)
            actual_quote_minute = pd.Timestamp(qr["minute"])
            underlying_close_val = qr.get("underlying_close")
            actual_underlying_close = float(underlying_close_val) if pd.notna(underlying_close_val) else None

        est_30m_usd = option_price(stop_spot, strike, 30.0 / YEAR_MINUTES, args.sigma, option_type)
        est_30m = option_price_btc(stop_spot, strike, 30.0 / YEAR_MINUTES, args.sigma, option_type)
        minutes_left = max((expiry_time - trigger_minute).total_seconds() / 60.0, 1.0)
        est_realmin_usd = option_price(stop_spot, strike, minutes_left / YEAR_MINUTES, args.sigma, option_type)
        est_realmin = option_price_btc(stop_spot, strike, minutes_left / YEAR_MINUTES, args.sigma, option_type)
        rows.append({
            "entry_date": entry_date,
            "instrument_name": symbol,
            "option_type": option_type,
            "stop_pct": args.stop_pct,
            "triggered": True,
            "entry_time": entry_time,
            "trigger_minute": trigger_minute,
            "quote_source": quote_source,
            "actual_quote_minute": actual_quote_minute,
            "entry_spot": entry_spot,
            "stop_spot": stop_spot,
            "strike_price": strike,
            "underlying_low_trigger_minute": float(trigger_row["underlying_low"]),
            "underlying_high_trigger_minute": float(trigger_row["underlying_high"]),
            "actual_stop_price": actual_price,
            "actual_stop_price_field": actual_price_field,
            "actual_underlying_close_quote_minute": actual_underlying_close,
            "estimated_stop_price_30m_usd": est_30m_usd,
            "estimated_stop_price_30m": est_30m,
            "estimated_stop_price_minutes_left_usd": est_realmin_usd,
            "estimated_stop_price_minutes_left": est_realmin,
            "actual_minus_est_30m": (actual_price - est_30m) if actual_price is not None else None,
            "actual_minus_est_minutes_left": (actual_price - est_realmin) if actual_price is not None else None,
            "actual_vs_est_30m_pct": ((actual_price / est_30m - 1.0) * 100.0) if (actual_price is not None and est_30m > 0) else None,
            "actual_vs_est_minutes_left_pct": ((actual_price / est_realmin - 1.0) * 100.0) if (actual_price is not None and est_realmin > 0) else None,
        })

    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
