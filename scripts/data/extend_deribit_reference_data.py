#!/usr/bin/env python3
"""Extend the Deribit reference data used by hourly backtests up to today.

Appends to the existing files instead of rebuilding them:
  * ``data/settlements/<ul>_settlements.parquet``  delivery prices
  * ``data/instruments/<ul>_instruments.parquet``  settled, listed and hourly-quoted options
  * ``data/underlying/<ul>_dvol_1D.parquet``       DVOL daily candles
  * ``data/underlying/<ul>_index_60.parquet``      hourly perpetual candles

Run from the directory that holds ``data/``:
    python scripts/data/extend_deribit_reference_data.py --underlying BTC --proxy http://127.0.0.1:7897
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from options_backtest.data.fetcher import parse_instrument_name  # noqa: E402

API = "https://www.deribit.com/api/v2/public/"
DATA_DIR = Path("data")


def atomic_write(path: Path, df: pd.DataFrame) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class Deribit:
    def __init__(self, proxy: str | None):
        self.session = requests.Session()
        self.session.trust_env = False
        if proxy:
            self.session.proxies = {"http": proxy, "https": proxy}

    def get(self, method: str, **params):
        last: Exception | None = None
        for attempt in range(6):
            try:
                r = self.session.get(API + method, params=params, timeout=30)
                if r.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                return r.json()["result"]
            except (requests.RequestException, KeyError, ValueError) as e:
                last = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"Deribit {method} failed: {last}")


def extend_settlements(api: Deribit, underlying: str) -> pd.DataFrame:
    path = DATA_DIR / "settlements" / f"{underlying.lower()}_settlements.parquet"
    existing = pd.read_parquet(path)
    last_ts = existing["settlement_timestamp"].max()
    rows, continuation = [], None
    while True:
        params = {"currency": underlying, "type": "delivery", "count": 1000}
        if continuation:
            params["continuation"] = continuation
        page = api.get("get_last_settlements_by_currency", **params)
        batch = page.get("settlements", [])
        fresh = [s for s in batch if pd.Timestamp(s["timestamp"], unit="ms", tz="UTC") > last_ts]
        rows.extend(fresh)
        continuation = page.get("continuation")
        # Pages run newest to oldest, so the first already-known record ends the scan.
        if len(fresh) < len(batch) or not batch or not continuation:
            break
    if not rows:
        print(f"settlements: already up to date ({last_ts})")
        return existing.iloc[0:0]
    new = pd.DataFrame({
        "instrument_name": [s["instrument_name"] for s in rows],
        "settlement_timestamp": pd.to_datetime([s["timestamp"] for s in rows], unit="ms", utc=True),
        "index_price": [float(s.get("index_price", 0.0)) for s in rows],
        "mark_price": [s.get("mark_price") for s in rows],
        "session_profit_loss": [s.get("session_profit_loss") for s in rows],
        "type": [s.get("type", "delivery") for s in rows],
    }).astype({"mark_price": "float64", "session_profit_loss": "float64"})
    new["settlement_timestamp"] = new["settlement_timestamp"].astype(existing["settlement_timestamp"].dtype)
    merged = pd.concat([new, existing], ignore_index=True).drop_duplicates(
        subset=["instrument_name", "settlement_timestamp"], keep="last")
    atomic_write(path, merged)
    print(f"settlements: +{len(new)} rows, {last_ts} -> {merged['settlement_timestamp'].max()}")
    return new


def hourly_symbols(underlying: str, since: pd.Timestamp) -> set[str]:
    names: set[str] = set()
    for p in sorted((DATA_DIR / "options_hourly" / underlying).glob("????-??.parquet")):
        if p.stem >= f"{since:%Y-%m}":
            names.update(pd.read_parquet(p, columns=["symbol"])["symbol"].unique())
    return names


def extend_instruments(api: Deribit, underlying: str, settled: pd.DataFrame, since: pd.Timestamp) -> None:
    path = DATA_DIR / "instruments" / f"{underlying.lower()}_instruments.parquet"
    existing = pd.read_parquet(path)
    known = set(existing["instrument_name"])
    active = {r["instrument_name"] for r in api.get("get_instruments", currency=underlying, kind="option", expired="false")}
    candidates = set(settled["instrument_name"]) | hourly_symbols(underlying, since) | active
    rows = []
    for name in sorted(candidates - known):
        parsed = parse_instrument_name(str(name))
        if parsed:
            parsed["is_active"] = True if name in active else None
            rows.append(parsed)
    if not rows:
        print("instruments: already up to date")
        return
    new = pd.DataFrame(rows)
    new["expiration_date"] = pd.to_datetime(new["expiration_date"], utc=True).astype(existing["expiration_date"].dtype)
    merged = pd.concat([existing, new[existing.columns]], ignore_index=True).sort_values(
        ["expiration_date", "strike_price", "option_type"]).reset_index(drop=True)
    atomic_write(path, merged)
    print(f"instruments: +{len(new)} rows, total {len(merged)}, max expiry {merged['expiration_date'].max()}")


def extend_dvol(api: Deribit, underlying: str) -> None:
    path = DATA_DIR / "underlying" / f"{underlying.lower()}_dvol_1D.parquet"
    if not path.exists():
        print("dvol: no existing file, skipped")
        return
    existing = pd.read_parquet(path)
    last_ts = existing["timestamp"].max()
    start_ms = int(last_ts.timestamp() * 1000)
    end_ms = int(time.time() * 1000)
    candles = []
    while True:
        page = api.get("get_volatility_index_data", currency=underlying, resolution="1D",
                       start_timestamp=start_ms, end_timestamp=end_ms)
        candles.extend(page.get("data", []))
        # Results are returned newest-first in pages; continuation is the next end timestamp.
        if not page.get("continuation"):
            break
        end_ms = int(page["continuation"])
    # The current UTC day is still forming; keep completed candles only.
    today = pd.Timestamp.now(tz="UTC").normalize()
    new = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close"])
    new["timestamp"] = pd.to_datetime(new["timestamp"], unit="ms", utc=True).astype(existing["timestamp"].dtype)
    new = new[(new["timestamp"] >= last_ts) & (new["timestamp"] < today)]
    merged = pd.concat([existing, new], ignore_index=True).drop_duplicates(
        subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)
    atomic_write(path, merged)
    print(f"dvol: +{len(merged) - len(existing)} rows, {last_ts} -> {merged['timestamp'].max()}")


def extend_underlying(api: Deribit, underlying: str) -> None:
    """Refresh the hourly underlying file with real perpetual candles.

    Bars after the last traded one (volume > 0) are replaced: rows synthesised
    from the option chain's median ``underlying_price`` sit above spot because
    far expiries quote their futures price, which skews strike selection.
    """
    path = DATA_DIR / "underlying" / f"{underlying.lower()}_index_60.parquet"
    if not path.exists():
        print("underlying: no existing file, skipped")
        return
    existing = pd.read_parquet(path)
    last_real = existing.loc[existing["volume"] > 0, "timestamp"].max()
    hour_ms = 3_600_000
    start_ms = int(last_real.timestamp() * 1000) + hour_ms
    # The current hour is still forming; keep completed candles only.
    end_ms = int(time.time() * 1000) // hour_ms * hour_ms - hour_ms
    frames = []
    while start_ms <= end_ms:
        chunk_end = min(start_ms + 4500 * hour_ms, end_ms)
        page = api.get("get_tradingview_chart_data", instrument_name=f"{underlying}-PERPETUAL",
                       start_timestamp=start_ms, end_timestamp=chunk_end, resolution="60")
        if page.get("status") != "ok" or not page.get("ticks"):
            break
        frames.append(pd.DataFrame({k: page[k] for k in ("ticks", "open", "high", "low", "close", "volume")}))
        start_ms = int(page["ticks"][-1]) + hour_ms
    if not frames:
        print(f"underlying: already up to date ({last_real})")
        return
    new = pd.concat(frames, ignore_index=True).rename(columns={"ticks": "timestamp"})
    new["timestamp"] = pd.to_datetime(new["timestamp"], unit="ms", utc=True).astype(existing["timestamp"].dtype)
    new["underlying"] = underlying
    new = new[new["timestamp"] > last_real]
    # Hours the exchange has no candle for keep their previous row.
    kept = existing[(existing["timestamp"] <= last_real) | ~existing["timestamp"].isin(new["timestamp"])]
    merged = pd.concat([kept, new[existing.columns]], ignore_index=True).drop_duplicates(
        subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)
    atomic_write(path, merged)
    synthetic = int((merged["volume"] <= 0).sum())
    print(f"underlying: {len(new)} real bars after {last_real}, max {merged['timestamp'].max()}, "
          f"{synthetic} bars without traded volume remain")

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--underlying", default="BTC")
    ap.add_argument("--proxy", default=None)
    args = ap.parse_args()
    underlying = args.underlying.upper()
    api = Deribit(args.proxy)
    settlements_path = DATA_DIR / "settlements" / f"{underlying.lower()}_settlements.parquet"
    since = pd.read_parquet(settlements_path, columns=["settlement_timestamp"])["settlement_timestamp"].max()
    settled = extend_settlements(api, underlying)
    extend_instruments(api, underlying, settled, since)
    extend_dvol(api, underlying)
    extend_underlying(api, underlying)


if __name__ == "__main__":
    main()
