"""
Targeted Deribit Trades Fetcher – 8:00 UTC snapshots only
==========================================================
For each day in the backtest range, fetch real trades in a narrow window
around 08:00 UTC for instruments that would be relevant to the strategy
(strikes within ±pct of spot, DTE 0-8).

Trades are aggregated into a single 1h OHLCV bar (07:00-08:00 UTC) per
instrument-day and merged into existing parquet files under
``data/market_data/60/``.

Uses: ``/public/get_last_trades_by_instrument_and_time``
"""
from __future__ import annotations

import asyncio
import argparse
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import aiohttp
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, "src")
from options_backtest.data.fetcher import parse_instrument_name

# ── Config ─────────────────────────────────────────────────────────────
BASE_URL = "https://www.deribit.com/api/v2"
MAX_RPS = 16           # slightly conservative
MAX_CONCURRENT = 12    # simultaneous connections
STRIKE_PCT = 0.10      # fetch strikes within ±10% of spot
MAX_DTE = 1            # fetch instruments expiring within 0-1 days (0DTE)
WINDOW_MINUTES = 30    # trades within [07:30, 08:30] UTC

DATA_DIR = Path("data")
OHLCV_DIR = DATA_DIR / "market_data" / "60"


# ── Rate Limiter ───────────────────────────────────────────────────────
class RateLimiter:
    def __init__(self, rps: int = MAX_RPS):
        self._tokens = float(rps)
        self._max = float(rps)
        self._rate = float(rps)
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        while True:
            async with self._lock:
                now = time.monotonic()
                self._tokens = min(self._max, self._tokens + (now - self._last) * self._rate)
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
            await asyncio.sleep(1.0 / self._rate)


# ── Fetch helpers ──────────────────────────────────────────────────────
_limiter = RateLimiter()
_session: aiohttp.ClientSession | None = None


async def _get_session() -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        conn = aiohttp.TCPConnector(limit=MAX_CONCURRENT, keepalive_timeout=30)
        _session = aiohttp.ClientSession(connector=conn)
    return _session


async def _get(endpoint: str, params: dict, retries: int = 4):
    session = await _get_session()
    url = f"{BASE_URL}{endpoint}"
    for attempt in range(retries):
        await _limiter.acquire()
        try:
            async with session.get(url, params=params,
                                   timeout=aiohttp.ClientTimeout(total=20)) as resp:
                if resp.status == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
                if resp.status >= 400:
                    return None
                data = await resp.json()
                return data.get("result")
        except (aiohttp.ClientError, asyncio.TimeoutError):
            await asyncio.sleep(2 ** attempt)
    return None


async def fetch_trades_window(instrument: str, start_ms: int, end_ms: int) -> list[dict]:
    """Fetch all trades for *instrument* between start_ms and end_ms.

    Handles pagination (Deribit returns max 1000 per call).
    """
    all_trades: list[dict] = []
    has_more = True

    params = {
        "instrument_name": instrument,
        "start_timestamp": start_ms,
        "end_timestamp": end_ms,
        "count": 1000,
        "sorting": "asc",
    }

    while has_more:
        result = await _get("/public/get_last_trades_by_instrument_and_time", params)
        if not result:
            break
        trades = result.get("trades", [])
        if not trades:
            break
        all_trades.extend(trades)
        has_more = result.get("has_more", False)
        if has_more and trades:
            # Continue from after the last trade
            params["start_timestamp"] = trades[-1]["timestamp"] + 1

    return all_trades


def trades_to_ohlcv(trades: list[dict], bar_ts: pd.Timestamp, instrument: str) -> dict | None:
    """Aggregate trades into a single OHLCV bar."""
    if not trades:
        return None
    prices = [t["price"] for t in trades]
    volumes = [t["amount"] for t in trades]
    return {
        "timestamp": bar_ts,
        "open": prices[0],
        "high": max(prices),
        "low": min(prices),
        "close": prices[-1],
        "volume": sum(volumes),
        "instrument_name": instrument,
        "trade_count": len(trades),
    }


# ── Main pipeline ─────────────────────────────────────────────────────
async def run(underlying: str, start_date: str, end_date: str,
              dry_run: bool = False, strike_pct: float = STRIKE_PCT,
              max_dte: int = MAX_DTE):
    """Main fetch pipeline."""

    # 1. Load instruments catalogue
    inst_path = DATA_DIR / "instruments" / f"{underlying.lower()}_instruments.parquet"
    if not inst_path.exists():
        print(f"ERROR: instruments file not found: {inst_path}")
        print("Run `options-bt fetch` first to get the instrument catalogue.")
        return
    inst_df = pd.read_parquet(inst_path)
    inst_df["expiration_date"] = pd.to_datetime(inst_df["expiration_date"], utc=True)
    print(f"Loaded {len(inst_df)} instruments from {inst_path}")

    # 2. Load underlying prices (to determine ATM strike each day)
    und_path = DATA_DIR / "underlying" / f"{underlying.lower()}_index_60.parquet"
    if not und_path.exists():
        print(f"ERROR: underlying file not found: {und_path}")
        return
    und_df = pd.read_parquet(und_path)
    und_df["timestamp"] = pd.to_datetime(und_df["timestamp"], utc=True)
    und_df = und_df.sort_values("timestamp").set_index("timestamp")
    print(f"Loaded {len(und_df)} underlying bars")

    # 3. Build day-by-day task list
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # Pre-index instruments by expiry date for fast lookup
    inst_df["_exp_date"] = inst_df["expiration_date"].dt.date
    exp_groups = {}
    for _, row in inst_df.iterrows():
        ed = row["_exp_date"]
        if ed not in exp_groups:
            exp_groups[ed] = []
        exp_groups[ed].append(row)

    tasks: list[tuple[str, int, int, pd.Timestamp]] = []  # (instrument, start_ms, end_ms, bar_ts)

    current = start_dt
    days_processed = 0
    days_with_tasks = 0

    while current <= end_dt:
        # Get spot price at 8:00 UTC this day
        target_ts = current.replace(hour=8, minute=0, second=0, microsecond=0)
        target_pd = pd.Timestamp(target_ts)

        # Find closest underlying price
        if target_pd in und_df.index:
            spot = float(und_df.loc[target_pd, "close"])
        else:
            # Find nearest bar
            idx = und_df.index.searchsorted(target_pd)
            if idx == 0:
                spot = float(und_df.iloc[0]["close"])
            elif idx >= len(und_df):
                spot = float(und_df.iloc[-1]["close"])
            else:
                spot = float(und_df.iloc[idx - 1]["close"])

        if spot <= 0:
            current += timedelta(days=1)
            continue

        # Strike range
        lo_strike = spot * (1 - strike_pct)
        hi_strike = spot * (1 + strike_pct)

        # Time window: 07:30 - 08:30 UTC
        win_start = target_ts - timedelta(minutes=WINDOW_MINUTES)
        win_end = target_ts + timedelta(minutes=WINDOW_MINUTES)
        start_ms = int(win_start.timestamp() * 1000)
        end_ms = int(win_end.timestamp() * 1000)

        # Bar timestamp = 07:00 UTC (the 1h candle covering 07:00-08:00)
        bar_ts = pd.Timestamp(target_ts - timedelta(hours=1))

        # Find relevant instruments: expire within max_dte days
        day_tasks = 0
        for dte_offset in range(0, max_dte + 1):
            exp_date = (current + timedelta(days=dte_offset)).date()
            candidates = exp_groups.get(exp_date, [])
            for row in candidates:
                strike = row["strike_price"]
                if lo_strike <= strike <= hi_strike:
                    tasks.append((row["instrument_name"], start_ms, end_ms, bar_ts))
                    day_tasks += 1

        if day_tasks > 0:
            days_with_tasks += 1
        days_processed += 1
        current += timedelta(days=1)

    print(f"\nDays scanned: {days_processed}")
    print(f"Days with targets: {days_with_tasks}")
    print(f"Total instrument-day pairs to fetch: {len(tasks)}")
    est_minutes = len(tasks) / MAX_RPS / 60
    print(f"Estimated time: {est_minutes:.1f} minutes")

    if dry_run:
        print("\n[DRY RUN] No requests will be made.")
        return

    # 4. Fetch trades and build OHLCV bars
    OHLCV_DIR.mkdir(parents=True, exist_ok=True)

    # Group tasks by instrument for efficient file I/O
    from collections import defaultdict
    by_instrument: dict[str, list] = defaultdict(list)
    for inst, s_ms, e_ms, b_ts in tasks:
        by_instrument[inst].append((s_ms, e_ms, b_ts))

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    total_bars = 0
    total_trades_found = 0
    instruments_with_data = 0
    failed = 0

    # Process instruments
    results_by_inst: dict[str, list[dict]] = {}
    all_items = []
    for inst, windows in by_instrument.items():
        for s_ms, e_ms, b_ts in windows:
            all_items.append((inst, s_ms, e_ms, b_ts))

    async def _fetch_one(item):
        inst, s_ms, e_ms, b_ts = item
        async with sem:
            trades = await fetch_trades_window(inst, s_ms, e_ms)
            bar = trades_to_ohlcv(trades, b_ts, inst)
            return inst, bar, len(trades)

    pbar = tqdm(total=len(all_items), desc="Fetching trades", unit="req")
    batch_size = 200  # process in batches to avoid too many concurrent tasks

    for batch_start in range(0, len(all_items), batch_size):
        batch = all_items[batch_start:batch_start + batch_size]
        coros = [_fetch_one(item) for item in batch]
        for coro in asyncio.as_completed(coros):
            inst, bar, n_trades = await coro
            if bar:
                if inst not in results_by_inst:
                    results_by_inst[inst] = []
                results_by_inst[inst].append(bar)
                total_bars += 1
                total_trades_found += n_trades
            pbar.update(1)

    pbar.close()

    # 5. Merge into parquet files
    print(f"\nBars collected: {total_bars}")
    print(f"Total trades found: {total_trades_found:,}")
    print(f"Instruments with data: {len(results_by_inst)}")

    saved = 0
    merged = 0
    for inst, bars in tqdm(results_by_inst.items(), desc="Saving parquet", unit="file"):
        new_df = pd.DataFrame(bars)
        new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], utc=True)

        out_path = OHLCV_DIR / f"{inst}.parquet"
        if out_path.exists():
            # Merge with existing data
            existing = pd.read_parquet(out_path)
            existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
            combined = pd.concat([existing, new_df[["timestamp", "open", "high", "low",
                                                     "close", "volume", "instrument_name"]]])
            combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
            combined.to_parquet(out_path, index=False)
            merged += 1
        else:
            new_df[["timestamp", "open", "high", "low", "close", "volume",
                     "instrument_name"]].to_parquet(out_path, index=False)
        saved += 1

    print(f"\nSaved: {saved} files ({merged} merged with existing data)")
    print(f"New files: {saved - merged}")

    # Close session
    global _session
    if _session and not _session.closed:
        await _session.close()

    # 6. Summary
    print("\n" + "=" * 50)
    print("  FETCH COMPLETE")
    print("=" * 50)
    print(f"  Period:           {start_date} → {end_date}")
    print(f"  Underlying:       {underlying}")
    print(f"  API requests:     {len(all_items):,}")
    print(f"  Bars saved:       {total_bars:,}")
    print(f"  Real trades:      {total_trades_found:,}")
    print(f"  Files written:    {saved}")
    print("=" * 50)


# ── CLI ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch 8:00 UTC trades for option backtest")
    parser.add_argument("-u", "--underlying", default="BTC", help="BTC or ETH")
    parser.add_argument("-s", "--start", default="2023-01-01", help="Start date")
    parser.add_argument("-e", "--end", default="2026-01-31", help="End date")
    parser.add_argument("--dry-run", action="store_true", help="Just count tasks, don't fetch")
    parser.add_argument("--strike-pct", type=float, default=STRIKE_PCT, help="Strike range ±pct of spot (default 0.10)")
    parser.add_argument("--max-dte", type=int, default=MAX_DTE, help="Max DTE to fetch (default 1 for 0DTE)")
    args = parser.parse_args()

    asyncio.run(run(args.underlying, args.start, args.end,
                    dry_run=args.dry_run,
                    strike_pct=args.strike_pct,
                    max_dte=args.max_dte))
