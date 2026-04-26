"""Download historical Deribit options OHLCV data from CryptoDataDownload API.

This script downloads all available BTC and ETH options OHLCV data from
CryptoDataDownload (CDD) and stores it as per-instrument parquet files
compatible with the existing backtest engine.

API: https://api.cryptodatadownload.com/v1/
Auth: Token-based (header or URL param)

Usage:
    python scripts/data/download_cdd_data.py [--currency BTC] [--years 2021,2022,2023,2024,2025]
    python scripts/data/download_cdd_data.py --currency BTC --years 2024 --concurrency 10
    python scripts/data/download_cdd_data.py --all  # Download BTC+ETH, all years
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE = "https://api.cryptodatadownload.com/v1"
API_TOKEN = "368ca0bd2ccf5620aa35c50f9a11a65943589b49"

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "market_data"
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / ".cache"

# Rate limiting
MAX_CONCURRENT = 1        # sequential requests (CDD is very strict)
REQUEST_DELAY = 3.0       # seconds between requests (safe for CDD)
MAX_RETRIES = 5           # per-request retries
RETRY_BACKOFF = 5.0       # exponential backoff multiplier
REQUEST_TIMEOUT = 60      # seconds


# ---------------------------------------------------------------------------
# Async Rate Limiter
# ---------------------------------------------------------------------------


class AsyncRateLimiter:
    """Token-bucket rate limiter for async requests."""

    def __init__(self, min_interval: float = 0.5):
        self._interval = min_interval
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = asyncio.get_event_loop().time()
            wait = self._interval - (now - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = asyncio.get_event_loop().time()


# ---------------------------------------------------------------------------
# CDD API Client
# ---------------------------------------------------------------------------


class CDDClient:
    """Async client for CryptoDataDownload API."""

    def __init__(self, token: str = API_TOKEN, max_concurrent: int = MAX_CONCURRENT):
        self.token = token
        self.headers = {"Authorization": f"Token {token}"}
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.limiter = AsyncRateLimiter(min_interval=REQUEST_DELAY)
        self._session: aiohttp.ClientSession | None = None
        self._stats = {"requests": 0, "retries": 0, "errors": 0, "saved": 0, "skipped": 0}
        self._rate_limited_count = 0

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
            connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT * 2, ssl=False)
            self._session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=timeout,
                connector=connector,
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def _request(self, endpoint: str, params: dict[str, Any]) -> dict | None:
        """Make a single API request with retries."""
        url = f"{API_BASE}{endpoint}"
        session = await self._get_session()

        for attempt in range(MAX_RETRIES):
            try:
                await self.limiter.acquire()
                async with self.semaphore:
                    self._stats["requests"] += 1
                    async with session.get(url, params=params) as resp:
                        if resp.status == 200:
                            # Gradually reduce interval on success
                            if self._rate_limited_count > 0:
                                self._rate_limited_count -= 1
                            if self.limiter._interval > REQUEST_DELAY:
                                self.limiter._interval = max(
                                    REQUEST_DELAY,
                                    self.limiter._interval * 0.9,
                                )
                            return await resp.json()
                        elif resp.status == 429:
                            self._rate_limited_count += 1
                            # Increase interval and wait
                            self.limiter._interval = min(
                                self.limiter._interval * 1.5, 10.0
                            )
                            wait = RETRY_BACKOFF * (attempt + 1) + self._rate_limited_count * 5
                            tqdm.write(
                                f"  ⚠ 429 (#{self._rate_limited_count}), "
                                f"wait {wait:.0f}s, interval→{self.limiter._interval:.1f}s"
                            )
                            await asyncio.sleep(wait)
                            self._stats["retries"] += 1
                        elif resp.status == 400:
                            return None
                        else:
                            self._stats["retries"] += 1
                            await asyncio.sleep(RETRY_BACKOFF ** attempt)
            except (aiohttp.ClientError, asyncio.TimeoutError):
                self._stats["retries"] += 1
                wait = RETRY_BACKOFF ** attempt + 2
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(wait)
                else:
                    self._stats["errors"] += 1
                    return None
        self._stats["errors"] += 1
        return None

    # ------------------------------------------------------------------
    # High-level API methods
    # ------------------------------------------------------------------

    async def get_available_instruments(self, year: int) -> list[str]:
        """Get all available instrument names for a given year."""
        data = await self._request(
            "/data/ohlc/deribit/options/available/",
            {"year": year},
        )
        if data and "result" in data:
            return data["result"]
        return []

    async def get_instrument_ohlcv(
        self, currency: str, symbol: str, limit: int = 5000
    ) -> list[dict]:
        """Fetch full OHLCV timeseries for a single instrument (paginated)."""
        all_rows: list[dict] = []
        offset = 0

        while True:
            data = await self._request(
                "/data/ohlc/deribit/options/",
                {
                    "currency": currency,
                    "symbol": symbol,
                    "limit": limit,
                    "offset": offset,
                },
            )
            if not data or "result" not in data:
                break

            rows = data["result"]
            all_rows.extend(rows)

            # Check pagination
            pagination = data.get("pagination", {})
            if pagination.get("has_more", False) and pagination.get("next_offset"):
                offset = pagination["next_offset"]
            else:
                break

        return all_rows

    async def get_date_snapshot(
        self, currency: str, date: str, limit: int = 5000
    ) -> list[dict]:
        """Fetch all instruments' OHLCV for a specific date."""
        all_rows: list[dict] = []
        offset = 0

        while True:
            data = await self._request(
                "/data/ohlc/deribit/options/",
                {
                    "currency": currency,
                    "date": date,
                    "limit": limit,
                    "offset": offset,
                },
            )
            if not data or "result" not in data:
                break

            rows = data["result"]
            all_rows.extend(rows)

            pagination = data.get("pagination", {})
            if pagination.get("has_more", False) and pagination.get("next_offset"):
                offset = pagination["next_offset"]
            else:
                break

        return all_rows


# ---------------------------------------------------------------------------
# Data conversion helpers
# ---------------------------------------------------------------------------


def cdd_rows_to_df(rows: list[dict], instrument_name: str | None = None) -> pd.DataFrame:
    """Convert CDD API rows to the backtest engine's parquet format.

    Expected output columns: timestamp, open, high, low, close, volume, instrument_name
    """
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Rename columns to match existing format
    rename_map = {"unix": "timestamp", "symbol": "instrument_name"}
    df = df.rename(columns=rename_map)

    # Convert unix ms → datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    # Keep only the columns the backtest engine expects
    cols = ["timestamp", "open", "high", "low", "close", "volume", "instrument_name"]
    for c in cols:
        if c not in df.columns:
            if c == "instrument_name" and instrument_name:
                df["instrument_name"] = instrument_name
            else:
                df[c] = 0.0

    df = df[cols].copy()

    # Sort by timestamp ascending
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def save_instrument_parquet(df: pd.DataFrame, out_dir: Path):
    """Save per-instrument parquet file, matching existing naming convention."""
    if df.empty:
        return 0

    saved = 0
    for name, group in df.groupby("instrument_name"):
        out_path = out_dir / f"{name}.parquet"
        group = group.sort_values("timestamp").reset_index(drop=True)
        group.to_parquet(out_path, index=False)
        saved += 1

    return saved


# ---------------------------------------------------------------------------
# Download strategies
# ---------------------------------------------------------------------------


async def download_by_symbol(
    client: CDDClient,
    currency: str,
    instruments: list[str],
    out_dir: Path,
    pbar: tqdm | None = None,
) -> int:
    """Download OHLCV for each instrument individually.

    Best for targeted downloads or resume after interruption.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter by currency
    prefix = currency.upper() + "-"
    instruments = [i for i in instruments if i.startswith(prefix)]

    # Skip already downloaded
    to_fetch = []
    for name in instruments:
        out_path = out_dir / f"{name}.parquet"
        if out_path.exists():
            client._stats["skipped"] += 1
        else:
            to_fetch.append(name)

    if not to_fetch:
        if pbar:
            pbar.update(len(instruments))
        return 0

    print(f"  {currency}: {len(instruments)} total, "
          f"{client._stats['skipped']} cached, {len(to_fetch)} to download")

    saved = 0

    async def _fetch_one(name: str) -> bool:
        rows = await client.get_instrument_ohlcv(currency, name)
        if rows:
            df = cdd_rows_to_df(rows, instrument_name=name)
            if not df.empty:
                out_path = out_dir / f"{name}.parquet"
                df.to_parquet(out_path, index=False)
                return True
        return False

    # Process in batches for progress tracking
    batch_size = MAX_CONCURRENT * 2
    for i in range(0, len(to_fetch), batch_size):
        batch = to_fetch[i:i + batch_size]
        results = await asyncio.gather(
            *[_fetch_one(name) for name in batch],
            return_exceptions=True,
        )
        for r in results:
            if r is True:
                saved += 1
                client._stats["saved"] += 1
        if pbar:
            pbar.update(len(batch))

    return saved


async def download_by_date(
    client: CDDClient,
    currency: str,
    start_date: str,
    end_date: str,
    out_dir: Path,
) -> int:
    """Download by iterating through dates. Most API-efficient for bulk downloads.

    Downloads one date at a time, accumulates per-instrument data, then saves.
    Saves a progress checkpoint after each date for resumability.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    from datetime import timedelta

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # Build date list
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    # Resume support: load checkpoint if exists
    checkpoint_path = CACHE_DIR / f"cdd_checkpoint_{currency}_{start.year}.json"
    completed_dates: set[str] = set()
    instrument_data: dict[str, list[dict]] = {}

    if checkpoint_path.exists():
        try:
            ckpt = json.loads(checkpoint_path.read_text())
            completed_dates = set(ckpt.get("completed_dates", []))
            print(f"  Resuming: {len(completed_dates)} dates already processed")
        except Exception:
            pass

    remaining_dates = [d for d in dates if d not in completed_dates]
    print(f"  {currency}: {len(dates)} total dates, "
          f"{len(completed_dates)} done, {len(remaining_dates)} remaining")

    if not remaining_dates:
        print(f"  All dates already processed!")
        return 0

    with tqdm(total=len(remaining_dates), desc=f"  {currency} dates", unit="day") as pbar:
        for date_str in remaining_dates:
            rows = await client.get_date_snapshot(currency, date_str)
            if rows:
                for row in rows:
                    sym = row.get("symbol", "")
                    if sym:
                        instrument_data.setdefault(sym, []).append(row)

            completed_dates.add(date_str)

            # Save checkpoint every 10 dates
            if len(completed_dates) % 10 == 0:
                checkpoint_path.write_text(json.dumps({
                    "completed_dates": sorted(completed_dates),
                    "instruments_accumulated": len(instrument_data),
                }))

            pbar.update(1)

    # Save final checkpoint
    checkpoint_path.write_text(json.dumps({
        "completed_dates": sorted(completed_dates),
        "instruments_accumulated": len(instrument_data),
        "status": "saving_parquets",
    }))

    # Save per-instrument parquet files
    print(f"  Saving {len(instrument_data)} instruments to parquet…")
    saved = 0
    for name, rows in tqdm(instrument_data.items(), desc="  Saving", unit="inst"):
        out_path = out_dir / f"{name}.parquet"
        df = cdd_rows_to_df(rows, instrument_name=name)
        if not df.empty:
            # If file already exists, merge with new data (keep unique timestamps)
            if out_path.exists():
                existing = pd.read_parquet(out_path)
                df = pd.concat([existing, df]).drop_duplicates(
                    subset=["timestamp", "instrument_name"]
                ).sort_values("timestamp").reset_index(drop=True)
            df.to_parquet(out_path, index=False)
            saved += 1
            client._stats["saved"] += 1

    # Mark as complete
    checkpoint_path.write_text(json.dumps({
        "completed_dates": sorted(completed_dates),
        "instruments_saved": saved,
        "status": "complete",
    }))

    return saved


# ---------------------------------------------------------------------------
# Main download orchestrator
# ---------------------------------------------------------------------------


async def run_download(
    currencies: list[str],
    years: list[int],
    max_concurrent: int = MAX_CONCURRENT,
    strategy: str = "symbol",
):
    """Main download orchestrator."""
    print("=" * 70)
    print("CryptoDataDownload - Deribit Options OHLCV Downloader")
    print("=" * 70)
    print(f"Currencies: {currencies}")
    print(f"Years: {years}")
    print(f"Strategy: {strategy}")
    print(f"Concurrency: {max_concurrent}")
    print(f"Output: {DATA_DIR}")
    print()

    client = CDDClient(max_concurrent=max_concurrent)
    start_time = time.time()

    try:
        for currency in currencies:
            print(f"\n{'='*50}")
            print(f"  {currency} - Fetching available instruments…")
            print(f"{'='*50}")

            all_instruments: list[str] = []
            for year in years:
                instruments = await client.get_available_instruments(year)
                # Filter by currency
                prefix = f"{currency}-"
                filtered = [i for i in instruments if i.startswith(prefix)]
                print(f"  {year}: {len(filtered)} {currency} instruments available")
                all_instruments.extend(filtered)
                await asyncio.sleep(0.5)  # Brief pause between year queries

            # Deduplicate (instruments may appear in multiple years)
            all_instruments = sorted(set(all_instruments))
            print(f"\n  Total unique {currency} instruments: {len(all_instruments)}")

            if not all_instruments:
                print(f"  No instruments found for {currency}")
                continue

            if strategy == "symbol":
                # Download by individual symbol
                print(f"\n  Downloading OHLCV by symbol (this may take a while)…")
                with tqdm(total=len(all_instruments),
                          desc=f"  {currency}", unit="inst") as pbar:
                    saved = await download_by_symbol(
                        client, currency, all_instruments, DATA_DIR, pbar
                    )
                print(f"\n  ✓ {currency}: saved {saved} new instrument files")

            elif strategy == "date":
                # Download by date
                for year in years:
                    start_date = f"{year}-01-01"
                    end_date = f"{year}-12-31"
                    # Don't go past today
                    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    if end_date > today:
                        end_date = today
                    print(f"\n  Year {year}:")
                    saved = await download_by_date(
                        client, currency, start_date, end_date, DATA_DIR
                    )
                    print(f"  ✓ {year}: saved {saved} instrument files")

        # Print summary
        elapsed = time.time() - start_time
        stats = client._stats
        print(f"\n{'='*70}")
        print(f"Download complete!")
        print(f"  Duration: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"  API Requests: {stats['requests']}")
        print(f"  Retries: {stats['retries']}")
        print(f"  Errors: {stats['errors']}")
        print(f"  Files saved: {stats['saved']}")
        print(f"  Files skipped (cached): {stats['skipped']}")
        print(f"{'='*70}")

        # Save download manifest
        manifest = {
            "download_time": datetime.now(timezone.utc).isoformat(),
            "currencies": currencies,
            "years": years,
            "strategy": strategy,
            "stats": stats,
            "elapsed_seconds": round(elapsed, 1),
        }
        manifest_path = CACHE_DIR / "cdd_download_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"  Manifest saved to {manifest_path}")

    finally:
        await client.close()


# ---------------------------------------------------------------------------
# CLI Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Download Deribit options OHLCV from CryptoDataDownload"
    )
    parser.add_argument(
        "--currency", "-c",
        type=str,
        default="BTC",
        help="Currency to download (BTC, ETH, or BTC,ETH). Default: BTC",
    )
    parser.add_argument(
        "--years", "-y",
        type=str,
        default="2021,2022,2023,2024,2025",
        help="Comma-separated years. Default: 2021,2022,2023,2024,2025",
    )
    parser.add_argument(
        "--strategy", "-s",
        type=str,
        choices=["symbol", "date"],
        default="date",
        help="Download strategy: 'symbol' (per-instrument) or 'date' (per-day). Default: date",
    )
    parser.add_argument(
        "--concurrency", "-n",
        type=int,
        default=MAX_CONCURRENT,
        help=f"Max concurrent requests. Default: {MAX_CONCURRENT}",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Download BTC+ETH for all available years (2021-2025)",
    )

    args = parser.parse_args()

    if args.all:
        currencies = ["BTC", "ETH"]
        years = [2021, 2022, 2023, 2024, 2025]
    else:
        currencies = [c.strip().upper() for c in args.currency.split(",")]
        years = [int(y.strip()) for y in args.years.split(",")]

    asyncio.run(run_download(
        currencies=currencies,
        years=years,
        max_concurrent=args.concurrency,
        strategy=args.strategy,
    ))


if __name__ == "__main__":
    main()
