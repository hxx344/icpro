"""Deribit public API data fetcher.

Fetches historical option instruments, OHLCV, index prices, and settlement
records via the Deribit public (unauthenticated) REST API.

Data-fetching strategy
----------------------
1. Underlying OHLCV: ``BTC-PERPETUAL`` via ``get_tradingview_chart_data``
   (the ``btc_usd`` index name does NOT work with this endpoint).
2. Instrument discovery: Deribit ``get_instruments(expired=true)`` only returns
   *today's* expired contracts – useless for historical back‑testing.  Instead we
   paginate ``get_last_settlements_by_currency`` to collect every instrument name
   that settled (expired) during the target period, then parse instrument names to
   extract expiry / strike / type metadata.
3. Active instruments: ``get_instruments(expired=false)`` for currently listed
   contracts.  Combined with the settlement‑based list.
4. (Optional) Individual option OHLCV via ``get_tradingview_chart_data``.

Rate limit: max 20 requests / second for public endpoints.
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd
from loguru import logger
from tqdm import tqdm

BASE_URL = "https://www.deribit.com/api/v2"
MAX_RPS = 18  # stay slightly below the 20 rps hard limit

_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

_INST_RE = re.compile(
    r"^(?P<underlying>[A-Z]+)-(?P<day>\d{1,2})(?P<mon>[A-Z]{3})(?P<yr>\d{2})"
    r"-(?P<strike>\d+)-(?P<cp>[CP])$"
)


def parse_instrument_name(name: str) -> dict[str, Any] | None:
    """Parse ``BTC-28MAR25-100000-C`` → metadata dict, or *None* on failure."""
    m = _INST_RE.match(name)
    if m is None:
        return None
    day = int(m.group("day"))
    mon = _MONTH_MAP.get(m.group("mon"))
    yr = 2000 + int(m.group("yr"))
    if mon is None:
        return None
    expiry = datetime(yr, mon, day, 8, 0, 0, tzinfo=timezone.utc)  # Deribit settles 08:00 UTC
    return {
        "instrument_name": name,
        "underlying": m.group("underlying"),
        "expiration_date": expiry,
        "strike_price": float(m.group("strike")),
        "option_type": "call" if m.group("cp") == "C" else "put",
    }


class TokenBucketLimiter:
    """Async token-bucket rate limiter that allows concurrent bursts.

    Unlike the old serial limiter, this maintains a pool of tokens that
    refill at *max_rps* per second.  Multiple coroutines can acquire a
    token simultaneously as long as the bucket is not empty, enabling
    true parallel request execution.
    """

    def __init__(self, max_rps: int = MAX_RPS):
        self._max_tokens = float(max_rps)
        self._tokens = float(max_rps)  # start full
        self._rate = float(max_rps)    # tokens per second
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(
                    self._max_tokens,
                    self._tokens + elapsed * self._rate,
                )
                self._last_refill = now

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return

            # Bucket empty – wait for ~1 token to refill
            await asyncio.sleep(1.0 / self._rate)


class DeribitFetcher:
    """Async fetcher for Deribit public market data."""

    def __init__(self, data_dir: str | Path = "data", base_url: str = BASE_URL):
        self.data_dir = Path(data_dir)
        self.base_url = base_url
        self._limiter = TokenBucketLimiter()
        self._session: aiohttp.ClientSession | None = None

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=30,               # max simultaneous connections
                keepalive_timeout=30,    # reuse idle connections
                enable_cleanup_closed=True,
            )
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _get(self, endpoint: str, params: dict[str, Any] | None = None,
                   max_retries: int = 5) -> Any:
        """GET with rate‑limiting and exponential back‑off retry."""
        session = await self._ensure_session()
        url = f"{self.base_url}{endpoint}"

        for attempt in range(max_retries):
            await self._limiter.acquire()
            try:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 429:
                        wait = 2 ** attempt
                        logger.warning(f"Rate limited, retrying in {wait}s …")
                        await asyncio.sleep(wait)
                        continue
                    if resp.status >= 400:
                        body = await resp.text()
                        logger.debug(f"HTTP {resp.status} from {endpoint}: {body[:200]}")
                        return None
                    data = await resp.json()
                    if "result" in data:
                        return data["result"]
                    return data
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                wait = 2 ** attempt
                logger.warning(f"Request failed ({exc}), retry {attempt+1}/{max_retries} in {wait}s")
                await asyncio.sleep(wait)

        raise RuntimeError(f"Failed after {max_retries} retries: {endpoint}")

    # ------------------------------------------------------------------
    # Underlying OHLCV  (uses BTC-PERPETUAL)
    # ------------------------------------------------------------------

    async def fetch_underlying_ohlcv(
        self,
        underlying: str = "BTC",
        resolution: str = "1D",
        start_date: str = "2025-01-01",
        end_date: str = "2025-12-31",
    ) -> pd.DataFrame:
        """Fetch underlying price via TradingView chart data for *BTC-PERPETUAL*."""
        instrument = f"{underlying}-PERPETUAL"
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d")
                       .replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d")
                     .replace(tzinfo=timezone.utc).timestamp() * 1000)

        logger.info(f"Fetching {instrument} OHLCV ({start_date} → {end_date}, res={resolution}) …")

        # Resolution → milliseconds per bar (approx) for chunk sizing
        _res_ms: dict[str, int] = {
            "1": 60_000, "3": 180_000, "5": 300_000, "10": 600_000,
            "15": 900_000, "30": 1_800_000, "60": 3_600_000,
            "120": 7_200_000, "180": 10_800_000, "360": 21_600_000,
            "720": 43_200_000, "1D": 86_400_000,
        }
        bar_ms = _res_ms.get(str(resolution), 86_400_000)
        # Request ~4500 bars per chunk to stay under Deribit's ~5000 limit
        chunk_ms = bar_ms * 4500

        all_bars: list[dict] = []
        current_start = start_ts

        with tqdm(desc=f"{instrument} OHLCV", unit="batch") as pbar:
            while current_start < end_ts:
                chunk_end = min(current_start + chunk_ms, end_ts)
                params = {
                    "instrument_name": instrument,
                    "start_timestamp": current_start,
                    "end_timestamp": chunk_end,
                    "resolution": resolution,
                }
                result = await self._get("/public/get_tradingview_chart_data", params)
                if not result or result.get("status") != "ok":
                    break

                ticks = result.get("ticks", [])
                if not ticks:
                    break

                n = len(ticks)
                for i in range(n):
                    all_bars.append({
                        "timestamp": ticks[i],
                        "open": result["open"][i],
                        "high": result["high"][i],
                        "low": result["low"][i],
                        "close": result["close"][i],
                        "volume": result["volume"][i],
                    })

                current_start = ticks[-1] + 1
                pbar.update(1)

                # If we got fewer bars than expected, this chunk is done;
                # advance to the next chunk boundary.
                if current_start < chunk_end and n < 100:
                    current_start = chunk_end + 1

        if not all_bars:
            logger.warning("No underlying OHLCV data returned")
            return pd.DataFrame()

        df = pd.DataFrame(all_bars)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["underlying"] = underlying
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        out_dir = self.data_dir / "underlying"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{underlying.lower()}_index_{resolution}.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"Saved {len(df)} underlying bars → {out_path}")
        return df

    # ------------------------------------------------------------------
    # Instrument discovery via settlement pagination
    # ------------------------------------------------------------------

    async def _discover_instruments_from_settlements(
        self,
        underlying: str = "BTC",
        start_date: str = "2025-01-01",
        end_date: str = "2025-12-31",
        page_size: int = 1000,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Paginate ``get_last_settlements_by_currency`` to discover all
        instrument names that expired within *[start_date, end_date]*.

        Returns (instruments_df, settlements_df).
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

        logger.info(f"Discovering {underlying} instruments via settlements ({start_date} → {end_date}) …")

        all_settlements: list[dict] = []
        seen_instruments: set[str] = set()
        continuation: str | None = None
        reached_start = False

        with tqdm(desc="Settlement pages", unit="page") as pbar:
            while not reached_start:
                params: dict[str, Any] = {
                    "currency": underlying,
                    "type": "delivery",
                    "count": page_size,
                }
                if continuation:
                    params["continuation"] = continuation

                result = await self._get("/public/get_last_settlements_by_currency", params)
                if not result:
                    break

                settlements = result.get("settlements", [])
                if not settlements:
                    break

                continuation = result.get("continuation")

                for s in settlements:
                    ts = s.get("timestamp", 0)
                    dt = datetime.utcfromtimestamp(ts / 1000).replace(tzinfo=timezone.utc)

                    if dt < start_dt:
                        reached_start = True
                        break

                    if dt <= end_dt:
                        name = s.get("instrument_name", "")
                        all_settlements.append({
                            "instrument_name": name,
                            "settlement_timestamp": dt,
                            "index_price": s.get("index_price", 0.0),
                            "mark_price": s.get("mark_price"),
                            "session_profit_loss": s.get("session_profit_loss"),
                            "type": s.get("type", "delivery"),
                        })
                        seen_instruments.add(name)

                pbar.update(1)

                if not continuation:
                    break

        logger.info(f"Discovered {len(seen_instruments)} unique instruments "
                    f"from {len(all_settlements)} settlement records")

        # -- Build instruments DataFrame by parsing names ----------------
        # Names without a strike/C/P suffix (e.g. BTC-28MAR25) are futures –
        # silently skip them.  Only warn on names that *look* like options but
        # still fail to parse.
        inst_rows: list[dict] = []
        skipped_futures = 0
        for name in sorted(seen_instruments):
            parsed = parse_instrument_name(name)
            if parsed:
                inst_rows.append(parsed)
            elif "-C" in name or "-P" in name:
                logger.debug(f"Could not parse option-like name: {name}")
            else:
                skipped_futures += 1
        if skipped_futures:
            logger.debug(f"Skipped {skipped_futures} futures contracts (not options)")

        instruments_df = pd.DataFrame(inst_rows) if inst_rows else pd.DataFrame()
        settlements_df = pd.DataFrame(all_settlements) if all_settlements else pd.DataFrame()

        return instruments_df, settlements_df

    # ------------------------------------------------------------------
    # Active instruments from API
    # ------------------------------------------------------------------

    async def _fetch_active_instruments(self, underlying: str = "BTC") -> pd.DataFrame:
        """Fetch currently active option instruments via ``get_instruments``."""
        logger.info(f"Fetching active {underlying} instruments …")
        params: dict[str, Any] = {
            "currency": underlying,
            "kind": "option",
            "expired": "false",
        }
        result = await self._get("/public/get_instruments", params)
        if not result:
            return pd.DataFrame()

        rows = []
        for r in result:
            parsed = parse_instrument_name(r.get("instrument_name", ""))
            if parsed:
                parsed["is_active"] = True
                rows.append(parsed)

        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        logger.info(f"Got {len(df)} active instruments")
        return df

    # ------------------------------------------------------------------
    # Combined instrument list
    # ------------------------------------------------------------------

    async def fetch_instruments(
        self,
        underlying: str = "BTC",
        start_date: str = "2025-01-01",
        end_date: str = "2025-12-31",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Discover instruments from settlements + active listings.

        Returns (instruments_df, settlements_df).
        ``instruments_df`` has columns:
            instrument_name, underlying, expiration_date, strike_price, option_type
        """
        settled_instruments, settlements_df = await self._discover_instruments_from_settlements(
            underlying, start_date, end_date,
        )
        active_instruments = await self._fetch_active_instruments(underlying)

        # Merge & deduplicate
        frames = [f for f in (settled_instruments, active_instruments) if not f.empty]
        if not frames:
            logger.warning("No instruments found at all")
            return pd.DataFrame(), settlements_df

        instruments = pd.concat(frames, ignore_index=True)
        instruments = instruments.drop_duplicates(subset=["instrument_name"]).reset_index(drop=True)
        instruments = instruments.sort_values(
            ["expiration_date", "strike_price", "option_type"]
        ).reset_index(drop=True)

        # Persist
        out_dir = self.data_dir / "instruments"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{underlying.lower()}_instruments.parquet"
        instruments.to_parquet(out_path, index=False)
        logger.info(f"Saved {len(instruments)} instruments → {out_path}")

        if not settlements_df.empty:
            sett_dir = self.data_dir / "settlements"
            sett_dir.mkdir(parents=True, exist_ok=True)
            sett_path = sett_dir / f"{underlying.lower()}_settlements.parquet"
            settlements_df.to_parquet(sett_path, index=False)
            logger.info(f"Saved {len(settlements_df)} settlement records → {sett_path}")

        return instruments, settlements_df

    # ------------------------------------------------------------------
    # Option OHLCV (single instrument)
    # ------------------------------------------------------------------

    async def fetch_option_ohlcv(
        self,
        instrument_name: str,
        resolution: str = "1D",
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV for a single option instrument."""
        params: dict[str, Any] = {
            "instrument_name": instrument_name,
            "resolution": resolution,
        }
        if start_ts is not None:
            params["start_timestamp"] = start_ts
        if end_ts is not None:
            params["end_timestamp"] = end_ts

        result = await self._get("/public/get_tradingview_chart_data", params)
        if not result or result.get("status") != "ok":
            return pd.DataFrame()

        ticks = result.get("ticks", [])
        if not ticks:
            return pd.DataFrame()

        rows = []
        for i in range(len(ticks)):
            rows.append({
                "timestamp": ticks[i],
                "open": result["open"][i],
                "high": result["high"][i],
                "low": result["low"][i],
                "close": result["close"][i],
                "volume": result["volume"][i],
            })

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["instrument_name"] = instrument_name
        return df

    # ------------------------------------------------------------------
    # Batch option OHLCV
    # ------------------------------------------------------------------

    async def fetch_all_option_ohlcv(
        self,
        instruments_df: pd.DataFrame,
        resolution: str = "1D",
        start_date: str = "2025-01-01",
        end_date: str = "2025-12-31",
        max_concurrent: int = 15,
    ) -> int:
        """Fetch OHLCV for all instruments.  Returns count of files saved.

        Optimizations applied:
        - Semaphore(15) + token-bucket limiter for true concurrency
        - Smart date-range filter: skip instruments whose expiry < start_date
        - Cache check: skip already-downloaded files
        """
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d")
                       .replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d")
                     .replace(tzinfo=timezone.utc).timestamp() * 1000)

        # -- Smart filtering: drop instruments that expired before start_date
        total_before = len(instruments_df)
        if "expiration_date" in instruments_df.columns:
            start_dt = pd.Timestamp(start_date, tz="UTC")
            mask = instruments_df["expiration_date"] >= start_dt
            instruments_df = instruments_df.loc[mask].copy()  # type: ignore[assignment]
            dropped = total_before - len(instruments_df)
            if dropped:
                logger.info(f"Smart filter: skipped {dropped} instruments "
                            f"(expired before {start_date})")

        names = instruments_df["instrument_name"].tolist()
        out_dir = self.data_dir / "market_data" / resolution
        out_dir.mkdir(parents=True, exist_ok=True)

        # Pre-check cache – skip names that already have files
        to_fetch = []
        cached = 0
        for name in names:
            out_path = out_dir / f"{name}.parquet"
            if out_path.exists():
                cached += 1
            else:
                to_fetch.append(name)
        if cached:
            logger.info(f"Cache hit: {cached}/{len(names)} instruments already on disk")

        sem = asyncio.Semaphore(max_concurrent)
        saved_count = cached  # count cached as saved

        async def _fetch_one(name: str) -> bool:
            async with sem:
                df = await self.fetch_option_ohlcv(name, resolution, start_ts, end_ts)
                if not df.empty:
                    out_path = out_dir / f"{name}.parquet"
                    df.to_parquet(out_path, index=False)
                    return True
                return False

        logger.info(f"Fetching OHLCV for {len(to_fetch)} option instruments "
                    f"(res={resolution}, concurrency={max_concurrent}) …")
        tasks = [_fetch_one(n) for n in to_fetch]
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Options OHLCV"):
            if await coro:
                saved_count += 1

        logger.info(f"Option OHLCV complete: {saved_count}/{len(names)} instruments saved "
                    f"({cached} cached, {saved_count - cached} new).")
        return saved_count

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    async def fetch_all(
        self,
        underlying: str = "BTC",
        start_date: str = "2025-01-01",
        end_date: str = "2025-12-31",
        resolution: str = "1D",
        fetch_option_ohlcv: bool = True,
    ) -> None:
        """Run the complete data‑fetching pipeline.

        Steps
        -----
        1. Underlying OHLCV (BTC-PERPETUAL)
        2. Instrument discovery via settlement pagination + active listing
        3. (Optional) Individual option OHLCV for every discovered instrument
        """
        try:
            # 1 – underlying price history
            underlying_df = await self.fetch_underlying_ohlcv(
                underlying, resolution, start_date, end_date,
            )

            # 2 – instrument catalogue + settlement records
            instruments, _settlements = await self.fetch_instruments(
                underlying, start_date, end_date
            )

            # 3 – option OHLCV (can be slow for many instruments)
            if fetch_option_ohlcv and not instruments.empty:
                await self.fetch_all_option_ohlcv(
                    instruments, resolution, start_date, end_date,
                )
        finally:
            await self.close()


# ---------------------------------------------------------------------------
# Sync entry‑point (called by CLI)
# ---------------------------------------------------------------------------

def run_fetch(
    underlying: str = "BTC",
    start_date: str = "2025-01-01",
    end_date: str = "2025-12-31",
    resolution: str = "1D",
    data_dir: str = "data",
    fetch_option_ohlcv: bool = True,
) -> None:
    """Synchronous wrapper to kick off the async fetcher."""
    fetcher = DeribitFetcher(data_dir=data_dir)
    asyncio.run(
        fetcher.fetch_all(
            underlying, start_date, end_date, resolution,
            fetch_option_ohlcv=fetch_option_ohlcv,
        )
    )
