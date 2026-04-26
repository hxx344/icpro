"""OKX public API data fetcher for historical options.

Fetches historical option instruments, mark-price candles, index prices, and
delivery/exercise records from OKX's public REST API.

OKX keeps ~3 months of history for expired contracts via:
  - /api/v5/market/history-candles
  - /api/v5/market/history-mark-price-candles
  - /api/v5/market/history-index-candles

Instrument names are converted from OKX format (ETH-USD-250328-2000-C) to
Deribit format (ETH-28MAR25-2000-C) so the existing loader / engine works
without modification.
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd
from loguru import logger
from tqdm import tqdm

OKX_BASE = "https://www.okx.com"
MAX_RPS = 18

_MONTH_NAMES = {
    1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
    7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC",
}

_OKX_RE = re.compile(
    r"^(?P<ul>[A-Z]+)-USD-(?P<date>\d{6})-(?P<strike>\d+)-(?P<cp>[CP])$"
)


def okx_to_deribit_name(inst_id: str) -> str | None:
    """Convert OKX instrument id to Deribit-style name.

    ``ETH-USD-250328-2000-C`` → ``ETH-28MAR25-2000-C``
    """
    m = _OKX_RE.match(inst_id)
    if not m:
        return None
    date_str = m.group("date")  # YYMMDD
    yr = int(date_str[:2])
    mon = int(date_str[2:4])
    day = int(date_str[4:6])
    mon_name = _MONTH_NAMES.get(mon)
    if mon_name is None:
        return None
    return f"{m.group('ul')}-{day}{mon_name}{yr:02d}-{m.group('strike')}-{m.group('cp')}"


def parse_okx_instrument(inst_id: str) -> dict[str, Any] | None:
    """Parse OKX option instId into metadata dict."""
    m = _OKX_RE.match(inst_id)
    if not m:
        return None
    date_str = m.group("date")
    yr = 2000 + int(date_str[:2])
    mon = int(date_str[2:4])
    day = int(date_str[4:6])
    expiry = datetime(yr, mon, day, 8, 0, 0, tzinfo=timezone.utc)
    deribit_name = okx_to_deribit_name(inst_id)
    if deribit_name is None:
        return None
    return {
        "instrument_name": deribit_name,
        "okx_inst_id": inst_id,
        "underlying": m.group("ul"),
        "expiration_date": expiry,
        "strike_price": float(m.group("strike")),
        "option_type": "call" if m.group("cp") == "C" else "put",
    }


# ---------------------------------------------------------------------------
# Rate limiter (same pattern as Deribit fetcher)
# ---------------------------------------------------------------------------

class _Limiter:
    def __init__(self, max_rps: int = MAX_RPS):
        self._max = float(max_rps)
        self._tokens = float(max_rps)
        self._rate = float(max_rps)
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                self._tokens = min(self._max, self._tokens + (now - self._last) * self._rate)
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
            await asyncio.sleep(1.0 / self._rate)


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

class OKXFetcher:
    """Async fetcher for OKX public options data."""

    def __init__(self, data_dir: str | Path = "data"):
        self.data_dir = Path(data_dir)
        self._limiter = _Limiter()
        self._session: aiohttp.ClientSession | None = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=30, keepalive_timeout=30),
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _get(self, path: str, params: dict | None = None,
                   max_retries: int = 5) -> Any:
        session = await self._ensure_session()
        url = f"{OKX_BASE}{path}"
        for attempt in range(max_retries):
            await self._limiter.acquire()
            try:
                async with session.get(url, params=params,
                                       timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 429:
                        wait = 2 ** attempt
                        logger.warning(f"OKX rate limited, retry in {wait}s")
                        await asyncio.sleep(wait)
                        continue
                    if resp.status >= 400:
                        body = await resp.text()
                        logger.debug(f"OKX HTTP {resp.status}: {body[:200]}")
                        return None
                    data = await resp.json()
                    if data.get("code") != "0":
                        msg = data.get("msg", "unknown")
                        logger.debug(f"OKX API error: {msg}")
                        return None
                    return data.get("data", [])
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                wait = 2 ** attempt
                logger.warning(f"OKX request failed ({exc}), retry {attempt+1} in {wait}s")
                await asyncio.sleep(wait)
        raise RuntimeError(f"OKX failed after {max_retries} retries: {path}")

    # ------------------------------------------------------------------
    # Underlying index candles
    # ------------------------------------------------------------------

    async def fetch_underlying_ohlcv(
        self,
        underlying: str = "ETH",
        resolution: str = "60",
        start_date: str = "2025-12-13",
        end_date: str = "2026-03-13",
    ) -> pd.DataFrame:
        """Fetch underlying index candles via OKX history-index-candles.

        Parameters
        ----------
        resolution : OKX bar size – "1m", "5m", "15m", "1H", "4H", "1D" etc.
                     We map our internal codes ("60"→"1H") automatically.
        """
        # Map internal resolution to OKX bar notation
        _bar_map = {
            "1": "1m", "5": "5m", "15": "15m", "30": "30m",
            "60": "1H", "120": "2H", "180": "3H", "240": "4H",
            "360": "6H", "720": "12H", "1D": "1D",
        }
        bar = _bar_map.get(str(resolution), "1H")
        inst_id = f"{underlying.upper()}-USD"

        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d")
                       .replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d")
                     .replace(tzinfo=timezone.utc).timestamp() * 1000)

        logger.info(f"OKX: fetching {inst_id} index candles ({start_date} → {end_date}, bar={bar})")

        all_bars: list[dict] = []
        # OKX returns max 100 candles per request, paginate backwards using `after`
        after = str(end_ts)

        with tqdm(desc=f"OKX {inst_id} index", unit="batch") as pbar:
            while True:
                params = {"instId": inst_id, "bar": bar, "after": after, "limit": "100"}
                result = await self._get("/api/v5/market/history-index-candles", params)
                if not result:
                    break
                for row in result:
                    ts = int(row[0])
                    if ts < start_ts:
                        continue
                    all_bars.append({
                        "timestamp": ts,
                        "open": float(row[1]),
                        "high": float(row[2]),
                        "low": float(row[3]),
                        "close": float(row[4]),
                        "volume": 0.0,  # index candles have no volume
                    })
                # OKX returns oldest-last; to paginate further back, set after = oldest ts
                oldest_ts = int(result[-1][0])
                if oldest_ts <= start_ts:
                    break
                after = str(oldest_ts)
                pbar.update(1)
                if len(result) < 100:
                    break

        if not all_bars:
            logger.warning("No underlying index data from OKX")
            return pd.DataFrame()

        df = pd.DataFrame(all_bars)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["underlying"] = underlying.upper()
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        out_dir = self.data_dir / "underlying"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{underlying.lower()}_index_{resolution}.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"OKX: saved {len(df)} underlying bars → {out_path}")
        return df

    # ------------------------------------------------------------------
    # Instrument discovery
    # ------------------------------------------------------------------

    async def _fetch_active_instruments(self, underlying: str = "ETH") -> list[dict]:
        """Fetch currently listed option instruments."""
        inst_family = f"{underlying.upper()}-USD"
        result = await self._get("/api/v5/public/instruments", {
            "instType": "OPTION", "instFamily": inst_family,
        })
        if not result:
            return []
        instruments = []
        for item in result:
            parsed = parse_okx_instrument(item.get("instId", ""))
            if parsed:
                instruments.append(parsed)
        logger.info(f"OKX: {len(instruments)} active {underlying} option instruments")
        return instruments

    async def _fetch_delivery_exercise_history(
        self,
        underlying: str = "ETH",
        start_date: str = "2025-12-13",
        end_date: str = "2026-03-13",
    ) -> tuple[list[dict], list[dict]]:
        """Fetch delivery/exercise history for instrument discovery + settlements.

        Returns (instruments, settlements).
        """
        inst_family = f"{underlying.upper()}-USD"
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        # Add 1 day to end_dt to include deliveries on the end_date (08:00 UTC)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)

        instruments: list[dict] = []
        settlements: list[dict] = []
        seen: set[str] = set()

        after = ""
        with tqdm(desc="OKX delivery history", unit="page") as pbar:
            while True:
                params: dict[str, str] = {
                    "instType": "OPTION",
                    "instFamily": inst_family,
                    "limit": "100",
                }
                if after:
                    params["after"] = after

                result = await self._get("/api/v5/public/delivery-exercise-history", params)
                if not result:
                    break

                stop = False
                for item in result:
                    ts = int(item.get("ts", 0))
                    dt = datetime.utcfromtimestamp(ts / 1000).replace(tzinfo=timezone.utc)
                    if dt < start_dt:
                        stop = True
                        break
                    if dt > end_dt:
                        continue

                    details = item.get("details", [])
                    for d in details:
                        inst_id = d.get("insId") or d.get("instId", "")
                        parsed = parse_okx_instrument(inst_id)
                        if not parsed:
                            continue

                        if parsed["instrument_name"] not in seen:
                            seen.add(parsed["instrument_name"])
                            instruments.append(parsed)

                        settlements.append({
                            "instrument_name": parsed["instrument_name"],
                            "settlement_timestamp": dt,
                            "delivery_price": float(d.get("px", 0)),
                            "type": d.get("type", "delivery"),
                        })

                if stop or len(result) < 100:
                    break
                after = result[-1].get("ts", "")
                pbar.update(1)

        logger.info(f"OKX: found {len(seen)} expired instruments, {len(settlements)} settlement records")
        return instruments, settlements

    async def fetch_instruments(
        self,
        underlying: str = "ETH",
        start_date: str = "2025-12-13",
        end_date: str = "2026-03-13",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Discover all instruments (active + expired) and save."""
        expired_insts, settlement_rows = await self._fetch_delivery_exercise_history(
            underlying, start_date, end_date,
        )
        active_insts = await self._fetch_active_instruments(underlying)

        # Merge and deduplicate
        all_insts: dict[str, dict] = {}
        for inst in expired_insts + active_insts:
            all_insts[inst["instrument_name"]] = inst

        if not all_insts:
            logger.warning("OKX: no instruments found")
            return pd.DataFrame(), pd.DataFrame()

        # Build instruments DF (drop okx_inst_id for storage)
        inst_rows = []
        for inst in all_insts.values():
            inst_rows.append({
                "instrument_name": inst["instrument_name"],
                "underlying": inst["underlying"],
                "expiration_date": inst["expiration_date"],
                "strike_price": inst["strike_price"],
                "option_type": inst["option_type"],
            })
        instruments_df = pd.DataFrame(inst_rows).sort_values(
            ["expiration_date", "strike_price", "option_type"]
        ).reset_index(drop=True)

        # Save
        out_dir = self.data_dir / "instruments"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{underlying.lower()}_instruments.parquet"
        instruments_df.to_parquet(out_path, index=False)
        logger.info(f"OKX: saved {len(instruments_df)} instruments → {out_path}")

        # Settlements
        settlements_df = pd.DataFrame()
        if settlement_rows:
            settlements_df = pd.DataFrame(settlement_rows)
            sett_dir = self.data_dir / "settlements"
            sett_dir.mkdir(parents=True, exist_ok=True)
            sett_path = sett_dir / f"{underlying.lower()}_settlements.parquet"
            settlements_df.to_parquet(sett_path, index=False)
            logger.info(f"OKX: saved {len(settlements_df)} settlements → {sett_path}")

        # Build okx_inst_id mapping for OHLCV fetching
        self._inst_id_map = {inst["instrument_name"]: inst["okx_inst_id"]
                             for inst in all_insts.values()}

        return instruments_df, settlements_df

    # ------------------------------------------------------------------
    # Option mark-price candles
    # ------------------------------------------------------------------

    async def fetch_option_ohlcv(
        self,
        okx_inst_id: str,
        deribit_name: str,
        resolution: str = "60",
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> pd.DataFrame:
        """Fetch mark-price candles for a single option instrument.

        Uses /api/v5/market/history-mark-price-candles which retains
        ~3 months of data for expired contracts.
        """
        _bar_map = {
            "1": "1m", "5": "5m", "15": "15m", "30": "30m",
            "60": "1H", "120": "2H", "240": "4H", "1D": "1D",
        }
        bar = _bar_map.get(str(resolution), "1H")

        all_bars: list[dict] = []
        after = str(end_ts) if end_ts else ""

        max_pages = 50  # safety limit
        for _ in range(max_pages):
            params: dict[str, str] = {
                "instId": okx_inst_id, "bar": bar, "limit": "100",
            }
            if after:
                params["after"] = after

            result = await self._get("/api/v5/market/history-mark-price-candles", params)
            if not result:
                break

            for row in result:
                ts = int(row[0])
                if start_ts and ts < start_ts:
                    continue
                all_bars.append({
                    "timestamp": ts,
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": 0.0,  # mark-price candles have no volume
                })

            oldest_ts = int(result[-1][0])
            if start_ts and oldest_ts <= start_ts:
                break
            after = str(oldest_ts)
            if len(result) < 100:
                break

        if not all_bars:
            return pd.DataFrame()

        df = pd.DataFrame(all_bars)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["instrument_name"] = deribit_name
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Batch option OHLCV
    # ------------------------------------------------------------------

    async def fetch_all_option_ohlcv(
        self,
        instruments_df: pd.DataFrame,
        resolution: str = "60",
        start_date: str = "2025-12-13",
        end_date: str = "2026-03-13",
        max_concurrent: int = 10,
    ) -> int:
        """Fetch mark-price candles for all instruments. Returns count saved."""
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d")
                       .replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d")
                     .replace(tzinfo=timezone.utc).timestamp() * 1000)

        out_dir = self.data_dir / "market_data" / resolution
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build list of (okx_inst_id, deribit_name) pairs
        pairs: list[tuple[str, str]] = []
        cached = 0
        for _, row in instruments_df.iterrows():
            deribit_name = row["instrument_name"]
            okx_id = self._inst_id_map.get(deribit_name)
            if not okx_id:
                continue
            out_path = out_dir / f"{deribit_name}.parquet"
            if out_path.exists():
                cached += 1
            else:
                pairs.append((okx_id, deribit_name))

        if cached:
            logger.info(f"OKX cache hit: {cached} instruments already on disk")

        sem = asyncio.Semaphore(max_concurrent)
        saved = cached

        async def _fetch_one(okx_id: str, deribit_name: str) -> bool:
            async with sem:
                df = await self.fetch_option_ohlcv(okx_id, deribit_name, resolution, start_ts, end_ts)
                if not df.empty:
                    out_path = out_dir / f"{deribit_name}.parquet"
                    df.to_parquet(out_path, index=False)
                    return True
                return False

        logger.info(f"OKX: fetching mark-price candles for {len(pairs)} instruments "
                    f"(res={resolution}, concurrency={max_concurrent})")
        tasks = [_fetch_one(okx_id, name) for okx_id, name in pairs]
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="OKX option candles"):
            if await coro:
                saved += 1

        logger.info(f"OKX option OHLCV: {saved}/{len(pairs)+cached} instruments "
                    f"({cached} cached, {saved-cached} new)")
        return saved

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    async def fetch_all(
        self,
        underlying: str = "ETH",
        start_date: str = "2025-12-13",
        end_date: str = "2026-03-13",
        resolution: str = "60",
        fetch_option_ohlcv: bool = True,
    ) -> None:
        """Run the complete OKX data-fetching pipeline."""
        try:
            # 1. Underlying index price
            await self.fetch_underlying_ohlcv(underlying, resolution, start_date, end_date)

            # 2. Instrument discovery + settlements
            instruments_df, _ = await self.fetch_instruments(underlying, start_date, end_date)

            # 3. Option mark-price candles
            if fetch_option_ohlcv and not instruments_df.empty:
                await self.fetch_all_option_ohlcv(
                    instruments_df, resolution, start_date, end_date,
                )
        finally:
            await self.close()


# ---------------------------------------------------------------------------
# Sync entry-point
# ---------------------------------------------------------------------------

def run_okx_fetch(
    underlying: str = "ETH",
    start_date: str = "2025-12-13",
    end_date: str = "2026-03-13",
    resolution: str = "60",
    data_dir: str = "data",
    fetch_option_ohlcv: bool = True,
) -> None:
    """Synchronous wrapper to run the OKX async fetcher."""
    fetcher = OKXFetcher(data_dir=data_dir)
    asyncio.run(
        fetcher.fetch_all(
            underlying, start_date, end_date, resolution,
            fetch_option_ohlcv=fetch_option_ohlcv,
        )
    )
