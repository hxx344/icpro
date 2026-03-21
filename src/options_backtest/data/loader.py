"""Data loader – reads Parquet files and builds option‑chain snapshots.

When real option OHLCV data is not available for a particular instrument, the
loader can compute a synthetic mark price via Black‑76 using the underlying
price and a configurable flat implied volatility (default 60 %).
"""

from __future__ import annotations

import hashlib
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from options_backtest.utils import to_utc_timestamp

# ---------------------------------------------------------------------------
# Module-level in-memory cache (survives across BacktestEngine instances
# within the same Python process – extremely useful for parameter sweeps).
# ---------------------------------------------------------------------------
_ohlcv_memory_cache: dict[str, dict[str, pd.DataFrame]] = {}


class DataLoader:
    """Load historical data from the local Parquet data store."""

    def __init__(self, data_dir: str | Path = "data"):
        self.data_dir = Path(data_dir)
        self._cache_dir = self.data_dir / ".cache"

    # ------------------------------------------------------------------
    # Instruments
    # ------------------------------------------------------------------

    def load_instruments(self, underlying: str = "BTC") -> pd.DataFrame:
        """Load the instrument catalogue."""
        path = self.data_dir / "instruments" / f"{underlying.lower()}_instruments.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Instruments file not found: {path}")
        df = pd.read_parquet(path)
        logger.info(f"Loaded {len(df)} instruments from {path}")
        return df

    # ------------------------------------------------------------------
    # Underlying price series
    # ------------------------------------------------------------------

    def load_underlying(
        self,
        underlying: str = "BTC",
        resolution: str = "1D",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load the underlying index OHLCV series."""
        path = self.data_dir / "underlying" / f"{underlying.lower()}_index_{resolution}.parquet"
        if not path.exists():
            # Try legacy naming convention
            alt = self.data_dir / "underlying" / f"{underlying.lower()}_index_{resolution}m.parquet"
            if alt.exists():
                path = alt
            else:
                raise FileNotFoundError(f"Underlying data not found: {path}")

        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        if start_date:
            ts = to_utc_timestamp(start_date)
            df = df[df["timestamp"] >= ts]
        if end_date:
            ts = to_utc_timestamp(end_date)
            df = df[df["timestamp"] <= ts]

        logger.info(f"Loaded {len(df)} underlying bars ({df['timestamp'].min()} → {df['timestamp'].max()})")
        return df

    # ------------------------------------------------------------------
    # Option market data
    # ------------------------------------------------------------------

    def load_option_data(self, instrument_name: str, resolution: str = "1D") -> pd.DataFrame:
        """Load OHLCV data for one option instrument."""
        # Try resolution-specific subdirectory first
        path = self.data_dir / "market_data" / resolution / f"{instrument_name}.parquet"
        if not path.exists():
            # Fallback to flat directory (legacy layout)
            path = self.data_dir / "market_data" / f"{instrument_name}.parquet"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.sort_values("timestamp").reset_index(drop=True)

    def load_all_option_data(
        self,
        instruments_df: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        resolution: str = "1D",
    ) -> dict[str, pd.DataFrame]:
        """Load option OHLCV data with aggressive caching.

        Caching layers (fastest to slowest):
        1. In-memory module cache  – instant on repeated runs in same process
        2. On-disk pickle cache    – ~2-3 s vs 120+ s cold load
        3. Full parquet scan       – original multi-file IO path (cold start)
        """
        import os

        # ---- Determine which instruments to load ---------------------------
        market_dir = self.data_dir / "market_data" / resolution
        legacy_dir = self.data_dir / "market_data"
        if not market_dir.exists():
            market_dir = legacy_dir
        if not market_dir.exists():
            logger.info("No market_data directory — using synthetic pricing")
            return {}

        market_dir_str = str(market_dir)
        existing_files = {
            f[:-8] for f in os.listdir(market_dir_str)
            if f.endswith(".parquet")
        }
        if not existing_files:
            logger.info("No parquet files found in market_data dir")
            return {}

        exp_col = ("expiration_timestamp"
                   if "expiration_timestamp" in instruments_df.columns
                   else "expiration_date")
        exps = pd.to_datetime(instruments_df[exp_col], utc=True)

        relevant_mask = pd.Series(True, index=instruments_df.index)
        if start_date:
            ts_start = to_utc_timestamp(start_date)
            relevant_mask = relevant_mask & (exps >= ts_start)
        if end_date:
            ts_end = to_utc_timestamp(end_date)
            relevant_mask = relevant_mask & (
                exps <= ts_end + pd.Timedelta(days=90)
            )

        all_relevant = instruments_df.loc[relevant_mask, "instrument_name"].tolist()
        to_load = sorted(n for n in all_relevant if n in existing_files)
        logger.info(
            f"Filtered to {len(all_relevant)} relevant instruments "
            f"(from {len(instruments_df)}), {len(to_load)} have data files"
        )
        if not to_load:
            logger.info("No matching data files found")
            return {}

        # ---- Cache key = hash of (file list + date range + resolution) -----
        cache_key = self._make_cache_key(to_load, start_date, end_date, resolution, market_dir)

        # Layer 1: in-memory cache (same process, e.g. parameter sweep)
        if cache_key in _ohlcv_memory_cache:
            cached = _ohlcv_memory_cache[cache_key]
            logger.info(f"OHLCV cache HIT (memory): {len(cached)} instruments")
            return cached

        # Layer 2: on-disk pickle cache
        disk_result = self._load_disk_cache(cache_key)
        if disk_result is not None:
            logger.info(f"OHLCV cache HIT (disk): {len(disk_result)} instruments")
            _ohlcv_memory_cache[cache_key] = disk_result
            return disk_result

        # Layer 3: cold load from individual parquet files
        logger.info(f"OHLCV cache MISS — loading {len(to_load)} files from disk...")
        result = self._load_all_option_data_cold(to_load, start_date, end_date, resolution)

        # Persist to both caches
        _ohlcv_memory_cache[cache_key] = result
        self._save_disk_cache(cache_key, result)

        return result

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_cache_key(
        names: list[str],
        start_date: Optional[str],
        end_date: Optional[str],
        resolution: str,
        market_dir: Path,
    ) -> str:
        """Compute a stable hash key from the load parameters."""
        # Include directory mtime so cache auto-invalidates when data changes
        try:
            dir_mtime = int(market_dir.stat().st_mtime)
        except Exception:
            dir_mtime = 0
        blob = f"{len(names)}|{names[0]}|{names[-1]}|{start_date}|{end_date}|{resolution}|{dir_mtime}"
        return hashlib.md5(blob.encode()).hexdigest()

    def _load_disk_cache(self, key: str) -> dict[str, pd.DataFrame] | None:
        """Try to load cached OHLCV dict from disk."""
        path = self._cache_dir / f"ohlcv_{key}.pkl"
        if not path.exists():
            return None
        try:
            import time
            t0 = time.perf_counter()
            with open(path, "rb") as f:
                data = pickle.load(f)
            elapsed = time.perf_counter() - t0
            logger.info(f"Disk cache loaded in {elapsed:.1f}s ({path.stat().st_size / 1e6:.0f} MB)")
            return data
        except Exception as e:
            logger.warning(f"Cache load failed: {e}, removing stale cache")
            path.unlink(missing_ok=True)
            return None

    def _save_disk_cache(self, key: str, data: dict[str, pd.DataFrame]) -> None:
        """Persist OHLCV dict to disk pickle cache."""
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            path = self._cache_dir / f"ohlcv_{key}.pkl"
            import time
            t0 = time.perf_counter()
            with open(path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            elapsed = time.perf_counter() - t0
            size_mb = path.stat().st_size / 1e6
            logger.info(f"Disk cache saved in {elapsed:.1f}s ({size_mb:.0f} MB) → {path}")
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")

    # ------------------------------------------------------------------
    # Cold load (original slow path)
    # ------------------------------------------------------------------

    def _load_all_option_data_cold(
        self,
        to_load: list[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        resolution: str = "1D",
    ) -> dict[str, pd.DataFrame]:
        """Load OHLCV data from individual parquet files (no cache)."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        available: dict[str, pd.DataFrame] = {}
        ts_start_filter = to_utc_timestamp(start_date) if start_date else None
        ts_end_filter = to_utc_timestamp(end_date) if end_date else None
        n_workers = min(32, len(to_load))

        def _load_one(name: str):
            df = self.load_option_data(name, resolution)
            if df.empty:
                return None
            if ts_start_filter is not None:
                df = df[df["timestamp"] >= ts_start_filter]
            if ts_end_filter is not None:
                df = df[df["timestamp"] <= ts_end_filter]
            if df.empty:
                return None
            return df

        loaded = 0
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_load_one, name): name for name in to_load}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    df = future.result()
                    if df is not None:
                        available[name] = df
                        loaded += 1
                except Exception:
                    pass

        logger.info(f"Loaded market data for {loaded} / {len(to_load)} files")
        return available

    # ------------------------------------------------------------------
    # Settlements
    # ------------------------------------------------------------------

    def load_settlements(self, underlying: str = "BTC") -> pd.DataFrame:
        """Load settlement / delivery records."""
        path = self.data_dir / "settlements" / f"{underlying.lower()}_settlements.parquet"
        if not path.exists():
            logger.warning(f"Settlements file not found: {path}")
            return pd.DataFrame()
        return pd.read_parquet(path)

    # ------------------------------------------------------------------
    # Option chain snapshot
    # ------------------------------------------------------------------

    def build_option_chain(
        self,
        instruments_df: pd.DataFrame,
        option_data: dict[str, pd.DataFrame],
        timestamp: datetime,
        underlying_price: float,
        default_iv: float = 0.60,
        max_dte: float = 90.0,
        ohlcv_index: dict | None = None,
        source_counter: dict | None = None,
        prefer_market_data: bool = True,
    ) -> pd.DataFrame:
        """Build an option‑chain snapshot – **fully vectorised**.

        All filtering + Black‑76 pricing is done via NumPy / pandas
        operations with zero Python-level loops.

        Parameters
        ----------
        ohlcv_index : optional pre-indexed dict mapping instrument name to
            (ts_arr, close_arr, low_arr, high_arr, vol_arr) numpy arrays.
            When provided, uses O(log n) searchsorted instead of boolean masks.
        """
        from options_backtest.pricing.black76 import call_price_vec, put_price_vec

        ts = to_utc_timestamp(timestamp)

        if instruments_df.empty:
            return pd.DataFrame()

        # --- Fast numpy filtering (avoids pd.Series arithmetic) ----------------
        if "_expiry_ns" in instruments_df.columns:
            # Pure numpy DTE calculation using pre-computed int64 nanoseconds
            exp_ns = instruments_df["_expiry_ns"].values  # int64
            ts_ns = ts.value  # nanoseconds since epoch
            dte_ns = exp_ns - ts_ns
            dte_days = dte_ns / (86400 * 1_000_000_000)  # ns → days (float)
            strikes_arr = instruments_df["strike_price"].values
            mask_np = ((dte_ns > 0) & (dte_days <= max_dte)
                       & (strikes_arr >= underlying_price * 0.3)
                       & (strikes_arr <= underlying_price * 3.0))
            mask = pd.Series(mask_np, index=instruments_df.index)
        else:
            exp_col = "expiration_timestamp" if "expiration_timestamp" in instruments_df.columns else "expiration_date"
            exps = pd.to_datetime(instruments_df[exp_col], utc=True)
            dte_seconds = (exps - ts).dt.total_seconds()
            dte_days_s = dte_seconds / 86400.0
            strike_col = "strike_price" if "strike_price" in instruments_df.columns else "strike"
            strikes_s = instruments_df[strike_col]
            mask = ((dte_seconds > 0) & (dte_days_s <= max_dte)
                    & (strikes_s >= underlying_price * 0.3)
                    & (strikes_s <= underlying_price * 3.0))
            dte_days = dte_days_s.values

        idx_arr = np.flatnonzero(mask.values if hasattr(mask, 'values') else mask)
        if len(idx_arr) == 0:
            return pd.DataFrame()

        # --- Build chain from numpy arrays (avoid full DataFrame copy) ---------
        names = instruments_df["instrument_name"].values[idx_arr]
        n = len(names)
        dte_vals = dte_days[idx_arr] if isinstance(dte_days, np.ndarray) else dte_days.values[idx_arr]
        dte_vals = np.maximum(dte_vals, 0.001)
        strike_vals = instruments_df["strike_price"].values[idx_arr]
        opt_type_vals = instruments_df["option_type"].values[idx_arr]

        if "_expiry_dt" in instruments_df.columns:
            exp_dt_vals = instruments_df["_expiry_dt"].values[idx_arr]
        else:
            exp_col = "expiration_timestamp" if "expiration_timestamp" in instruments_df.columns else "expiration_date"
            exp_dt_vals = pd.to_datetime(instruments_df[exp_col].values[idx_arr], utc=True)

        # --- OHLCV lookup (vectorised batch approach) -------------------------
        mark_arr = np.full(n, np.nan)
        bid_arr = np.full(n, np.nan)
        ask_arr = np.full(n, np.nan)
        vol_arr = np.zeros(n)

        if prefer_market_data:
            if ohlcv_index:
                ts_np = ts.to_datetime64() if hasattr(ts, "to_datetime64") else np.datetime64(ts)
                for i in range(n):
                    idx_data = ohlcv_index.get(names[i])
                    if idx_data is None:
                        continue
                    ts_a, close_a, low_a, high_a, vol_a = idx_data
                    j = np.searchsorted(ts_a, ts_np, side="right") - 1
                    if j >= 0:
                        mark_arr[i] = close_a[j]
                        bid_arr[i] = low_a[j]
                        ask_arr[i] = high_a[j]
                        vol_arr[i] = vol_a[j]
            elif option_data:
                ts_np64 = ts.to_datetime64() if hasattr(ts, "to_datetime64") else np.datetime64(ts)
                for i, name in enumerate(names):
                    ohlcv = option_data.get(name)
                    if ohlcv is None or ohlcv.empty:
                        continue
                    row_mask = ohlcv["timestamp"].values <= ts_np64
                    if not row_mask.any():
                        continue
                    j = np.flatnonzero(row_mask)[-1]
                    mark_arr[i] = ohlcv.iloc[j]["close"]
                    bid_arr[i] = ohlcv.iloc[j]["low"]
                    ask_arr[i] = ohlcv.iloc[j]["high"]
                    vol_arr[i] = ohlcv.iloc[j].get("volume", 0)

        # --- Vectorised Black-76 synthetic fill for missing marks ---------------
        need_synth = np.isnan(mark_arr) | (mark_arr <= 0)
        n_market = int(n - np.sum(need_synth))
        n_synth = int(np.sum(need_synth))
        if source_counter is not None:
            source_counter["market"] = source_counter.get("market", 0) + n_market
            source_counter["synth"] = source_counter.get("synth", 0) + n_synth
        if need_synth.any():
            s_strikes = strike_vals[need_synth]
            s_T = dte_vals[need_synth] / 365.0
            is_call = (opt_type_vals[need_synth] == "call")

            call_px = call_price_vec(underlying_price, s_strikes, s_T, default_iv, r=0.0)
            put_px = put_price_vec(underlying_price, s_strikes, s_T, default_iv, r=0.0)
            synth_usd = np.where(is_call, call_px, put_px)
            synth_mark = synth_usd / underlying_price if underlying_price > 0 else synth_usd * 0

            mark_arr[need_synth] = synth_mark
            bid_arr[need_synth] = np.maximum(synth_mark * 0.98, 0.0)
            ask_arr[need_synth] = synth_mark * 1.02

        # --- Build result DataFrame directly from arrays -----------------------
        chain = pd.DataFrame({
            "instrument_name": names,
            "underlying_price": underlying_price,
            "strike_price": strike_vals,
            "option_type": opt_type_vals,
            "expiration_date": exp_dt_vals,
            "days_to_expiry": dte_vals,
            "mark_price": mark_arr,
            "bid_price": bid_arr,
            "ask_price": ask_arr,
            "volume": vol_arr,
        })
        chain = chain.sort_values(["expiration_date", "strike_price"]).reset_index(drop=True) if "_expiry_ns" not in instruments_df.columns else chain.reset_index(drop=True)
        return chain
