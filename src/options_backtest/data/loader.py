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


class _ValuesView:
    """Tiny proxy so that ``chain['col'].values`` returns the array itself."""
    __slots__ = ("_arr",)

    def __init__(self, arr: np.ndarray):
        self._arr = arr

    @property
    def values(self) -> np.ndarray:
        return self._arr

    # forward .astype so ``chain['col'].values.astype(float)`` works
    def astype(self, dtype):
        return self._arr.astype(dtype)

    # pandas-like .str accessor for long_call.py compatibility
    @property
    def str(self):
        return pd.Series(self._arr).str

    # Comparison operators (return numpy bool arrays for boolean indexing)
    def __eq__(self, other):  return self._arr == other  # noqa: E704
    def __ne__(self, other):  return self._arr != other  # noqa: E704
    def __ge__(self, other):  return self._arr >= other  # noqa: E704
    def __le__(self, other):  return self._arr <= other  # noqa: E704
    def __gt__(self, other):  return self._arr > other   # noqa: E704
    def __lt__(self, other):  return self._arr < other   # noqa: E704
    def __hash__(self):       return id(self)             # noqa: E704


class ArrayChain:
    """Lightweight option-chain container backed by numpy arrays.

    Supports the same ``chain['col'].values`` and ``chain.empty``
    interface used by all strategies, but avoids the ~5 s overhead of
    constructing a pandas DataFrame every bar.
    """
    __slots__ = ("_data", "_len")

    def __init__(self, data: dict[str, np.ndarray], length: int):
        self._data = data
        self._len = length

    # chain['col'] returns a _ValuesView or raw array
    def __getitem__(self, key):
        if isinstance(key, str):
            arr = self._data[key]
            return _ValuesView(arr)
        # Boolean indexing: chain[mask] → convert to DataFrame and filter
        return self._to_dataframe()[key]

    # chain.empty
    @property
    def empty(self) -> bool:
        return self._len == 0

    def __len__(self) -> int:
        return self._len

    # Pandas boolean-index filter support for long_call.py:
    #   chain[chain['option_type'].str.lower().str.startswith('c')]
    def __getattr__(self, name):
        # Fallback: convert to DataFrame for legacy code paths
        return getattr(self._to_dataframe(), name)

    def _to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._data)

    # Support iteration over column names (rarely needed)
    @property
    def columns(self):
        return list(self._data.keys())

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
        ohlcv_cursor: dict | None = None,
        ohlcv_arith: dict | None = None,
        inst_arrays: dict | None = None,
    ) -> "ArrayChain":
        """Build an option‑chain snapshot – **fully vectorised**.

        All filtering + Black‑76 pricing is done via NumPy / pandas
        operations with zero Python-level loops.

        Parameters
        ----------
        ohlcv_index : optional pre-indexed dict mapping instrument name to
            (ts_arr, close_arr, low_arr, high_arr, vol_arr) numpy arrays.
            When provided, uses O(log n) searchsorted instead of boolean masks.
        ohlcv_arith : optional pre-built dict mapping instrument name to
            (start_ns, step_ns, length, close, low, high, vol) for O(1)
            arithmetic index lookups on regular hourly grids.
        inst_arrays : optional pre-extracted dict of numpy arrays from
            instruments_df, avoids per-call DataFrame column lookups.
        """
        from options_backtest.pricing.black76 import call_price_vec, put_price_vec

        ts = to_utc_timestamp(timestamp)

        if instruments_df.empty:
            return ArrayChain({}, 0)

        # --- Use pre-extracted arrays if available, else pull from DataFrame ---
        _ia = inst_arrays
        if _ia is not None and _ia.get("_expiry_ns") is not None:
            exp_ns = _ia["_expiry_ns"]
            strikes_arr = _ia["strike_price"]
            all_names = _ia["instrument_name"]
            all_opt_types = _ia["option_type"]
            all_exp_dt = _ia.get("_expiry_dt")
        else:
            exp_ns = instruments_df["_expiry_ns"].values if "_expiry_ns" in instruments_df.columns else None
            strikes_arr = instruments_df["strike_price"].values
            all_names = instruments_df["instrument_name"].values
            all_opt_types = instruments_df["option_type"].values
            all_exp_dt = instruments_df["_expiry_dt"].values if "_expiry_dt" in instruments_df.columns else None

        # --- Fast numpy filtering (avoids pd.Series arithmetic) ----------------
        if exp_ns is not None:
            # Pure numpy DTE calculation using pre-computed int64 nanoseconds
            ts_ns = ts.value  # nanoseconds since epoch
            dte_ns = exp_ns - ts_ns
            dte_days = dte_ns / (86400 * 1_000_000_000)  # ns → days (float)
            mask_np = ((dte_ns > 0) & (dte_days <= max_dte)
                       & (strikes_arr >= underlying_price * 0.3)
                       & (strikes_arr <= underlying_price * 3.0))
        else:
            exp_col = "expiration_timestamp" if "expiration_timestamp" in instruments_df.columns else "expiration_date"
            exps = pd.to_datetime(instruments_df[exp_col], utc=True)
            dte_seconds = (exps - ts).dt.total_seconds()
            dte_days_s = dte_seconds / 86400.0
            mask_np = ((dte_seconds > 0).values & (dte_days_s <= max_dte).values
                       & (strikes_arr >= underlying_price * 0.3)
                       & (strikes_arr <= underlying_price * 3.0))
            dte_days = dte_days_s.values

        idx_arr = np.flatnonzero(mask_np)
        if len(idx_arr) == 0:
            return ArrayChain({}, 0)

        # --- Build chain from numpy arrays (avoid full DataFrame copy) ---------
        names = all_names[idx_arr]
        n = len(names)
        dte_vals = dte_days[idx_arr] if isinstance(dte_days, np.ndarray) else dte_days.values[idx_arr]
        dte_vals = np.maximum(dte_vals, 0.001)
        strike_vals = strikes_arr[idx_arr]
        opt_type_vals = all_opt_types[idx_arr]

        if all_exp_dt is not None:
            exp_dt_vals = all_exp_dt[idx_arr]
        else:
            exp_col = "expiration_timestamp" if "expiration_timestamp" in instruments_df.columns else "expiration_date"
            exp_dt_vals = pd.to_datetime(instruments_df[exp_col].values[idx_arr], utc=True)

        # --- OHLCV lookup (vectorised batch approach) -------------------------
        mark_arr = np.full(n, np.nan)
        bid_arr = np.full(n, np.nan)
        ask_arr = np.full(n, np.nan)
        vol_arr = np.zeros(n)

        if prefer_market_data:
            if ohlcv_arith or ohlcv_index:
                ts_np = ts.to_datetime64() if hasattr(ts, "to_datetime64") else np.datetime64(ts)
                ts_ns_val = int(np.datetime64(ts_np, "ns").view("int64"))
                _arith_get = ohlcv_arith.get if ohlcv_arith else None
                _ohlcv_get = ohlcv_index.get if ohlcv_index else None
                for i in range(n):
                    nm = names[i]
                    # Fast path: O(1) arithmetic index for regular grids
                    if _arith_get is not None:
                        arith = _arith_get(nm)
                        if arith is not None:
                            start_ns, step_ns, a_len, a_close, a_low, a_high, a_vol = arith
                            j = (ts_ns_val - start_ns) // step_ns
                            if 0 <= j < a_len:
                                mark_arr[i] = a_close[j]
                                bid_arr[i] = a_low[j]
                                ask_arr[i] = a_high[j]
                                vol_arr[i] = a_vol[j]
                            elif j >= a_len:
                                # past end: use last available
                                mark_arr[i] = a_close[a_len - 1]
                                bid_arr[i] = a_low[a_len - 1]
                                ask_arr[i] = a_high[a_len - 1]
                                vol_arr[i] = a_vol[a_len - 1]
                            continue
                    # Fallback: O(log n) searchsorted for irregular grids
                    if _ohlcv_get is not None:
                        idx_data = _ohlcv_get(nm)
                        if idx_data is not None:
                            ts_a, close_a, low_a, high_a, vol_a_arr = idx_data
                            j = int(np.searchsorted(ts_a, ts_np, side="right")) - 1
                            if j >= 0:
                                mark_arr[i] = close_a[j]
                                bid_arr[i] = low_a[j]
                                ask_arr[i] = high_a[j]
                                vol_arr[i] = vol_a_arr[j]
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

        # --- Build lightweight ArrayChain (no pandas overhead) ----------------
        chain = ArrayChain({
            "instrument_name": names,
            "underlying_price": np.full(n, underlying_price),
            "strike_price": strike_vals,
            "option_type": opt_type_vals,
            "expiration_date": exp_dt_vals,
            "days_to_expiry": dte_vals,
            "mark_price": mark_arr,
            "bid_price": bid_arr,
            "ask_price": ask_arr,
            "volume": vol_arr,
        }, n)
        return chain
