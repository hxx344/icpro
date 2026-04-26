from __future__ import annotations

import hashlib
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from options_backtest.utils import to_utc_timestamp

_HOURLY_STORE_CACHE: dict[str, "HourlyOptionStore"] = {}
_HOURLY_STORE_DISK_CACHE_VERSION = 2
_HOURLY_STORE_MONTH_CACHE_VERSION = 1

_RAW_COLUMNS = [
    "symbol", "type", "strike_price", "expiration", "open_interest",
    "last_price", "bid_price", "ask_price", "mark_price", "mark_iv",
    "bid_iv", "ask_iv", "underlying_price", "delta", "gamma", "vega",
    "theta", "hourly_pick", "hour",
]

_KEEP_COLS = [
    "instrument_name", "hour", "hourly_pick", "option_type", "strike_price",
    "expiration_date", "underlying_price", "mark_price", "bid_price", "ask_price",
    "last_price", "open_interest", "mark_iv", "bid_iv", "ask_iv", "delta",
    "gamma", "vega", "theta",
]


def _to_utc_ns(timestamp) -> int:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.value)


def _series_to_ns(values: pd.Series) -> np.ndarray:
    return values.to_numpy(dtype="datetime64[ns]").astype("int64")


@dataclass
class HourlyOptionStore:
    underlying: str
    frame: pd.DataFrame
    hour_index: dict[tuple[int, str], tuple[int, int]]
    quote_index: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]
    available_ts_ns: dict[str, np.ndarray]
    _last_snapshot_key: tuple[int, str] | None = field(default=None, init=False, repr=False)
    _last_snapshot_frame: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _last_quote_key: tuple[int, str] | None = field(default=None, init=False, repr=False)
    _last_quote_map: dict[str, tuple[float | None, float | None, float | None]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def get_snapshot(self, timestamp, pick: str = "close") -> pd.DataFrame:
        ts_ns = _to_utc_ns(timestamp)
        pick_key = str(pick).lower()
        cache_key = (ts_ns, pick_key)
        if self._last_snapshot_key == cache_key and self._last_snapshot_frame is not None:
            return self._last_snapshot_frame

        bounds = self.hour_index.get(cache_key)
        if bounds is None:
            empty = self.frame.iloc[0:0]
            self._last_snapshot_key = cache_key
            self._last_snapshot_frame = empty
            return empty
        start, end = bounds
        snap = self.frame.iloc[start:end]
        self._last_snapshot_key = cache_key
        self._last_snapshot_frame = snap
        return snap

    def get_quote(self, instrument_name: str, timestamp, pick: str = "close") -> tuple[float | None, float | None, float | None]:
        pick_key = str(pick).lower()
        by_symbol = self.quote_index.get(pick_key, {})
        data = by_symbol.get(str(instrument_name))
        if data is not None:
            ts_arr, bid_arr, ask_arr, mark_arr = data
            ts_ns = _to_utc_ns(timestamp)
            j = int(np.searchsorted(ts_arr, ts_ns, side="right") - 1)
            if j < 0:
                return None, None, None
            bid = float(bid_arr[j]) if np.isfinite(bid_arr[j]) and bid_arr[j] > 0 else None
            ask = float(ask_arr[j]) if np.isfinite(ask_arr[j]) and ask_arr[j] > 0 else None
            mark = float(mark_arr[j]) if np.isfinite(mark_arr[j]) and mark_arr[j] > 0 else None
            if mark is None and bid is not None and ask is not None:
                mark = (bid + ask) / 2.0
            elif mark is None:
                mark = bid or ask
            return bid, ask, mark

        ts_ns = _to_utc_ns(timestamp)
        cache_key = (ts_ns, pick_key)
        if self._last_quote_key != cache_key:
            snap = self.get_snapshot(timestamp, pick=pick_key)
            if snap.empty:
                self._last_quote_map = {}
            else:
                inst_arr = snap["instrument_name"].astype(str).to_numpy()
                bid_arr = snap["bid_price"].astype(float).to_numpy()
                ask_arr = snap["ask_price"].astype(float).to_numpy()
                mark_arr = snap["mark_price"].astype(float).to_numpy()
                quote_map: dict[str, tuple[float | None, float | None, float | None]] = {}
                for i, name in enumerate(inst_arr):
                    bid = float(bid_arr[i]) if np.isfinite(bid_arr[i]) and bid_arr[i] > 0 else None
                    ask = float(ask_arr[i]) if np.isfinite(ask_arr[i]) and ask_arr[i] > 0 else None
                    mark = float(mark_arr[i]) if np.isfinite(mark_arr[i]) and mark_arr[i] > 0 else None
                    if mark is None and bid is not None and ask is not None:
                        mark = (bid + ask) / 2.0
                    elif mark is None:
                        mark = bid or ask
                    quote_map[name] = (bid, ask, mark)
                self._last_quote_map = quote_map
            self._last_quote_key = cache_key

        return self._last_quote_map.get(str(instrument_name), (None, None, None))

    def available_timestamps(self, pick: str = "close") -> np.ndarray:
        return self.available_ts_ns.get(str(pick).lower(), np.array([], dtype=np.int64))


def _month_starts(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> list[pd.Timestamp]:
    start_month = pd.Timestamp(year=start_ts.year, month=start_ts.month, day=1, tz="UTC")
    end_month = pd.Timestamp(year=end_ts.year, month=end_ts.month, day=1, tz="UTC")
    return list(pd.date_range(start=start_month, end=end_month, freq="MS", tz="UTC"))


def _store_cache_key(file_paths: list[Path], start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> str:
    parts: list[str] = [str(start_ts.value), str(end_ts.value)]
    for path in file_paths:
        stat = path.stat()
        parts.append(f"{path}:{int(stat.st_mtime_ns)}:{stat.st_size}")
    return "|".join(parts)


def _disk_cache_paths(data_dir: str | Path, underlying: str, cache_key: str) -> tuple[Path, Path]:
    digest = hashlib.sha256(
        f"v{_HOURLY_STORE_DISK_CACHE_VERSION}|{cache_key}".encode("utf-8")
    ).hexdigest()[:24]
    cache_root = Path(data_dir) / ".cache" / "options_hourly_store" / str(underlying).upper()
    return cache_root / f"{digest}.parquet", cache_root / f"{digest}.meta.pkl"


def _monthly_cache_path(data_dir: str | Path, underlying: str, source_path: Path) -> Path:
    stat = source_path.stat()
    digest = hashlib.sha256(
        (
            f"v{_HOURLY_STORE_MONTH_CACHE_VERSION}|{source_path.as_posix()}|"
            f"{int(stat.st_mtime_ns)}|{int(stat.st_size)}"
        ).encode("utf-8")
    ).hexdigest()[:16]
    cache_root = Path(data_dir) / ".cache" / "options_hourly_monthly" / str(underlying).upper()
    return cache_root / f"{source_path.stem}.{digest}.parquet"


def _atomic_write_pickle(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with open(tmp, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _atomic_write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        df.to_parquet(tmp, engine="pyarrow", compression="zstd", index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _load_disk_cached_store(
    data_dir: str | Path,
    underlying: str,
    cache_key: str,
) -> HourlyOptionStore | None:
    frame_path, meta_path = _disk_cache_paths(data_dir, underlying, cache_key)
    if not frame_path.exists() or not meta_path.exists():
        return None

    try:
        with open(meta_path, "rb") as fh:
            meta = pickle.load(fh)
        if not isinstance(meta, dict) or int(meta.get("version", -1)) != _HOURLY_STORE_DISK_CACHE_VERSION:
            return None

        df = pd.read_parquet(frame_path)
        df["hour"] = pd.to_datetime(df["hour"], utc=True)
        df["expiration_date"] = pd.to_datetime(df["expiration_date"], utc=True)
        available_ts_ns = meta.get("available_ts_ns") or {"open": np.array([], dtype=np.int64), "close": np.array([], dtype=np.int64)}
        hour_index = meta.get("hour_index") or {}

        logger.info(
            f"options_hourly disk cache HIT for {str(underlying).upper()}: {frame_path.name}"
        )
        return HourlyOptionStore(
            underlying=str(underlying).upper(),
            frame=df,
            hour_index=hour_index,
            quote_index={"open": {}, "close": {}},
            available_ts_ns=available_ts_ns,
        )
    except Exception as exc:
        logger.warning(f"options_hourly disk cache load failed: {frame_path.name} ({exc})")
        for stale in (frame_path, meta_path):
            try:
                if stale.exists():
                    stale.unlink()
            except OSError:
                pass
        return None


def _save_disk_cached_store(
    data_dir: str | Path,
    underlying: str,
    cache_key: str,
    df: pd.DataFrame,
    hour_index: dict[tuple[int, str], tuple[int, int]],
    available_ts_ns: dict[str, np.ndarray],
) -> None:
    frame_path, meta_path = _disk_cache_paths(data_dir, underlying, cache_key)
    meta = {
        "version": _HOURLY_STORE_DISK_CACHE_VERSION,
        "hour_index": hour_index,
        "available_ts_ns": available_ts_ns,
        "rows": int(len(df)),
    }
    try:
        _atomic_write_parquet(frame_path, df)
        _atomic_write_pickle(meta_path, meta)
    except Exception as exc:
        logger.warning(f"options_hourly disk cache save failed: {frame_path.name} ({exc})")
        for stale in (frame_path, meta_path):
            try:
                if stale.exists():
                    stale.unlink()
            except OSError:
                pass


def _normalize_hourly_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=_KEEP_COLS)

    df = df.copy()
    df["hour"] = pd.to_datetime(df["hour"], utc=True)
    df["instrument_name"] = df["symbol"].astype(str)
    df["option_type"] = df["type"].astype(str).str.lower()
    df["expiration_date"] = pd.to_datetime(df["expiration"], unit="us", utc=True)
    df["hourly_pick"] = df["hourly_pick"].astype(str).str.lower()

    numeric_cols = [
        "strike_price", "open_interest", "last_price", "bid_price", "ask_price",
        "mark_price", "mark_iv", "bid_iv", "ask_iv", "underlying_price",
        "delta", "gamma", "vega", "theta",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    mid = (df["bid_price"] + df["ask_price"]) / 2.0
    df["mark_price"] = df["mark_price"].where(df["mark_price"] > 0, mid)
    df["mark_price"] = df["mark_price"].where(df["mark_price"] > 0, df["last_price"])
    df["bid_price"] = df["bid_price"].where(df["bid_price"] > 0)
    df["ask_price"] = df["ask_price"].where(df["ask_price"] > 0)
    df["mark_price"] = df["mark_price"].fillna(0.0)
    df["underlying_price"] = df["underlying_price"].ffill().bfill()

    return df[_KEEP_COLS].sort_values(["hour", "hourly_pick"]).reset_index(drop=True)


def _load_or_build_month_cache(
    data_dir: str | Path,
    underlying: str,
    source_path: Path,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    cache_path = _monthly_cache_path(data_dir, underlying, source_path)
    if not cache_path.exists():
        logger.info(f"options_hourly month cache MISS for {str(underlying).upper()}: {source_path.name}")
        raw_df = pd.read_parquet(source_path, columns=_RAW_COLUMNS)
        month_df = _normalize_hourly_frame(raw_df)
        _atomic_write_parquet(cache_path, month_df)

    filters = [
        ("hour", ">=", start_ts.to_pydatetime()),
        ("hour", "<=", end_ts.to_pydatetime()),
    ]
    month_df = pd.read_parquet(cache_path, filters=filters)
    if month_df.empty:
        return month_df

    month_df["hour"] = pd.to_datetime(month_df["hour"], utc=True)
    month_df["expiration_date"] = pd.to_datetime(month_df["expiration_date"], utc=True)
    return month_df.reset_index(drop=True)


def _build_hour_index(df: pd.DataFrame) -> dict[tuple[int, str], tuple[int, int]]:
    hour_ns = _series_to_ns(df["hour"])
    pick_arr = df["hourly_pick"].to_numpy(copy=False)
    index: dict[tuple[int, str], tuple[int, int]] = {}
    if len(df) == 0:
        return index

    start = 0
    current_hour = int(hour_ns[0])
    current_pick = str(pick_arr[0])
    for i in range(1, len(df)):
        hour_i = int(hour_ns[i])
        pick_i = str(pick_arr[i])
        if hour_i != current_hour or pick_i != current_pick:
            index[(current_hour, current_pick)] = (start, i)
            start = i
            current_hour = hour_i
            current_pick = pick_i
    index[(current_hour, current_pick)] = (start, len(df))
    return index


def _build_quote_index(df: pd.DataFrame) -> tuple[dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]], dict[str, np.ndarray]]:
    quote_index: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {"open": {}, "close": {}}
    available_ts_ns: dict[str, np.ndarray] = {"open": np.array([], dtype=np.int64), "close": np.array([], dtype=np.int64)}

    for pick in ("open", "close"):
        sub = df[df["hourly_pick"] == pick]
        if sub.empty:
            continue
        available_ts_ns[pick] = np.sort(np.unique(_series_to_ns(sub["hour"])))
        grouped = sub.sort_values(["instrument_name", "hour"])
        for name, grp in grouped.groupby("instrument_name", sort=False):
            quote_index[pick][str(name)] = (
            _series_to_ns(grp["hour"]),
                grp["bid_price"].astype(float).to_numpy(),
                grp["ask_price"].astype(float).to_numpy(),
                grp["mark_price"].astype(float).to_numpy(),
            )
    return quote_index, available_ts_ns


def load_hourly_option_store(
    data_dir: str | Path,
    underlying: str,
    start_date,
    end_date,
) -> HourlyOptionStore:
    root = Path(data_dir) / "options_hourly" / str(underlying).upper()
    if not root.exists():
        raise FileNotFoundError(f"options_hourly directory not found: {root}")

    start_ts = to_utc_timestamp(start_date)
    end_ts = to_utc_timestamp(end_date)
    months = _month_starts(start_ts, end_ts)
    paths = [root / f"{m.year:04d}-{m.month:02d}.parquet" for m in months]
    existing_paths = [p for p in paths if p.exists()]
    if not existing_paths:
        raise FileNotFoundError(f"No options_hourly parquet files found for {underlying} in {root}")

    cache_key = _store_cache_key(existing_paths, start_ts, end_ts)
    cached = _HOURLY_STORE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    disk_cached = _load_disk_cached_store(data_dir, underlying, cache_key)
    if disk_cached is not None:
        _HOURLY_STORE_CACHE[cache_key] = disk_cached
        return disk_cached

    frames: list[pd.DataFrame] = []
    for path in existing_paths:
        month_df = _load_or_build_month_cache(data_dir, underlying, path, start_ts, end_ts)
        if not month_df.empty:
            frames.append(month_df)
    if not frames:
        raise RuntimeError(f"No data loaded from options_hourly for {underlying}")

    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise RuntimeError(f"No options_hourly rows found for {underlying} between {start_ts} and {end_ts}")
    df["underlying_price"] = df["underlying_price"].ffill().bfill()
    df = df.reset_index(drop=True)

    hour_ns = _series_to_ns(df["hour"])
    pick_arr = df["hourly_pick"].to_numpy(copy=False)
    available_ts_ns = {
        pick: np.sort(np.unique(hour_ns[pick_arr == pick]))
        for pick in ("open", "close")
    }
    hour_index = _build_hour_index(df)

    _save_disk_cached_store(data_dir, underlying, cache_key, df, hour_index, available_ts_ns)

    store = HourlyOptionStore(
        underlying=str(underlying).upper(),
        frame=df,
        hour_index=hour_index,
        quote_index={"open": {}, "close": {}},
        available_ts_ns=available_ts_ns,
    )
    _HOURLY_STORE_CACHE[cache_key] = store
    return store
