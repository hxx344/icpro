from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")


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


def _load_existing_underlying(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Underlying file not found: {path}")
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def _build_extension_rows(underlying: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    hourly_root = DATA_DIR / "options_hourly" / underlying.upper()
    if not hourly_root.exists():
        raise FileNotFoundError(f"options_hourly dir not found: {hourly_root}")

    month_starts = pd.date_range(
        start=pd.Timestamp(year=start_ts.year, month=start_ts.month, day=1, tz="UTC"),
        end=pd.Timestamp(year=end_ts.year, month=end_ts.month, day=1, tz="UTC"),
        freq="MS",
        tz="UTC",
    )

    frames: list[pd.DataFrame] = []
    for month in month_starts:
        path = hourly_root / f"{month.year:04d}-{month.month:02d}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["hour", "hourly_pick", "underlying_price"])
        df["hour"] = pd.to_datetime(df["hour"], utc=True)
        df = df[(df["hour"] >= start_ts) & (df["hour"] <= end_ts)].copy()
        if df.empty:
            continue
        df["hourly_pick"] = df["hourly_pick"].astype(str).str.lower()
        df["underlying_price"] = pd.to_numeric(df["underlying_price"], errors="coerce")
        df = df.loc[df["underlying_price"].notna()].copy()
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "underlying"])

    raw = pd.concat(frames, ignore_index=True)
    agg = (
        raw.groupby(["hour", "hourly_pick"], sort=True)["underlying_price"]
        .median()
        .unstack("hourly_pick")
        .sort_index()
    )

    if "open" not in agg.columns:
        agg["open"] = pd.NA
    if "close" not in agg.columns:
        agg["close"] = pd.NA

    agg["open"] = pd.to_numeric(agg["open"], errors="coerce")
    agg["close"] = pd.to_numeric(agg["close"], errors="coerce")
    agg["open"] = agg["open"].fillna(agg["close"])
    agg["close"] = agg["close"].fillna(agg["open"])
    agg = agg.dropna(subset=["open", "close"])
    if agg.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "underlying"])

    out = pd.DataFrame({
        "timestamp": agg.index,
        "open": agg["open"].astype(float),
        "high": agg[["open", "close"]].max(axis=1).astype(float),
        "low": agg[["open", "close"]].min(axis=1).astype(float),
        "close": agg["close"].astype(float),
        "volume": 0.0,
        "underlying": underlying.upper(),
    }).reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Extend underlying hourly parquet from options_hourly median underlying_price")
    parser.add_argument("--underlying", default="BTC")
    parser.add_argument("--start")
    parser.add_argument("--end")
    args = parser.parse_args()

    underlying = str(args.underlying).upper()
    path = DATA_DIR / "underlying" / f"{underlying.lower()}_index_60.parquet"
    existing = _load_existing_underlying(path)
    last_ts = existing["timestamp"].max()

    if args.start:
        start_ts = pd.Timestamp(args.start, tz="UTC")
    else:
        start_ts = last_ts + pd.Timedelta(hours=1)

    if args.end:
        end_ts = pd.Timestamp(args.end, tz="UTC")
    else:
        options_last = pd.Timestamp.min.tz_localize("UTC")
        hourly_root = DATA_DIR / "options_hourly" / underlying
        for p in sorted(hourly_root.glob("*.parquet")):
            df = pd.read_parquet(p, columns=["hour"])
            hour_series = pd.Series(pd.to_datetime(df["hour"], utc=True), copy=False)
            cur = hour_series.max()
            if pd.notna(cur) and cur > options_last:
                options_last = cur
        end_ts = options_last

    if start_ts > end_ts:
        print({"status": "noop", "start": str(start_ts), "end": str(end_ts)})
        return

    extension = _build_extension_rows(underlying, start_ts, end_ts)
    if extension.empty:
        print({"status": "noop", "reason": "no extension rows", "start": str(start_ts), "end": str(end_ts)})
        return

    merged = pd.concat([existing, extension], ignore_index=True)
    merged = merged.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    _atomic_write_parquet(path, merged)

    print({
        "status": "ok",
        "underlying": underlying,
        "added_rows": int(len(extension)),
        "new_max": str(merged["timestamp"].max()),
        "start": str(start_ts),
        "end": str(end_ts),
    })


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print({"status": "error", "error": str(exc)})
        raise
