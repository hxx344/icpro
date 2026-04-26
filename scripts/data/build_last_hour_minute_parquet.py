#!/usr/bin/env python3
from __future__ import annotations

"""Build minute-level BTC option quote snapshots for a configurable UTC window.

This script reuses the same Deribit tick-level daily `.zst` source as
`scripts/build_hourly_parquet.py`, but only keeps BTC options in a configurable
intraday time window (default: full day 00:00-00:00 UTC), filters contracts to
<= 7 calendar days to expiry and <= 10% OTM, then downsamples them to
per-minute quote snapshots.

Output layout:
    data/options_minute_daily/{UNDERLYING}/{YYYY-MM-DD}.parquet

Each output row is one `symbol` in one `minute`, carrying both the last quote
fields and per-minute option/underlying OHLC columns. The result is suitable
for rough stop-loss replay against real minute quotes.
"""

import argparse
import concurrent.futures
import io
import json
import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time as time_mod
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import zstandard as zstd

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_hourly_parquet as hourly_builder


LOG = logging.getLogger("build_last_hour_minute")
DEFAULT_OUTPUT_DIR = Path("data/options_minute_daily")
DEFAULT_DOWNLOAD_WORKERS = 8
DEFAULT_PROCESS_WORKERS = 2
MIN_MINUTE_AVAILABILITY_PCT = 100.0
AVAILABILITY_SCAN_CACHE_VERSION = 1
DEFAULT_MAX_DTE_DAYS = 7.0
DEFAULT_MAX_OTM_PCT = 10.0
CSV_COLUMNS = [
    "timestamp",
    "symbol",
    "type",
    "strike_price",
    "expiration",
    "last_price",
    "bid_price",
    "ask_price",
    "mark_price",
    "bid_iv",
    "ask_iv",
    "mark_iv",
    "underlying_price",
    "delta",
]
CSV_DTYPES = {
    "timestamp": "int64",
    "symbol": "string",
    "type": "string",
    "strike_price": "float64",
    "expiration": "int64",
    "last_price": "float64",
    "bid_price": "float64",
    "ask_price": "float64",
    "mark_price": "float64",
    "bid_iv": "float64",
    "ask_iv": "float64",
    "mark_iv": "float64",
    "underlying_price": "float64",
    "delta": "float64",
}


def _polars_schema_overrides_for_minute(pl):
    return {
        "timestamp": pl.Int64,
        "symbol": pl.Utf8,
        "type": pl.Utf8,
        "strike_price": pl.Float64,
        "expiration": pl.Int64,
        "last_price": pl.Float64,
        "bid_price": pl.Float64,
        "ask_price": pl.Float64,
        "mark_price": pl.Float64,
        "bid_iv": pl.Float64,
        "ask_iv": pl.Float64,
        "mark_iv": pl.Float64,
        "underlying_price": pl.Float64,
        "delta": pl.Float64,
    }


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _parse_hhmm(value: str) -> time:
    return datetime.strptime(value, "%H:%M").time().replace(tzinfo=timezone.utc)


def _daterange(start: date, end: date) -> Iterable[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _normalize_underlyings(value: str) -> list[str]:
    items = [s.strip().upper() for s in str(value).split(",") if s.strip()]
    return sorted(dict.fromkeys(items))


def _window_bounds_us(target_date: date, start_t: time, end_t: time) -> tuple[int, int]:
    start_dt = datetime.combine(target_date, start_t).astimezone(timezone.utc)
    end_dt = datetime.combine(target_date, end_t).astimezone(timezone.utc)
    if end_dt <= start_dt:
        end_dt += timedelta(days=1)
    start_us = int(pd.Timestamp(start_dt).value // 1_000)
    end_us = int(pd.Timestamp(end_dt).value // 1_000)
    return start_us, end_us


def _expected_window_minutes(target_date: date, start_t: time, end_t: time) -> int:
    start_dt = datetime.combine(target_date, start_t).astimezone(timezone.utc)
    end_dt = datetime.combine(target_date, end_t).astimezone(timezone.utc)
    if end_dt <= start_dt:
        end_dt += timedelta(days=1)
    minutes = int((end_dt - start_dt).total_seconds() // 60)
    if minutes <= 0:
        raise ValueError(f"Invalid window: {start_t} -> {end_t}")
    return minutes


def _build_underlying_ohlc(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["underlying", "minute", "underlying_open", "underlying_high", "underlying_low", "underlying_close"])

    agg = (
        raw.sort_values("timestamp")
        .groupby(["underlying", "minute"], as_index=False, sort=False)
        .agg(
            first_ts=("timestamp", "first"),
            last_ts=("timestamp", "last"),
            underlying_open=("underlying_price", "first"),
            underlying_close=("underlying_price", "last"),
            underlying_high=("underlying_price", "max"),
            underlying_low=("underlying_price", "min"),
        )
    )
    return agg[["underlying", "minute", "underlying_open", "underlying_high", "underlying_low", "underlying_close"]]


def _build_option_ohlc(raw: pd.DataFrame) -> pd.DataFrame:
    columns = ["underlying", "symbol", "minute", "option_open", "option_high", "option_low", "option_close"]
    if raw.empty:
        return pd.DataFrame(columns=columns)

    source = raw.dropna(subset=["mark_price"])
    if source.empty:
        return pd.DataFrame(columns=columns)

    agg = (
        source.sort_values("timestamp")
        .groupby(["underlying", "symbol", "minute"], as_index=False, sort=False)
        .agg(
            option_open=("mark_price", "first"),
            option_high=("mark_price", "max"),
            option_low=("mark_price", "min"),
            option_close=("mark_price", "last"),
        )
    )
    return agg[columns]


def _filter_near_otm_polars(chunk, pl, *, max_dte_days: float, max_otm_pct: float):
    max_expiry_us = int(max_dte_days * 24.0 * 60.0 * 60.0 * 1_000_000)
    otm_frac = max_otm_pct / 100.0
    option_type = pl.col("type").str.to_lowercase()
    return chunk.filter(
        ((pl.col("expiration") - pl.col("timestamp")) > 0)
        & ((pl.col("expiration") - pl.col("timestamp")) <= max_expiry_us)
        & (
            (
                option_type.eq("call")
                & (pl.col("strike_price") >= pl.col("underlying_price"))
                & (pl.col("strike_price") <= pl.col("underlying_price") * (1.0 + otm_frac))
            )
            |
            (
                option_type.eq("put")
                & (pl.col("strike_price") <= pl.col("underlying_price"))
                & (pl.col("strike_price") >= pl.col("underlying_price") * (1.0 - otm_frac))
            )
        )
    )


def _filter_near_otm_pandas(chunk: pd.DataFrame, *, max_dte_days: float, max_otm_pct: float) -> pd.DataFrame:
    if chunk.empty:
        return chunk
    max_expiry_us = max_dte_days * 24.0 * 60.0 * 60.0 * 1_000_000
    expiry_delta_us = chunk["expiration"] - chunk["timestamp"]
    option_type = chunk["type"].astype(str).str.lower()
    otm_frac = max_otm_pct / 100.0
    call_mask = (
        option_type.eq("call")
        & chunk["strike_price"].ge(chunk["underlying_price"])
        & chunk["strike_price"].le(chunk["underlying_price"] * (1.0 + otm_frac))
    )
    put_mask = (
        option_type.eq("put")
        & chunk["strike_price"].le(chunk["underlying_price"])
        & chunk["strike_price"].ge(chunk["underlying_price"] * (1.0 - otm_frac))
    )
    mask = expiry_delta_us.gt(0) & expiry_delta_us.le(max_expiry_us) & (call_mask | put_mask)
    return chunk.loc[mask].copy()


def _finalize_minute_parts(quotes: pd.DataFrame, underlying_ohlc: pd.DataFrame, target_date: date) -> pd.DataFrame:
    if quotes.empty:
        return pd.DataFrame()

    option_ohlc = (
        quotes.groupby(["underlying", "symbol", "minute"], as_index=False, sort=False)
        .agg(
            option_open=("option_open", "first"),
            option_high=("option_high", "max"),
            option_low=("option_low", "min"),
            option_close=("option_close", "last"),
        )
    )

    quotes = quotes.sort_values(["underlying", "symbol", "minute", "last_tick_time"]).drop_duplicates(
        subset=["underlying", "symbol", "minute"], keep="last"
    )
    quotes = quotes.drop(columns=["option_open", "option_high", "option_low", "option_close"], errors="ignore")
    quotes = quotes.merge(option_ohlc, on=["underlying", "symbol", "minute"], how="left")

    if underlying_ohlc.empty:
        out = quotes.copy()
    else:
        underlying_ohlc = (
            underlying_ohlc.groupby(["underlying", "minute"], as_index=False, sort=False)
            .agg(
                underlying_open=("underlying_open", "first"),
                underlying_high=("underlying_high", "max"),
                underlying_low=("underlying_low", "min"),
                underlying_close=("underlying_close", "last"),
            )
        )
        out = quotes.merge(underlying_ohlc, on=["underlying", "minute"], how="left")

    out.insert(0, "date", pd.Timestamp(target_date))
    return out.sort_values(["underlying", "minute", "symbol"]).reset_index(drop=True)


def _delete_output_for_date(output_dir: Path, target_date: date, underlyings: list[str]) -> None:
    for underlying in underlyings:
        path = output_dir / str(underlying).upper() / f"{target_date.isoformat()}.parquet"
        if path.exists():
            try:
                path.unlink()
                LOG.warning("Deleted incomplete minute output: %s", path)
            except OSError:
                pass


def _validate_minute_output_frame(
    df: pd.DataFrame,
    *,
    target_date: date,
    underlyings: list[str],
    window_start: time,
    window_end: time,
    max_dte_days: float,
    max_otm_pct: float,
    source_label: str,
) -> None:
    expected_minutes = _expected_window_minutes(target_date, window_start, window_end)
    expected_start_dt = datetime.combine(target_date, window_start).astimezone(timezone.utc)
    expected_end_dt = datetime.combine(target_date, window_end).astimezone(timezone.utc)
    if expected_end_dt <= expected_start_dt:
        expected_end_dt += timedelta(days=1)
    expected_start = pd.Timestamp(expected_start_dt)
    expected_end = pd.Timestamp(expected_end_dt) - pd.Timedelta(minutes=1)

    if df.empty:
        raise RuntimeError(f"{source_label}: no minute rows for {target_date}")

    work = df.copy()
    work["minute"] = pd.to_datetime(work["minute"], utc=True, errors="coerce")
    work["last_tick_time"] = pd.to_datetime(work["last_tick_time"], utc=True, errors="coerce")
    work["expiration"] = pd.to_datetime(work["expiration"], utc=True, errors="coerce")
    bad_rows = work[work["minute"].isna()]
    if not bad_rows.empty:
        raise RuntimeError(f"{source_label}: invalid minute timestamps for {target_date}")
    if work["last_tick_time"].isna().any() or work["expiration"].isna().any():
        raise RuntimeError(f"{source_label}: invalid last_tick_time/expiration timestamps for {target_date}")

    required_option_ohlc = ["option_open", "option_high", "option_low", "option_close"]
    missing_option_ohlc = [col for col in required_option_ohlc if col not in work.columns]
    if missing_option_ohlc:
        raise RuntimeError(f"{source_label}: missing option OHLC columns for {target_date}: {missing_option_ohlc}")

    missing_option_ohlc_rows = ~work[required_option_ohlc].notna().all(axis=1)
    if missing_option_ohlc_rows.any():
        raise RuntimeError(f"{source_label}: missing option OHLC row(s) for {target_date}")

    max_expiry_delta = pd.Timedelta(days=max_dte_days)
    otm_frac = max_otm_pct / 100.0
    option_type = work["type"].astype(str).str.lower()
    expiry_delta = work["expiration"] - work["last_tick_time"]
    call_mask = (
        option_type.eq("call")
        & work["strike_price"].ge(work["underlying_price"])
        & work["strike_price"].le(work["underlying_price"] * (1.0 + otm_frac))
    )
    put_mask = (
        option_type.eq("put")
        & work["strike_price"].le(work["underlying_price"])
        & work["strike_price"].ge(work["underlying_price"] * (1.0 - otm_frac))
    )
    within_filter = expiry_delta.gt(pd.Timedelta(0)) & expiry_delta.le(max_expiry_delta) & (call_mask | put_mask)
    if (~within_filter).any():
        raise RuntimeError(f"{source_label}: found row(s) outside <= {max_dte_days:g} day / <= {max_otm_pct:g}% OTM filter for {target_date}")

    problems: list[str] = []
    for underlying in underlyings:
        sub = work[work["underlying"].astype(str).str.upper() == underlying]
        if sub.empty:
            problems.append(f"{underlying}=missing")
            continue
        unique_minutes = pd.Index(sub["minute"].dropna().unique()).sort_values()
        minute_count = len(unique_minutes)
        min_minute = unique_minutes[0] if minute_count else None
        max_minute = unique_minutes[-1] if minute_count else None
        missing_ohlc = 1.0 - float(sub[["underlying_open", "underlying_high", "underlying_low", "underlying_close"]].notna().all(axis=1).mean())
        if minute_count != expected_minutes or min_minute != expected_start or max_minute != expected_end or missing_ohlc > 0.0:
            problems.append(
                f"{underlying}={minute_count}/{expected_minutes}m range={min_minute}..{max_minute} missing_ohlc_rows={missing_ohlc:.2%}"
            )

    if problems:
        raise RuntimeError(f"{source_label}: incomplete minute availability for {target_date}: " + ", ".join(problems))


def _existing_outputs_complete(
    output_dir: Path,
    *,
    target_date: date,
    underlyings: list[str],
    window_start: time,
    window_end: time,
    max_dte_days: float,
    max_otm_pct: float,
) -> bool:
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for underlying in underlyings:
        path = output_dir / underlying / f"{target_date.isoformat()}.parquet"
        if not path.exists():
            missing.append(underlying)
            continue
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            LOG.warning("Could not read existing minute output %s: %s", path, exc)
            _delete_output_for_date(output_dir, target_date, [underlying])
            return False
        frames.append(frame)

    if missing:
        return False

    try:
        merged = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        _validate_minute_output_frame(
            merged,
            target_date=target_date,
            underlyings=underlyings,
            window_start=window_start,
            window_end=window_end,
            max_dte_days=max_dte_days,
            max_otm_pct=max_otm_pct,
            source_label="existing output",
        )
        return True
    except Exception as exc:
        LOG.warning("Existing minute output for %s is incomplete, will rebuild: %s", target_date, exc)
        _delete_output_for_date(output_dir, target_date, underlyings)
        return False


def process_zst_to_last_hour_minute_polars_stream(
    zst_path: Path,
    target_date: date,
    underlyings: list[str],
    window_start: time,
    window_end: time,
    max_dte_days: float,
    max_otm_pct: float,
    chunk_mb: int = 64,
) -> pd.DataFrame:
    start_us, end_us = _window_bounds_us(target_date, window_start, window_end)
    LOG.info(
        "Processing %s for %s %s-%s UTC via polars-stream",
        zst_path,
        target_date,
        window_start.strftime("%H:%M"),
        window_end.strftime("%H:%M"),
    )

    pl = hourly_builder._import_polars()
    schema_overrides = _polars_schema_overrides_for_minute(pl)
    chunk_mb = max(16, int(chunk_mb))
    allowed_underlyings = set(underlyings)
    quote_parts: list[pd.DataFrame] = []
    underlying_parts: list[pd.DataFrame] = []
    rows_scanned = 0
    rows_kept = 0

    source = None
    zst_fh = None
    zst_stream = None
    zstd_proc = None
    stream_mode = ""
    try:
        source, zst_fh, zst_stream, zstd_proc, stream_mode = hourly_builder._open_streaming_zstd_source(zst_path)
        chunk_iter = hourly_builder._iter_csv_chunk_bytes_from_stream(source, chunk_mb)
        for chunk_blob in chunk_iter:
            chunk = pl.read_csv(
                io.BytesIO(chunk_blob),
                columns=CSV_COLUMNS,
                schema_overrides=schema_overrides,
                infer_schema_length=0,
                null_values=hourly_builder.CSV_NULL_TOKENS,
                low_memory=True,
                batch_size=50_000,
            )
            rows_scanned += int(chunk.height)
            if chunk.height == 0:
                continue

            chunk = (
                chunk.filter((pl.col("timestamp") >= start_us) & (pl.col("timestamp") < end_us))
                .with_columns([
                    pl.col("symbol").str.split("-").list.get(0).str.to_uppercase().alias("underlying"),
                    pl.from_epoch(pl.col("timestamp"), time_unit="us").dt.replace_time_zone("UTC").alias("ts"),
                    pl.from_epoch(pl.col("expiration"), time_unit="us").dt.replace_time_zone("UTC").alias("expiration_dt"),
                ])
                .filter(pl.col("underlying").is_in(sorted(allowed_underlyings)))
            )
            chunk = _filter_near_otm_polars(chunk, pl, max_dte_days=max_dte_days, max_otm_pct=max_otm_pct)
            if chunk.height == 0:
                continue

            rows_kept += int(chunk.height)
            chunk = chunk.with_columns(pl.col("ts").dt.truncate("1m").alias("minute")).sort("timestamp")

            quote_part = (
                chunk.group_by(["underlying", "symbol", "minute"], maintain_order=True)
                .last()
                .select([
                    "underlying",
                    "symbol",
                    "minute",
                    pl.col("ts").alias("last_tick_time"),
                    "type",
                    "strike_price",
                    pl.col("expiration_dt").alias("expiration"),
                    "last_price",
                    "bid_price",
                    "ask_price",
                    "mark_price",
                    "bid_iv",
                    "ask_iv",
                    "mark_iv",
                    "underlying_price",
                    "delta",
                    pl.col("mark_price").first().alias("option_open"),
                    pl.col("mark_price").max().alias("option_high"),
                    pl.col("mark_price").min().alias("option_low"),
                    pl.col("mark_price").last().alias("option_close"),
                ])
            )
            if quote_part.height > 0:
                quote_parts.append(quote_part.to_pandas())

            underlying_part = (
                chunk.drop_nulls("underlying_price")
                .group_by(["underlying", "minute"], maintain_order=True)
                .agg([
                    pl.col("underlying_price").first().alias("underlying_open"),
                    pl.col("underlying_price").max().alias("underlying_high"),
                    pl.col("underlying_price").min().alias("underlying_low"),
                    pl.col("underlying_price").last().alias("underlying_close"),
                ])
            )
            if underlying_part.height > 0:
                underlying_parts.append(underlying_part.to_pandas())
    finally:
        if zst_stream is not None:
            try:
                zst_stream.close()
            except Exception:
                pass
        if zst_fh is not None:
            try:
                zst_fh.close()
            except Exception:
                pass
        if zstd_proc is not None:
            if zstd_proc.stdout is not None:
                try:
                    zstd_proc.stdout.close()
                except Exception:
                    pass
            hourly_builder._finalize_streaming_zstd_proc(zstd_proc, timeout=30.0)

    if not quote_parts:
        LOG.warning("No rows kept for %s", target_date)
        return pd.DataFrame()

    quotes = pd.concat(quote_parts, ignore_index=True)
    underlying_ohlc = pd.concat(underlying_parts, ignore_index=True) if underlying_parts else pd.DataFrame()
    out = _finalize_minute_parts(quotes, underlying_ohlc, target_date)
    LOG.info(
        "  stream_mode=%s scanned=%s kept=%s minute_rows=%s symbols=%s",
        stream_mode,
        f"{rows_scanned:,}",
        f"{rows_kept:,}",
        f"{len(out):,}",
        out["symbol"].nunique(),
    )
    return out


def process_zst_to_last_hour_minute(
    zst_path: Path,
    target_date: date,
    underlyings: list[str],
    window_start: time,
    window_end: time,
    max_dte_days: float,
    max_otm_pct: float,
    chunk_rows: int = 500_000,
) -> pd.DataFrame:
    start_us, end_us = _window_bounds_us(target_date, window_start, window_end)
    LOG.info("Processing %s for %s %s-%s UTC", zst_path, target_date, window_start.strftime("%H:%M"), window_end.strftime("%H:%M"))

    quote_parts: list[pd.DataFrame] = []
    underlying_parts: list[pd.DataFrame] = []
    rows_scanned = 0
    rows_kept = 0

    with zst_path.open("rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for chunk in pd.read_csv(
                text_stream,
                usecols=CSV_COLUMNS,
                dtype=CSV_DTYPES,  # type: ignore[arg-type]
                chunksize=chunk_rows,
                low_memory=False,
            ):
                rows_scanned += len(chunk)
                if chunk.empty:
                    continue

                chunk = chunk[(chunk["timestamp"] >= start_us) & (chunk["timestamp"] < end_us)].copy()
                if chunk.empty:
                    continue

                chunk["underlying"] = chunk["symbol"].str.split("-").str[0].str.upper()
                chunk = chunk[chunk["underlying"].isin(underlyings)].copy()
                if chunk.empty:
                    continue

                chunk = _filter_near_otm_pandas(chunk, max_dte_days=max_dte_days, max_otm_pct=max_otm_pct)
                if chunk.empty:
                    continue

                rows_kept += len(chunk)
                chunk["ts"] = pd.to_datetime(chunk["timestamp"], unit="us", utc=True)
                chunk["minute"] = chunk["ts"].dt.floor("min")
                chunk["expiration"] = pd.to_datetime(chunk["expiration"], unit="us", utc=True, errors="coerce")
                chunk = chunk.sort_values("timestamp")

                underlying_part = _build_underlying_ohlc(
                    chunk[["underlying", "minute", "timestamp", "underlying_price"]].dropna(subset=["underlying_price"])
                )
                if not underlying_part.empty:
                    underlying_parts.append(underlying_part)

                option_part = _build_option_ohlc(chunk)

                quote_part = (
                    chunk.groupby(["underlying", "symbol", "minute"], as_index=False, sort=False)
                    .last()
                )
                if not quote_part.empty:
                    if not option_part.empty:
                        quote_part = quote_part.merge(option_part, on=["underlying", "symbol", "minute"], how="left")
                    quote_part = quote_part.rename(columns={"ts": "last_tick_time"})
                    quote_parts.append(
                        quote_part[
                            [
                                "underlying",
                                "symbol",
                                "minute",
                                "last_tick_time",
                                "type",
                                "strike_price",
                                "expiration",
                                "last_price",
                                "bid_price",
                                "ask_price",
                                "mark_price",
                                "bid_iv",
                                "ask_iv",
                                "mark_iv",
                                "underlying_price",
                                "delta",
                                "option_open",
                                "option_high",
                                "option_low",
                                "option_close",
                            ]
                        ]
                    )

    if not quote_parts:
        LOG.warning("No rows kept for %s", target_date)
        return pd.DataFrame()

    quotes = pd.concat(quote_parts, ignore_index=True)
    underlying_ohlc = pd.concat(underlying_parts, ignore_index=True) if underlying_parts else pd.DataFrame()
    out = _finalize_minute_parts(quotes, underlying_ohlc, target_date)
    LOG.info("  scanned=%s kept=%s minute_rows=%s symbols=%s", f"{rows_scanned:,}", f"{rows_kept:,}", f"{len(out):,}", out["symbol"].nunique())
    return out


def _write_daily_parquet(df: pd.DataFrame, output_dir: Path, target_date: date, overwrite: bool) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for underlying, sub in df.groupby("underlying"):
        ul_dir = output_dir / str(underlying).upper()
        ul_dir.mkdir(parents=True, exist_ok=True)
        path = ul_dir / f"{target_date.isoformat()}.parquet"
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        if path.exists() and not overwrite:
            existing = pd.read_parquet(path)
            merged = pd.concat([existing, sub], ignore_index=True)
            merged = merged.sort_values(["symbol", "minute", "last_tick_time"]).drop_duplicates(
                subset=["symbol", "minute"], keep="last"
            )
            merged.to_parquet(tmp_path, index=False)
        else:
            sub.to_parquet(tmp_path, index=False)
        tmp_path.replace(path)
        written.append(path)
    return written


def _checkpoint_file(output_dir: Path) -> Path:
    return output_dir / ".checkpoint"


def _availability_scan_cache_file(output_dir: Path) -> Path:
    return output_dir / ".availability_scan_cache.json"


def load_checkpoint(output_dir: Path) -> set[str]:
    path = _checkpoint_file(output_dir)
    if path.exists():
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return set()
        return {line for line in raw.split("\n") if line.strip()}
    return set()


def save_checkpoint(output_dir: Path, processed: set[str]) -> None:
    path = _checkpoint_file(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{int(time_mod.time() * 1000)}")
    try:
        tmp.write_text("\n".join(sorted(processed)), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def _availability_summary_to_records(summary: pd.DataFrame) -> list[dict[str, object]]:
    if summary.empty:
        return []

    records: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        records.append({
            "date": str(row["date"]),
            "underlying": str(row["underlying"]),
            "unique_minutes": int(row["unique_minutes"]),
            "rows": int(row["rows"]),
            "unique_symbols": int(row["unique_symbols"]),
            "missing_ohlc_rows": int(row["missing_ohlc_rows"]),
        })
    return records


def _load_availability_scan_cache(output_dir: Path) -> dict[str, object]:
    path = _availability_scan_cache_file(output_dir)
    if not path.exists():
        return {"version": AVAILABILITY_SCAN_CACHE_VERSION, "files": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOG.warning("Could not read availability scan cache %s: %s", path, exc)
        return {"version": AVAILABILITY_SCAN_CACHE_VERSION, "files": {}}
    if not isinstance(payload, dict) or int(payload.get("version", -1)) != AVAILABILITY_SCAN_CACHE_VERSION:
        return {"version": AVAILABILITY_SCAN_CACHE_VERSION, "files": {}}
    files = payload.get("files")
    if not isinstance(files, dict):
        return {"version": AVAILABILITY_SCAN_CACHE_VERSION, "files": {}}
    return {"version": AVAILABILITY_SCAN_CACHE_VERSION, "files": files}


def _save_availability_scan_cache(output_dir: Path, payload: dict[str, object]) -> None:
    path = _availability_scan_cache_file(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{int(time_mod.time() * 1000)}")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _summarize_target_date_minute_availability(
    df: pd.DataFrame,
    target_date: date,
    window_start: time,
    window_end: time,
) -> pd.DataFrame:
    columns = [
        "date",
        "underlying",
        "unique_minutes",
        "rows",
        "unique_symbols",
        "missing_ohlc_rows",
        "availability_pct",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    work = df.copy()
    work["minute"] = pd.to_datetime(work["minute"], utc=True, errors="coerce")
    work = work[work["minute"].dt.strftime("%Y-%m-%d") == target_date.isoformat()].copy()
    if work.empty:
        return pd.DataFrame(columns=columns)

    expected_minutes = _expected_window_minutes(target_date, window_start, window_end)
    work["missing_ohlc_flag"] = ~work[["underlying_open", "underlying_high", "underlying_low", "underlying_close"]].notna().all(axis=1)
    summary = (
        work.groupby("underlying", as_index=False, sort=False)
        .agg(
            unique_minutes=("minute", "nunique"),
            rows=("minute", "size"),
            unique_symbols=("symbol", "nunique"),
            missing_ohlc_rows=("missing_ohlc_flag", "sum"),
        )
    )
    summary.insert(0, "date", target_date.isoformat())
    summary["availability_pct"] = summary["unique_minutes"].astype(float) / float(expected_minutes) * 100.0
    return pd.DataFrame(summary[columns])


def _scan_existing_daily_availability(
    output_dir: Path,
    processed_dates: set[str],
    underlyings: list[str],
    window_start: time,
    window_end: time,
) -> pd.DataFrame:
    columns = ["date", "underlying", "unique_minutes", "rows", "unique_symbols", "missing_ohlc_rows", "availability_pct"]
    if not processed_dates:
        return pd.DataFrame(columns=columns)

    cache_payload = _load_availability_scan_cache(output_dir)
    raw_cache_files = cache_payload.get("files", {})
    cache_files: dict[str, object] = raw_cache_files if isinstance(raw_cache_files, dict) else {}
    scanned_keys: set[str] = set()
    cache_hits = 0
    cache_misses = 0
    rows: list[pd.DataFrame] = []

    for underlying in underlyings:
        for d_str in sorted(processed_dates):
            pq_path = output_dir / underlying / f"{d_str}.parquet"
            cache_key = f"{underlying}/{d_str}"
            scanned_keys.add(cache_key)
            if not pq_path.exists():
                cache_files.pop(cache_key, None)
                continue
            try:
                stat = pq_path.stat()
            except Exception as exc:
                cache_files.pop(cache_key, None)
                LOG.warning("Availability scan skipped %s: %s", pq_path, exc)
                continue

            cached_entry = cache_files.get(cache_key)
            if (
                isinstance(cached_entry, dict)
                and int(cached_entry.get("size", -1)) == int(stat.st_size)
                and int(cached_entry.get("mtime_ns", -1)) == int(stat.st_mtime_ns)
            ):
                summary = pd.DataFrame(cached_entry.get("rows") or [])
                if not summary.empty:
                    summary["availability_pct"] = summary["unique_minutes"].astype(float) / float(_expected_window_minutes(date.fromisoformat(d_str), window_start, window_end)) * 100.0
                cache_hits += 1
            else:
                try:
                    df = pd.read_parquet(pq_path)
                except Exception as exc:
                    LOG.warning("Availability scan skipped %s: %s", pq_path, exc)
                    cache_files.pop(cache_key, None)
                    continue
                summary = _summarize_target_date_minute_availability(df, date.fromisoformat(d_str), window_start, window_end)
                cache_files[cache_key] = {
                    "path": pq_path.as_posix(),
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "rows": _availability_summary_to_records(summary),
                }
                cache_misses += 1

            if summary.empty:
                continue
            rows.append(summary)

    report = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)
    missing_rows: list[dict[str, object]] = []
    for underlying in underlyings:
        seen_dates = set(report.loc[report["underlying"] == underlying, "date"]) if not report.empty else set()
        for d_str in sorted(processed_dates):
            if d_str not in seen_dates:
                missing_rows.append({
                    "date": d_str,
                    "underlying": underlying,
                    "unique_minutes": 0,
                    "rows": 0,
                    "unique_symbols": 0,
                    "missing_ohlc_rows": 0,
                    "availability_pct": 0.0,
                })
    if missing_rows:
        report = pd.concat([report, pd.DataFrame(missing_rows)], ignore_index=True)

    stale_keys = [key for key in list(cache_files.keys()) if key in scanned_keys and not (output_dir / key).with_suffix(".parquet").exists()]
    for key in stale_keys:
        cache_files.pop(key, None)
    _save_availability_scan_cache(output_dir, {"version": AVAILABILITY_SCAN_CACHE_VERSION, "files": cache_files})
    LOG.info("Startup availability scan cache: %d hit(s), %d miss(es), %d target file(s)", cache_hits, cache_misses, len(scanned_keys))
    return pd.DataFrame(report[columns]).reset_index(drop=True) if not report.empty else pd.DataFrame(columns=columns)


def process_single_date(
    d: date,
    *,
    underlyings: list[str],
    window_start: time,
    window_end: time,
    output_dir: Path,
    csv_read_mode: str,
    chunk_rows: int,
    chunk_mb: int,
    force: bool,
    cache_only: bool,
    keep_cache: bool,
    overwrite: bool,
    availability_threshold_pct: float,
    max_dte_days: float,
    max_otm_pct: float,
    prefetched_zst: Path | None,
) -> str:
    max_attempts = 2 if not cache_only else 1
    zst: Path | None = prefetched_zst

    for attempt in range(1, max_attempts + 1):
        try:
            if cache_only:
                zst = hourly_builder.cache_path(d)
                if not zst.exists():
                    LOG.warning("Cache missing for %s; a fresh download is required", d)
                    return "redownload"
            elif zst is None or attempt > 1:
                zst = hourly_builder.download_file(d, force=(force or attempt > 1), quiet=True)

            if csv_read_mode == "polars-stream":
                df = process_zst_to_last_hour_minute_polars_stream(
                    zst_path=zst,
                    target_date=d,
                    underlyings=underlyings,
                    window_start=window_start,
                    window_end=window_end,
                    max_dte_days=max_dte_days,
                    max_otm_pct=max_otm_pct,
                    chunk_mb=chunk_mb,
                )
            else:
                df = process_zst_to_last_hour_minute(
                    zst_path=zst,
                    target_date=d,
                    underlyings=underlyings,
                    window_start=window_start,
                    window_end=window_end,
                    max_dte_days=max_dte_days,
                    max_otm_pct=max_otm_pct,
                    chunk_rows=chunk_rows,
                )

            _validate_minute_output_frame(
                df,
                target_date=d,
                underlyings=underlyings,
                window_start=window_start,
                window_end=window_end,
                max_dte_days=max_dte_days,
                max_otm_pct=max_otm_pct,
                source_label="new output",
            )

            availability = _summarize_target_date_minute_availability(df, d, window_start, window_end)
            if availability.empty:
                raise RuntimeError(f"No minute availability rows found for {d}")
            bad_availability = availability[
                (availability["availability_pct"] < float(availability_threshold_pct))
                | (availability["missing_ohlc_rows"] > 0)
            ].copy()
            if not bad_availability.empty:
                details = ", ".join(
                    f"{row['underlying']}={float(row['availability_pct']):.2f}% ({int(row['unique_minutes'])}/{_expected_window_minutes(d, window_start, window_end)}m, missing_ohlc={int(row['missing_ohlc_rows'])})"
                    for _, row in bad_availability.iterrows()
                )
                raise RuntimeError(
                    f"Minute availability below {availability_threshold_pct:.0f}% for {d}: {details}"
                )

            details = ", ".join(
                f"{row['underlying']}={float(row['availability_pct']):.2f}% ({int(row['unique_minutes'])}/{_expected_window_minutes(d, window_start, window_end)}m, symbols={int(row['unique_symbols'])})"
                for _, row in availability.iterrows()
            )
            LOG.info("  Availability %s: %s", d, details)

            written = _write_daily_parquet(df, output_dir, d, overwrite=(overwrite or force or attempt > 1))
            for path in written:
                LOG.info("Wrote %s", path)

            if (not keep_cache) and (not cache_only) and zst.exists() and hourly_builder.CACHE_DIR in zst.parents:
                try:
                    zst.unlink()
                except OSError:
                    pass
            return "ok"
        except (zstd.ZstdError, RuntimeError, subprocess.CalledProcessError) as exc:
            _delete_output_for_date(output_dir, d, underlyings)
            if zst and zst.exists() and hourly_builder.CACHE_DIR in zst.parents:
                try:
                    zst.unlink()
                    LOG.warning("Deleted corrupted cache file: %s", zst)
                except OSError:
                    pass
            if cache_only:
                LOG.warning("Processing detected bad cache for %s (%s); rerun without --cache-only to re-download", d, exc)
                return "redownload"
            if attempt < max_attempts:
                LOG.warning("Attempt %d/%d failed for %s (%s), retrying with fresh download...", attempt, max_attempts, d, exc)
                zst = None
                continue
            LOG.exception("Failed to process %s: %s", d, exc)
            return "failed"
        except Exception as exc:
            _delete_output_for_date(output_dir, d, underlyings)
            if zst and zst.exists() and hourly_builder.CACHE_DIR in zst.parents:
                try:
                    zst.unlink()
                    LOG.warning("Deleted corrupted cache file: %s", zst)
                except OSError:
                    pass
            LOG.exception("Failed to process %s: %s", d, exc)
            return "failed"

    return "failed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build minute option data from Deribit daily zst files.")
    parser.add_argument("--dates", type=str, help="Comma-separated YYYY-MM-DD dates. Overrides --start/--end.")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD.")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD.")
    parser.add_argument("--underlyings", type=str, default="BTC", help="Comma-separated underlyings, default BTC")
    parser.add_argument("--window-start", type=str, default="00:00", help="UTC window start, default 00:00")
    parser.add_argument("--window-end", type=str, default="00:00", help="UTC window end (exclusive), default next-day 00:00")
    parser.add_argument("--max-dte-days", type=float, default=DEFAULT_MAX_DTE_DAYS, help="Keep contracts with at most this many days to expiry")
    parser.add_argument("--max-otm-pct", type=float, default=DEFAULT_MAX_OTM_PCT, help="Keep contracts within this OTM percent")
    parser.add_argument("--chunk-rows", type=int, default=500_000, help="CSV chunk size for streaming parse")
    parser.add_argument("--chunk-mb", type=int, default=64, help="Polars-stream chunk size in MB")
    parser.add_argument("--csv-read-mode", type=str, default="polars-stream", choices=["polars-stream", "pandas-stream"], help="Minute builder CSV processing mode")
    parser.add_argument("--download-workers", type=int, default=DEFAULT_DOWNLOAD_WORKERS, help="Concurrent download workers, default 8")
    parser.add_argument("--process-workers", type=int, default=DEFAULT_PROCESS_WORKERS, help="Concurrent processing subprocess workers")
    parser.add_argument("--availability-threshold-pct", type=float, default=MIN_MINUTE_AVAILABILITY_PCT, help="Required minute availability percent per underlying")
    parser.add_argument("--cache-only", action="store_true", help="Only use local zst cache, do not download")
    parser.add_argument("--keep-cache", action="store_true", help="Keep downloaded zst cache files")
    parser.add_argument("--force", action="store_true", help="Force re-download / overwrite")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing parquet outputs for the same date")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output root dir")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--_worker-date", dest="_worker_date", type=str, help=argparse.SUPPRESS)
    return parser.parse_args()


def _resolve_dates(args: argparse.Namespace) -> list[date]:
    if args.dates:
        return [_parse_date(x.strip()) for x in args.dates.split(",") if x.strip()]
    if not args.start or not args.end:
        raise SystemExit("Either --dates or both --start/--end are required")
    start = _parse_date(args.start)
    end = _parse_date(args.end)
    if end < start:
        raise SystemExit("--end must be >= --start")
    return list(_daterange(start, end))


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    underlyings = ["BTC"]
    window_start = _parse_hhmm(args.window_start)
    window_end = _parse_hhmm(args.window_end)
    output_dir = Path(args.output_dir)
    hourly_builder._download_segments = 1

    if args._worker_date:
        outcome = process_single_date(
            date.fromisoformat(args._worker_date),
            underlyings=underlyings,
            window_start=window_start,
            window_end=window_end,
            output_dir=output_dir,
            csv_read_mode=args.csv_read_mode,
            chunk_rows=args.chunk_rows,
            chunk_mb=args.chunk_mb,
            force=args.force,
            cache_only=args.cache_only,
            keep_cache=args.keep_cache,
            overwrite=args.overwrite,
            availability_threshold_pct=args.availability_threshold_pct,
            max_dte_days=args.max_dte_days,
            max_otm_pct=args.max_otm_pct,
            prefetched_zst=None,
        )
        if outcome == "ok":
            sys.exit(0)
        if outcome == "redownload":
            sys.exit(2)
        sys.exit(1)

    dates = _resolve_dates(args)

    if args.csv_read_mode == "polars-stream":
        LOG.info("Minute builder mode: polars-stream (aligned with build_hourly_parquet zst streaming path)")
    else:
        LOG.info("Minute builder mode: pandas-stream")
    LOG.info(
        "Scope: underlyings=%s, download_segments=%d, max_dte_days=%s, max_otm_pct=%s",
        ",".join(underlyings),
        hourly_builder._download_segments,
        args.max_dte_days,
        args.max_otm_pct,
    )

    processed = load_checkpoint(output_dir) if not args.force else set()
    requested_dates = {d.isoformat() for d in dates}
    processed_in_range = {d_str for d_str in processed if d_str in requested_dates}

    if processed_in_range:
        LOG.info(
            "Startup availability scan: checking %d processed date(s) against %.0f%% threshold...",
            len(processed_in_range),
            args.availability_threshold_pct,
        )
        availability_report = _scan_existing_daily_availability(
            output_dir,
            processed_in_range,
            underlyings,
            window_start,
            window_end,
        )
        bad_rows = availability_report[
            (availability_report["availability_pct"] < float(args.availability_threshold_pct))
            | (availability_report["missing_ohlc_rows"] > 0)
        ].copy()
        if not bad_rows.empty:
            bad_dates = sorted(set(str(x) for x in bad_rows["date"]))
            for d_str in bad_dates:
                _delete_output_for_date(output_dir, date.fromisoformat(d_str), underlyings)
                processed.discard(d_str)
            save_checkpoint(output_dir, processed)
            sample_details = ", ".join(
                f"{row['date']}:{row['underlying']}={float(row['availability_pct']):.2f}% missing_ohlc={int(row['missing_ohlc_rows'])}"
                for _, row in bad_rows.head(10).iterrows()
            )
            LOG.warning(
                "Startup availability scan found %d low-availability sample(s) across %d date(s); they will be rebuilt. Samples: %s",
                len(bad_rows),
                len(bad_dates),
                sample_details or "-",
            )
        else:
            LOG.info("Startup availability scan passed: no date below %.0f%%", args.availability_threshold_pct)

    remaining: list[date] = []
    already_processed = 0
    for d in dates:
        key = d.isoformat()
        if args.force or args.overwrite:
            remaining.append(d)
            continue
        if _existing_outputs_complete(
            output_dir,
            target_date=d,
            underlyings=underlyings,
            window_start=window_start,
            window_end=window_end,
            max_dte_days=args.max_dte_days,
            max_otm_pct=args.max_otm_pct,
        ):
            processed.add(key)
            already_processed += 1
            LOG.info("Skip %s: minute parquet already complete", d)
            continue
        processed.discard(key)
        remaining.append(d)

    save_checkpoint(output_dir, processed)
    LOG.info("Total dates: %d, already processed: %d, remaining: %d", len(dates), already_processed, len(remaining))
    if not remaining:
        LOG.info("Done. ok=0 failed=0 redownload=0")
        return

    n_dl_workers = max(1, int(args.download_workers))
    n_proc_workers = max(1, int(args.process_workers))
    max_ahead = max(2, n_dl_workers * 2)

    ready_q: queue.Queue = queue.Queue(maxsize=max_ahead)
    download_q: queue.Queue = queue.Queue()
    pipeline_cancel = threading.Event()
    download_stop = threading.Event()
    state_lock = threading.Lock()
    outstanding_dates: set[str] = {d.isoformat() for d in remaining}
    queued_for_download: set[str] = set()
    queued_for_process: set[str] = set()
    redownload_attempts: dict[str, int] = {}
    max_parent_redownloads = 1

    def _enqueue_download_request(d: date, force_download: bool = False, reason: str = "") -> bool:
        key = d.isoformat()
        with state_lock:
            if key not in outstanding_dates:
                return False
            if key in queued_for_download or key in queued_for_process:
                return False
            queued_for_download.add(key)
        if reason:
            LOG.info("Queueing download for %s (%s)", d, reason)
        download_q.put((d, force_download))
        return True

    def _mark_failed(d: date) -> None:
        key = d.isoformat()
        with state_lock:
            queued_for_download.discard(key)
            queued_for_process.discard(key)
            outstanding_dates.discard(key)

    def _download_worker() -> None:
        while not (pipeline_cancel.is_set() or download_stop.is_set()):
            try:
                d, force_download = download_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                key = d.isoformat()
                try:
                    if args.cache_only:
                        zst = hourly_builder.cache_path(d)
                        if not zst.exists():
                            LOG.warning("Cache missing for %s", d)
                            _mark_failed(d)
                            continue
                    else:
                        zst = hourly_builder.download_file(d, force=(args.force or force_download), quiet=True)
                        LOG.info("Downloaded %s -> %s", d, zst)
                except Exception as exc:
                    LOG.error("Download failed for %s: %s", d, exc)
                    _mark_failed(d)
                    continue

                with state_lock:
                    queued_for_download.discard(key)
                    if key in outstanding_dates:
                        queued_for_process.add(key)
                while not (pipeline_cancel.is_set() or download_stop.is_set()):
                    try:
                        ready_q.put(d, timeout=0.5)
                        break
                    except queue.Full:
                        continue
            finally:
                download_q.task_done()

    for d in remaining:
        _enqueue_download_request(d, force_download=args.force, reason="initial")

    def _build_worker_cmd(d: date) -> list[str]:
        cmd = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--_worker-date",
            d.isoformat(),
            "--underlyings",
            "BTC",
            "--window-start",
            args.window_start,
            "--window-end",
            args.window_end,
            "--max-dte-days",
            str(args.max_dte_days),
            "--max-otm-pct",
            str(args.max_otm_pct),
            "--chunk-rows",
            str(args.chunk_rows),
            "--chunk-mb",
            str(args.chunk_mb),
            "--csv-read-mode",
            args.csv_read_mode,
            "--availability-threshold-pct",
            str(args.availability_threshold_pct),
            "--output-dir",
            str(output_dir),
            "--cache-only",
        ]
        if args.keep_cache:
            cmd.append("--keep-cache")
        if args.force:
            cmd.append("--force")
        if args.overwrite:
            cmd.append("--overwrite")
        if args.verbose:
            cmd.append("-v")
        return cmd

    LOG.info("Pipeline: %d download thread(s), queue depth %d, up to %d process worker(s), %d date(s)", n_dl_workers, max_ahead, n_proc_workers, len(remaining))

    dl_threads: list[threading.Thread] = []
    for _ in range(n_dl_workers):
        t = threading.Thread(target=_download_worker, daemon=True)
        t.start()
        dl_threads.append(t)

    active_workers: dict[str, tuple[date, subprocess.Popen]] = {}
    ok = 0
    failed = 0
    redownload = 0

    try:
        while True:
            while len(active_workers) < n_proc_workers:
                try:
                    d = ready_q.get(timeout=0.2)
                except queue.Empty:
                    break
                proc = subprocess.Popen(_build_worker_cmd(d), cwd=str(Path(__file__).resolve().parents[2]))
                active_workers[d.isoformat()] = (d, proc)

            finished_keys: list[str] = []
            for key, (d, proc) in list(active_workers.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                finished_keys.append(key)
                with state_lock:
                    queued_for_process.discard(key)
                if rc == 0:
                    ok += 1
                    processed.add(key)
                    save_checkpoint(output_dir, processed)
                    with state_lock:
                        outstanding_dates.discard(key)
                elif rc == 2:
                    if args.cache_only:
                        redownload += 1
                        _mark_failed(d)
                    else:
                        attempts = redownload_attempts.get(key, 0)
                        if attempts < max_parent_redownloads:
                            redownload_attempts[key] = attempts + 1
                            _enqueue_download_request(d, force_download=True, reason=f"worker requested redownload attempt {attempts + 1}")
                        else:
                            redownload += 1
                            _mark_failed(d)
                else:
                    failed += 1
                    _mark_failed(d)

            for key in finished_keys:
                active_workers.pop(key, None)

            with state_lock:
                done = not outstanding_dates and not active_workers
            if done:
                break
            time_mod.sleep(0.2)
    finally:
        pipeline_cancel.set()
        download_stop.set()
        for _, proc in active_workers.values():
            if proc.poll() is None:
                proc.terminate()
        for t in dl_threads:
            t.join(timeout=1.0)

    LOG.info("Done. ok=%d failed=%d redownload=%d", ok, failed, redownload)


if __name__ == "__main__":
    main()
