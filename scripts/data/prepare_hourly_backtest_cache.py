"""Prepare and verify reusable monthly and range caches for hourly backtests.

Run from the repository root. Defaults to all locally available ETH history.
Uses the same cache paths, normalization and indexes as the backtest loader.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from options_backtest.data import hourly_store as hourly
from options_backtest.utils import to_utc_timestamp


def assert_same_frame(actual, expected):
    # Older pandas caches serialize UTC dates as ns, newer ones may use us.
    # Compare exact instants in ns, while retaining exact checks for other types.
    actual, expected = actual.copy(deep=False), expected.copy(deep=False)
    for column in ("hour", "expiration_date"):
        actual[column] = actual[column].astype("datetime64[ns, UTC]")
        expected[column] = expected[column].astype("datetime64[ns, UTC]")
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def inspect_source(path):
    metadata = pq.read_metadata(path)
    column = metadata.schema.names.index("hour")
    bounds = [metadata.row_group(i).column(column).statistics
              for i in range(metadata.num_row_groups)]
    if bounds and all(s is not None and s.has_min_max for s in bounds):
        start = min(s.min for s in bounds)
        end = max(s.max for s in bounds)
    else:
        hours = pd.read_parquet(path, columns=["hour"]).hour
        start, end = hours.min(), hours.max()
    if pd.isna(start) or pd.isna(end):
        raise ValueError(f"Empty hourly source: {path}")
    stat = path.stat()
    return {"source": str(path), "rows": metadata.num_rows,
            "start": str(to_utc_timestamp(start)), "end": str(to_utc_timestamp(end)),
            "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def verify_month(data_dir, underlying, source, start, end):
    path = Path(source["source"])
    cache_path = hourly._monthly_cache_path(data_dir, underlying, path)
    existed = cache_path.exists()
    began = perf_counter()
    cached = hourly._load_or_build_month_cache(data_dir, underlying, path, start, end)
    cache_seconds = perf_counter() - began
    began = perf_counter()
    expected = hourly._normalize_hourly_frame(pd.read_parquet(path, columns=hourly._RAW_COLUMNS))
    expected = expected.loc[expected.hour.between(start, end)].reset_index(drop=True)
    normalize_seconds = perf_counter() - began
    assert_same_frame(cached, expected)
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (source["bytes"], source["mtime_ns"]):
        raise RuntimeError(f"Source changed while preparing cache: {path}")
    return {**source, "cache": str(cache_path), "reused": existed,
            "selected_rows": len(cached), "cache_bytes": cache_path.stat().st_size,
            "cache_prepare_seconds": cache_seconds,
            "raw_read_normalize_seconds": normalize_seconds, "exact_match": True}


def verify_range(store, months, start, end):
    offset, expected_prices = 0, []
    quotes_checked = 0
    for month in months:
        frame = pd.read_parquet(month["cache"], filters=[
            ("hour", ">=", start.to_pydatetime()), ("hour", "<=", end.to_pydatetime())])
        frame = frame.reset_index(drop=True)
        actual = store.frame.iloc[offset:offset + len(frame)].reset_index(drop=True)
        # The loader fills index-price gaps across month boundaries after concat.
        assert_same_frame(actual.drop(columns="underlying_price"),
                          frame.drop(columns="underlying_price"))
        expected_prices.append(frame.underlying_price)
        for pick in ("open", "close"):
            times = frame.loc[frame.hourly_pick == pick, "hour"].drop_duplicates()
            for timestamp in times.iloc[sorted(set([0, len(times) // 2, len(times) - 1]))] if len(times) else []:
                expected = frame.loc[(frame.hour == timestamp) & (frame.hourly_pick == pick)]
                snapshot = store.get_snapshot(timestamp, pick)
                assert_same_frame(snapshot.drop(columns="underlying_price").reset_index(drop=True),
                    expected.drop(columns="underlying_price").reset_index(drop=True))
                # Last-row-wins semantics for duplicate symbols are intentional.
                for name in expected.instrument_name.iloc[[0, -1]].unique():
                    row = expected.loc[expected.instrument_name == name].iloc[-1]
                    bid, ask, mark = [float(row[c]) if np.isfinite(row[c]) and row[c] > 0 else None
                                      for c in ("bid_price", "ask_price", "mark_price")]
                    if mark is None:
                        mark = (bid + ask) / 2 if bid is not None and ask is not None else bid or ask
                    assert store.get_quote(name, timestamp, pick) == (bid, ask, mark)
                    quotes_checked += 1
        offset += len(frame)
    assert offset == len(store.frame), "Range row count mismatch"
    pd.testing.assert_series_equal(store.frame.underlying_price.reset_index(drop=True),
        pd.concat(expected_prices, ignore_index=True).ffill().bfill(), check_exact=True)
    assert store.hour_index == hourly._build_hour_index(store.frame), "Hour index mismatch"
    for pick in ("open", "close"):
        expected = hourly._series_to_ns(store.frame.loc[store.frame.hourly_pick == pick, "hour"])
        np.testing.assert_array_equal(store.available_timestamps(pick), np.unique(expected))
    return {"all_rows_and_columns_exact": True, "hour_index_exact": True,
            "sampled_quote_checks": quotes_checked}


def prepare(data_dir, underlying, start_date=None, end_date=None):
    started = perf_counter()
    data_dir, underlying = Path(data_dir), underlying.upper()
    paths = sorted((data_dir / "options_hourly" / underlying).glob("????-??.parquet"))
    if not paths:
        raise FileNotFoundError(f"No monthly hourly data for {underlying}")
    sources = [inspect_source(p) for p in paths]
    coverage_start = min(to_utc_timestamp(s["start"]) for s in sources)
    coverage_end = max(to_utc_timestamp(s["end"]) for s in sources)
    start = to_utc_timestamp(start_date) if start_date else coverage_start
    end = to_utc_timestamp(end_date) if end_date else coverage_end
    if start > end or start < coverage_start or end > coverage_end:
        raise ValueError(f"Requested range must be inside {coverage_start} .. {coverage_end}")
    # Match the loader's month selection, including gaps within selected months.
    names = {f"{m:%Y-%m}.parquet" for m in hourly._month_starts(start, end)}
    sources = [s for s in sources if Path(s["source"]).name in names]
    months = []
    for i, source in enumerate(sources, 1):
        result = verify_month(data_dir, underlying, source, start, end)
        months.append(result)
        print(f"[{i}/{len(sources)}] {Path(source['source']).name}: "
              f"{result['selected_rows']:,} rows verified; reused={result['reused']}", flush=True)
    monthly_seconds = perf_counter() - started
    hourly._HOURLY_STORE_CACHE.clear()
    began = perf_counter()
    store = hourly.load_hourly_option_store(data_dir, underlying, start, end)
    range_prepare_seconds = perf_counter() - began
    cache_key = hourly._store_cache_key([Path(s["source"]) for s in sources], start, end)
    frame_path, meta_path = hourly._disk_cache_paths(data_dir, underlying, cache_key)
    if not frame_path.exists() or not meta_path.exists():
        raise RuntimeError("Range cache was not saved successfully")
    print(f"Range cache ready: {len(store.frame):,} rows; {range_prepare_seconds:.2f}s", flush=True)
    del store
    hourly._HOURLY_STORE_CACHE.clear()
    gc.collect()
    began = perf_counter()
    # Direct disk load proves this measurement cannot fall back to source data.
    store = hourly._load_disk_cached_store(data_dir, underlying, cache_key)
    disk_load_seconds = perf_counter() - began
    if store is None:
        raise RuntimeError("Prepared range cache could not be loaded")
    hourly._HOURLY_STORE_CACHE[cache_key] = store
    began = perf_counter()
    assert hourly.load_hourly_option_store(data_dir, underlying, start, end) is store
    memory_load_seconds = perf_counter() - began
    print(f"Disk reload: {disk_load_seconds:.2f}s; verifying full range", flush=True)
    verification = verify_range(store, months, start, end)
    for source in sources:
        stat = Path(source["source"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (source["bytes"], source["mtime_ns"]):
            raise RuntimeError(f"Source changed during verification: {source['source']}")
    stats = {
        "underlying": underlying, "start": str(start), "end": str(end),
        "source_months": len(sources), "rows": len(store.frame),
        "source_bytes": sum(s["bytes"] for s in sources),
        "monthly_cache_bytes": sum(m["cache_bytes"] for m in months),
        "range_cache_bytes": frame_path.stat().st_size + meta_path.stat().st_size,
        "range_frame": str(frame_path), "range_metadata": str(meta_path),
        "monthly_prepare_and_verify_seconds": monthly_seconds,
        "range_prepare_seconds": range_prepare_seconds, "disk_load_seconds": disk_load_seconds,
        "memory_load_seconds": memory_load_seconds, "total_seconds": perf_counter() - started,
        "available_hours": {p: len(store.available_timestamps(p)) for p in ("open", "close")},
        "missing_hours": {p: len(pd.date_range(start.ceil("h"), end.floor("h"), freq="h").difference(
            pd.to_datetime(store.available_timestamps(p), unit="ns", utc=True))) for p in ("open", "close")},
        "verification": verification, "months": months,
    }
    hourly._HOURLY_STORE_CACHE.clear()
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--underlying", default="ETH")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--report-dir")
    args = parser.parse_args()
    stats = prepare(args.data_dir, args.underlying, args.start, args.end)
    output = Path(args.report_dir or f"reports/cache_preparation/{args.underlying.lower()}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "preparation.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "months"}, indent=2), flush=True)
    print(f"Report: {output / 'preparation.json'}", flush=True)


if __name__ == "__main__":
    main()
