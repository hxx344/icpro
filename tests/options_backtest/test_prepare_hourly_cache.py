"""The preparation command must create caches the real loader can reuse."""
import os

import numpy as np
import pandas as pd
import pytest

from options_backtest.data import hourly_store as hourly
from scripts.data.prepare_hourly_backtest_cache import prepare


@pytest.fixture
def sources(tmp_path):
    root = tmp_path / "options_hourly/ETH"
    root.mkdir(parents=True)
    for month, day, price in [("2025-01", "2025-01-31", np.nan),
                              ("2025-02", "2025-02-01", 2000.)]:
        rows = []
        for hour in [0, 1]:
            for pick in ["open", "close"]:
                row = {c: 0. for c in hourly._RAW_COLUMNS}
                row.update(symbol="ETH-7FEB25-2000-C", type="call", strike_price=2000.,
                    expiration=pd.Timestamp("2025-02-07 08:00Z").value // 1000,
                    hour=pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=hour),
                    hourly_pick=pick, bid_price=0. if pick == "open" else .02,
                    ask_price=.04, mark_price=.03, underlying_price=price)
                rows.append(row)
        pd.DataFrame(rows).to_parquet(root / f"{month}.parquet", index=False)
    yield tmp_path
    hourly._HOURLY_STORE_CACHE.clear()


def test_full_preparation_roundtrip_reuses_loader_caches(sources, monkeypatch):
    first = prepare(sources, "eth")
    assert first["rows"] == 8
    assert first["source_months"] == 2
    assert first["available_hours"] == {"open": 4, "close": 4}
    assert first["missing_hours"] == {"open": 22, "close": 22}
    assert first["verification"]["all_rows_and_columns_exact"]
    assert all(m["exact_match"] and not m["reused"] for m in first["months"])

    def unexpected_write(*args, **kwargs):
        raise AssertionError("Valid cache should be reused")

    monkeypatch.setattr(hourly, "_atomic_write_parquet", unexpected_write)
    second = prepare(sources, "ETH")
    assert all(m["reused"] for m in second["months"])
    # Match the public API used by DataLoader, including its source signature key.
    store = hourly.load_hourly_option_store(sources, "ETH", first["start"], first["end"])
    assert len(store.frame) == 8
    assert store.frame.underlying_price.eq(2000.).all()
    assert store.get_quote("ETH-7FEB25-2000-C", "2025-01-31 00:00Z", "open") == (None, .04, .03)


def test_source_change_rebuilds_only_affected_month_and_range(sources):
    first = prepare(sources, "ETH")
    path = sources / "options_hourly/ETH/2025-02.parquet"
    old_mtime = path.stat().st_mtime_ns
    frame = pd.read_parquet(path)
    frame["mark_price"] = .05
    frame.to_parquet(path, index=False)
    os.utime(path, ns=(old_mtime + 1_000_000_000, old_mtime + 1_000_000_000))
    second = prepare(sources, "ETH")
    assert [m["reused"] for m in second["months"]] == [True, False]
    assert second["range_frame"] != first["range_frame"]


def test_partial_range_has_inclusive_hour_boundaries(sources):
    result = prepare(sources, "ETH", "2025-01-31 01:00Z", "2025-02-01 00:00Z")
    assert result["rows"] == 4
    assert result["available_hours"] == {"open": 2, "close": 2}
    with pytest.raises(ValueError, match="inside"):
        prepare(sources, "ETH", "2024-01-01", None)


def test_legacy_timestamp_units_preserve_exact_instants(sources):
    path = sources / "options_hourly/ETH/2025-01.parquet"
    cache = hourly._monthly_cache_path(sources, "ETH", path)
    frame = hourly._normalize_hourly_frame(pd.read_parquet(path))
    frame["expiration_date"] = frame.expiration_date.astype("datetime64[ns, UTC]")
    hourly._atomic_write_parquet(cache, frame)
    result = prepare(sources, "ETH")
    assert result["months"][0]["reused"]
    assert result["verification"]["all_rows_and_columns_exact"]
