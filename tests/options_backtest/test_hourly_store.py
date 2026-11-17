"""Quote lookup semantics shared by all cached-hourly backtests."""
import numpy as np
import pandas as pd
import pytest

from options_backtest.data.hourly_store import HourlyOptionStore, _build_hour_index


NOW = pd.Timestamp("2025-01-03 21:00Z")


def make_store(rows):
    frame = pd.DataFrame(rows, columns=[
        "hour", "hourly_pick", "instrument_name", "bid_price", "ask_price", "mark_price"])
    frame["hour"] = pd.to_datetime(frame.hour, utc=True)
    return HourlyOptionStore("BTC", frame, _build_hour_index(frame), {}, {})


@pytest.mark.parametrize("bid,ask,mark,expected", [
    (1., 3., 2.5, (1., 3., 2.5)),
    (1., 3., np.nan, (1., 3., 2.)),
    (1., np.nan, 0., (1., None, 1.)),
    (0., 3., -1., (None, 3., 3.)),
    (np.inf, -1., np.nan, (None, None, None)),
    (np.nan, np.nan, 2., (None, None, 2.)),
])
@pytest.mark.parametrize("indexed", [False, True])
def test_quote_fallbacks_identical_for_snapshot_and_symbol_index(bid, ask, mark, expected, indexed):
    store = make_store([(NOW, "open", "A", bid, ask, mark)])
    if indexed:
        store.quote_index = {"open": {"A": (np.array([NOW.value]), np.array([bid]),
                                               np.array([ask]), np.array([mark]))}}
    assert store.get_quote("A", NOW, "open") == expected
    assert store.get_quote("A", NOW, "open") == expected


def test_hour_and_pick_switch_cannot_reuse_stale_or_future_quote():
    later = NOW + pd.Timedelta(hours=1)
    store = make_store([
        (NOW, "close", "A", 2., 4., 3.),
        (NOW, "open", "A", 1., 3., 2.),
        (later, "open", "B", 4., 6., 5.),
    ])
    assert store.get_quote("A", NOW, "OPEN") == (1., 3., 2.)
    assert store.get_quote("A", NOW, "close") == (2., 4., 3.)
    assert store.get_quote("A", later, "open") == (None, None, None)
    assert store.get_quote("B", NOW, "open") == (None, None, None)
    assert store.get_quote("B", later, "open") == (4., 6., 5.)
    assert store.get_quote("B", later + pd.Timedelta(hours=1), "open") == (None, None, None)
    assert store.get_quote("A", NOW, "open") == (1., 3., 2.)


def test_duplicate_symbol_uses_last_row_and_nonzero_dataframe_index():
    store = make_store([(NOW, "open", "A", 1., 2., 1.5),
                        (NOW, "open", "A", 3., 4., 3.5)])
    store.frame.index = [10, 20]
    assert store.get_quote("A", NOW, "open") == (3., 4., 3.5)


def test_hour_index_boundaries_include_both_picks_and_last_group():
    later = NOW + pd.Timedelta(hours=1)
    store = make_store([(NOW, "close", "A", 1, 2, 1.5),
                        (NOW, "close", "B", 1, 2, 1.5),
                        (NOW, "open", "A", 1, 2, 1.5),
                        (later, "open", "A", 1, 2, 1.5)])
    assert store.hour_index == {
        (NOW.value, "close"): (0, 2), (NOW.value, "open"): (2, 3),
        (later.value, "open"): (3, 4),
    }
    assert make_store([]).hour_index == {}
    one = make_store([(NOW, "open", "A", 1, 2, 1.5)])
    assert one.hour_index == {(NOW.value, "open"): (0, 1)}


def test_symbol_index_retains_past_only_search_behavior():
    store = make_store([])
    store.quote_index = {"open": {"A": (np.array([NOW.value]), np.array([1.]),
                                           np.array([3.]), np.array([2.]))}}
    assert store.get_quote("A", NOW - pd.Timedelta(seconds=1), "open") == (None, None, None)
    assert store.get_quote("A", NOW + pd.Timedelta(hours=1), "open") == (1., 3., 2.)

def test_microsecond_expiry_is_normalized_and_matched_by_weekend_filter():
    from options_backtest.data.hourly_store import _normalize_hourly_frame
    from options_backtest.strategy.weekend_vol import WeekendVolStrategy

    friday = pd.Timestamp("2026-10-02 18:00Z")
    sunday = pd.Timestamp("2026-10-04 08:00Z")
    raw = pd.DataFrame({
        "symbol": ["A", "B"], "type": ["call", "put"], "strike_price": [1., 1.],
        "expiration": [sunday.value // 1000, (sunday + pd.Timedelta(days=1)).value // 1000],
        "hourly_pick": ["close", "close"], "hour": [friday, friday],
        **{c: [1., 1.] for c in ("open_interest", "last_price", "bid_price", "ask_price",
                                 "mark_price", "mark_iv", "bid_iv", "ask_iv",
                                 "underlying_price", "delta", "gamma", "vega", "theta")},
    })
    frame = _normalize_hourly_frame(raw)
    assert str(frame["expiration_date"].dtype) == "datetime64[ns, UTC]"
    assert str(frame["hour"].dtype) == "datetime64[ns, UTC]"

    chain = pd.DataFrame({"expiration_date": frame["expiration_date"].dt.as_unit("us"),
                          "instrument_name": frame["instrument_name"]})
    strategy = WeekendVolStrategy({"close_day": "sunday", "close_time_utc": "08:00",
                                   "expiry_tolerance_hours": 2.0})
    assert list(strategy._filter_target_expiry(chain, friday)["instrument_name"]) == ["A"]
