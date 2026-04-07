from __future__ import annotations

from datetime import datetime, timezone

from trader.bybit_client import OptionTicker
from trader.dashboard_expiry import nearest_weekday_expiry, resolve_test_order_expiry_target


def _make_ticker(symbol: str, expiry: datetime) -> OptionTicker:
    return OptionTicker(
        symbol=symbol,
        underlying="BTC",
        strike=100000.0,
        option_type="call" if symbol.endswith("-C") else "put",
        expiry=expiry,
        bid_price=100.0,
        ask_price=101.0,
        mark_price=100.5,
        last_price=100.5,
        underlying_price=100000.0,
        volume_24h=1.0,
        open_interest=1.0,
        delta=0.25,
        mark_iv=0.6,
    )


def test_resolve_test_order_expiry_target_prefers_sunday_when_available() -> None:
    now_utc = datetime(2026, 1, 8, 10, 0, tzinfo=timezone.utc)  # Thursday
    sunday_expiry = nearest_weekday_expiry(now_utc, weekday=6)
    friday_expiry = nearest_weekday_expiry(now_utc, weekday=4)
    tickers = [
        _make_ticker("BTC-09JAN26-100000-C", friday_expiry),
        _make_ticker("BTC-11JAN26-100000-C", sunday_expiry),
    ]

    selected, sunday_target, friday_target = resolve_test_order_expiry_target(tickers, now_utc)

    assert selected.expiry == sunday_expiry
    assert not selected.is_fallback
    assert len(selected.tickers) == 1
    assert sunday_target.tickers[0].symbol == "BTC-11JAN26-100000-C"
    assert friday_target.tickers[0].symbol == "BTC-09JAN26-100000-C"


def test_resolve_test_order_expiry_target_falls_back_to_friday_when_sunday_missing() -> None:
    now_utc = datetime(2026, 1, 8, 10, 0, tzinfo=timezone.utc)  # Thursday
    friday_expiry = nearest_weekday_expiry(now_utc, weekday=4)
    tickers = [
        _make_ticker("BTC-09JAN26-100000-P", friday_expiry),
    ]

    selected, sunday_target, friday_target = resolve_test_order_expiry_target(tickers, now_utc)

    assert sunday_target.tickers == []
    assert selected.is_fallback
    assert selected.expiry == friday_expiry
    assert selected.tickers == friday_target.tickers
    assert selected.tickers[0].symbol == "BTC-09JAN26-100000-P"