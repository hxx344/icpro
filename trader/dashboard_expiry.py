from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from trader.bybit_client import OptionTicker


@dataclass(frozen=True)
class ExpiryTickerTarget:
    expiry: datetime
    label: str
    tickers: list[OptionTicker]
    is_fallback: bool = False


def nearest_weekday_expiry(now_utc: datetime, weekday: int, hour_utc: int = 8) -> datetime:
    """Return the nearest upcoming weekday expiry timestamp at the given UTC hour."""
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    days_ahead = (weekday - now_utc.weekday()) % 7
    if days_ahead == 0 and now_utc.hour >= hour_utc:
        days_ahead = 7
    return now_utc.replace(hour=hour_utc, minute=0, second=0, microsecond=0) + timedelta(days=days_ahead)


def filter_tickers_for_expiry(
    tickers: list[OptionTicker],
    target_expiry: datetime,
    tolerance_hours: float = 2.0,
) -> list[OptionTicker]:
    tolerance_seconds = tolerance_hours * 3600.0
    return [
        ticker
        for ticker in tickers
        if abs((ticker.expiry - target_expiry).total_seconds()) < tolerance_seconds
    ]


def resolve_test_order_expiry_target(
    tickers: list[OptionTicker],
    now_utc: datetime,
    tolerance_hours: float = 2.0,
) -> tuple[ExpiryTickerTarget, ExpiryTickerTarget, ExpiryTickerTarget]:
    sunday_expiry = nearest_weekday_expiry(now_utc, weekday=6)
    sunday_target = ExpiryTickerTarget(
        expiry=sunday_expiry,
        label=f"周日到期 ({sunday_expiry.strftime('%Y-%m-%d %H:%M')} UTC)",
        tickers=filter_tickers_for_expiry(tickers, sunday_expiry, tolerance_hours=tolerance_hours),
        is_fallback=False,
    )

    friday_expiry = nearest_weekday_expiry(now_utc, weekday=4)
    friday_target = ExpiryTickerTarget(
        expiry=friday_expiry,
        label=f"最近周五到期 ({friday_expiry.strftime('%Y-%m-%d %H:%M')} UTC)",
        tickers=filter_tickers_for_expiry(tickers, friday_expiry, tolerance_hours=tolerance_hours),
        is_fallback=True,
    )

    selected_target = sunday_target if sunday_target.tickers else friday_target
    return selected_target, sunday_target, friday_target