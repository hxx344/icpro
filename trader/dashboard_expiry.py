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


@dataclass(frozen=True)
class ExpirySummary:
    expiry: datetime
    label: str
    contract_count: int


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


def _distinct_expiries(tickers: list[OptionTicker]) -> list[datetime]:
    seen: dict[datetime, None] = {}
    for ticker in tickers:
        seen.setdefault(ticker.expiry, None)
    return sorted(seen)


def summarize_available_expiries(tickers: list[OptionTicker]) -> list[ExpirySummary]:
    counts: dict[datetime, int] = {}
    for ticker in tickers:
        counts[ticker.expiry] = counts.get(ticker.expiry, 0) + 1

    return [
        ExpirySummary(
            expiry=expiry,
            label=f"{expiry.strftime('%Y-%m-%d %H:%M')} UTC ({expiry.strftime('%A')})",
            contract_count=counts[expiry],
        )
        for expiry in sorted(counts)
    ]


def _nearest_available_expiry(
    tickers: list[OptionTicker],
    *,
    target_expiry: datetime,
    weekday: int,
) -> datetime | None:
    candidates = [expiry for expiry in _distinct_expiries(tickers) if expiry.weekday() == weekday]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda expiry: (
            abs((expiry - target_expiry).total_seconds()),
            0 if expiry <= target_expiry else 1,
            expiry,
        ),
    )


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

    friday_expiry = _nearest_available_expiry(
        tickers,
        target_expiry=sunday_expiry,
        weekday=4,
    ) or nearest_weekday_expiry(now_utc, weekday=4)
    friday_target = ExpiryTickerTarget(
        expiry=friday_expiry,
        label=f"最近可用周五到期 ({friday_expiry.strftime('%Y-%m-%d %H:%M')} UTC)",
        tickers=filter_tickers_for_expiry(tickers, friday_expiry, tolerance_hours=tolerance_hours),
        is_fallback=True,
    )

    selected_target = sunday_target if sunday_target.tickers else friday_target
    return selected_target, sunday_target, friday_target