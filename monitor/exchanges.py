"""Exchange API clients for options data.

Fetches real-time option quotes from Deribit, OKX, and Binance,
normalises them into a unified OptionQuote format for comparison.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests
from loguru import logger

# ---------------------------------------------------------------------------
# Unified quote model
# ---------------------------------------------------------------------------

MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


@dataclass
class OptionQuote:
    """Normalised option quote across exchanges."""

    exchange: str           # "Deribit" / "OKX" / "Binance"
    underlying: str         # "BTC" / "ETH"
    instrument: str         # exchange-native instrument name
    option_type: str        # "call" / "put"
    strike: float
    expiry: datetime
    dte_hours: float        # hours to expiry
    bid_usd: float          # best bid in USD
    ask_usd: float          # best ask in USD
    mark_usd: float         # mark price in USD
    underlying_price: float # spot/index price in USD
    iv: float = 0.0         # implied volatility (0-1 scale)
    volume_24h: float = 0.0
    open_interest: float = 0.0

    # -- derived properties --

    @property
    def dte(self) -> float:
        """Days to expiry."""
        return max(self.dte_hours / 24.0, 0.001)

    @property
    def mid_usd(self) -> float:
        if self.bid_usd > 0 and self.ask_usd > 0:
            return (self.bid_usd + self.ask_usd) / 2
        return self.mark_usd

    @property
    def spread_usd(self) -> float:
        if self.bid_usd > 0 and self.ask_usd > 0:
            return self.ask_usd - self.bid_usd
        return 0.0

    @property
    def spread_pct(self) -> float:
        mid = self.mid_usd
        return (self.spread_usd / mid * 100) if mid > 0 else 0.0

    @property
    def strategy(self) -> str:
        return "CC" if self.option_type == "call" else "SP"

    @property
    def intrinsic_value(self) -> float:
        """Intrinsic value of the option."""
        if self.option_type == "call":
            return max(0.0, self.underlying_price - self.strike)
        else:
            return max(0.0, self.strike - self.underlying_price)

    @property
    def extrinsic_bid(self) -> float:
        """Extrinsic (time) value based on bid price."""
        return max(0.0, self.bid_usd - self.intrinsic_value)

    @property
    def extrinsic_mark(self) -> float:
        """Extrinsic (time) value based on mark price."""
        return max(0.0, self.mark_usd - self.intrinsic_value)

    @property
    def apr(self) -> float:
        """Annualised premium rate (%) based on extrinsic (time) value only.

        CC: extrinsic / underlying_price × 365/DTE
        SP: extrinsic / strike × 365/DTE
        """
        if self.dte <= 0:
            return 0.0
        premium = self.extrinsic_bid  # only time value counts
        if premium <= 0:
            return 0.0
        base = self.underlying_price if self.option_type == "call" else self.strike
        if base <= 0:
            return 0.0
        return (premium / base) * (365 / self.dte) * 100

    @property
    def net_apr(self) -> float:
        """APR after deducting estimated fee (~0.03%), based on extrinsic value."""
        if self.dte <= 0 or self.underlying_price <= 0:
            return 0.0
        premium = self.extrinsic_bid
        fee_usd = self.underlying_price * 0.0003  # approximate taker fee
        net_premium = premium - fee_usd
        if net_premium <= 0:
            return 0.0
        base = self.underlying_price if self.option_type == "call" else self.strike
        if base <= 0:
            return 0.0
        return (net_premium / base) * (365 / self.dte) * 100

    @property
    def moneyness_pct(self) -> float:
        """(Strike / Spot - 1) × 100.  0 = ATM, >0 = OTM call / ITM put."""
        if self.underlying_price <= 0:
            return 0
        return (self.strike / self.underlying_price - 1) * 100

    @property
    def daily_yield_per_unit(self) -> float:
        """Estimated daily yield in USD per 1 unit of underlying (extrinsic only)."""
        if self.dte <= 0:
            return 0.0
        return self.extrinsic_bid / self.dte

    def yield_estimate(self, capital_usd: float) -> dict:
        """Estimate yield for given capital (extrinsic value only)."""
        if self.underlying_price <= 0 or self.dte <= 0:
            return {"daily": 0, "weekly": 0, "monthly": 0, "yearly": 0}
        contracts = capital_usd / self.underlying_price
        total_premium = self.extrinsic_bid * contracts
        daily = total_premium / self.dte
        return {
            "daily": daily,
            "weekly": daily * 7,
            "monthly": daily * 30,
            "yearly": daily * 365,
        }


# ---------------------------------------------------------------------------
# Deribit
# ---------------------------------------------------------------------------

DERIBIT_BASE = "https://www.deribit.com/api/v2"
DERIBIT_RE = re.compile(
    r"^(?P<ul>[A-Z]+)-(?P<day>\d{1,2})(?P<mon>[A-Z]{3})(?P<yr>\d{2})"
    r"-(?P<strike>\d+)-(?P<cp>[CP])$"
)


def _parse_deribit_name(name: str) -> dict | None:
    m = DERIBIT_RE.match(name)
    if not m:
        return None
    day = int(m.group("day"))
    mon = MONTH_MAP.get(m.group("mon"))
    yr = 2000 + int(m.group("yr"))
    if mon is None:
        return None
    expiry = datetime(yr, mon, day, 8, 0, 0, tzinfo=timezone.utc)
    return {
        "underlying": m.group("ul"),
        "expiry": expiry,
        "strike": float(m.group("strike")),
        "option_type": "call" if m.group("cp") == "C" else "put",
    }


def fetch_deribit(currency: str = "BTC", timeout: int = 10) -> list[OptionQuote]:
    """Fetch all option book summaries from Deribit."""
    url = f"{DERIBIT_BASE}/public/get_book_summary_by_currency"
    params = {"currency": currency.upper(), "kind": "option"}
    quotes: list[OptionQuote] = []
    now = datetime.now(timezone.utc)

    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"Deribit {currency} fetch failed: {e}")
        return quotes

    results = data.get("result", [])
    for item in results:
        name = item.get("instrument_name", "")
        parsed = _parse_deribit_name(name)
        if not parsed:
            continue

        underlying_price = item.get("underlying_price", 0) or 0
        if underlying_price <= 0:
            continue

        dte_hours = (parsed["expiry"] - now).total_seconds() / 3600
        if dte_hours < 0:
            continue

        # Deribit prices are in BTC/ETH (coin-margined) → convert to USD
        bid_coin = item.get("bid_price") or 0
        ask_coin = item.get("ask_price") or 0
        mark_coin = item.get("mark_price") or 0

        quotes.append(OptionQuote(
            exchange="Deribit",
            underlying=parsed["underlying"],
            instrument=name,
            option_type=parsed["option_type"],
            strike=parsed["strike"],
            expiry=parsed["expiry"],
            dte_hours=dte_hours,
            bid_usd=bid_coin * underlying_price,
            ask_usd=ask_coin * underlying_price,
            mark_usd=mark_coin * underlying_price,
            underlying_price=underlying_price,
            iv=item.get("mark_iv", 0) or 0,
            volume_24h=item.get("volume", 0) or 0,
            open_interest=item.get("open_interest", 0) or 0,
        ))

    logger.info(f"Deribit {currency}: {len(quotes)} option quotes")
    return quotes


# ---------------------------------------------------------------------------
# OKX
# ---------------------------------------------------------------------------

OKX_BASE = "https://www.okx.com"
OKX_RE = re.compile(
    r"^(?P<ul>[A-Z]+)-USD-(?P<date>\d{6})-(?P<strike>\d+)-(?P<cp>[CP])$"
)


def _parse_okx_name(inst_id: str) -> dict | None:
    m = OKX_RE.match(inst_id)
    if not m:
        return None
    date_str = m.group("date")  # YYMMDD
    yr = 2000 + int(date_str[:2])
    mon = int(date_str[2:4])
    day = int(date_str[4:6])
    expiry = datetime(yr, mon, day, 8, 0, 0, tzinfo=timezone.utc)
    return {
        "underlying": m.group("ul"),
        "expiry": expiry,
        "strike": float(m.group("strike")),
        "option_type": "call" if m.group("cp") == "C" else "put",
    }


def _fetch_okx_spot(underlying: str = "BTC", timeout: int = 10) -> float:
    """Get OKX spot index price."""
    try:
        url = f"{OKX_BASE}/api/v5/market/index-tickers"
        params = {"instId": f"{underlying}-USD"}
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if data:
            return float(data[0].get("idxPx", 0))
    except Exception as e:
        logger.warning(f"OKX spot {underlying} fetch failed: {e}")
    return 0.0


def fetch_okx(underlying: str = "BTC", timeout: int = 10) -> list[OptionQuote]:
    """Fetch OKX option tickers."""
    quotes: list[OptionQuote] = []
    now = datetime.now(timezone.utc)
    inst_family = f"{underlying.upper()}-USD"

    # 1. Get spot price
    spot = _fetch_okx_spot(underlying, timeout)

    # 2. Get option tickers
    url = f"{OKX_BASE}/api/v5/market/tickers"
    params = {"instType": "OPTION", "instFamily": inst_family}
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"OKX {underlying} options fetch failed: {e}")
        return quotes

    tickers = data.get("data", [])

    # 3. Get instrument details for expiry info (batch)
    inst_url = f"{OKX_BASE}/api/v5/public/instruments"
    inst_params = {"instType": "OPTION", "instFamily": inst_family}
    inst_map: dict[str, dict] = {}
    try:
        inst_resp = requests.get(inst_url, params=inst_params, timeout=timeout)
        inst_resp.raise_for_status()
        for inst in inst_resp.json().get("data", []):
            inst_map[inst["instId"]] = inst
    except Exception:
        pass  # fall back to name parsing

    # 4. Get IV / Greeks from opt-summary
    iv_map: dict[str, dict] = {}
    try:
        summary_url = f"{OKX_BASE}/api/v5/public/opt-summary"
        summary_params = {"instFamily": inst_family}
        summary_resp = requests.get(summary_url, params=summary_params, timeout=timeout)
        summary_resp.raise_for_status()
        for item in summary_resp.json().get("data", []):
            iv_map[item["instId"]] = item
    except Exception:
        pass  # IV will remain 0

    for ticker in tickers:
        inst_id = ticker.get("instId", "")
        parsed = _parse_okx_name(inst_id)
        if not parsed:
            continue

        dte_hours = (parsed["expiry"] - now).total_seconds() / 3600
        if dte_hours < 0:
            continue

        # OKX option prices are in BTC/ETH (coin-margined) → multiply by spot
        bid_coin = float(ticker.get("bidPx") or 0)
        ask_coin = float(ticker.get("askPx") or 0)
        last_coin = float(ticker.get("last") or 0)

        ul_price = spot
        if ul_price <= 0:
            continue

        # Get IV from opt-summary (markVol is annualised, 0-1 scale)
        summary = iv_map.get(inst_id, {})
        mark_iv = float(summary.get("markVol") or 0)

        quotes.append(OptionQuote(
            exchange="OKX",
            underlying=parsed["underlying"],
            instrument=inst_id,
            option_type=parsed["option_type"],
            strike=parsed["strike"],
            expiry=parsed["expiry"],
            dte_hours=dte_hours,
            bid_usd=bid_coin * ul_price,
            ask_usd=ask_coin * ul_price,
            mark_usd=(last_coin if last_coin > 0 else
                      (bid_coin + ask_coin) / 2 if bid_coin > 0 and ask_coin > 0
                      else 0) * ul_price,
            underlying_price=ul_price,
            iv=mark_iv,
            volume_24h=float(ticker.get("vol24h") or 0),
            open_interest=float(ticker.get("oi") or 0),
        ))

    logger.info(f"OKX {underlying}: {len(quotes)} option quotes")
    return quotes


# ---------------------------------------------------------------------------
# Binance
# ---------------------------------------------------------------------------

BINANCE_EAPI = "https://eapi.binance.com"
BINANCE_RE = re.compile(
    r"^(?P<ul>[A-Z]+)-(?P<date>\d{6})-(?P<strike>\d+)-(?P<cp>[CP])$"
)


def _parse_binance_name(symbol: str) -> dict | None:
    m = BINANCE_RE.match(symbol)
    if not m:
        return None
    date_str = m.group("date")  # YYMMDD
    yr = 2000 + int(date_str[:2])
    mon = int(date_str[2:4])
    day = int(date_str[4:6])
    expiry = datetime(yr, mon, day, 8, 0, 0, tzinfo=timezone.utc)
    return {
        "underlying": m.group("ul"),
        "expiry": expiry,
        "strike": float(m.group("strike")),
        "option_type": "call" if m.group("cp") == "C" else "put",
    }


def fetch_binance(underlying: str = "BTC", timeout: int = 10) -> list[OptionQuote]:
    """Fetch Binance European options tickers."""
    quotes: list[OptionQuote] = []
    now = datetime.now(timezone.utc)
    ul = underlying.upper()

    # Get spot price from Binance spot API
    spot_price = 0.0
    try:
        resp = requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbol": f"{ul}USDT"},
            timeout=timeout,
        )
        resp.raise_for_status()
        spot_price = float(resp.json().get("price", 0))
    except Exception as e:
        logger.warning(f"Binance spot {ul} fetch failed: {e}")

    # Get option tickers
    url = f"{BINANCE_EAPI}/eapi/v1/ticker"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"Binance options fetch failed: {e}")
        return quotes

    for item in data:
        symbol = item.get("symbol", "")
        if not symbol.startswith(ul):
            continue

        parsed = _parse_binance_name(symbol)
        if not parsed:
            continue

        dte_hours = (parsed["expiry"] - now).total_seconds() / 3600
        if dte_hours < 0:
            continue

        bid = float(item.get("bidPrice") or 0)
        ask = float(item.get("askPrice") or 0)
        mark = float(item.get("markPrice") or 0)
        # Binance provides exercisePrice = underlying index price
        ul_price = float(item.get("exercisePrice") or 0)
        if ul_price <= 0:
            ul_price = spot_price

        quotes.append(OptionQuote(
            exchange="Binance",
            underlying=parsed["underlying"],
            instrument=symbol,
            option_type=parsed["option_type"],
            strike=parsed["strike"],
            expiry=parsed["expiry"],
            dte_hours=dte_hours,
            bid_usd=bid,
            ask_usd=ask,
            mark_usd=mark,
            underlying_price=ul_price,
            iv=0,
            volume_24h=float(item.get("volume") or 0),
            open_interest=float(item.get("openInterest") or 0),
        ))

    logger.info(f"Binance {ul}: {len(quotes)} option quotes")
    return quotes


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def fetch_all_quotes(
    underlyings: list[str] | None = None,
    exchanges: list[str] | None = None,
    timeout: int = 10,
) -> list[OptionQuote]:
    """Fetch and merge quotes from all selected exchanges.

    Parameters
    ----------
    underlyings : list of "BTC", "ETH" etc. Default: ["BTC", "ETH"]
    exchanges : list of "Deribit", "OKX", "Binance". Default: all.
    timeout : HTTP request timeout in seconds.
    """
    if underlyings is None:
        underlyings = ["BTC", "ETH"]
    if exchanges is None:
        exchanges = ["Deribit", "OKX", "Binance"]

    all_quotes: list[OptionQuote] = []

    fetchers = {
        "Deribit": fetch_deribit,
        "OKX": fetch_okx,
        "Binance": fetch_binance,
    }

    for exch_name in exchanges:
        fetcher = fetchers.get(exch_name)
        if fetcher is None:
            continue
        for ul in underlyings:
            try:
                quotes = fetcher(ul, timeout=timeout)
                all_quotes.extend(quotes)
            except Exception as e:
                logger.error(f"Failed to fetch {exch_name} {ul}: {e}")

    return all_quotes
