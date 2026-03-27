"""Binance European Options API client.

Provides a unified interface for the trader modules:
- Market data (spot/index, option tickers, mark prices)
- Account summary
- Order placement / cancellation

Binance European Options API:
  Production: https://eapi.binance.com
  Testnet:    https://testnet.binancefuture.com
"""

from __future__ import annotations

import hashlib
import hmac
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlencode

import requests
from loguru import logger

from trader.config import ExchangeConfig


# ---------------------------------------------------------------------------
# Shared data models
# ---------------------------------------------------------------------------


@dataclass
class OptionTicker:
    symbol: str
    underlying: str
    strike: float
    option_type: str
    expiry: datetime
    bid_price: float
    ask_price: float
    mark_price: float
    last_price: float
    underlying_price: float
    volume_24h: float
    open_interest: float
    delta: float = 0.0
    mark_iv: float = 0.0

    @property
    def mid_price(self) -> float:
        if self.bid_price > 0 and self.ask_price > 0:
            return (self.bid_price + self.ask_price) / 2.0
        return self.mark_price

    @property
    def spread(self) -> float:
        if self.bid_price > 0 and self.ask_price > 0:
            return self.ask_price - self.bid_price
        return 0.0

    @property
    def dte_hours(self) -> float:
        now = datetime.now(timezone.utc)
        return max((self.expiry - now).total_seconds() / 3600.0, 0.0)

    @property
    def moneyness_pct(self) -> float:
        if self.underlying_price <= 0:
            return 0.0
        return (self.strike / self.underlying_price - 1.0) * 100.0


@dataclass
class OrderResult:
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    avg_price: float
    status: str
    fee: float
    raw: dict


@dataclass
class AccountInfo:
    total_balance: float
    available_balance: float
    unrealized_pnl: float
    raw: dict


# ---------------------------------------------------------------------------
# Symbol parsing - Binance format: ETH-260321-2000-C
# ---------------------------------------------------------------------------

BINANCE_SYMBOL_RE = re.compile(
    r"^(?P<ul>[A-Z]+)-(?P<yymmdd>\d{6})-(?P<strike>\d+(?:\.\d+)?)-(?P<cp>[CP])$"
)


def _parse_symbol(symbol: str):
    """Parse Binance option symbol like ETH-260321-2000-C."""
    m = BINANCE_SYMBOL_RE.match(symbol)
    if not m:
        return None
    yymmdd = m.group("yymmdd")
    yy = int(yymmdd[:2])
    mm = int(yymmdd[2:4])
    dd = int(yymmdd[4:6])
    year = 2000 + yy
    expiry = datetime(year, mm, dd, 8, 0, 0, tzinfo=timezone.utc)
    return {
        "underlying": m.group("ul"),
        "expiry": expiry,
        "strike": float(m.group("strike")),
        "option_type": "call" if m.group("cp") == "C" else "put",
    }


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class BinanceOptionsClient:
    """REST client for Binance European Options API."""

    def __init__(self, config: ExchangeConfig):
        self.cfg = config
        self.session = requests.Session()
        self._base = self.cfg.base_url.rstrip("/")

        if self.cfg.api_key:
            self.session.headers.update({"X-MBX-APIKEY": self.cfg.api_key})

    # ------------------------------------------------------------------
    # Low-level HTTP helpers
    # ------------------------------------------------------------------

    def _sign(self, params: dict) -> dict:
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000
        query = urlencode(params)
        sig = hmac.new(
            self.cfg.api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        params["signature"] = sig
        return params

    def _public_get(self, path: str, params=None):
        url = f"{self._base}{path}"
        try:
            resp = self.session.get(url, params=params or {}, timeout=self.cfg.timeout)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and data.get("code") and int(data["code"]) < 0:
                raise RuntimeError(f"Binance error {data['code']}: {data.get('msg', '')}")
            return data
        except Exception as e:
            logger.error(f"Binance GET {path} failed: {e}")
            raise

    def _private_get(self, path: str, params=None):
        if self.cfg.simulate_private:
            return None
        signed = self._sign(dict(params or {}))
        url = f"{self._base}{path}"
        try:
            resp = self.session.get(url, params=signed, timeout=self.cfg.timeout)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and data.get("code") and int(data["code"]) < 0:
                raise RuntimeError(f"Binance error {data['code']}: {data.get('msg', '')}")
            return data
        except Exception as e:
            logger.error(f"Binance private GET {path} failed: {e}")
            raise

    def _private_post(self, path: str, params=None):
        if self.cfg.simulate_private:
            return None
        signed = self._sign(dict(params or {}))
        url = f"{self._base}{path}"
        try:
            resp = self.session.post(url, params=signed, timeout=self.cfg.timeout)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and data.get("code") and int(data["code"]) < 0:
                raise RuntimeError(f"Binance error {data['code']}: {data.get('msg', '')}")
            return data
        except Exception as e:
            logger.error(f"Binance private POST {path} failed: {e}")
            raise

    def _private_delete(self, path: str, params=None):
        if self.cfg.simulate_private:
            return None
        signed = self._sign(dict(params or {}))
        url = f"{self._base}{path}"
        try:
            resp = self.session.delete(url, params=signed, timeout=self.cfg.timeout)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and data.get("code") and int(data["code"]) < 0:
                raise RuntimeError(f"Binance error {data['code']}: {data.get('msg', '')}")
            return data
        except Exception as e:
            logger.error(f"Binance DELETE {path} failed: {e}")
            raise

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_spot_price(self, underlying: str = "ETH") -> float:
        ul = underlying.upper()
        try:
            result = self._public_get("/eapi/v1/index", {"underlying": f"{ul}USDT"})
            return float(result.get("indexPrice", 0.0))
        except Exception:
            return 0.0

    def get_tickers(self, underlying: str = "ETH") -> list[OptionTicker]:
        ul = underlying.upper()
        try:
            result = self._public_get("/eapi/v1/ticker")
        except Exception as e:
            logger.error(f"Failed to fetch tickers: {e}")
            return []

        tickers: list[OptionTicker] = []
        if not isinstance(result, list):
            return tickers

        for item in result:
            symbol = str(item.get("symbol", ""))
            if not symbol.startswith(f"{ul}-"):
                continue
            parsed = _parse_symbol(symbol)
            if not parsed:
                continue

            spot = float(item.get("underlyingPrice") or item.get("exercisePrice") or 0)
            bid_price = float(item.get("bidPrice") or 0)
            ask_price = float(item.get("askPrice") or 0)
            last_price = float(item.get("lastPrice") or 0)
            mark_price = (bid_price + ask_price) / 2 if bid_price > 0 and ask_price > 0 else last_price

            tickers.append(OptionTicker(
                symbol=symbol,
                underlying=parsed["underlying"],
                strike=parsed["strike"],
                option_type=parsed["option_type"],
                expiry=parsed["expiry"],
                bid_price=bid_price,
                ask_price=ask_price,
                mark_price=mark_price,
                last_price=last_price,
                underlying_price=spot,
                volume_24h=float(item.get("volume") or 0),
                open_interest=float(item.get("amount") or 0),
            ))

        return tickers

    def enrich_greeks(self, tickers: list[OptionTicker], underlying: str = "ETH") -> None:
        """Fetch Greeks from /eapi/v1/mark and populate delta/mark_iv on tickers."""
        greeks = self.get_greeks(underlying)
        if not greeks:
            return
        for ticker in tickers:
            if ticker.symbol in greeks:
                g = greeks[ticker.symbol]
                ticker.delta = g["delta"]
                ticker.mark_iv = g["mark_iv"]

    def get_greeks(self, underlying: str = "ETH") -> dict[str, dict[str, float]]:
        """Fetch delta and mark IV for all options from /eapi/v1/mark."""
        ul = underlying.upper()
        try:
            result = self._public_get("/eapi/v1/mark")
        except Exception:
            return {}

        greeks: dict[str, dict[str, float]] = {}
        if not isinstance(result, list):
            return greeks

        for item in result:
            symbol = str(item.get("symbol", ""))
            if not symbol.startswith(f"{ul}-"):
                continue
            greeks[symbol] = {
                "delta": float(item.get("delta") or 0),
                "mark_iv": float(item.get("markIV") or 0),
            }

        return greeks

    def get_mark_prices(self, underlying: str = "ETH") -> dict[str, float]:
        ul = underlying.upper()
        try:
            result = self._public_get("/eapi/v1/mark")
        except Exception:
            return {}

        prices: dict[str, float] = {}
        if not isinstance(result, list):
            return prices

        for item in result:
            symbol = str(item.get("symbol", ""))
            if not symbol.startswith(f"{ul}-"):
                continue
            prices[symbol] = float(item.get("markPrice") or 0)

        return prices

    def get_account(self) -> AccountInfo:
        if self.cfg.simulate_private or not self.cfg.api_key or not self.cfg.api_secret:
            return AccountInfo(
                total_balance=1.0,
                available_balance=1.0,
                unrealized_pnl=0.0,
                raw={"simulated": True},
            )

        try:
            result = self._private_get("/eapi/v1/marginAccount")
        except Exception as e:
            logger.error(f"Failed to get account: {e}")
            return AccountInfo(
                total_balance=0.0,
                available_balance=0.0,
                unrealized_pnl=0.0,
                raw={"error": str(e)},
            )

        if not isinstance(result, dict):
            logger.warning(f"Unexpected account response type: {type(result)}")
            return AccountInfo(
                total_balance=0.0,
                available_balance=0.0,
                unrealized_pnl=0.0,
                raw={"unexpected": str(result)[:200]},
            )

        assets = result.get("asset", [])
        if not isinstance(assets, list):
            assets = []

        total_balance = 0.0
        available = 0.0
        upnl = 0.0

        for asset in assets:
            if not isinstance(asset, dict):
                continue
            total_balance += float(asset.get("marginBalance") or 0)
            available += float(asset.get("available") or asset.get("availableBalance") or 0)
            upnl += float(asset.get("unrealizedPNL") or asset.get("unrealizedPnl") or 0)

        logger.debug(
            f"Account: balance={total_balance:.4f}, available={available:.4f}, "
            f"upnl={upnl:.4f}, assets_count={len(assets)}"
        )

        return AccountInfo(
            total_balance=total_balance,
            available_balance=available,
            unrealized_pnl=upnl,
            raw=result,
        )

    def get_positions(self, underlying: str = "ETH") -> list[dict]:
        """Get open option positions from exchange (filtered by underlying)."""
        if self.cfg.simulate_private or not self.cfg.api_key or not self.cfg.api_secret:
            return []

        try:
            result = self._private_get("/eapi/v1/position")
        except Exception as e:
            logger.error(f"Failed to query positions: {e}")
            return []

        if not isinstance(result, list):
            return []

        ul = underlying.upper()
        positions: list[dict] = []
        for item in result:
            if not isinstance(item, dict):
                continue

            symbol = str(item.get("symbol", ""))
            if ul and not symbol.startswith(f"{ul}-"):
                continue

            qty = float(
                item.get("quantity")
                or item.get("positionQty")
                or item.get("positionAmount")
                or 0
            )
            if abs(qty) <= 0:
                continue

            positions.append({
                "symbol": symbol,
                "side": str(item.get("side") or ("LONG" if qty > 0 else "SHORT")).upper(),
                "quantity": qty,
                "entryPrice": float(item.get("entryPrice") or 0),
                "unrealizedPnl": float(item.get("unrealizedPNL") or item.get("unrealizedPnl") or 0),
            })

        return positions

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price=None,
        time_in_force: Optional[str] = None,
        reduce_only: bool = False,
        client_order_id=None,
    ) -> OrderResult:
        side_norm = side.upper()
        order_type_norm = order_type.upper()

        if self.cfg.simulate_private:
            import uuid
            return OrderResult(
                order_id=f"SIM-{uuid.uuid4().hex[:8]}",
                symbol=symbol,
                side=side_norm,
                quantity=quantity,
                price=price or 0.0,
                avg_price=price or 0.0,
                status="FILLED",
                fee=0.0,
                raw={
                    "simulated": True,
                    "requestedType": order_type_norm,
                    "timeInForce": time_in_force,
                },
            )

        effective_type = order_type_norm
        effective_price = price
        effective_tif = time_in_force

        if order_type_norm == "MARKET":
            quote = self.get_ticker(symbol)
            if quote is None:
                raise RuntimeError(
                    f"No live quote available for synthetic market order: {symbol}"
                )

            effective_price = quote.ask_price if side_norm == "BUY" else quote.bid_price
            effective_price = float(effective_price)
            if effective_price <= 0:
                raise RuntimeError(
                    f"Invalid synthetic market reference price for {symbol}: "
                    f"side={side_norm}, bid={quote.bid_price}, ask={quote.ask_price}"
                )

            effective_type = "LIMIT"
            effective_tif = effective_tif or "IOC"
            logger.info(
                f"Using synthetic marketable limit for {symbol}: "
                f"side={side_norm}, px={effective_price:.4f}, tif={effective_tif}"
            )

        params = {
            "symbol": symbol,
            "side": side_norm,
            "type": effective_type,
            "quantity": str(quantity),
        }
        if effective_price is not None and effective_type == "LIMIT":
            params["price"] = str(effective_price)
            params["timeInForce"] = effective_tif or "GTC"
        if reduce_only:
            params["reduceOnly"] = "true"
        if client_order_id:
            params["newClientOrderId"] = client_order_id

        try:
            result = self._private_post("/eapi/v1/order", params)
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            raise

        if not isinstance(result, dict):
            result = {"raw": result}

        avg_price = float(result.get("avgPrice") or result.get("price") or 0.0)
        fee = float(result.get("fee") or 0.0)
        status = str(result.get("status", "")).upper()
        if status in ("FILLED", "PARTIALLY_FILLED"):
            pass
        elif status in ("NEW", "PENDING"):
            status = "NEW"
        else:
            status = status or "FILLED"

        return OrderResult(
            order_id=str(result.get("orderId", "")),
            symbol=symbol,
            side=side_norm,
            quantity=float(result.get("executedQty") or quantity),
            price=float(result.get("price") or 0.0),
            avg_price=avg_price,
            status=status,
            fee=fee,
            raw=result,
        )

    def close_position(self, symbol: str, side: str, quantity: float) -> OrderResult:
        close_side = "SELL" if side == "LONG" else "BUY"
        return self.place_order(
            symbol=symbol,
            side=close_side,
            quantity=quantity,
            order_type="MARKET",
            reduce_only=True,
        )

    # ------------------------------------------------------------------
    # Order query / cancel
    # ------------------------------------------------------------------

    def query_order(self, symbol: str, order_id: str) -> OrderResult:
        """Query the status of an existing order."""
        if self.cfg.simulate_private:
            return OrderResult(
                order_id=order_id,
                symbol=symbol,
                side="",
                quantity=0.0,
                price=0.0,
                avg_price=0.0,
                status="FILLED",
                fee=0.0,
                raw={"simulated": True},
            )

        try:
            result = self._private_get(
                "/eapi/v1/order",
                {"symbol": symbol, "orderId": order_id},
            )
        except Exception as e:
            logger.error(f"Query order {order_id} failed: {e}")
            raise

        if not isinstance(result, dict):
            result = {"raw": result}

        return OrderResult(
            order_id=str(result.get("orderId", order_id)),
            symbol=symbol,
            side=str(result.get("side", "")).upper(),
            quantity=float(result.get("executedQty") or 0.0),
            price=float(result.get("price") or 0.0),
            avg_price=float(result.get("avgPrice") or result.get("price") or 0.0),
            status=str(result.get("status", "")).upper(),
            fee=float(result.get("fee") or 0.0),
            raw=result,
        )

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an open order. Returns True on success."""
        if self.cfg.simulate_private:
            return True

        try:
            self._private_delete(
                "/eapi/v1/order",
                {"symbol": symbol, "orderId": order_id},
            )
            return True
        except Exception as e:
            logger.error(f"Cancel order {order_id} failed: {e}")
            return False

    def get_ticker(self, symbol: str) -> OptionTicker | None:
        """Get a single ticker by exact symbol name."""
        parsed = _parse_symbol(symbol)
        if not parsed:
            return None

        try:
            result = self._public_get("/eapi/v1/ticker", {"symbol": symbol})
        except Exception:
            return None

        items = result if isinstance(result, list) else [result]
        for item in items:
            if not isinstance(item, dict):
                continue
            sym = str(item.get("symbol", ""))
            if sym != symbol:
                continue

            spot = float(item.get("underlyingPrice") or item.get("exercisePrice") or 0)
            bid_price = float(item.get("bidPrice") or 0)
            ask_price = float(item.get("askPrice") or 0)
            last_price = float(item.get("lastPrice") or 0)
            mark_price = (bid_price + ask_price) / 2 if bid_price > 0 and ask_price > 0 else last_price

            return OptionTicker(
                symbol=symbol,
                underlying=parsed["underlying"],
                strike=parsed["strike"],
                option_type=parsed["option_type"],
                expiry=parsed["expiry"],
                bid_price=bid_price,
                ask_price=ask_price,
                mark_price=mark_price,
                last_price=last_price,
                underlying_price=spot,
                volume_24h=float(item.get("volume") or 0),
                open_interest=float(item.get("amount") or 0),
                delta=float(item.get("delta") or 0),
                mark_iv=float(item.get("markIV") or 0),
            )

        return None
