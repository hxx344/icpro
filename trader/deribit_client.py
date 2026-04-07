"""Deribit Options API client – Deribit 期权 API 客户端.

Provides a unified interface for the trader modules:
- Market data (spot/index, option tickers, mark prices)
- Account summary
- Order placement / cancellation
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests
from loguru import logger

from trader.config import ExchangeConfig


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
    mark_iv: float = 0.0                 # mark implied vol (decimal, e.g. 0.80)

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


MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
    "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

DERIBIT_SYMBOL_RE = re.compile(
    r"^(?P<ul>[A-Z]+)-(?P<day>\d{1,2})(?P<mon>[A-Z]{3})(?P<yy>\d{2})-(?P<strike>\d+(?:\.\d+)*)-(?P<cp>[CP])$"
)


def _parse_symbol(symbol: str) -> dict | None:
    """Parse Deribit option symbol like ETH-28MAR26-2200-C."""
    m = DERIBIT_SYMBOL_RE.match(symbol)
    if not m:
        return None
    day = int(m.group("day"))
    mon = MONTH_MAP.get(m.group("mon"), 1)
    yy = int(m.group("yy"))
    year = 2000 + yy
    expiry = datetime(year, mon, day, 8, 0, 0, tzinfo=timezone.utc)
    return {
        "underlying": m.group("ul"),
        "expiry": expiry,
        "strike": float(m.group("strike")),
        "option_type": "call" if m.group("cp") == "C" else "put",
    }


class DeribitOptionsClient:
    """REST client for Deribit options API v2."""

    def __init__(self, config: ExchangeConfig):
        self.cfg = config
        self.session = requests.Session()
        self._base = self.cfg.base_url.rstrip("/")
        self._access_token: str = ""
        self._token_expire_at: float = 0.0

    def _rpc(self, method: str, params: dict | None = None) -> Any:
        url = f"{self._base}/api/v2/{method}"
        try:
            resp = self.session.get(url, params=params or {}, timeout=self.cfg.timeout)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data and data["error"]:
                raise RuntimeError(str(data["error"]))
            return data.get("result")
        except Exception as e:
            logger.error(f"Deribit RPC {method} failed: {e}")
            raise

    def _ensure_token(self) -> None:
        if self.cfg.simulate_private:
            return
        now = time.time()
        if self._access_token and now < self._token_expire_at - 30:
            return

        if not self.cfg.client_id or not self.cfg.client_secret:
            raise RuntimeError("Deribit client_id/client_secret not configured")

        result = self._rpc("public/auth", {
            "grant_type": "client_credentials",
            "client_id": self.cfg.client_id,
            "client_secret": self.cfg.client_secret,
        })
        self._access_token = str(result.get("access_token", ""))
        expires_in = int(result.get("expires_in", 0))
        self._token_expire_at = now + max(expires_in, 60)

    def _private_rpc(self, method: str, params: dict | None = None) -> Any:
        self._ensure_token()
        merged = dict(params or {})
        if self._access_token:
            merged["access_token"] = self._access_token
        return self._rpc(method, merged)

    def get_spot_price(self, underlying: str = "ETH") -> float:
        index_name = f"{underlying.lower()}_usd"
        try:
            result = self._rpc("public/get_index_price", {"index_name": index_name})
            return float(result.get("index_price", 0.0))
        except Exception:
            return 0.0

    def get_tickers(self, underlying: str = "ETH") -> list[OptionTicker]:
        currency = underlying.upper()
        result = self._rpc("public/get_book_summary_by_currency", {
            "currency": currency,
            "kind": "option",
        })
        tickers: list[OptionTicker] = []

        for item in result if isinstance(result, list) else []:
            symbol = str(item.get("instrument_name", ""))
            parsed = _parse_symbol(symbol)
            if not parsed:
                continue

            tickers.append(OptionTicker(
                symbol=symbol,
                underlying=parsed["underlying"],
                strike=parsed["strike"],
                option_type=parsed["option_type"],
                expiry=parsed["expiry"],
                bid_price=float(item.get("bid_price") or 0.0),
                ask_price=float(item.get("ask_price") or 0.0),
                mark_price=float(item.get("mark_price") or 0.0),
                last_price=float(item.get("last") or 0.0),
                underlying_price=float(item.get("underlying_price") or 0.0),
                volume_24h=float(item.get("volume") or 0.0),
                open_interest=float(item.get("open_interest") or 0.0),
                mark_iv=float(item.get("mark_iv") or 0.0) / 100.0,  # API returns %, convert to decimal
            ))

        return tickers

    def get_mark_prices(self, underlying: str = "ETH") -> dict[str, float]:
        return {t.symbol: t.mark_price for t in self.get_tickers(underlying)}

    def get_account(self) -> AccountInfo:
        if self.cfg.simulate_private or not self.cfg.client_id or not self.cfg.client_secret:
            return AccountInfo(
                total_balance=1.0,
                available_balance=1.0,
                unrealized_pnl=0.0,
                raw={"simulated": True},
            )

        currency = self.cfg.account_currency.upper()
        result = self._private_rpc("private/get_account_summary", {
            "currency": currency,
            "extended": True,
        })
        return AccountInfo(
            total_balance=float(result.get("equity", 0.0)),
            available_balance=float(result.get("available_funds", 0.0)),
            unrealized_pnl=float(result.get("total_pl", 0.0)),
            raw=result,
        )

    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        *,
        reduce_only: bool = False,
        client_order_id: str | None = None,
    ) -> OrderResult:
        side_norm = side.upper()
        if self.cfg.simulate_private:
            import uuid
            return OrderResult(
                order_id=f"SIM-{uuid.uuid4().hex[:8]}",
                symbol=symbol,
                side=side_norm,
                quantity=quantity,
                price=0.0,
                avg_price=0.0,
                status="FILLED",
                fee=0.0,
                raw={"simulated": True},
            )

        method = "private/buy" if side_norm == "BUY" else "private/sell"
        params: dict[str, Any] = {
            "instrument_name": symbol,
            "amount": quantity,
            "type": "market",
            "reduce_only": bool(reduce_only),
        }
        if client_order_id:
            params["label"] = client_order_id

        result = self._private_rpc(method, params)
        order = result.get("order", {}) if isinstance(result, dict) else {}
        avg_price = float(order.get("average_price") or order.get("price") or 0.0)
        fee = float(order.get("commission") or 0.0)
        state = str(order.get("order_state", "")).upper()
        status = "FILLED" if state in {"FILLED", "CANCELLED", "REJECTED"} else state

        return OrderResult(
            order_id=str(order.get("order_id", "")),
            symbol=symbol,
            side=side_norm,
            quantity=float(order.get("amount") or quantity),
            price=float(order.get("price") or 0.0),
            avg_price=avg_price,
            status=status or "FILLED",
            fee=fee,
            raw=result if isinstance(result, dict) else {"result": result},
        )

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        *,
        reduce_only: bool = False,
        client_order_id: str | None = None,
    ) -> OrderResult:
        return self.submit_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            reduce_only=reduce_only,
            client_order_id=client_order_id,
        )

    def close_position(self, symbol: str, side: str, quantity: float) -> OrderResult:
        close_side = "SELL" if side == "LONG" else "BUY"
        return self.submit_order(
            symbol=symbol,
            side=close_side,
            quantity=quantity,
            reduce_only=True,
        )

    # ------------------------------------------------------------------
    # Additional methods for PositionManager compatibility
    # ------------------------------------------------------------------

    def get_ticker(self, symbol: str) -> OptionTicker | None:
        """Get a single ticker by instrument name."""
        parsed = _parse_symbol(symbol)
        if not parsed:
            return None

        try:
            result = self._rpc("public/ticker", {"instrument_name": symbol})
        except Exception:
            return None

        if not isinstance(result, dict):
            return None

        mark_iv_raw = float(result.get("mark_iv") or 0.0)
        return OptionTicker(
            symbol=symbol,
            underlying=parsed["underlying"],
            strike=parsed["strike"],
            option_type=parsed["option_type"],
            expiry=parsed["expiry"],
            bid_price=float(result.get("best_bid_price") or 0.0),
            ask_price=float(result.get("best_ask_price") or 0.0),
            mark_price=float(result.get("mark_price") or 0.0),
            last_price=float(result.get("last_price") or 0.0),
            underlying_price=float(result.get("underlying_price") or 0.0),
            volume_24h=float(result.get("stats", {}).get("volume") or 0.0) if isinstance(result.get("stats"), dict) else 0.0,
            open_interest=float(result.get("open_interest") or 0.0),
            mark_iv=mark_iv_raw / 100.0,
        )

    def get_positions(self, underlying: str = "BTC") -> list[dict]:
        """Get open option positions from Deribit."""
        if self.cfg.simulate_private or not self.cfg.client_id or not self.cfg.client_secret:
            return []

        try:
            result = self._private_rpc("private/get_positions", {
                "currency": underlying.upper(),
                "kind": "option",
            })
        except Exception as e:
            logger.error(f"Failed to query positions: {e}")
            return []

        if not isinstance(result, list):
            return []

        positions: list[dict] = []
        for item in result:
            if not isinstance(item, dict):
                continue
            size = float(item.get("size") or 0)
            if abs(size) <= 0:
                continue
            positions.append({
                "symbol": str(item.get("instrument_name", "")),
                "side": "LONG" if size > 0 else "SHORT",
                "quantity": abs(size),
                "entryPrice": float(item.get("average_price") or 0),
                "unrealizedPnl": float(item.get("floating_profit_loss") or 0),
            })

        return positions

    def query_order(self, symbol: str, order_id: str) -> OrderResult:
        """Query a single order by order_id."""
        if self.cfg.simulate_private:
            return OrderResult(
                order_id=order_id, symbol=symbol, side="",
                quantity=0.0, price=0.0, avg_price=0.0,
                status="FILLED", fee=0.0, raw={"simulated": True},
            )

        try:
            result = self._private_rpc("private/get_order_state", {
                "order_id": order_id,
            })
        except Exception as e:
            logger.error(f"Query order {order_id} failed: {e}")
            raise

        order = result if isinstance(result, dict) else {}
        state = str(order.get("order_state", "")).upper()
        status_map = {
            "FILLED": "FILLED",
            "OPEN": "NEW",
            "CANCELLED": "CANCELLED",
            "REJECTED": "REJECTED",
            "UNTRIGGERED": "NEW",
        }
        status = status_map.get(state, state)

        return OrderResult(
            order_id=str(order.get("order_id", order_id)),
            symbol=symbol,
            side=str(order.get("direction", "")).upper(),
            quantity=float(order.get("filled_amount") or 0.0),
            price=float(order.get("price") or 0.0),
            avg_price=float(order.get("average_price") or 0.0),
            status=status,
            fee=float(order.get("commission") or 0.0),
            raw=order,
        )

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an open order. Returns True on success."""
        if self.cfg.simulate_private:
            return True

        try:
            self._private_rpc("private/cancel", {"order_id": order_id})
            return True
        except Exception as e:
            logger.error(f"Cancel order {order_id} failed: {e}")
            return False
