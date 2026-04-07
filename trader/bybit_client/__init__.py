"""Bybit Options API client."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlencode

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


class OrderNotFoundError(RuntimeError):
    """Raised when Bybit reports that an order no longer exists."""

    def __init__(
        self,
        symbol: str,
        order_id: str | None = None,
        client_order_id: str | None = None,
        message: str = "Order does not exist",
    ):
        lookup = order_id or client_order_id or ""
        super().__init__(f"Order not found for {symbol}: {lookup} ({message})")
        self.symbol = symbol
        self.order_id = order_id
        self.client_order_id = client_order_id
        self.message = message


MONTH_MAP = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}

BYBIT_SYMBOL_RE = re.compile(
    r"^(?P<ul>[A-Z]+)-(?P<day>\d{1,2})(?P<mon>[A-Z]{3})(?P<yy>\d{2})-(?P<strike>\d+(?:\.\d+)?)-(?P<cp>[CP])(?:-(?P<settle>[A-Z]+))?$"
)

LEGACY_SYMBOL_RE = re.compile(
    r"^(?P<ul>[A-Z]+)-(?P<yymmdd>\d{6})-(?P<strike>\d+(?:\.\d+)?)-(?P<cp>[CP])$"
)


def _parse_symbol(symbol: str):
    m = BYBIT_SYMBOL_RE.match(symbol)
    if m:
        day = int(m.group("day"))
        month = MONTH_MAP.get(m.group("mon"), 1)
        year = 2000 + int(m.group("yy"))
        return {
            "underlying": m.group("ul"),
            "expiry": datetime(year, month, day, 8, 0, 0, tzinfo=timezone.utc),
            "strike": float(m.group("strike")),
            "option_type": "call" if m.group("cp") == "C" else "put",
        }

    legacy = LEGACY_SYMBOL_RE.match(symbol)
    if not legacy:
        return None

    yymmdd = legacy.group("yymmdd")
    year = 2000 + int(yymmdd[:2])
    month = int(yymmdd[2:4])
    day = int(yymmdd[4:6])
    return {
        "underlying": legacy.group("ul"),
        "expiry": datetime(year, month, day, 8, 0, 0, tzinfo=timezone.utc),
        "strike": float(legacy.group("strike")),
        "option_type": "call" if legacy.group("cp") == "C" else "put",
    }


class BybitOptionsClient:
    GET_RETRY_ATTEMPTS = 3
    GET_RETRY_BACKOFF_SEC = 0.25
    RECV_WINDOW = 5000

    def __init__(self, config: ExchangeConfig):
        self.cfg = config
        self.session = requests.Session()
        self._base = self.cfg.base_url.rstrip("/")
        self.session.headers.update({"Content-Type": "application/json"})

    @staticmethod
    def _normalize_iv(value: Any) -> float:
        try:
            iv = float(value or 0.0)
        except Exception:
            return 0.0
        if iv > 3.0:
            iv /= 100.0
        return max(iv, 0.0)

    @staticmethod
    def _side_to_exchange(side: str) -> str:
        return "Buy" if str(side or "").upper() == "BUY" else "Sell"

    @staticmethod
    def _status_from_exchange(status: str) -> str:
        value = str(status or "").strip().lower()
        mapping = {
            "new": "NEW",
            "partiallyfilled": "PARTIALLY_FILLED",
            "filled": "FILLED",
            "cancelled": "CANCELLED",
            "rejected": "REJECTED",
            "partiallyfilledcanceled": "PARTIALLY_FILLED",
            "pendingcancel": "PENDING_CANCEL",
            "untriggered": "NEW",
            "triggered": "NEW",
            "deactivated": "CANCELLED",
        }
        return mapping.get(value.replace("_", ""), str(status or "").upper() or "NEW")

    @staticmethod
    def _normalize_client_order_id(client_order_id: str | None) -> str | None:
        if not client_order_id:
            return None
        normalized = re.sub(r"[^A-Za-z0-9_-]+", "-", str(client_order_id)).strip("-")
        if not normalized:
            normalized = f"oid-{uuid.uuid4().hex[:20]}"
        if len(normalized) <= 36:
            return normalized
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:8]
        prefix_len = 20
        suffix_len = 7
        compressed = f"{normalized[:prefix_len]}-{digest}-{normalized[-suffix_len:]}"
        return compressed[:36]

    def _sign(self, payload: str, timestamp_ms: int) -> str:
        origin = f"{timestamp_ms}{self.cfg.api_key}{self.RECV_WINDOW}{payload}"
        return hmac.new(self.cfg.api_secret.encode("utf-8"), origin.encode("utf-8"), hashlib.sha256).hexdigest()

    @staticmethod
    def _has_api_error(data: Any) -> bool:
        if not isinstance(data, dict):
            return True
        try:
            return int(data.get("retCode", -1)) != 0
        except Exception:
            return True

    def _is_retryable_get_error(self, exc: Exception) -> bool:
        if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
            return True
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            return exc.response.status_code in {429, 500, 502, 503, 504}
        if isinstance(exc, RuntimeError):
            return any(token in str(exc) for token in ("10016", "10006", "timeout", "Too many visits"))
        return False

    @staticmethod
    def _extract_api_error(exc: Exception) -> tuple[int | None, str | None]:
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            try:
                payload = exc.response.json()
            except Exception:
                return None, None
            if isinstance(payload, dict):
                code = payload.get("retCode")
                msg = payload.get("retMsg")
                try:
                    code = int(code) if code is not None else None
                except Exception:
                    code = None
                return code, str(msg) if msg is not None else None
        match = re.search(r"Bybit error\s+(-?\d+)\s*:\s*(.*)", str(exc), re.IGNORECASE)
        if match:
            try:
                code = int(match.group(1))
            except Exception:
                code = None
            return code, (match.group(2) or "").strip() or None
        return None, None

    @staticmethod
    def _format_http_error(exc: Exception) -> str:
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            resp = exc.response
            detail = ""
            try:
                payload = resp.json()
                detail = f" | body={payload}"
            except Exception:
                text = (resp.text or "").strip()
                if text:
                    detail = f" | body={text[:300]}"
            return f"{exc}{detail}"
        return str(exc)

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        private: bool = False,
        retry_get: bool = False,
    ) -> Any:
        url = f"{self._base}{path}"
        attempts = self.GET_RETRY_ATTEMPTS if retry_get else 1
        params = dict(params or {})
        body = dict(body or {})

        for attempt in range(1, attempts + 1):
            try:
                headers: dict[str, str] = {}
                request_kwargs: dict[str, Any] = {"timeout": self.cfg.timeout}
                payload = ""
                filtered_params = {k: v for k, v in params.items() if v is not None}
                if private:
                    timestamp_ms = int(time.time() * 1000)
                    if method.upper() == "GET":
                        payload = urlencode(list(filtered_params.items()))
                    else:
                        payload = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
                    headers.update(
                        {
                            "X-BAPI-API-KEY": self.cfg.api_key,
                            "X-BAPI-TIMESTAMP": str(timestamp_ms),
                            "X-BAPI-RECV-WINDOW": str(self.RECV_WINDOW),
                            "X-BAPI-SIGN": self._sign(payload, timestamp_ms),
                            "X-BAPI-SIGN-TYPE": "2",
                        }
                    )

                if method.upper() == "GET":
                    request_kwargs["params"] = filtered_params
                    resp = self.session.get(url, headers=headers, **request_kwargs)
                elif method.upper() == "POST":
                    request_kwargs["data"] = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
                    resp = self.session.post(url, headers=headers, **request_kwargs)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                resp.raise_for_status()
                data = resp.json()
                if self._has_api_error(data):
                    raise RuntimeError(f"Bybit error {data.get('retCode')}: {data.get('retMsg', '')}")
                return data
            except Exception as e:
                if retry_get and attempt < attempts and self._is_retryable_get_error(e):
                    sleep_sec = self.GET_RETRY_BACKOFF_SEC * (2 ** (attempt - 1))
                    logger.warning(
                        f"Bybit {method} {path} failed on attempt {attempt}/{attempts}: {self._format_http_error(e)}; retrying in {sleep_sec:.2f}s"
                    )
                    time.sleep(sleep_sec)
                    continue
                raise

    def _public_get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        try:
            return self._request_json("GET", path, params=params, retry_get=True)
        except Exception as e:
            logger.error(f"Bybit GET {path} failed: {self._format_http_error(e)}")
            raise

    def _private_get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if self.cfg.simulate_private:
            return None
        try:
            return self._request_json("GET", path, params=params, private=True, retry_get=True)
        except Exception as e:
            logger.error(f"Bybit private GET {path} failed: {self._format_http_error(e)}")
            raise

    def _private_post(self, path: str, body: dict[str, Any] | None = None) -> Any:
        if self.cfg.simulate_private:
            return None
        try:
            return self._request_json("POST", path, body=body, private=True)
        except Exception as e:
            logger.error(f"Bybit private POST {path} failed: {self._format_http_error(e)}")
            raise

    @staticmethod
    def _get_result_list(data: Any) -> list[dict[str, Any]]:
        if not isinstance(data, dict):
            return []
        result = data.get("result")
        if not isinstance(result, dict):
            return []
        items = result.get("list")
        return items if isinstance(items, list) else []

    @classmethod
    def _item_to_ticker(cls, item: dict[str, Any], parsed: dict[str, Any]) -> OptionTicker:
        bid_price = float(item.get("bid1Price") or 0.0)
        ask_price = float(item.get("ask1Price") or 0.0)
        last_price = float(item.get("lastPrice") or 0.0)
        mark_price = float(item.get("markPrice") or 0.0)
        if mark_price <= 0 and bid_price > 0 and ask_price > 0:
            mark_price = (bid_price + ask_price) / 2.0
        return OptionTicker(
            symbol=str(item.get("symbol", "")),
            underlying=parsed["underlying"],
            strike=parsed["strike"],
            option_type=parsed["option_type"],
            expiry=parsed["expiry"],
            bid_price=bid_price,
            ask_price=ask_price,
            mark_price=mark_price,
            last_price=last_price,
            underlying_price=float(item.get("underlyingPrice") or item.get("indexPrice") or 0.0),
            volume_24h=float(item.get("volume24h") or 0.0),
            open_interest=float(item.get("openInterest") or 0.0),
            delta=float(item.get("delta") or 0.0),
            mark_iv=cls._normalize_iv(item.get("markIv") or item.get("markIV") or 0.0),
        )

    def get_spot_price(self, underlying: str = "BTC") -> float:
        if self.cfg.simulate_private:
            return 0.0
        try:
            data = self._public_get(
                "/v5/market/tickers",
                {"category": "spot", "symbol": f"{underlying.upper()}USDT"},
            )
        except Exception:
            return 0.0
        items = self._get_result_list(data)
        if not items:
            return 0.0
        item = items[0]
        return float(item.get("lastPrice") or item.get("markPrice") or item.get("indexPrice") or 0.0)

    def get_hourly_index_prices(self, underlying: str = "BTC", limit: int = 25) -> list[tuple[datetime, float]]:
        if self.cfg.simulate_private:
            return []
        limit = max(int(limit), 3)
        try:
            data = self._public_get(
                "/v5/market/kline",
                {
                    "category": "spot",
                    "symbol": f"{underlying.upper()}USDT",
                    "interval": "60",
                    "limit": limit,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to fetch hourly prices from Bybit kline: {self._format_http_error(e)}")
            return []

        rows = self._get_result_list(data)
        prices: list[tuple[datetime, float]] = []
        for row_any in reversed(rows):
            if not isinstance(row_any, (list, tuple)) or len(row_any) < 5:
                continue
            row = list(row_any)
            try:
                ts = datetime.fromtimestamp(float(row[0]) / 1000.0, tz=timezone.utc)
                close = float(row[4])
            except Exception:
                continue
            if close > 0 and math.isfinite(close):
                prices.append((ts, close))
        return prices

    def get_realized_vol(self, underlying: str = "BTC", lookback_hours: int = 24) -> float | None:
        if lookback_hours < 2:
            return None
        window = self.get_hourly_index_prices(underlying, limit=lookback_hours + 1)
        if len(window) < 3:
            return None
        prices = [px for _, px in window if px > 0 and math.isfinite(px)]
        if len(prices) < 3:
            return None
        log_returns: list[float] = []
        for prev, curr in zip(prices[:-1], prices[1:]):
            if prev <= 0 or curr <= 0:
                continue
            try:
                log_returns.append(math.log(curr / prev))
            except Exception:
                continue
        if len(log_returns) < 2:
            return None
        mean = sum(log_returns) / len(log_returns)
        var = sum((x - mean) ** 2 for x in log_returns) / (len(log_returns) - 1)
        if var < 0 or not math.isfinite(var):
            return None
        return math.sqrt(var) * math.sqrt(24.0 * 365.25)

    def get_tickers(self, underlying: str = "BTC") -> list[OptionTicker]:
        if self.cfg.simulate_private:
            return []
        try:
            result = self._public_get(
                "/v5/market/tickers",
                {"category": "option", "baseCoin": underlying.upper()},
            )
        except Exception as e:
            logger.error(f"Failed to fetch Bybit option tickers: {e}")
            return []
        out: list[OptionTicker] = []
        for item in self._get_result_list(result):
            symbol = str(item.get("symbol", ""))
            parsed = _parse_symbol(symbol)
            if parsed and parsed["underlying"] == underlying.upper():
                out.append(self._item_to_ticker(item, parsed))
        return out

    def get_tickers_for_symbols(self, symbols: list[str]) -> dict[str, OptionTicker]:
        wanted = {str(s) for s in symbols if s}
        if not wanted:
            return {}
        grouped: dict[str, set[str]] = {}
        for symbol in wanted:
            parsed = _parse_symbol(symbol)
            if parsed:
                grouped.setdefault(parsed["underlying"], set()).add(symbol)
        quotes: dict[str, OptionTicker] = {}
        for underlying, group in grouped.items():
            for ticker in self.get_tickers(underlying):
                if ticker.symbol in group:
                    quotes[ticker.symbol] = ticker
        for symbol in wanted - set(quotes):
            ticker = self.get_ticker(symbol)
            if ticker is not None:
                quotes[symbol] = ticker
        return quotes

    def get_greeks(self, underlying: str = "BTC") -> dict[str, dict[str, float]]:
        return {
            ticker.symbol: {"delta": float(ticker.delta or 0.0), "mark_iv": float(ticker.mark_iv or 0.0)}
            for ticker in self.get_tickers(underlying)
        }

    def enrich_greeks(self, tickers: list[OptionTicker], underlying: str = "BTC") -> None:
        greeks = self.get_greeks(underlying)
        for ticker in tickers:
            if ticker.symbol in greeks:
                ticker.delta = greeks[ticker.symbol]["delta"]
                ticker.mark_iv = greeks[ticker.symbol]["mark_iv"]

    def get_mark_prices(self, underlying: str = "BTC") -> dict[str, float]:
        return {ticker.symbol: float(ticker.mark_price or 0.0) for ticker in self.get_tickers(underlying)}

    def get_account(self) -> AccountInfo:
        if self.cfg.simulate_private or not self.cfg.api_key or not self.cfg.api_secret:
            return AccountInfo(1.0, 1.0, 0.0, {"simulated": True})
        try:
            result = self._private_get(
                "/v5/account/wallet-balance",
                {"accountType": "UNIFIED", "coin": self.cfg.account_currency.upper()},
            )
        except Exception as e:
            logger.error(f"Failed to get Bybit account: {e}")
            return AccountInfo(0.0, 0.0, 0.0, {"error": str(e)})
        entries = self._get_result_list(result)
        if not entries:
            return AccountInfo(0.0, 0.0, 0.0, result if isinstance(result, dict) else {"result": result})
        info = entries[0]
        coins = info.get("coin") if isinstance(info, dict) else []
        if not isinstance(coins, list):
            coins = []
        coin_upnl = sum(float((coin or {}).get("unrealisedPnl") or 0.0) for coin in coins if isinstance(coin, dict))
        total_balance = float(info.get("totalEquity") or info.get("totalMarginBalance") or info.get("totalWalletBalance") or 0.0)
        available_balance = float(info.get("totalAvailableBalance") or 0.0)
        unrealized_pnl = float(info.get("totalPerpUPL") or coin_upnl or 0.0)
        return AccountInfo(total_balance, available_balance, unrealized_pnl, result if isinstance(result, dict) else {"result": result})

    def get_positions(self, underlying: str = "") -> list[dict]:
        if self.cfg.simulate_private or not self.cfg.api_key or not self.cfg.api_secret:
            return []
        params: dict[str, Any] = {"category": "option"}
        if underlying:
            params["baseCoin"] = underlying.upper()
        try:
            result = self._private_get("/v5/position/list", params)
        except Exception as e:
            logger.error(f"Failed to query Bybit positions: {e}")
            return []
        out: list[dict] = []
        for item in self._get_result_list(result):
            if not isinstance(item, dict):
                continue
            qty = float(item.get("size") or 0.0)
            if qty <= 0:
                continue
            side_raw = str(item.get("side") or "")
            side = "LONG" if side_raw.lower() == "buy" else "SHORT"
            symbol = str(item.get("symbol") or "")
            if underlying:
                parsed = _parse_symbol(symbol)
                if not parsed or parsed["underlying"] != underlying.upper():
                    continue
            out.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "quantity": qty,
                    "entryPrice": float(item.get("avgPrice") or 0.0),
                    "unrealizedPnl": float(item.get("unrealisedPnl") or 0.0),
                    "positionIM": float(item.get("positionIM") or item.get("positionIMByMp") or 0.0),
                    "positionMM": float(item.get("positionMM") or item.get("positionMMByMp") or 0.0),
                    "delta": float(item.get("delta") or 0.0),
                }
            )
        return out

    def get_order_book(self, symbol: str, limit: int = 10) -> dict[str, list[tuple[float, float]]]:
        if self.cfg.simulate_private:
            ticker = self.get_ticker(symbol)
            if ticker is None:
                return {"bids": [], "asks": []}
            synthetic_depth = 999.0
            return {
                "bids": [(float(ticker.bid_price or 0.0), synthetic_depth)],
                "asks": [(float(ticker.ask_price or 0.0), synthetic_depth)],
            }
        normalized_limit = max(1, min(int(limit or 1), 25))
        try:
            result = self._public_get(
                "/v5/market/orderbook",
                {"category": "option", "symbol": symbol, "limit": normalized_limit},
            )
        except Exception as e:
            logger.error(f"Failed to fetch Bybit order book for {symbol}: {e}")
            return {"bids": [], "asks": []}
        if not isinstance(result, dict):
            return {"bids": [], "asks": []}
        result_data = result.get("result")
        data = result_data if isinstance(result_data, dict) else {}

        def _parse_side(rows: Any) -> list[tuple[float, float]]:
            parsed: list[tuple[float, float]] = []
            if not isinstance(rows, list):
                return parsed
            for row in rows:
                if not isinstance(row, (list, tuple)) or len(row) < 2:
                    continue
                try:
                    price = float(row[0])
                    qty = float(row[1])
                except Exception:
                    continue
                if price > 0 and qty > 0:
                    parsed.append((price, qty))
            return parsed

        return {"bids": _parse_side(data.get("b")), "asks": _parse_side(data.get("a"))}

    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        *,
        reduce_only: bool = False,
        client_order_id=None,
    ) -> OrderResult:
        side_norm = side.upper()
        if self.cfg.simulate_private:
            sim_price = 0.0
            return OrderResult(
                order_id=f"SIM-{uuid.uuid4().hex[:8]}",
                symbol=symbol,
                side=side_norm,
                quantity=float(quantity or 0.0),
                price=sim_price,
                avg_price=sim_price,
                status="FILLED",
                fee=0.0,
                raw={"simulated": True},
            )

        order_link_id = self._normalize_client_order_id(client_order_id) or f"oid-{uuid.uuid4().hex[:28]}"
        body: dict[str, Any] = {
            "category": "option",
            "symbol": symbol,
            "side": self._side_to_exchange(side_norm),
            "orderType": "Market",
            "qty": str(float(quantity or 0.0)),
            "orderLinkId": order_link_id,
            "reduceOnly": bool(reduce_only),
        }
        try:
            result = self._private_post("/v5/order/create", body)
        except Exception as e:
            logger.error(f"Bybit order placement failed: {e}")
            raise
        order_info = result.get("result") if isinstance(result, dict) else {}
        order_id = str((order_info or {}).get("orderId") or "")
        try:
            return self.query_order(symbol, order_id=order_id or None, client_order_id=order_link_id)
        except Exception:
            return OrderResult(
                order_id=order_id,
                symbol=symbol,
                side=side_norm,
                quantity=0.0,
                price=0.0,
                avg_price=0.0,
                status="NEW",
                fee=0.0,
                raw=result if isinstance(result, dict) else {"result": result, "orderLinkId": order_link_id},
            )

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        *,
        reduce_only: bool = False,
        client_order_id=None,
    ) -> OrderResult:
        return self.submit_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            reduce_only=reduce_only,
            client_order_id=client_order_id,
        )

    def close_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        client_order_id: str | None = None,
    ) -> OrderResult:
        close_side = "SELL" if side == "LONG" else "BUY"
        return self.submit_order(
            symbol,
            close_side,
            abs(float(quantity)),
            reduce_only=True,
            client_order_id=client_order_id,
        )

    def query_order(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> OrderResult:
        lookup_client_id = self._normalize_client_order_id(client_order_id)
        lookup = order_id or lookup_client_id or ""
        if self.cfg.simulate_private:
            return OrderResult(lookup, symbol, "", 0.0, 0.0, 0.0, "FILLED", 0.0, {"simulated": True})
        params: dict[str, Any] = {"category": "option", "symbol": symbol}
        if order_id:
            params["orderId"] = order_id
        elif lookup_client_id:
            params["orderLinkId"] = lookup_client_id
        else:
            raise ValueError("Either order_id or client_order_id is required")
        try:
            result = self._private_get("/v5/order/realtime", params)
        except Exception as e:
            code, msg = self._extract_api_error(e)
            msg_norm = str(msg or e).lower()
            if code in {110001, 170213} or "order not" in msg_norm:
                raise OrderNotFoundError(
                    symbol=symbol,
                    order_id=order_id,
                    client_order_id=lookup_client_id,
                    message=msg or "Order does not exist",
                ) from e
            raise
        items = self._get_result_list(result)
        if not items:
            raise OrderNotFoundError(
                symbol=symbol,
                order_id=order_id,
                client_order_id=lookup_client_id,
                message="Order does not exist",
            )
        item = items[0]
        return OrderResult(
            order_id=str(item.get("orderId", lookup)),
            symbol=symbol,
            side=str(item.get("side", "")).upper(),
            quantity=float(item.get("cumExecQty") or 0.0),
            price=float(item.get("price") or 0.0),
            avg_price=float(item.get("avgPrice") or item.get("price") or 0.0),
            status=self._status_from_exchange(str(item.get("orderStatus") or "")),
            fee=float(item.get("cumExecFee") or 0.0),
            raw=item if isinstance(item, dict) else {"result": item},
        )

    def cancel_order(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> bool:
        if self.cfg.simulate_private:
            return True
        lookup_client_id = self._normalize_client_order_id(client_order_id)
        body: dict[str, Any] = {"category": "option", "symbol": symbol}
        lookup = order_id or lookup_client_id or ""
        if order_id:
            body["orderId"] = order_id
        elif lookup_client_id:
            body["orderLinkId"] = lookup_client_id
        else:
            raise ValueError("Either order_id or client_order_id is required")
        try:
            self._request_json("POST", "/v5/order/cancel", body=body, private=True)
            return True
        except Exception as e:
            code, msg = self._extract_api_error(e)
            msg_norm = str(msg or e).lower()
            if code in {110001, 170213} or "order not" in msg_norm or "too late to cancel" in msg_norm:
                logger.debug(f"Cancel order {lookup} ignored: {msg or 'order already gone'}")
                return False
            logger.error(f"Cancel order {lookup} failed: {self._format_http_error(e)}")
            return False

    def get_ticker(self, symbol: str) -> OptionTicker | None:
        parsed = _parse_symbol(symbol)
        if not parsed:
            return None
        try:
            result = self._public_get(
                "/v5/market/tickers",
                {"category": "option", "symbol": symbol},
            )
        except Exception:
            return None
        items = self._get_result_list(result)
        for item in items:
            if isinstance(item, dict) and str(item.get("symbol", "")) == symbol:
                return self._item_to_ticker(item, parsed)
        return None
