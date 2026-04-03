"""Binance European Options API client."""

from __future__ import annotations

import hashlib
import hmac
import math
import re
import time
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
    """Raised when Binance reports that an order no longer exists."""

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


BINANCE_SYMBOL_RE = re.compile(
    r"^(?P<ul>[A-Z]+)-(?P<yymmdd>\d{6})-(?P<strike>\d+(?:\.\d+)?)-(?P<cp>[CP])$"
)


def _parse_symbol(symbol: str):
    m = BINANCE_SYMBOL_RE.match(symbol)
    if not m:
        return None
    yymmdd = m.group("yymmdd")
    return {
        "underlying": m.group("ul"),
        "expiry": datetime(
            2000 + int(yymmdd[:2]),
            int(yymmdd[2:4]),
            int(yymmdd[4:6]),
            8,
            0,
            0,
            tzinfo=timezone.utc,
        ),
        "strike": float(m.group("strike")),
        "option_type": "call" if m.group("cp") == "C" else "put",
    }


class BinanceOptionsClient:
    GET_RETRY_ATTEMPTS = 3
    GET_RETRY_BACKOFF_SEC = 0.25

    def __init__(self, config: ExchangeConfig):
        self.cfg = config
        self.session = requests.Session()
        self._base = self.cfg.base_url.rstrip("/")
        if self.cfg.api_key:
            self.session.headers.update({"X-MBX-APIKEY": self.cfg.api_key})

    def _sign(self, params: dict) -> dict:
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000
        query = urlencode(params)
        params["signature"] = hmac.new(
            self.cfg.api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        return params

    @staticmethod
    def _has_api_error(data: Any) -> bool:
        return bool(isinstance(data, dict) and data.get("code") and int(data["code"]) < 0)

    def _is_retryable_get_error(self, exc: Exception) -> bool:
        if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
            return True
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            return exc.response.status_code in {429, 500, 502, 503, 504}
        if isinstance(exc, RuntimeError):
            return any(code in str(exc) for code in ("-1001", "-1003", "-1007"))
        return False

    @staticmethod
    def _extract_api_error(exc: Exception) -> tuple[int | None, str | None]:
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            try:
                payload = exc.response.json()
            except Exception:
                return None, None
            if isinstance(payload, dict):
                code = payload.get("code")
                msg = payload.get("msg")
                try:
                    code = int(code) if code is not None else None
                except Exception:
                    code = None
                return code, str(msg) if msg is not None else None
            return None, None
        if isinstance(exc, RuntimeError):
            match = re.search(r"Binance error\s+(-?\d+):\s*(.*)", str(exc))
            if match:
                try:
                    return int(match.group(1)), match.group(2).strip() or None
                except Exception:
                    return None, None
        return None, None

    @staticmethod
    def _format_http_error(exc: Exception) -> str:
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            resp = exc.response
            detail = ""
            try:
                payload = resp.json()
                if isinstance(payload, dict):
                    code = payload.get("code")
                    msg = payload.get("msg")
                    if code is not None or msg:
                        detail = f" | code={code} msg={msg}"
                    else:
                        detail = f" | body={payload}"
                else:
                    detail = f" | body={payload}"
            except Exception:
                text = (resp.text or "").strip()
                if text:
                    detail = f" | body={text[:300]}"
            return f"{exc}{detail}"
        return str(exc)

    def _request_json(self, method: str, path: str, params=None, retry_get: bool = False):
        url = f"{self._base}{path}"
        attempts = self.GET_RETRY_ATTEMPTS if retry_get else 1
        for attempt in range(1, attempts + 1):
            try:
                if method == "GET":
                    resp = self.session.get(url, params=params or {}, timeout=self.cfg.timeout)
                elif method == "POST":
                    resp = self.session.post(url, params=params or {}, timeout=self.cfg.timeout)
                elif method == "DELETE":
                    resp = self.session.delete(url, params=params or {}, timeout=self.cfg.timeout)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                resp.raise_for_status()
                data = resp.json()
                if self._has_api_error(data):
                    raise RuntimeError(f"Binance error {data['code']}: {data.get('msg', '')}")
                return data
            except Exception as e:
                if retry_get and attempt < attempts and self._is_retryable_get_error(e):
                    sleep_sec = self.GET_RETRY_BACKOFF_SEC * (2 ** (attempt - 1))
                    logger.warning(
                        f"Binance {method} {path} failed on attempt {attempt}/{attempts}: {self._format_http_error(e)}; retrying in {sleep_sec:.2f}s"
                    )
                    time.sleep(sleep_sec)
                    continue
                raise

    def _public_get(self, path: str, params=None):
        try:
            return self._request_json("GET", path, params=params, retry_get=True)
        except Exception as e:
            logger.error(f"Binance GET {path} failed: {e}")
            raise

    def _private_get(self, path: str, params=None):
        if self.cfg.simulate_private:
            return None
        try:
            return self._request_json("GET", path, params=self._sign(dict(params or {})), retry_get=True)
        except Exception as e:
            logger.error(f"Binance private GET {path} failed: {self._format_http_error(e)}")
            raise

    def _private_post(self, path: str, params=None):
        if self.cfg.simulate_private:
            return None
        try:
            return self._request_json("POST", path, params=self._sign(dict(params or {})))
        except Exception as e:
            logger.error(f"Binance private POST {path} failed: {self._format_http_error(e)}")
            raise

    def _private_delete(self, path: str, params=None):
        if self.cfg.simulate_private:
            return None
        try:
            return self._request_json("DELETE", path, params=self._sign(dict(params or {})))
        except Exception as e:
            logger.error(f"Binance DELETE {path} failed: {self._format_http_error(e)}")
            raise

    @staticmethod
    def _item_to_ticker(item: dict, parsed: dict) -> OptionTicker:
        bid_price = float(item.get("bidPrice") or 0)
        ask_price = float(item.get("askPrice") or 0)
        last_price = float(item.get("lastPrice") or 0)
        mark_price = (bid_price + ask_price) / 2 if bid_price > 0 and ask_price > 0 else last_price
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
            underlying_price=float(item.get("underlyingPrice") or item.get("exercisePrice") or 0),
            volume_24h=float(item.get("volume") or 0),
            open_interest=float(item.get("amount") or 0),
            delta=float(item.get("delta") or 0),
            mark_iv=float(item.get("markIV") or 0),
        )

    def get_spot_price(self, underlying: str = "ETH") -> float:
        try:
            data = self._public_get("/eapi/v1/index", {"underlying": f"{underlying.upper()}USDT"})
            if not isinstance(data, dict):
                return 0.0
            return float(data.get("indexPrice", 0.0))
        except Exception:
            return 0.0

    def _get_external_json(self, url: str, params: dict[str, Any]) -> Any:
        resp = self.session.get(url, params=params, timeout=self.cfg.timeout)
        resp.raise_for_status()
        data = resp.json()
        if self._has_api_error(data):
            raise RuntimeError(f"Binance error {data['code']}: {data.get('msg', '')}")
        return data

    def get_hourly_index_prices(self, underlying: str = "ETH", limit: int = 25) -> list[tuple[datetime, float]]:
        limit = max(int(limit), 3)
        ul = underlying.upper()
        candidates = [
            (
                "https://fapi.binance.com/fapi/v1/indexPriceKlines",
                {"pair": f"{ul}USDT", "interval": "1h", "limit": limit},
            ),
            (
                "https://dapi.binance.com/dapi/v1/indexPriceKlines",
                {"pair": f"{ul}USD", "interval": "1h", "limit": limit},
            ),
            (
                "https://api.binance.com/api/v3/klines",
                {"symbol": f"{ul}USDT", "interval": "1h", "limit": limit},
            ),
        ]

        for url, params in candidates:
            try:
                result = self._get_external_json(url, params)
            except Exception as e:
                logger.warning(f"Failed to fetch hourly prices from {url}: {self._format_http_error(e)}")
                continue

            if not isinstance(result, list):
                continue

            prices: list[tuple[datetime, float]] = []
            for row in result:
                if not isinstance(row, list) or len(row) < 5:
                    continue
                try:
                    ts = datetime.fromtimestamp(float(row[0]) / 1000.0, tz=timezone.utc)
                    close = float(row[4])
                except Exception:
                    continue
                if close > 0 and math.isfinite(close):
                    prices.append((ts, close))
            if len(prices) >= 3:
                return prices

        return []

    def get_realized_vol(self, underlying: str = "ETH", lookback_hours: int = 24) -> float | None:
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

    def get_tickers(self, underlying: str = "ETH") -> list[OptionTicker]:
        try:
            result = self._public_get("/eapi/v1/ticker")
        except Exception as e:
            logger.error(f"Failed to fetch tickers: {e}")
            return []
        if not isinstance(result, list):
            return []
        prefix = f"{underlying.upper()}-"
        out: list[OptionTicker] = []
        for item in result:
            symbol = str(item.get("symbol", ""))
            if not symbol.startswith(prefix):
                continue
            parsed = _parse_symbol(symbol)
            if parsed:
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

    def get_greeks(self, underlying: str = "ETH") -> dict[str, dict[str, float]]:
        try:
            result = self._public_get("/eapi/v1/mark")
        except Exception:
            return {}
        if not isinstance(result, list):
            return {}
        prefix = f"{underlying.upper()}-"
        greeks = {}
        for item in result:
            symbol = str(item.get("symbol", ""))
            if symbol.startswith(prefix):
                greeks[symbol] = {
                    "delta": float(item.get("delta") or 0),
                    "mark_iv": float(item.get("markIV") or 0),
                }
        return greeks

    def enrich_greeks(self, tickers: list[OptionTicker], underlying: str = "ETH") -> None:
        greeks = self.get_greeks(underlying)
        for ticker in tickers:
            if ticker.symbol in greeks:
                ticker.delta = greeks[ticker.symbol]["delta"]
                ticker.mark_iv = greeks[ticker.symbol]["mark_iv"]

    def get_mark_prices(self, underlying: str = "ETH") -> dict[str, float]:
        try:
            result = self._public_get("/eapi/v1/mark")
        except Exception:
            return {}
        if not isinstance(result, list):
            return {}
        prefix = f"{underlying.upper()}-"
        return {
            str(item.get("symbol", "")): float(item.get("markPrice") or 0)
            for item in result
            if str(item.get("symbol", "")).startswith(prefix)
        }

    def get_account(self) -> AccountInfo:
        if self.cfg.simulate_private or not self.cfg.api_key or not self.cfg.api_secret:
            return AccountInfo(1.0, 1.0, 0.0, {"simulated": True})
        try:
            result = self._private_get("/eapi/v1/marginAccount")
        except Exception as e:
            logger.error(f"Failed to get account: {e}")
            return AccountInfo(0.0, 0.0, 0.0, {"error": str(e)})
        if not isinstance(result, dict):
            return AccountInfo(0.0, 0.0, 0.0, {"unexpected": str(result)[:200]})
        assets = result.get("asset", [])
        if not isinstance(assets, list):
            assets = []
        total = sum(
            float(
                a.get("equity")
                or a.get("totalEquity")
                or a.get("balance")
                or a.get("marginBalance")
                or 0
            )
            for a in assets
            if isinstance(a, dict)
        )
        avail = sum(float(a.get("available") or a.get("availableBalance") or 0) for a in assets if isinstance(a, dict))
        upnl = sum(float(a.get("unrealizedPNL") or a.get("unrealizedPnl") or 0) for a in assets if isinstance(a, dict))
        return AccountInfo(total, avail, upnl, result)

    def get_positions(self, underlying: str = "ETH") -> list[dict]:
        if self.cfg.simulate_private or not self.cfg.api_key or not self.cfg.api_secret:
            return []
        try:
            result = self._private_get("/eapi/v1/position")
        except Exception as e:
            logger.error(f"Failed to query positions: {e}")
            return []
        if not isinstance(result, list):
            return []
        prefix = f"{underlying.upper()}-"
        out = []
        for item in result:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol", ""))
            if underlying and not symbol.startswith(prefix):
                continue
            qty = float(item.get("quantity") or item.get("positionQty") or item.get("positionAmount") or 0)
            if abs(qty) <= 0:
                continue
            out.append({
                "symbol": symbol,
                "side": str(item.get("side") or ("LONG" if qty > 0 else "SHORT")).upper(),
                "quantity": qty,
                "entryPrice": float(item.get("entryPrice") or 0),
                "unrealizedPnl": float(item.get("unrealizedPNL") or item.get("unrealizedPnl") or 0),
            })
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

        try:
            requested_limit = int(limit)
        except Exception:
            requested_limit = 10
        valid_limits = (10, 20, 50, 100, 500, 1000)
        normalized_limit = next((value for value in valid_limits if requested_limit <= value), valid_limits[-1])

        try:
            result = self._public_get("/eapi/v1/depth", {"symbol": symbol, "limit": normalized_limit})
        except Exception as e:
            logger.error(f"Failed to fetch order book for {symbol}: {e}")
            return {"bids": [], "asks": []}

        if not isinstance(result, dict):
            return {"bids": [], "asks": []}

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

        return {
            "bids": _parse_side(result.get("bids")),
            "asks": _parse_side(result.get("asks")),
        }

    def submit_order(
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
            sim_price = float(price or 0.0)
            return OrderResult(
                order_id=f"SIM-{uuid.uuid4().hex[:8]}",
                symbol=symbol,
                side=side_norm,
                quantity=quantity,
                price=sim_price,
                avg_price=sim_price,
                status="FILLED",
                fee=0.0,
                raw={"simulated": True, "requestedType": order_type_norm, "timeInForce": time_in_force},
            )
        effective_type = order_type_norm
        effective_price = price
        effective_tif = time_in_force
        if order_type_norm == "MARKET":
            quote = self.get_ticker(symbol)
            if quote is None:
                raise RuntimeError(f"No live quote available for synthetic market order: {symbol}")
            effective_price = float(quote.ask_price if side_norm == "BUY" else quote.bid_price)
            if effective_price <= 0:
                raise RuntimeError(
                    f"Invalid synthetic market reference price for {symbol}: side={side_norm}, bid={quote.bid_price}, ask={quote.ask_price}"
                )
            effective_type = "LIMIT"
            effective_tif = effective_tif or "IOC"
        params = {"symbol": symbol, "side": side_norm, "type": effective_type, "quantity": str(quantity)}
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
        status = str(result.get("status", "")).upper()
        if status in ("NEW", "PENDING"):
            status = "NEW"
        elif status not in ("FILLED", "PARTIALLY_FILLED"):
            status = status or "FILLED"
        return OrderResult(
            order_id=str(result.get("orderId", "")),
            symbol=symbol,
            side=side_norm,
            quantity=float(result.get("executedQty") or quantity),
            price=float(result.get("price") or 0.0),
            avg_price=float(result.get("avgPrice") or result.get("price") or 0.0),
            status=status,
            fee=float(result.get("fee") or 0.0),
            raw=result,
        )

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
        return self.submit_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price,
            time_in_force=time_in_force,
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
            order_type="MARKET",
            reduce_only=True,
            client_order_id=client_order_id,
        )

    def query_order(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> OrderResult:
        lookup = order_id or client_order_id or ""
        if self.cfg.simulate_private:
            return OrderResult(lookup, symbol, "", 0.0, 0.0, 0.0, "FILLED", 0.0, {"simulated": True})

        def _query(params: dict) -> dict:
            _result = self._private_get("/eapi/v1/order", params)
            if not isinstance(_result, dict):
                return {"raw": _result}
            return _result

        def _raise_not_found(err: Exception) -> None:
            code, msg = self._extract_api_error(err)
            if code == -2013:
                raise OrderNotFoundError(
                    symbol=symbol,
                    order_id=order_id,
                    client_order_id=client_order_id,
                    message=msg or "Order does not exist",
                ) from err
            raise err

        result: dict = {}
        if order_id:
            try:
                result = _query({"symbol": symbol, "orderId": order_id})
            except Exception as e:
                if client_order_id:
                    code, _ = self._extract_api_error(e)
                    log_fn = logger.debug if code == -2013 else logger.warning
                    log_fn(
                        f"Query order by orderId failed for {symbol} orderId={order_id}: "
                        f"{self._format_http_error(e)}; retrying with clientOrderId"
                    )
                    try:
                        result = _query({"symbol": symbol, "clientOrderId": client_order_id})
                    except Exception as fallback_exc:
                        try:
                            result = _query({"symbol": symbol, "origClientOrderId": client_order_id})
                        except Exception:
                            _raise_not_found(fallback_exc)
                else:
                    _raise_not_found(e)
        elif client_order_id:
            try:
                result = _query({"symbol": symbol, "clientOrderId": client_order_id})
            except Exception as e:
                try:
                    result = _query({"symbol": symbol, "origClientOrderId": client_order_id})
                except Exception:
                    _raise_not_found(e)
        else:
            raise ValueError("Either order_id or client_order_id is required")

        if not isinstance(result, dict):
            result = {"raw": result}
        return OrderResult(
            order_id=str(result.get("orderId", lookup)),
            symbol=symbol,
            side=str(result.get("side", "")).upper(),
            quantity=float(result.get("executedQty") or 0.0),
            price=float(result.get("price") or 0.0),
            avg_price=float(result.get("avgPrice") or result.get("price") or 0.0),
            status=str(result.get("status", "")).upper(),
            fee=float(result.get("fee") or 0.0),
            raw=result,
        )

    def cancel_order(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> bool:
        if self.cfg.simulate_private:
            return True
        params = {"symbol": symbol}
        lookup = order_id or client_order_id or ""
        if order_id:
            params["orderId"] = order_id
        elif client_order_id:
            params["clientOrderId"] = client_order_id
        else:
            raise ValueError("Either order_id or client_order_id is required")
        try:
            self._private_delete("/eapi/v1/order", params)
            return True
        except Exception as e:
            if client_order_id:
                try:
                    self._private_delete("/eapi/v1/order", {"symbol": symbol, "origClientOrderId": client_order_id})
                    return True
                except Exception:
                    pass
            code, msg = self._extract_api_error(e)
            if code == -2013:
                logger.debug(
                    f"Cancel order {lookup} ignored because exchange reports missing order: {msg or 'Order does not exist'}"
                )
                return False
            logger.error(f"Cancel order {lookup} failed: {self._format_http_error(e)}")
            return False

    def get_ticker(self, symbol: str) -> OptionTicker | None:
        parsed = _parse_symbol(symbol)
        if not parsed:
            return None
        try:
            result = self._public_get("/eapi/v1/ticker", {"symbol": symbol})
        except Exception:
            return None
        items = result if isinstance(result, list) else [result]
        for item in items:
            if isinstance(item, dict) and str(item.get("symbol", "")) == symbol:
                return self._item_to_ticker(item, parsed)
        return None
