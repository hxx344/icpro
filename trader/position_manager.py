"""Position manager – 仓位管理.

Tracks open multi-leg option positions, maps exchange positions to
local trade records, and provides aggregated risk views.
"""

from __future__ import annotations

import re
import time as _time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, cast

from loguru import logger

from trader.binance_client import (
    OrderResult,
    _parse_symbol,
)
from trader.limit_chaser import ChaserConfig, LegOrder, LimitChaser
from trader.storage import Storage


# ---------------------------------------------------------------------------
# Position model
# ---------------------------------------------------------------------------

@dataclass
class CondorLeg:
    """Single option leg inside a grouped position."""
    symbol: str
    side: str              # "SELL" or "BUY"
    option_type: str       # "call" / "put"
    strike: float
    quantity: float
    entry_price: float
    trade_id: int          # DB row id in trades table
    order_id: str = ""


@dataclass
class IronCondorPosition:
    """Multi-leg option position.

    Legs:
        - sell_put: short OTM put (collect premium)
        - buy_put: long further OTM put (protection)
        - sell_call: short OTM call (collect premium)
        - buy_call: long further OTM call (protection)

    For short strangle positions, `buy_put` and `buy_call` remain `None`.
    """
    group_id: str               # unique group identifier
    entry_time: datetime
    underlying_price: float     # spot at entry
    sell_put: CondorLeg | None = None
    buy_put: CondorLeg | None = None
    sell_call: CondorLeg | None = None
    buy_call: CondorLeg | None = None
    is_open: bool = True
    total_premium: float = 0.0  # net premium collected

    @property
    def legs(self) -> list[CondorLeg]:
        return [l for l in [self.sell_put, self.buy_put,
                            self.sell_call, self.buy_call] if l is not None]

    @property
    def max_profit(self) -> float:
        """Max profit = net premium collected."""
        return self.total_premium

    @property
    def max_loss(self) -> float:
        """Max loss = wing width - net premium (per unit).

        For an iron condor: max_loss = (short_strike - long_strike) - premium.
        """
        if self.sell_put and self.buy_put:
            put_width = self.sell_put.strike - self.buy_put.strike
        else:
            put_width = float("inf")

        if self.sell_call and self.buy_call:
            call_width = self.buy_call.strike - self.sell_call.strike
        else:
            call_width = float("inf")

        width = max(put_width, call_width)
        if width == float("inf"):
            return float("inf")
        return width - self.total_premium


# ---------------------------------------------------------------------------
# Position Manager
# ---------------------------------------------------------------------------

class PositionManager:
    """Manages iron condor positions with exchange integration.

    Responsibilities:
    - Open new iron condor positions (4-leg order)
    - Track all open positions
    - Monitor PnL and trigger TP/SL
    - Close positions (4-leg close)
    - Sync with exchange positions
    - Record all trades to persistent storage
    """

    MARKET_BATCH_MAX_QTY = 1.0
    MARKET_BATCH_MAX_SPREAD_USD = 5.0
    MARKET_BATCH_SECONDARY_MAX_SPREAD_USD = 10.0
    MARKET_BATCH_RELAXED_MAX_SPREAD_USD = 15.0
    MARKET_BATCH_DEPTH_BUFFER_MULT = 1.5
    MARKET_BATCH_STAGE_WAIT_SEC = 300.0
    MARKET_BATCH_POLL_SEC = 0.5
    MARKET_PARTIAL_RETRY_WINDOW_SEC = 600.0

    def __init__(
        self,
        client: Any,
        storage: Storage,
        chaser_config: ChaserConfig | None = None,
    ):
        self.client = client
        self.storage = storage
        self.chaser = LimitChaser(client, chaser_config)
        self.open_condors: dict[str, IronCondorPosition] = {}  # group_id -> position
        self._load_open_positions()

    def _build_position_from_trades(
        self,
        group_id: str,
        trades: list[dict[str, Any]],
    ) -> IronCondorPosition | None:
        if not trades:
            return None

        condor = IronCondorPosition(
            group_id=group_id,
            entry_time=datetime.fromisoformat(trades[0]["timestamp"]),
            underlying_price=0.0,
            is_open=True,
        )

        total_premium = 0.0
        for t in trades:
            import json
            meta = json.loads(t.get("meta", "{}"))
            leg = CondorLeg(
                symbol=t["symbol"],
                side=t["side"],
                option_type=meta.get("option_type", ""),
                strike=meta.get("strike", 0),
                quantity=t["quantity"],
                entry_price=t["price"],
                trade_id=t["id"],
                order_id=t.get("order_id", ""),
            )

            leg_role = meta.get("leg_role", "")
            if leg_role == "sell_put":
                condor.sell_put = leg
                total_premium += leg.entry_price * leg.quantity
            elif leg_role == "buy_put":
                condor.buy_put = leg
                total_premium -= leg.entry_price * leg.quantity
            elif leg_role == "sell_call":
                condor.sell_call = leg
                total_premium += leg.entry_price * leg.quantity
            elif leg_role == "buy_call":
                condor.buy_call = leg
                total_premium -= leg.entry_price * leg.quantity

            condor.underlying_price = meta.get("underlying_price", condor.underlying_price)

        condor.total_premium = total_premium
        return condor

    def _load_open_positions(self) -> None:
        """Recover open positions from storage on restart."""
        self.open_condors = {}
        open_groups = self.storage.get_open_trade_groups()
        for group_id in open_groups:
            trades = self.storage.get_open_trades(group_id)
            condor = self._build_position_from_trades(group_id, trades)
            if condor is not None:
                self.open_condors[group_id] = condor

        if self.open_condors:
            logger.info(f"Recovered {len(self.open_condors)} open condor position(s)")

    def _ensure_group_loaded(self, group_id: str) -> bool:
        if group_id in self.open_condors:
            return True
        trades = self.storage.get_open_trades(group_id)
        condor = self._build_position_from_trades(group_id, trades)
        if condor is None:
            return False
        self.open_condors[group_id] = condor
        logger.info(f"Recovered open position from storage for manual action: {group_id}")
        return True

    @staticmethod
    def _leg_role_from_exchange_position(symbol: str, exchange_side: str) -> str:
        parsed = _parse_symbol(symbol) or {}
        option_type = str(parsed.get("option_type") or "").lower()
        exchange_side_norm = str(exchange_side or "").upper()
        if option_type == "call":
            return "sell_call" if exchange_side_norm == "SHORT" else "buy_call"
        return "sell_put" if exchange_side_norm == "SHORT" else "buy_put"

    def sync_exchange_positions_to_local(
        self,
        underlying: str,
        replace_on_conflict: bool = True,
    ) -> dict[str, Any]:
        """Restore live exchange positions into local open trade groups.

        When local open trades conflict with exchange positions, exchange state wins.
        """
        get_positions = getattr(self.client, "get_positions", None)
        if not callable(get_positions):
            return {"restored": False, "reason": "client_has_no_get_positions"}

        underlying_norm = str(underlying or "").upper()
        raw_positions = get_positions(underlying_norm)
        positions = raw_positions if isinstance(raw_positions, list) else []
        exchange_symbols = {
            str(pos.get("symbol") or "").upper()
            for pos in positions
            if str(pos.get("symbol") or "").upper().startswith(f"{underlying_norm}-")
        }
        local_open_trades = [
            trade for trade in self.storage.get_open_trades()
            if str(trade.get("symbol") or "").upper().startswith(f"{underlying_norm}-")
        ]
        local_symbols = {str(trade.get("symbol") or "").upper() for trade in local_open_trades}

        if not exchange_symbols:
            return {
                "restored": False,
                "reason": "no_exchange_positions",
                "local_symbols": sorted(local_symbols),
            }

        if local_symbols == exchange_symbols and local_symbols:
            if not self.open_condors:
                self._load_open_positions()
            return {
                "restored": False,
                "reason": "already_in_sync",
                "symbols": sorted(exchange_symbols),
            }

        if local_symbols and local_symbols != exchange_symbols and not replace_on_conflict:
            return {
                "restored": False,
                "reason": "conflict_detected",
                "local_symbols": sorted(local_symbols),
                "exchange_symbols": sorted(exchange_symbols),
            }

        deleted = 0
        if local_symbols != exchange_symbols:
            deleted = self.storage.delete_open_trades(symbol_prefix=f"{underlying_norm}-")
            self._load_open_positions()

        try:
            spot_price = float(getattr(self.client, "get_spot_price", lambda _u: 0.0)(underlying_norm) or 0.0)
        except Exception:
            spot_price = 0.0

        group_ids_by_expiry: dict[str, str] = {}
        used_roles: dict[str, set[str]] = {}
        recovered_symbols: list[str] = []
        for pos in positions:
            symbol = str(pos.get("symbol") or "").upper()
            if not symbol.startswith(f"{underlying_norm}-"):
                continue
            parsed = _parse_symbol(symbol) or {}
            expiry = parsed.get("expiry")
            expiry_key = expiry.strftime("%Y%m%d_%H%M%S") if expiry else symbol.replace("-", "_")
            base_group_id = group_ids_by_expiry.setdefault(
                expiry_key,
                f"RECOVERED_{underlying_norm}_{expiry_key}",
            )
            exchange_side = str(pos.get("side") or "").upper()
            leg_role = self._leg_role_from_exchange_position(symbol, exchange_side)
            group_id = base_group_id
            used = used_roles.setdefault(group_id, set())
            if leg_role in used:
                group_id = f"{base_group_id}_{symbol.replace('-', '_')}"
                used = used_roles.setdefault(group_id, set())
            used.add(leg_role)

            open_side = "SELL" if exchange_side == "SHORT" else "BUY"
            quantity = abs(float(pos.get("quantity") or 0.0))
            if quantity <= 1e-9:
                continue
            entry_price = float(pos.get("entryPrice") or 0.0)
            meta = {
                "leg_role": leg_role,
                "option_type": parsed.get("option_type", ""),
                "strike": parsed.get("strike", 0.0),
                "underlying_price": spot_price,
                "group_id": group_id,
                "recovered_from_exchange": True,
                "exchange_side": exchange_side,
            }
            self.storage.record_trade(
                trade_group=group_id,
                symbol=symbol,
                side=open_side,
                quantity=quantity,
                price=entry_price,
                fee=0.0,
                order_id=str(pos.get("orderId") or ""),
                meta=meta,
            )
            recovered_symbols.append(symbol)

        self._load_open_positions()
        logger.warning(
            f"Recovered {len(recovered_symbols)} exchange position leg(s) into local storage for {underlying_norm}: {recovered_symbols}"
        )
        return {
            "restored": bool(recovered_symbols),
            "symbols": recovered_symbols,
            "deleted_open_trades": deleted,
            "group_count": len({trade.get('trade_group') for trade in self.storage.get_open_trades() if str(trade.get('symbol') or '').upper().startswith(f'{underlying_norm}-')}),
        }

    @staticmethod
    def _infer_underlying(symbol: str) -> str:
        return str(symbol).split("-", 1)[0].upper() if symbol else "ETH"

    def _submit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: float | None = None,
        time_in_force: str | None = None,
        reduce_only: bool = False,
        client_order_id: str | None = None,
    ) -> OrderResult:
        submit_fn = getattr(self.client, "submit_order", None)
        if submit_fn is None:
            submit_fn = self.client.place_order
        return submit_fn(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price,
            time_in_force=time_in_force,
            reduce_only=reduce_only,
            client_order_id=client_order_id,
        )

    def _extract_api_error(self, exc: Exception) -> tuple[int | None, str | None]:
        extractor = getattr(self.client, "_extract_api_error", None)
        if callable(extractor):
            try:
                typed_extractor = cast(Callable[[Exception], tuple[int | None, str | None]], extractor)
                code, msg = typed_extractor(exc)
                return code, msg
            except Exception:
                pass

        message = str(exc or "")
        for pattern in (
            r"Binance error\s+(-?\d+):\s*(.*)",
            r"code\s*[=:]\s*(-?\d+)",
        ):
            match = re.search(pattern, message, re.IGNORECASE)
            if not match:
                continue
            try:
                code = int(match.group(1))
            except Exception:
                code = None
            if match.lastindex and match.lastindex >= 2:
                detail = (match.group(2) or "").strip()
                return code, detail or message or None
            return code, message or None
        return None, message or None

    def _is_margin_insufficient_error(self, exc: Exception) -> bool:
        code, msg = self._extract_api_error(exc)
        msg_norm = str(msg or exc).lower()
        return code == -2027 or "margin is insuff" in msg_norm or "insufficient margin" in msg_norm

    def _set_leg_last_error(self, leg: LegOrder, exc: Exception) -> None:
        code, msg = self._extract_api_error(exc)
        setattr(leg, "last_error_code", code)
        setattr(leg, "last_error_message", msg or str(exc))

    @staticmethod
    def _filled_leg_count(leg_orders: list[LegOrder]) -> int:
        return sum(1 for leg in leg_orders if PositionManager._remaining_leg_qty(leg) <= 1e-9)

    @staticmethod
    def _filled_leg_orders(results: list[LegOrder]) -> list[LegOrder]:
        return [r for r in results if float(r.filled_qty or 0.0) > 1e-9]

    @staticmethod
    def _has_margin_insufficient_failure(results: list[LegOrder]) -> bool:
        return any(getattr(r, "abort_reason", "") == "margin_insufficient" for r in results)

    def _persist_position_from_results(
        self,
        group_id: str,
        underlying_price: float,
        results: list[LegOrder],
        requested_quantity: float,
    ) -> tuple[IronCondorPosition | None, float]:
        filled_results = self._filled_leg_orders(results)
        if not filled_results:
            return None, 0.0

        condor = IronCondorPosition(
            group_id=group_id,
            entry_time=datetime.now(timezone.utc),
            underlying_price=underlying_price,
        )
        total_premium = 0.0

        for leg_order in filled_results:
            leg_qty = leg_order.filled_qty or requested_quantity
            meta = {
                "leg_role": leg_order.leg_role,
                "option_type": leg_order.option_type,
                "strike": leg_order.strike,
                "underlying_price": underlying_price,
                "group_id": group_id,
            }
            trade_id = self.storage.record_trade(
                trade_group=group_id,
                symbol=leg_order.symbol,
                side=leg_order.side,
                quantity=leg_qty,
                price=leg_order.avg_price,
                fee=leg_order.fee,
                order_id=leg_order.order_id,
                meta=meta,
            )

            leg = CondorLeg(
                symbol=leg_order.symbol,
                side=leg_order.side,
                option_type=leg_order.option_type,
                strike=leg_order.strike,
                quantity=leg_qty,
                entry_price=leg_order.avg_price,
                trade_id=trade_id,
                order_id=leg_order.order_id,
            )
            setattr(condor, leg_order.leg_role, leg)
            total_premium += leg_order.avg_price * leg_qty * (1.0 if leg_order.side == "SELL" else -1.0)

        condor.total_premium = total_premium
        setattr(condor, "expected_leg_count", len(results))
        setattr(condor, "is_partial_open", len(filled_results) != len(results))
        self.open_condors[group_id] = condor
        return condor, total_premium

    @staticmethod
    def _remaining_leg_qty(leg: LegOrder) -> float:
        return max(float(leg.quantity or 0.0) - float(leg.filled_qty or 0.0), 0.0)

    @staticmethod
    def _serialize_leg_status(
        leg: LegOrder,
        snapshots: dict[str, dict[str, float | bool | str]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "leg_role": leg.leg_role,
            "symbol": leg.symbol,
            "side": leg.side,
            "quantity": leg.quantity,
            "filled_qty": leg.filled_qty,
            "status": leg.status,
            "avg_price": leg.avg_price,
            "attempts": leg.attempts,
        }
        if snapshots is not None:
            snapshot = snapshots.get(leg.symbol) or {}
            payload["spread"] = float(snapshot.get("spread") or 0.0)
            payload["top_qty"] = float(snapshot.get("available_qty") or 0.0)
        return payload

    @staticmethod
    def _fill_ratio(leg_orders: list[LegOrder]) -> float:
        filled = sum(float(leg.filled_qty or 0.0) for leg in leg_orders)
        total = sum(float(leg.quantity or 0.0) for leg in leg_orders)
        return filled / max(total, 1e-9)

    def _serialize_open_request_legs(self, leg_orders: list[LegOrder]) -> list[dict[str, Any]]:
        return [
            {
                "leg_role": leg.leg_role,
                "symbol": leg.symbol,
                "side": leg.side,
                "quantity": leg.quantity,
                "status": leg.status,
            }
            for leg in leg_orders
        ]

    def _serialize_leg_results(self, results: list[LegOrder]) -> list[dict[str, Any]]:
        return [self._serialize_leg_status(result) for result in results]

    def _emit_execution_event(
        self,
        *,
        event_type: str,
        message: str,
        group_id: str = "",
        execution_mode: str = "",
        execution_state: str,
        severity: str = "info",
        underlying: str = "",
        legs: list[dict[str, Any]] | None = None,
        status_callback: Callable[[dict[str, Any]], None] | None = None,
        **payload: Any,
    ) -> None:
        event_underlying = underlying or ""
        if not event_underlying and legs:
            first_symbol = str((legs[0] or {}).get("symbol") or "")
            event_underlying = self._infer_underlying(first_symbol)

        event_payload: dict[str, Any] = {
            "event": event_type,
            "message": message,
            "group_id": group_id,
            "execution_mode": execution_mode,
            "execution_state": execution_state,
            "severity": severity,
            "underlying": event_underlying,
        }
        if legs is not None:
            event_payload["legs"] = legs
        event_payload.update(payload)

        if status_callback is not None:
            status_callback(event_payload)

        meta = {
            key: value
            for key, value in event_payload.items()
            if key not in {
                "event",
                "message",
                "group_id",
                "execution_mode",
                "execution_state",
                "severity",
                "underlying",
                "legs",
            }
        }
        try:
            self.storage.record_execution_event(
                event_type=event_type,
                execution_state=execution_state,
                severity=severity,
                group_id=group_id,
                underlying=event_underlying,
                execution_mode=execution_mode,
                message=message,
                legs=legs,
                meta=meta,
            )
        except Exception as e:
            logger.debug(f"Failed to persist execution event {event_type} for {group_id}: {e}")

    @staticmethod
    def _rollback_failure_message(
        rollback_due_to_margin: bool,
        rollback_failures: list[str],
    ) -> str:
        if rollback_due_to_margin and rollback_failures:
            return "保证金不足导致至少有一条腿未成交，且回滚失败，可能存在残留仓位：" + ", ".join(rollback_failures)
        if rollback_due_to_margin:
            return "保证金不足导致至少有一条腿未成交，已完成回滚"
        if rollback_failures:
            return "至少有一条腿未成交，且回滚失败，可能存在残留仓位：" + ", ".join(rollback_failures)
        return "至少有一条腿未成交，已完成回滚"

    @staticmethod
    def _merge_leg_fill(leg: LegOrder, result: OrderResult) -> None:
        filled_qty = max(float(result.quantity or 0.0), 0.0)
        avg_price = float(result.avg_price or result.price or 0.0)
        fee = max(float(result.fee or 0.0), 0.0)

        prev_qty = float(leg.filled_qty or 0.0)
        add_qty = min(filled_qty, max(float(leg.quantity or 0.0) - prev_qty, 0.0))
        if add_qty > 1e-9 and avg_price > 0:
            total_qty = prev_qty + add_qty
            total_notional = float(leg.avg_price or 0.0) * prev_qty + avg_price * add_qty
            leg.avg_price = total_notional / total_qty
            leg.filled_qty = total_qty
        elif add_qty > 1e-9:
            leg.filled_qty = prev_qty + add_qty

        leg.fee = float(leg.fee or 0.0) + fee
        if result.order_id:
            leg.order_id = result.order_id

        remaining = PositionManager._remaining_leg_qty(leg)
        status = str(result.status or "").upper()
        if remaining <= 1e-9:
            leg.status = "FILLED"
        elif status in {"FAILED", "ERROR", "REJECTED", "EXPIRED", "CANCELED", "CANCELLED"} and filled_qty > 0:
            leg.status = "PARTIALLY_FILLED"
        elif status in {"PARTIALLY_FILLED", "NEW", "PENDING"}:
            leg.status = status
        elif status in {"FAILED", "ERROR", "REJECTED", "EXPIRED", "CANCELED", "CANCELLED"}:
            leg.status = "FAILED"
        elif filled_qty > 0:
            leg.status = "PARTIALLY_FILLED"
        else:
            leg.status = status or leg.status or "PENDING"

    def _get_market_book_snapshot(self, leg: LegOrder) -> dict[str, float | bool | str]:
        get_order_book = getattr(self.client, "get_order_book", None)
        if not get_order_book:
            return {
                "symbol": leg.symbol,
                "valid": False,
                "spread": float("inf"),
                "best_bid": 0.0,
                "best_ask": 0.0,
                "available_qty": 0.0,
            }

        book = get_order_book(leg.symbol, limit=5)
        bids = list(book.get("bids") or [])
        asks = list(book.get("asks") or [])
        best_bid, bid_qty = bids[0] if bids else (0.0, 0.0)
        best_ask, ask_qty = asks[0] if asks else (0.0, 0.0)
        spread = (float(best_ask) - float(best_bid)) if best_bid > 0 and best_ask > 0 else float("inf")
        available_qty = float(ask_qty if str(leg.side).upper() == "BUY" else bid_qty)
        return {
            "symbol": leg.symbol,
            "valid": best_bid > 0 and best_ask > 0,
            "spread": spread,
            "best_bid": float(best_bid),
            "best_ask": float(best_ask),
            "available_qty": available_qty,
        }

    def _wait_until_market_batch_ready(
        self,
        group_id: str,
        legs: list[LegOrder],
        batch_qty: float,
        status_callback: Callable[[dict[str, Any]], None] | None = None,
        max_spread_usd: float | None = None,
        wait_sec: float | None = None,
    ) -> dict[str, dict[str, float | bool | str]] | None:
        def _collect_blockers(
            max_spread_usd: float,
        ) -> tuple[list[dict[str, Any]], dict[str, dict[str, float | bool | str]]]:
            _blockers: list[dict[str, Any]] = []
            _snapshots: dict[str, dict[str, float | bool | str]] = {}
            for leg in legs:
                snap = self._get_market_book_snapshot(leg)
                _snapshots[leg.symbol] = snap
                spread = float(snap.get("spread") or float("inf"))
                available_qty = float(snap.get("available_qty") or 0.0)
                valid = bool(snap.get("valid"))
                if (not valid) or spread > max_spread_usd or available_qty + 1e-9 < required_depth:
                    _blockers.append({
                        "symbol": leg.symbol,
                        "leg_role": leg.leg_role,
                        "spread": spread,
                        "available_qty": available_qty,
                    })
            return _blockers, _snapshots

        required_depth = batch_qty * self.MARKET_BATCH_DEPTH_BUFFER_MULT
        stage_wait_sec = float(wait_sec or self.MARKET_BATCH_STAGE_WAIT_SEC)
        if max_spread_usd is not None:
            spread_stages: list[tuple[float, float | None]] = [
                (float(max_spread_usd), stage_wait_sec),
            ]
        else:
            spread_stages = [
                (float(self.MARKET_BATCH_MAX_SPREAD_USD), stage_wait_sec),
                (float(self.MARKET_BATCH_SECONDARY_MAX_SPREAD_USD), stage_wait_sec),
                (float(self.MARKET_BATCH_RELAXED_MAX_SPREAD_USD), None),
            ]
        stage_index = 0
        current_max_spread, current_wait_sec = spread_stages[stage_index]
        deadline = (
            _time.monotonic() + current_wait_sec
            if current_wait_sec is not None
            else None
        )
        poll_idx = 0

        while True:
            poll_idx += 1
            blockers, snapshots = _collect_blockers(current_max_spread)

            if not blockers:
                return snapshots

            now = _time.monotonic()
            if deadline is not None and now >= deadline:
                if stage_index + 1 < len(spread_stages):
                    prev_max_spread = current_max_spread
                    stage_index += 1
                    current_max_spread, current_wait_sec = spread_stages[stage_index]
                    deadline = (
                        now + current_wait_sec
                        if current_wait_sec is not None
                        else None
                    )
                    logger.warning(
                        f"Market batch {group_id} timed out at spread<={prev_max_spread:.2f}; "
                        f"relaxing to spread<={current_max_spread:.2f} for batch_qty={batch_qty:.4f}"
                    )
                    self._emit_execution_event(
                        event_type="market_batch_spread_relaxed",
                        message=f"等待超时，放宽最大价差到 {current_max_spread:.0f} USD：批次={batch_qty:.4f}",
                        group_id=group_id,
                        execution_mode="market",
                        execution_state="opening",
                        severity="warning",
                        batch_qty=batch_qty,
                        required_depth=required_depth,
                        max_spread_usd=current_max_spread,
                        legs=[self._serialize_leg_status(leg) for leg in legs],
                        status_callback=status_callback,
                    )
                    continue
                self._emit_execution_event(
                    event_type="market_batch_timeout",
                    message=f"批量下单等待超时：批次={batch_qty:.4f}",
                    group_id=group_id,
                    execution_mode="market",
                    execution_state="partial_exposure_timeout" if self._filled_leg_count(legs) > 0 else "opening",
                    severity="warning",
                    batch_qty=batch_qty,
                    required_depth=required_depth,
                    max_spread_usd=current_max_spread,
                    blockers=blockers,
                    legs=[self._serialize_leg_status(leg) for leg in legs],
                    status_callback=status_callback,
                )
                return None

            if status_callback is not None:
                status_callback({
                    "event": "market_batch_wait",
                    "message": f"等待盘口满足批量下单：批次={batch_qty:.4f}",
                    "group_id": group_id,
                    "execution_mode": "market",
                    "batch_qty": batch_qty,
                    "required_depth": required_depth,
                    "max_spread_usd": current_max_spread,
                    "poll_index": poll_idx,
                    "blockers": blockers,
                    "legs": [
                        {
                            "leg_role": leg.leg_role,
                            "symbol": leg.symbol,
                            "side": leg.side,
                            "quantity": leg.quantity,
                            "filled_qty": leg.filled_qty,
                            "status": leg.status,
                            "avg_price": leg.avg_price,
                            "attempts": leg.attempts,
                        }
                        for leg in legs
                    ],
                })
            _time.sleep(self.MARKET_BATCH_POLL_SEC)

    def _refresh_order_result(
        self,
        symbol: str,
        result: OrderResult,
        client_order_id: str | None = None,
    ) -> OrderResult:
        """Refresh submitted order result from exchange when immediate status is ambiguous."""
        status = str(result.status or "").upper()
        if status not in {"NEW", "PENDING", "PARTIALLY_FILLED"}:
            return result
        if not getattr(self.client, "query_order", None):
            return result

        last_error: Exception | None = None
        for attempt in range(1, 5):
            try:
                refreshed = self.client.query_order(
                    symbol,
                    order_id=result.order_id or None,
                    client_order_id=client_order_id or None,
                )
            except Exception as e:
                last_error = e
                logger.debug(
                    f"Order refresh attempt {attempt}/4 skipped for {symbol} orderId={result.order_id}: {e}"
                )
                if attempt < 4:
                    _time.sleep(0.5)
                continue

            if refreshed.order_id:
                result.order_id = refreshed.order_id
            result.quantity = refreshed.quantity or result.quantity
            result.price = refreshed.price or result.price
            result.avg_price = refreshed.avg_price or result.avg_price
            result.status = refreshed.status or result.status
            result.fee = refreshed.fee or result.fee
            result.raw = refreshed.raw or result.raw

            refreshed_status = str(result.status or "").upper()
            if refreshed_status not in {"NEW", "PENDING", "PARTIALLY_FILLED"}:
                break
            if attempt < 4:
                _time.sleep(0.5)

        if last_error is not None and str(result.status or "").upper() in {"NEW", "PENDING", "PARTIALLY_FILLED"}:
            logger.debug(
                f"Order refresh exhausted for {symbol} orderId={result.order_id}; will rely on exchange position reconciliation"
            )
        return result

    def _cancel_leg_order(self, leg: LegOrder) -> bool:
        cancel_fn = getattr(self.client, "cancel_order", None)
        order_id = str(leg.order_id or "").strip() or None
        client_order_id = str(getattr(leg, "client_order_id", "") or "").strip() or None
        if cancel_fn is None or (order_id is None and client_order_id is None):
            return False
        try:
            cancelled = bool(
                cancel_fn(
                    leg.symbol,
                    order_id=order_id,
                    client_order_id=client_order_id,
                )
            )
        except Exception as e:
            logger.debug(
                f"Cancel unresolved order skipped for {leg.symbol} orderId={order_id} clientOrderId={client_order_id}: {e}"
            )
            cancelled = False
        leg.order_id = ""
        leg.client_order_id = ""
        return cancelled

    @staticmethod
    def _expected_position_side_for_leg(leg: LegOrder, reduce_only: bool = False) -> str:
        if reduce_only:
            return "LONG" if str(leg.side).upper() == "SELL" else "SHORT"
        return "SHORT" if str(leg.side).upper() == "SELL" else "LONG"

    def _reconcile_market_results_with_exchange_positions(
        self,
        leg_orders: list[LegOrder],
        reduce_only: bool = False,
        max_polls: int = 3,
        poll_delay_sec: float = 0.25,
        group_id: str = "",
        execution_mode: str = "market",
        status_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Reconcile order results against live exchange positions.

        If live positions still do not match the expected result after a batch,
        cancel any unresolved current order and leave the residual quantity to be
        supplemented by the next batch.
        """
        if not leg_orders or not getattr(self.client, "get_positions", None):
            return

        underlying = self._infer_underlying(leg_orders[0].symbol)
        before_state = {
            leg.symbol: (float(leg.filled_qty or 0.0), str(leg.status or ""))
            for leg in leg_orders
        }
        target_legs = list(leg_orders)
        for poll_idx in range(max_polls):
            try:
                positions = self.client.get_positions(underlying)
            except Exception as e:
                logger.debug(f"Market result reconcile skipped for {underlying}: {e}")
                return

            positions_by_symbol_side = {
                (
                    str(pos.get("symbol") or "").upper(),
                    str(pos.get("side") or "").upper(),
                ): pos
                for pos in positions
                if abs(float(pos.get("quantity") or 0.0)) > 0
            }

            unresolved = []
            for leg in target_legs:
                expected_side = self._expected_position_side_for_leg(leg, reduce_only=reduce_only)
                pos = positions_by_symbol_side.get((str(leg.symbol or "").upper(), expected_side))
                pos_qty = abs(float((pos or {}).get("quantity") or 0.0))
                target_qty = float(leg.quantity or 0.0)

                if reduce_only:
                    remaining_on_exchange = min(pos_qty, target_qty)
                    progress_qty = max(target_qty - remaining_on_exchange, 0.0)
                else:
                    progress_qty = min(pos_qty, target_qty)
                    pos_entry = float((pos or {}).get("entryPrice") or 0.0)
                    if pos_entry > 0:
                        leg.avg_price = pos_entry

                leg.filled_qty = progress_qty
                remaining_qty = self._remaining_leg_qty(leg)
                if remaining_qty <= 1e-9:
                    leg.status = "FILLED"
                    continue

                unresolved.append(leg)

            if not unresolved:
                return
            target_legs = unresolved
            if poll_idx < max_polls - 1:
                _time.sleep(poll_delay_sec)

        for leg in target_legs:
            if self._remaining_leg_qty(leg) <= 1e-9:
                leg.status = "FILLED"
                continue
            self._cancel_leg_order(leg)
            leg.status = "PENDING"

        adjusted_legs = []
        for leg in leg_orders:
            prev_qty, prev_status = before_state.get(leg.symbol, (0.0, ""))
            curr_qty = float(leg.filled_qty or 0.0)
            curr_status = str(leg.status or "")
            if curr_qty > prev_qty + 1e-9 or curr_status != prev_status:
                adjusted_legs.append(self._serialize_leg_status(leg))

        if adjusted_legs:
            remaining_exists = any(self._remaining_leg_qty(leg) > 1e-9 for leg in leg_orders)
            self._emit_execution_event(
                event_type="exchange_reconcile_adjustment",
                message="交易所持仓复核已更新批次成交结果",
                group_id=group_id,
                execution_mode=execution_mode,
                execution_state="partial_exposure" if remaining_exists else "opening",
                severity="warning" if remaining_exists else "info",
                underlying=underlying,
                adjusted_legs=adjusted_legs,
                legs=[self._serialize_leg_status(leg) for leg in leg_orders],
                status_callback=status_callback,
            )

    # ------------------------------------------------------------------
    # Open a new short strangle (2-leg naked sell)
    # ------------------------------------------------------------------

    def open_short_strangle(
        self,
        sell_call_symbol: str,
        sell_put_symbol: str,
        sell_call_strike: float,
        sell_put_strike: float,
        quantity: float,
        underlying_price: float,
        execution_mode: str = "market",
        status_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> IronCondorPosition | None:
        """Open a naked short strangle (2 legs).

        Returns the IronCondorPosition (with buy legs = None) if all legs
        fill, or None on failure.
        """
        group_id = f"SS_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        execution_mode_norm = str(execution_mode or "market").strip().lower()
        if execution_mode_norm not in {"chaser", "market"}:
            raise ValueError(f"Unsupported execution_mode: {execution_mode}")

        logger.info(
            f"Opening Short Strangle {group_id} ({execution_mode_norm}): "
            f"sell_put={sell_put_strike} "
            f"sell_call={sell_call_strike} "
            f"qty={quantity} spot={underlying_price}"
        )

        leg_orders = [
            LegOrder(
                leg_role="sell_put", symbol=sell_put_symbol, side="SELL",
                quantity=quantity, strike=sell_put_strike, option_type="put",
                client_order_prefix=f"{group_id}:sell_put",
            ),
            LegOrder(
                leg_role="sell_call", symbol=sell_call_symbol, side="SELL",
                quantity=quantity, strike=sell_call_strike, option_type="call",
                client_order_prefix=f"{group_id}:sell_call",
            ),
        ]
        underlying = self._infer_underlying(sell_call_symbol or sell_put_symbol)

        self._emit_execution_event(
            event_type="position_open_start",
            message=f"开始提交 Short Strangle {group_id}（{execution_mode_norm}）",
            group_id=group_id,
            execution_mode=execution_mode_norm,
            execution_state="opening",
            underlying=underlying,
            legs=self._serialize_open_request_legs(leg_orders),
            total_legs=len(leg_orders),
            status_callback=status_callback,
        )

        if execution_mode_norm == "market":
            results = self._execute_market_legs(
                group_id=group_id,
                leg_orders=leg_orders,
                quantity=quantity,
                status_callback=status_callback,
            )
            if results is None:
                return None
        else:
            try:
                results = self.chaser.execute_legs(
                    leg_orders,
                    underlying=underlying,
                    status_callback=status_callback,
                )
            except Exception as e:
                logger.error(f"Short Strangle {group_id}: execute_legs exception: {e}")
                return None

        failed = [r for r in results if r.status != "FILLED"]
        if failed:
            logger.error(
                f"Short Strangle {group_id}: {len(failed)} leg(s) failed to fill – "
                + ", ".join(f"{r.leg_role}={r.status}" for r in failed)
            )
            rollback_due_to_margin = self._has_margin_insufficient_failure(failed)
            if execution_mode_norm == "market" and not rollback_due_to_margin:
                filled_results = self._filled_leg_orders(results)
                if filled_results:
                    try:
                        condor, total_premium = self._persist_position_from_results(
                            group_id=group_id,
                            underlying_price=underlying_price,
                            results=results,
                            requested_quantity=quantity,
                        )
                    except Exception as e:
                        logger.error(f"Short Strangle {group_id}: partial persistence failed: {e}")
                        return None

                    if condor is not None:
                        logger.warning(
                            f"Short Strangle {group_id}: partial fill kept live without rollback; "
                            f"filled={len(filled_results)}/{len(results)}"
                        )
                        self._emit_execution_event(
                            event_type="position_open_partial",
                            message="已有成交腿保留，未再执行回滚；剩余腿未完成",
                            group_id=group_id,
                            execution_mode=execution_mode_norm,
                            execution_state="partial_exposure",
                            severity="warning",
                            underlying=underlying,
                            failed_legs=[r.leg_role for r in failed],
                            legs=self._serialize_leg_results(results),
                            status_callback=status_callback,
                        )
                        return condor
            filled_results = [
                (r.leg_role, OrderResult(
                    order_id=r.order_id, symbol=r.symbol, side=r.side,
                    quantity=r.filled_qty, price=r.avg_price,
                    avg_price=r.avg_price, status="FILLED",
                    fee=r.fee, raw={},
                ))
                for r in results if r.status == "FILLED"
            ]
            rollback_failures = self._rollback_legs(filled_results)
            self._emit_execution_event(
                event_type="position_open_failed",
                message=self._rollback_failure_message(rollback_due_to_margin, rollback_failures),
                group_id=group_id,
                execution_mode=execution_mode_norm,
                execution_state="partial_exposure" if rollback_failures else "rolled_back",
                severity="error" if rollback_failures else "warning",
                underlying=underlying,
                failed_legs=[r.leg_role for r in failed],
                abort_reason="margin_insufficient" if rollback_due_to_margin else "submit_failed",
                rollback_failed_legs=rollback_failures,
                legs=self._serialize_leg_results(results),
                status_callback=status_callback,
            )
            return None

        try:
            condor, total_premium = self._persist_position_from_results(
                group_id=group_id,
                underlying_price=underlying_price,
                results=results,
                requested_quantity=quantity,
            )
            if condor is None:
                raise RuntimeError("No filled legs to persist after successful execution")
        except Exception as e:
            logger.error(f"Short Strangle {group_id}: persistence failed after fills: {e}")
            rollback_failures = self._rollback_legs([
                (
                    r.leg_role,
                    OrderResult(
                        order_id=r.order_id,
                        symbol=r.symbol,
                        side=r.side,
                        quantity=r.filled_qty or quantity,
                        price=r.avg_price,
                        avg_price=r.avg_price,
                        status="FILLED",
                        fee=r.fee,
                        raw={},
                    ),
                )
                for r in results if (r.filled_qty or 0) > 0
            ])
            _msg = (
                f"成交后本地记账失败，且回滚失败，可能有残留仓位：{', '.join(rollback_failures)}"
                if rollback_failures
                else f"成交后本地记账失败，已尝试全部回滚：{e}"
            )
            self._emit_execution_event(
                event_type="position_open_error",
                message=_msg,
                group_id=group_id,
                execution_mode=execution_mode_norm,
                execution_state="partial_exposure" if rollback_failures else "rolled_back",
                severity="error",
                underlying=underlying,
                rollback_failed_legs=rollback_failures,
                status_callback=status_callback,
            )
            return None

        logger.info(
            f"Short Strangle {group_id} opened via {execution_mode_norm}. "
            f"Net premium: {total_premium:.4f} USD"
        )

        self._emit_execution_event(
            event_type="position_open_success",
            message=f"Short Strangle {group_id} 已全部成交",
            group_id=group_id,
            execution_mode=execution_mode_norm,
            execution_state="open",
            underlying=underlying,
            net_premium=total_premium,
            legs=self._serialize_leg_results(results),
            status_callback=status_callback,
        )

        return condor

    # ------------------------------------------------------------------
    # Open a new iron condor
    # ------------------------------------------------------------------

    def open_iron_condor(
        self,
        sell_call_symbol: str,
        buy_call_symbol: str,
        sell_put_symbol: str,
        buy_put_symbol: str,
        sell_call_strike: float,
        buy_call_strike: float,
        sell_put_strike: float,
        buy_put_strike: float,
        quantity: float,
        underlying_price: float,
        execution_mode: str = "market",
        status_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> IronCondorPosition | None:
        """Open a full iron condor.

        Default `execution_mode="market"` submits each leg as a market order.
        `execution_mode="chaser"` remains available as an optional adaptive limit chaser.
        """
        group_id = f"IC_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        execution_mode_norm = str(execution_mode or "market").strip().lower()
        if execution_mode_norm not in {"chaser", "market"}:
            raise ValueError(f"Unsupported execution_mode: {execution_mode}")

        logger.info(
            f"Opening Iron Condor {group_id} ({execution_mode_norm}): "
            f"sell_put={sell_put_strike} buy_put={buy_put_strike} "
            f"sell_call={sell_call_strike} buy_call={buy_call_strike} "
            f"qty={quantity} spot={underlying_price}"
        )

        # Build LegOrder list – protection legs first, then short legs
        leg_orders = [
            LegOrder(
                leg_role="buy_put", symbol=buy_put_symbol, side="BUY",
                quantity=quantity, strike=buy_put_strike, option_type="put",
                client_order_prefix=f"{group_id}:buy_put",
            ),
            LegOrder(
                leg_role="buy_call", symbol=buy_call_symbol, side="BUY",
                quantity=quantity, strike=buy_call_strike, option_type="call",
                client_order_prefix=f"{group_id}:buy_call",
            ),
            LegOrder(
                leg_role="sell_put", symbol=sell_put_symbol, side="SELL",
                quantity=quantity, strike=sell_put_strike, option_type="put",
                client_order_prefix=f"{group_id}:sell_put",
            ),
            LegOrder(
                leg_role="sell_call", symbol=sell_call_symbol, side="SELL",
                quantity=quantity, strike=sell_call_strike, option_type="call",
                client_order_prefix=f"{group_id}:sell_call",
            ),
        ]

        underlying = self._infer_underlying(sell_call_symbol or sell_put_symbol)
        self._emit_execution_event(
            event_type="position_open_start",
            message=f"开始提交 Iron Condor {group_id}（{execution_mode_norm}）",
            group_id=group_id,
            execution_mode=execution_mode_norm,
            execution_state="opening",
            underlying=underlying,
            legs=self._serialize_open_request_legs(leg_orders),
            total_legs=len(leg_orders),
            status_callback=status_callback,
        )

        if execution_mode_norm == "market":
            results = self._execute_market_iron_condor(
                group_id=group_id,
                leg_orders=leg_orders,
                quantity=quantity,
                status_callback=status_callback,
            )
            if results is None:
                return None
        else:
            # Execute via limit chaser (blocks up to window_seconds)
            try:
                results = self.chaser.execute_legs(
                    leg_orders,
                    underlying=underlying,
                    status_callback=status_callback,
                )
            except Exception as e:
                logger.error(f"Iron Condor {group_id}: execute_legs exception: {e}")
                self._emit_execution_event(
                    event_type="position_open_error",
                    message=f"追单异常: {e}",
                    group_id=group_id,
                    execution_mode=execution_mode_norm,
                    execution_state="error",
                    severity="error",
                    underlying=underlying,
                    status_callback=status_callback,
                )
                return None

        # Check for failures
        failed = [r for r in results if r.status != "FILLED"]
        if failed:
            logger.error(
                f"Iron Condor {group_id}: {len(failed)} leg(s) failed to fill – "
                + ", ".join(f"{r.leg_role}={r.status}" for r in failed)
            )
            rollback_due_to_margin = self._has_margin_insufficient_failure(failed)
            if execution_mode_norm == "market" and not rollback_due_to_margin:
                filled_results = self._filled_leg_orders(results)
                if filled_results:
                    try:
                        condor, total_premium = self._persist_position_from_results(
                            group_id=group_id,
                            underlying_price=underlying_price,
                            results=results,
                            requested_quantity=quantity,
                        )
                    except Exception as e:
                        logger.error(f"Iron Condor {group_id}: partial persistence failed: {e}")
                        return None

                    if condor is not None:
                        logger.warning(
                            f"Iron Condor {group_id}: partial fill kept live without rollback; "
                            f"filled={len(filled_results)}/{len(results)}"
                        )
                        self._emit_execution_event(
                            event_type="position_open_partial",
                            message="已有成交腿保留，未再执行回滚；剩余腿未完成",
                            group_id=group_id,
                            execution_mode=execution_mode_norm,
                            execution_state="partial_exposure",
                            severity="warning",
                            underlying=underlying,
                            failed_legs=[r.leg_role for r in failed],
                            legs=self._serialize_leg_results(results),
                            status_callback=status_callback,
                        )
                        return condor
            # Rollback filled legs
            filled_results = [
                (r.leg_role, OrderResult(
                    order_id=r.order_id, symbol=r.symbol, side=r.side,
                    quantity=r.filled_qty, price=r.avg_price,
                    avg_price=r.avg_price, status="FILLED",
                    fee=r.fee, raw={},
                ))
                for r in results if r.status == "FILLED"
            ]
            rollback_failures = self._rollback_legs(filled_results)
            self._emit_execution_event(
                event_type="position_open_failed",
                message=self._rollback_failure_message(rollback_due_to_margin, rollback_failures),
                group_id=group_id,
                execution_mode=execution_mode_norm,
                execution_state="partial_exposure" if rollback_failures else "rolled_back",
                severity="error" if rollback_failures else "warning",
                underlying=underlying,
                failed_legs=[r.leg_role for r in failed],
                abort_reason="margin_insufficient" if rollback_due_to_margin else "submit_failed",
                rollback_failed_legs=rollback_failures,
                legs=self._serialize_leg_results(results),
                status_callback=status_callback,
            )
            return None

        try:
            condor, total_premium = self._persist_position_from_results(
                group_id=group_id,
                underlying_price=underlying_price,
                results=results,
                requested_quantity=quantity,
            )
            if condor is None:
                raise RuntimeError("No filled legs to persist after successful execution")
        except Exception as e:
            logger.error(f"Iron Condor {group_id}: persistence failed after fills: {e}")
            rollback_failures = self._rollback_legs([
                (
                    r.leg_role,
                    OrderResult(
                        order_id=r.order_id,
                        symbol=r.symbol,
                        side=r.side,
                        quantity=r.filled_qty or quantity,
                        price=r.avg_price,
                        avg_price=r.avg_price,
                        status="FILLED",
                        fee=r.fee,
                        raw={},
                    ),
                )
                for r in results if (r.filled_qty or 0) > 0
            ])
            _msg = (
                f"成交后本地记账失败，且回滚失败，可能有残留仓位：{', '.join(rollback_failures)}"
                if rollback_failures
                else f"成交后本地记账失败，已尝试全部回滚：{e}"
            )
            self._emit_execution_event(
                event_type="position_open_error",
                message=_msg,
                group_id=group_id,
                execution_mode=execution_mode_norm,
                execution_state="partial_exposure" if rollback_failures else "rolled_back",
                severity="error",
                underlying=underlying,
                rollback_failed_legs=rollback_failures,
                status_callback=status_callback,
            )
            return None

        logger.info(
            f"Iron Condor {group_id} opened via {execution_mode_norm}. "
            f"Net premium: {total_premium:.4f} USD"
        )

        self._emit_execution_event(
            event_type="position_open_success",
            message=f"Iron Condor {group_id} 已全部成交",
            group_id=group_id,
            execution_mode=execution_mode_norm,
            execution_state="open",
            underlying=underlying,
            net_premium=total_premium,
            legs=[self._serialize_leg_status(r) for r in results],
            status_callback=status_callback,
        )

        return condor

    def _execute_market_legs(
        self,
        group_id: str,
        leg_orders: list[LegOrder],
        quantity: float,
        status_callback: Callable[[dict[str, Any]], None] | None = None,
        reduce_only: bool = False,
        client_order_tag: str = "mkt",
    ) -> list[LegOrder] | None:
        """Submit market legs in synchronized <=1 BTC batches after spread/depth checks."""
        if not leg_orders:
            return []

        total_legs = len(leg_orders)
        batch_index = 0
        completion_deadline: float | None = None
        for leg in leg_orders:
            leg.status = leg.status or "PENDING"
            leg.filled_qty = float(leg.filled_qty or 0.0)
            leg.avg_price = float(leg.avg_price or 0.0)
            leg.fee = float(leg.fee or 0.0)
            leg.attempts = int(leg.attempts or 0)

        while True:
            pending = [leg for leg in leg_orders if self._remaining_leg_qty(leg) > 1e-9]
            if not pending:
                break

            partial_mode = self._filled_leg_count(leg_orders) > 0
            if partial_mode and completion_deadline is None:
                completion_deadline = _time.monotonic() + self.MARKET_PARTIAL_RETRY_WINDOW_SEC

            batch_index += 1
            batch_qty = min(
                self.MARKET_BATCH_MAX_QTY,
                min(self._remaining_leg_qty(leg) for leg in pending),
            )
            snapshots = self._wait_until_market_batch_ready(
                group_id=group_id,
                legs=pending,
                batch_qty=batch_qty,
                status_callback=status_callback,
                max_spread_usd=self.MARKET_BATCH_RELAXED_MAX_SPREAD_USD if partial_mode else None,
            )
            if snapshots is None:
                if partial_mode and completion_deadline is not None and _time.monotonic() < completion_deadline:
                    self._emit_execution_event(
                        event_type="market_batch_retry_pending",
                        message=f"已有单腿成交，继续放宽条件补齐剩余腿：批次={batch_qty:.4f}",
                        group_id=group_id,
                        execution_mode="market",
                        execution_state="partial_exposure",
                        severity="warning",
                        batch_index=batch_index,
                        batch_qty=batch_qty,
                        max_spread_usd=self.MARKET_BATCH_RELAXED_MAX_SPREAD_USD,
                        retry_window_sec=self.MARKET_PARTIAL_RETRY_WINDOW_SEC,
                        legs=[self._serialize_leg_status(leg) for leg in leg_orders],
                        status_callback=status_callback,
                    )
                    continue
                for leg in pending:
                    if self._remaining_leg_qty(leg) > 1e-9:
                        leg.status = "FAILED"
                break

            self._emit_execution_event(
                event_type="market_batch_submit",
                message=f"第 {batch_index} 批开始同步下单，批次={batch_qty:.4f}",
                group_id=group_id,
                execution_mode="market",
                execution_state="partial_exposure" if partial_mode else "opening",
                batch_index=batch_index,
                batch_qty=batch_qty,
                filled_legs=sum(1 for leg in leg_orders if self._remaining_leg_qty(leg) <= 1e-9),
                total_legs=total_legs,
                fill_ratio=self._fill_ratio(leg_orders),
                legs=[self._serialize_leg_status(leg, snapshots=snapshots) for leg in leg_orders],
                status_callback=status_callback,
            )

            fatal_error: tuple[LegOrder, Exception] | None = None
            for leg in pending:
                client_order_id = f"{leg.client_order_prefix}:{client_order_tag}:{batch_index:02d}"
                try:
                    leg.attempts += 1
                    leg.client_order_id = client_order_id
                    order_result = self._submit_order(
                        symbol=leg.symbol,
                        side=leg.side,
                        quantity=batch_qty,
                        order_type="MARKET",
                        reduce_only=reduce_only,
                        client_order_id=client_order_id,
                    )
                    order_result = self._refresh_order_result(
                        leg.symbol,
                        order_result,
                        client_order_id=client_order_id,
                    )
                    self._merge_leg_fill(leg, order_result)
                except Exception as e:
                    self._set_leg_last_error(leg, e)
                    logger.error(f"Position {group_id}: batched market order failed for {leg.leg_role}: {e}")
                    partial_fill_started = partial_mode or self._filled_leg_count(leg_orders) > 0
                    if self._is_margin_insufficient_error(e):
                        setattr(leg, "abort_reason", "margin_insufficient")
                        leg.status = "FAILED"
                        fatal_error = (leg, e)
                        break
                    if partial_fill_started:
                        if completion_deadline is None:
                            completion_deadline = _time.monotonic() + self.MARKET_PARTIAL_RETRY_WINDOW_SEC
                        leg.status = "PENDING"
                        continue
                    leg.status = "FAILED"
                    fatal_error = (leg, e)
                    break

            self._reconcile_market_results_with_exchange_positions(
                leg_orders,
                reduce_only=reduce_only,
                group_id=group_id,
                execution_mode="market",
                status_callback=status_callback,
            )

            filled_count = sum(1 for leg in leg_orders if self._remaining_leg_qty(leg) <= 1e-9)
            self._emit_execution_event(
                event_type="market_batch_result",
                message=f"第 {batch_index} 批完成",
                group_id=group_id,
                execution_mode="market",
                execution_state=(
                    "open"
                    if filled_count == total_legs
                    else ("partial_exposure" if filled_count > 0 else "opening")
                ),
                severity="warning" if 0 < filled_count < total_legs else "info",
                batch_index=batch_index,
                batch_qty=batch_qty,
                filled_legs=filled_count,
                total_legs=total_legs,
                fill_ratio=self._fill_ratio(leg_orders),
                legs=[self._serialize_leg_status(leg) for leg in leg_orders],
                status_callback=status_callback,
            )

            if fatal_error is not None:
                failed_leg, exc = fatal_error
                partial_fill_started = partial_mode or self._filled_leg_count(leg_orders) > 0
                self._emit_execution_event(
                    event_type="position_open_error",
                    message=f"批量市价单提交失败: {failed_leg.leg_role} | {exc}",
                    group_id=group_id,
                    execution_mode="market",
                    execution_state="partial_exposure" if partial_fill_started else "error",
                    severity="error",
                    failed_leg=failed_leg.leg_role,
                    abort_reason=getattr(failed_leg, "abort_reason", "submit_failed"),
                    legs=[self._serialize_leg_status(leg) for leg in leg_orders],
                    status_callback=status_callback,
                )
                if partial_fill_started and getattr(failed_leg, "abort_reason", "") != "margin_insufficient":
                    if completion_deadline is not None and _time.monotonic() < completion_deadline:
                        failed_leg.status = "PENDING"
                        continue
                break

            if any(str(leg.status or "").upper() == "FAILED" for leg in leg_orders):
                partial_fill_started = partial_mode or self._filled_leg_count(leg_orders) > 0
                if partial_fill_started and completion_deadline is not None and _time.monotonic() < completion_deadline:
                    for leg in leg_orders:
                        if self._remaining_leg_qty(leg) > 1e-9 and str(leg.status or "").upper() == "FAILED":
                            leg.status = "PENDING"
                    continue
                break

        for leg in leg_orders:
            if self._remaining_leg_qty(leg) <= 1e-9:
                leg.status = "FILLED"
            elif str(leg.status or "").upper() not in {"FAILED", "ERROR"}:
                leg.status = "FAILED"

        return leg_orders

    def _execute_market_iron_condor(
        self,
        group_id: str,
        leg_orders: list[LegOrder],
        quantity: float,
        status_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[LegOrder] | None:
        """Backward-compatible wrapper for iron condor market execution."""
        return self._execute_market_legs(group_id, leg_orders, quantity, status_callback)

    def _rollback_legs(
        self,
        filled_legs: list[tuple[str, OrderResult]],
    ) -> list[str]:
        """Attempt to close already-filled legs on partial failure.

        Returns the leg roles that failed to rollback.
        """
        logger.warning(f"Rolling back {len(filled_legs)} filled leg(s)")
        failed_rollbacks: list[str] = []
        for leg_role, result in filled_legs:
            try:
                close_side = "SELL" if result.side == "BUY" else "BUY"
                self._submit_order(
                    symbol=result.symbol,
                    side=close_side,
                    quantity=result.quantity,
                    order_type="MARKET",
                    reduce_only=True,
                )
                logger.info(f"Rolled back {leg_role}")
            except Exception as e:
                logger.error(f"Failed to rollback {leg_role}: {e}")
                failed_rollbacks.append(leg_role)
        return failed_rollbacks

    # ------------------------------------------------------------------
    # Close an iron condor
    # ------------------------------------------------------------------

    def close_iron_condor(self, group_id: str, reason: str = "", execution_mode: str = "market") -> float:
        """Close all legs of an iron condor.

        Returns total realized PnL.
        """
        condor = self.open_condors.get(group_id)
        if not condor or not condor.is_open:
            logger.warning(f"Condor {group_id} not found or already closed")
            return 0.0

        execution_mode_norm = str(execution_mode or "market").strip().lower()
        if execution_mode_norm not in {"chaser", "market"}:
            raise ValueError(f"Unsupported close execution_mode: {execution_mode}")

        logger.info(
            f"Closing Iron Condor {group_id}, reason: {reason}, mode: {execution_mode_norm}"
        )

        # Build close legs
        close_legs: list[LegOrder] = []
        for leg in condor.legs:
            close_side = "SELL" if leg.side == "BUY" else "BUY"
            close_legs.append(LegOrder(
                leg_role=f"close_{leg.option_type}_{leg.side.lower()}",
                symbol=leg.symbol,
                side=close_side,
                quantity=leg.quantity,
                strike=leg.strike,
                option_type=leg.option_type,
            ))

        # Execute close via limit chaser or market orders
        underlying = self._infer_underlying(condor.legs[0].symbol) if condor.legs else "ETH"
        if execution_mode_norm == "market":
            for close_leg in close_legs:
                close_leg.client_order_prefix = f"{group_id}:{close_leg.leg_role}"
            results = self._execute_market_legs(
                group_id=group_id,
                leg_orders=close_legs,
                quantity=0.0,
                status_callback=None,
                reduce_only=True,
                client_order_tag="mkt_close",
            ) or close_legs
        else:
            results = self.chaser.execute_legs(close_legs, underlying=underlying)

        total_pnl = 0.0
        for leg, close_result in zip(condor.legs, results):
            if close_result.status != "FILLED":
                logger.error(
                    f"Failed to close {leg.symbol}: {close_result.status}"
                )
                continue

            # Compute PnL for this leg
            if leg.side == "SELL":
                leg_pnl = (leg.entry_price - close_result.avg_price) * leg.quantity
            else:
                leg_pnl = (close_result.avg_price - leg.entry_price) * leg.quantity

            leg_pnl -= close_result.fee
            total_pnl += leg_pnl

            # Update storage
            self.storage.close_trade(
                trade_id=leg.trade_id,
                close_price=close_result.avg_price,
                pnl=leg_pnl,
            )

            logger.info(
                f"  Closed {leg.side} {leg.option_type} K={leg.strike}: "
                f"entry={leg.entry_price:.4f} close={close_result.avg_price:.4f} "
                f"pnl={leg_pnl:.4f}"
            )

        condor.is_open = False
        del self.open_condors[group_id]

        logger.info(
            f"Iron Condor {group_id} closed. "
            f"Total PnL: {total_pnl:.4f} USD. Reason: {reason}"
        )

        return total_pnl

    def close_all(self, reason: str = "close_all", execution_mode: str = "market") -> float:
        """Close all open iron condors."""
        total = 0.0
        group_ids = list(dict.fromkeys([
            *self.open_condors.keys(),
            *self.storage.get_open_trade_groups(),
        ]))
        for gid in group_ids:
            self._ensure_group_loaded(gid)
            total += self.close_iron_condor(gid, reason=reason, execution_mode=execution_mode)
        return total

    def close_all_exchange_positions(self, underlying: str = "", reason: str = "close_all_exchange") -> float:
        """Emergency close using live exchange positions only.

        This path ignores local tracking for order discovery and closes whatever
        Binance currently reports as open option positions for the target underlying.
        After exchange close, local storage and in-memory groups are best-effort synced.
        """
        if not getattr(self.client, "get_positions", None):
            logger.warning("Exchange-driven close-all unavailable: client has no get_positions()")
            return 0.0

        positions = self.client.get_positions(underlying) if underlying else self.client.get_positions()
        if not positions:
            logger.info(
                f"Exchange-driven close-all skipped: no live option positions for {underlying or 'ALL'}"
            )
            return 0.0

        open_trades = self.storage.get_open_trades()
        open_trades_by_symbol: dict[str, list[dict[str, Any]]] = {}
        for trade in open_trades:
            open_trades_by_symbol.setdefault(str(trade.get("symbol") or ""), []).append(trade)

        total_pnl = 0.0
        affected_groups: set[str] = set()

        attempted_symbols: set[str] = set()
        symbol_close_prices: dict[str, float] = {}
        max_passes = 3

        for attempt in range(1, max_passes + 1):
            positions = self.client.get_positions(underlying) if underlying else self.client.get_positions()
            if not positions:
                break

            close_legs: list[LegOrder] = []
            for pos in positions:
                symbol = str(pos.get("symbol") or "")
                side = str(pos.get("side") or "").upper()
                qty = abs(float(pos.get("quantity") or 0.0))
                if not symbol or qty <= 0:
                    continue
                attempted_symbols.add(symbol)
                close_side = "SELL" if side == "LONG" else "BUY"
                option_type = "call" if symbol.upper().endswith("-C") else "put"
                close_legs.append(LegOrder(
                    leg_role=f"closeall_{attempt}_{symbol}",
                    symbol=symbol,
                    side=close_side,
                    quantity=qty,
                    strike=0.0,
                    option_type=option_type,
                    client_order_prefix=f"closeall:{reason}:{attempt}:{symbol}",
                ))
                logger.warning(
                    f"Exchange-driven close-all: pass={attempt}/{max_passes} queued {symbol} "
                    f"side={side} qty={qty:.4f} reason={reason}"
                )

            if not close_legs:
                break

            results = self._execute_market_legs(
                group_id=f"closeall:{reason}:{attempt}",
                leg_orders=close_legs,
                quantity=0.0,
                status_callback=None,
                reduce_only=True,
                client_order_tag="mkt_closeall",
            ) or close_legs

            for close_leg in results:
                close_price = float(close_leg.avg_price or 0.0)
                if close_price > 0:
                    symbol_close_prices[close_leg.symbol] = close_price

            if attempt < max_passes:
                _time.sleep(0.35)

        remaining_positions = self.client.get_positions(underlying) if underlying else self.client.get_positions()
        remaining_symbols = {
            str(pos.get("symbol") or "")
            for pos in remaining_positions
            if abs(float(pos.get("quantity") or 0.0)) > 0
        }
        closed_symbols = attempted_symbols - remaining_symbols

        for symbol in sorted(closed_symbols):
            close_price = float(symbol_close_prices.get(symbol) or 0.0)
            if close_price <= 0:
                logger.warning(
                    f"Exchange-driven close verified {symbol} is no longer open, but no close price was captured; local pnl sync skipped"
                )
                continue

            symbol_pnl = 0.0
            for trade in open_trades_by_symbol.get(symbol, []):
                trade_qty = float(trade.get("quantity") or 0.0)
                trade_side = str(trade.get("side") or "").upper()
                entry_price = float(trade.get("price") or 0.0)
                fee = float(trade.get("fee") or 0.0)
                if trade_side == "SELL":
                    leg_pnl = (entry_price - close_price) * trade_qty
                else:
                    leg_pnl = (close_price - entry_price) * trade_qty
                leg_pnl -= fee
                self.storage.close_trade(
                    trade_id=int(trade["id"]),
                    close_price=close_price,
                    pnl=leg_pnl,
                )
                symbol_pnl += leg_pnl
                affected_groups.add(str(trade.get("trade_group") or ""))

            total_pnl += symbol_pnl
            logger.info(
                f"Exchange-driven close verified for {symbol}: close={close_price:.4f}, synced_pnl={symbol_pnl:.4f}"
            )

        if remaining_symbols:
            logger.warning(
                f"Exchange-driven close-all incomplete after {max_passes} pass(es): remaining={sorted(remaining_symbols)}"
            )

        for gid in list(affected_groups):
            condor = self.open_condors.get(gid)
            if condor is not None:
                condor.is_open = False
                del self.open_condors[gid]

        return total_pnl

    # ------------------------------------------------------------------
    # Position monitoring
    # ------------------------------------------------------------------

    def get_unrealized_pnl(self, mark_prices: dict[str, float]) -> float:
        """Compute total unrealized PnL across all open condors.

        Parameters
        ----------
        mark_prices : dict of symbol -> current mark price.
        """
        total_upnl = 0.0
        for condor in self.open_condors.values():
            for leg in condor.legs:
                mark = mark_prices.get(leg.symbol, leg.entry_price)
                if leg.side == "SELL":
                    leg_upnl = (leg.entry_price - mark) * leg.quantity
                else:
                    leg_upnl = (mark - leg.entry_price) * leg.quantity
                total_upnl += leg_upnl
        return total_upnl

    @property
    def open_position_count(self) -> int:
        return len(self.open_condors)

    def summary(self, mark_prices: dict[str, float] | None = None) -> dict:
        """Return summary of current position state."""
        upnl = self.get_unrealized_pnl(mark_prices or {})
        return {
            "open_condors": self.open_position_count,
            "unrealized_pnl": upnl,
            "condors": {
                gid: {
                    "entry_time": c.entry_time.isoformat(),
                    "underlying_price": c.underlying_price,
                    "total_premium": c.total_premium,
                    "legs": len(c.legs),
                }
                for gid, c in self.open_condors.items()
            },
        }
