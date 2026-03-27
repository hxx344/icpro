"""Limit-order chaser – 限价追单引擎.

Manages limit orders that gradually drift toward market price over a
configurable time window to maximise fill probability while minimising
slippage.

Strategy:
  - SELL legs: start at (ask - tick), drift toward bid over time
  - BUY  legs: start at (bid + tick), drift toward ask over time
  - At deadline: switch to market order to guarantee fill

Time-based price interpolation:
  elapsed_ratio = elapsed / window_seconds        (0.0 → 1.0)
  aggression    = elapsed_ratio ^ 2               (slow start, fast finish)

  SELL: price = ask - tick × (1 - aggression) × spread_ticks
        i.e. starts near ask, ends at bid
  BUY:  price = bid + tick × (1 - aggression) × spread_ticks
        i.e. starts near bid, ends at ask
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from loguru import logger

from trader.binance_client import OrderNotFoundError, OrderResult, OptionTicker


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class LegOrder:
    """Tracks a single leg's limit order through its lifecycle."""
    leg_role: str       # "buy_put", "buy_call", "sell_put", "sell_call"
    symbol: str
    side: str           # "BUY" or "SELL"
    quantity: float
    strike: float
    option_type: str    # "call" or "put"

    # Current order state
    order_id: str = ""
    client_order_id: str = ""
    client_order_prefix: str = ""
    current_price: float = 0.0
    status: str = "PENDING"     # PENDING / NEW / PARTIALLY_FILLED / FILLED / FAILED
    avg_price: float = 0.0
    fee: float = 0.0
    filled_qty: float = 0.0      # cumulative filled quantity across all child orders
    current_order_filled_qty: float = 0.0
    current_order_avg_price: float = 0.0
    current_order_fee: float = 0.0

    # Tracking
    attempts: int = 0
    start_time: float = 0.0     # time.monotonic()


@dataclass
class ChaserConfig:
    """Configuration for the limit-order chaser."""
    window_seconds: int = 1800          # 30 minutes total window
    poll_interval_sec: int = 60         # check / amend every 60 seconds
    tick_size_usdt: float = 5.0         # min price increment in USD
    market_fallback_sec: int = 60       # switch to market order last N seconds
    max_amend_attempts: int = 180       # safety cap on re-pricing loops


# ---------------------------------------------------------------------------
# Limit Chaser Engine
# ---------------------------------------------------------------------------

class LimitChaser:
    """Execute multi-leg limit orders with time-based price drift.

    Usage:
        chaser = LimitChaser(client, config)
        results = chaser.execute_legs(legs)
        # results: list[LegOrder] – each with status FILLED or FAILED
    """

    def __init__(
        self,
        client: Any,
        config: Optional[ChaserConfig] = None,
    ):
        self.client = client
        self.cfg = config or ChaserConfig()
        self._quote_cache: dict[str, tuple[OptionTicker, float]] = {}
        self._quote_cache_ttl_sec = min(max(self.cfg.poll_interval_sec, 1), 5)
        self._market_fallback_steps = 3

    def _legs_snapshot(self, legs: list[LegOrder]) -> list[dict[str, Any]]:
        return [
            {
                "leg_role": leg.leg_role,
                "symbol": leg.symbol,
                "side": leg.side,
                "quantity": leg.quantity,
                "filled_qty": leg.filled_qty,
                "remaining_qty": max(leg.quantity - leg.filled_qty, 0.0),
                "status": leg.status,
                "attempts": leg.attempts,
                "current_price": leg.current_price,
                "avg_price": leg.avg_price,
                "order_id": leg.order_id,
                "client_order_id": leg.client_order_id,
            }
            for leg in legs
        ]

    def _emit_progress(
        self,
        status_callback: Optional[Callable[[dict[str, Any]], None]],
        event: str,
        legs: list[LegOrder],
        **extra: Any,
    ) -> None:
        if status_callback is None:
            return

        total_qty = sum(max(float(leg.quantity or 0.0), 0.0) for leg in legs)
        filled_qty = sum(min(max(float(leg.filled_qty or 0.0), 0.0), max(float(leg.quantity or 0.0), 0.0)) for leg in legs)
        fill_ratio = (filled_qty / total_qty) if total_qty > 0 else 0.0
        payload = {
            "event": event,
            "legs": self._legs_snapshot(legs),
            "filled_legs": sum(1 for leg in legs if leg.status == "FILLED"),
            "failed_legs": sum(1 for leg in legs if leg.status == "FAILED"),
            "done_legs": sum(1 for leg in legs if leg.status in ("FILLED", "FAILED")),
            "total_legs": len(legs),
            "fill_ratio": fill_ratio,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        payload.update(extra)
        try:
            status_callback(payload)
        except Exception as e:
            logger.debug(f"[Chaser] progress callback failed: {e}")

    def _remaining_qty(self, leg: LegOrder) -> float:
        """Return the quantity still needing execution."""
        remaining = leg.quantity - leg.filled_qty
        if remaining <= 1e-9:
            return 0.0
        return remaining

    def _record_fill_progress(self, leg: LegOrder, result: OrderResult) -> None:
        """Accumulate fills reported for the current child order."""
        executed_qty = max(float(result.quantity or 0.0), 0.0)
        prev_qty = leg.current_order_filled_qty
        if executed_qty < prev_qty:
            executed_qty = prev_qty

        delta_qty = executed_qty - prev_qty
        if delta_qty > 1e-9:
            prev_notional = leg.current_order_avg_price * prev_qty
            new_notional = float(result.avg_price or 0.0) * executed_qty
            delta_notional = max(new_notional - prev_notional, 0.0)

            total_qty = leg.filled_qty + delta_qty
            if total_qty > 1e-9:
                total_notional = leg.avg_price * leg.filled_qty + delta_notional
                leg.avg_price = total_notional / total_qty
            leg.filled_qty = total_qty

        reported_fee = max(float(result.fee or 0.0), 0.0)
        delta_fee = reported_fee - leg.current_order_fee
        if delta_fee > 1e-9:
            leg.fee += delta_fee

        leg.current_order_filled_qty = executed_qty
        leg.current_order_avg_price = float(result.avg_price or leg.current_order_avg_price or 0.0)
        leg.current_order_fee = reported_fee

    def _start_new_child_order(self, leg: LegOrder) -> None:
        """Reset per-order fill trackers before placing a replacement order."""
        leg.current_order_filled_qty = 0.0
        leg.current_order_avg_price = 0.0
        leg.current_order_fee = 0.0

    def _mark_filled_if_done(self, leg: LegOrder) -> bool:
        """Mark the leg filled when the target quantity has been reached."""
        if self._remaining_qty(leg) <= 1e-9:
            leg.status = "FILLED"
            return True
        return False

    def _make_client_order_id(self, leg: LegOrder, intent: str) -> str:
        """Build a deterministic short client order id for idempotent recovery."""
        base = leg.client_order_prefix or (
            f"{leg.leg_role}|{leg.symbol}|{leg.side}|{leg.strike}|{leg.quantity}|{leg.start_time:.6f}"
        )
        digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12].upper()
        seq = leg.attempts + 1
        return f"LC{intent}{digest}{seq:03d}"[:32]

    def _adopt_submitted_order(
        self,
        leg: LegOrder,
        result: OrderResult,
        price: float,
        client_order_id: str,
    ) -> None:
        """Apply a successfully accepted order result to the leg state."""
        leg.order_id = result.order_id
        leg.client_order_id = client_order_id
        leg.current_price = price

        if result.status == "FILLED":
            self._record_fill_progress(leg, result)
            leg.status = "FILLED"
            logger.info(
                f"[Chaser] {leg.leg_role} {leg.symbol} FILLED immediately "
                f"at {result.avg_price:.4f}"
            )
        elif result.status == "PARTIALLY_FILLED":
            self._record_fill_progress(leg, result)
            leg.status = "PARTIALLY_FILLED"
            logger.info(
                f"[Chaser] {leg.leg_role} {leg.symbol} partially filled "
                f"{leg.filled_qty:.4f}/{leg.quantity:.4f} @ {result.avg_price:.4f}"
            )
        else:
            leg.status = "NEW"
            logger.info(
                f"[Chaser] {leg.leg_role} {leg.symbol} limit {leg.side} "
                f"@ {price:.4f} for {self._remaining_qty(leg):.4f} placed "
                f"(attempt #{leg.attempts})"
            )

    def _recover_submit_error(
        self,
        leg: LegOrder,
        price: float,
        client_order_id: str,
    ) -> bool:
        """After a submit error, try to recover the order by client id."""
        try:
            result = self.client.query_order(
                leg.symbol,
                client_order_id=client_order_id,
            )
        except Exception:
            return False

        self._adopt_submitted_order(leg, result, price, client_order_id)
        return True

    def execute_legs(
        self,
        legs: list[LegOrder],
        underlying: str = "ETH",
        status_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> list[LegOrder]:
        """Execute all legs concurrently with adaptive limit orders.

        Legs are processed in parallel: all limits placed simultaneously,
        then a polling loop amends unfilled orders until all fill or timeout.

        Parameters
        ----------
        legs : list of LegOrder to execute
        underlying : underlying symbol for ticker lookups

        Returns
        -------
        List of LegOrder with updated status / fill info.
        """
        if not legs:
            return legs

        t0 = time.monotonic()
        for leg in legs:
            leg.start_time = t0
            leg.status = "PENDING"

        self._emit_progress(
            status_callback,
            "started",
            legs,
            message=f"开始追单，共 {len(legs)} 条腿",
            elapsed_sec=0.0,
            remaining_sec=self.cfg.window_seconds,
        )

        # Phase 1: Place initial limit orders for all legs
        self._place_initial_orders(legs, underlying, status_callback=status_callback)

        # Phase 2: Poll + amend loop
        self._chase_loop(legs, underlying, t0, status_callback=status_callback)

        self._emit_progress(
            status_callback,
            "finished",
            legs,
            message="追单结束",
            elapsed_sec=max(time.monotonic() - t0, 0.0),
            remaining_sec=0.0,
        )

        return legs

    # ------------------------------------------------------------------
    # Phase 1: Initial placement
    # ------------------------------------------------------------------

    def _place_initial_orders(
        self,
        legs: list[LegOrder],
        underlying: str,
        status_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> None:
        """Place the first limit order for each leg."""
        # Fetch all tickers once
        quotes = self._get_quotes(legs)

        for leg in legs:
            q = quotes.get(leg.symbol)
            if not q or q.bid_price <= 0 or q.ask_price <= 0:
                logger.warning(
                    f"[Chaser] No valid quote for {leg.symbol}, "
                    f"using market order fallback"
                )
                self._market_fallback(leg)
                self._emit_progress(
                    status_callback,
                    "market_fallback",
                    legs,
                    message=f"{leg.leg_role} 无有效报价，切换到兜底成交",
                    leg_role=leg.leg_role,
                    symbol=leg.symbol,
                )
                continue

            price = self._compute_limit_price(leg, q, elapsed_ratio=0.0)
            self._place_or_amend(leg, price)
            self._emit_progress(
                status_callback,
                "initial_order",
                legs,
                message=f"{leg.leg_role} 已提交初始委托",
                leg_role=leg.leg_role,
                symbol=leg.symbol,
                limit_price=price,
            )

    # ------------------------------------------------------------------
    # Phase 2: Chase loop
    # ------------------------------------------------------------------

    def _chase_loop(
        self,
        legs: list[LegOrder],
        underlying: str,
        t0: float,
        status_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> None:
        """Poll unfilled orders and amend prices until all fill or timeout."""
        deadline = t0 + self.cfg.window_seconds
        market_time = t0 + self.cfg.window_seconds - self.cfg.market_fallback_sec
        iteration = 0

        while True:
            iteration += 1
            # Check if all legs are done
            pending = [l for l in legs if l.status not in ("FILLED", "FAILED")]
            if not pending:
                logger.info("[Chaser] All legs filled")
                break

            now = time.monotonic()
            elapsed = now - t0

            # Past deadline – should not happen but safety check
            if now > deadline + 5:
                logger.warning("[Chaser] Past deadline, forcing market fills")
                for leg in pending:
                    self._cancel_and_market(leg)
                self._emit_progress(
                    status_callback,
                    "deadline_forced_market",
                    legs,
                    message="超过追单截止时间，强制进入兜底成交",
                    elapsed_sec=elapsed,
                    remaining_sec=max(deadline - now, 0.0),
                    iteration=iteration,
                )
                break

            # Check order statuses
            for leg in pending:
                self._check_fill(leg)

            self._emit_progress(
                status_callback,
                "poll",
                legs,
                message=f"第 {iteration} 轮检查：{sum(1 for leg in legs if leg.status == 'FILLED')}/{len(legs)} 条腿已成交",
                elapsed_sec=elapsed,
                remaining_sec=max(deadline - now, 0.0),
                iteration=iteration,
            )

            # Re-check pending after status updates
            pending = [l for l in legs if l.status not in ("FILLED", "FAILED")]
            if not pending:
                break

            # Market fallback zone: last N seconds
            if now >= market_time:
                logger.info(
                    f"[Chaser] Entering market fallback zone "
                    f"({self.cfg.market_fallback_sec}s remaining)"
                )
                for leg in pending:
                    self._cancel_and_market(leg)
                self._emit_progress(
                    status_callback,
                    "market_fallback_zone",
                    legs,
                    message=f"进入最后 {self.cfg.market_fallback_sec} 秒兜底成交区间",
                    elapsed_sec=elapsed,
                    remaining_sec=max(deadline - now, 0.0),
                    iteration=iteration,
                )
                break

            # Amend prices based on elapsed time
            elapsed_ratio = min(elapsed / (self.cfg.window_seconds - self.cfg.market_fallback_sec), 1.0)
            quotes = self._get_quotes(pending)

            for leg in pending:
                if leg.status == "FILLED":
                    continue
                if leg.attempts >= self.cfg.max_amend_attempts:
                    logger.warning(f"[Chaser] Max attempts for {leg.symbol}, market fill")
                    self._cancel_and_market(leg)
                    continue

                q = quotes.get(leg.symbol)
                if not q or q.bid_price <= 0 or q.ask_price <= 0:
                    continue

                new_price = self._compute_limit_price(leg, q, elapsed_ratio)

                # Only amend if price changed meaningfully
                if abs(new_price - leg.current_price) >= self.cfg.tick_size_usdt * 0.5:
                    self._cancel_and_replace(leg, new_price)

            self._emit_progress(
                status_callback,
                "amend_cycle",
                legs,
                message=f"第 {iteration} 轮改价完成，等待下一次轮询",
                elapsed_sec=elapsed,
                remaining_sec=max(deadline - now, 0.0),
                iteration=iteration,
            )

            # Sleep before next poll
            time.sleep(self.cfg.poll_interval_sec)

    # ------------------------------------------------------------------
    # Price computation
    # ------------------------------------------------------------------

    def _compute_limit_price(
        self,
        leg: LegOrder,
        quote: OptionTicker,
        elapsed_ratio: float,
    ) -> float:
        """Compute limit price with time-based drift.

        SELL: start at ask - 1 tick, drift toward bid
        BUY:  start at bid + 1 tick, drift toward ask

        If the spread is only 1 tick, prefer the passive price first
        instead of crossing immediately.

        Uses quadratic aggression curve: slow at start, fast at end.
        """
        bid = quote.bid_price   # Binance option limit prices are quoted in USD/USDT
        ask = quote.ask_price
        tick = self.cfg.tick_size_usdt

        # Safety: ensure bid < ask
        if bid >= ask:
            mid = (bid + ask) / 2
            bid = mid - tick
            ask = mid + tick

        # Quadratic aggression: 0 at start → 1 at deadline
        aggression = elapsed_ratio ** 2
        one_tick_spread = (ask - bid) <= (tick + 1e-9)

        if leg.side == "SELL":
            # For 1-tick spreads, stay passive at the ask first.
            # Otherwise start just under the ask.
            # End:   bid (take the bid if needed)
            start_price = ask if one_tick_spread else ask - tick
            end_price = bid
            price = start_price - aggression * (start_price - end_price)
            # Never below bid
            price = max(price, bid)
        else:  # BUY
            # For 1-tick spreads, stay passive at the bid first.
            # Otherwise start just above the bid.
            # End:   ask (take the ask if needed)
            start_price = bid if one_tick_spread else bid + tick
            end_price = ask
            price = start_price + aggression * (end_price - start_price)
            # Never above ask
            price = min(price, ask)

        # Round to tick size
        price = round(price / tick) * tick
        price = max(price, tick)  # never zero

        return price

    # ------------------------------------------------------------------
    # Order operations
    # ------------------------------------------------------------------

    def _place_or_amend(self, leg: LegOrder, price: float) -> None:
        """Place a new limit order for a leg."""
        remaining_qty = self._remaining_qty(leg)
        if remaining_qty <= 1e-9:
            self._mark_filled_if_done(leg)
            return

        client_order_id = self._make_client_order_id(leg, "L")
        leg.attempts += 1

        try:
            self._start_new_child_order(leg)
            result = self.client.place_order(
                symbol=leg.symbol,
                side=leg.side,
                quantity=remaining_qty,
                order_type="LIMIT",
                price=price,
                client_order_id=client_order_id,
            )
            self._adopt_submitted_order(leg, result, price, client_order_id)

        except Exception as e:
            if self._recover_submit_error(leg, price, client_order_id):
                logger.warning(
                    f"[Chaser] Recovered {leg.leg_role} via client_order_id "
                    f"after submit error"
                )
                return
            logger.error(f"[Chaser] Failed to place {leg.leg_role}: {e}")
            leg.status = "FAILED"

    def _check_fill(self, leg: LegOrder) -> None:
        """Query exchange for current order status."""
        if not leg.order_id or leg.status in ("FILLED", "FAILED"):
            return

        try:
            result = self.client.query_order(
                leg.symbol,
                order_id=leg.order_id,
                client_order_id=leg.client_order_id or None,
            )

            if result.status in ("FILLED", "PARTIALLY_FILLED", "CANCELLED", "EXPIRED"):
                self._record_fill_progress(leg, result)

            if result.status == "FILLED":
                leg.status = "FILLED"
                logger.info(
                    f"[Chaser] {leg.leg_role} {leg.symbol} FILLED "
                    f"at {result.avg_price:.4f}"
                )
            elif result.status == "PARTIALLY_FILLED":
                leg.status = "PARTIALLY_FILLED"
                logger.debug(
                    f"[Chaser] {leg.leg_role} partial fill: "
                    f"{result.quantity}/{leg.quantity}"
                )
            elif result.status in ("CANCELLED", "EXPIRED", "REJECTED"):
                if self._mark_filled_if_done(leg):
                    return
                # Order was killed externally
                leg.status = "PENDING"
                leg.order_id = ""
                logger.warning(
                    f"[Chaser] {leg.leg_role} order {result.status}, will retry"
                )

            self._mark_filled_if_done(leg)

        except OrderNotFoundError:
            if self._mark_filled_if_done(leg):
                return
            logger.debug(
                f"[Chaser] {leg.leg_role} order not found on exchange, treating it as inactive"
            )
            leg.status = "PENDING"
            leg.order_id = ""
            leg.client_order_id = ""
        except Exception as e:
            logger.error(f"[Chaser] Query failed for {leg.leg_role}: {e}")

    def _cancel_and_replace(self, leg: LegOrder, new_price: float) -> None:
        """Cancel current order and place a new one at new_price."""
        if leg.order_id:
            try:
                cancelled = self.client.cancel_order(
                    leg.symbol,
                    order_id=leg.order_id or None,
                    client_order_id=leg.client_order_id or None,
                )
                self._check_fill(leg)
                if leg.status == "FILLED":
                    return
                if cancelled:
                    logger.debug(
                        f"[Chaser] Cancelled {leg.leg_role} order {leg.order_id}"
                    )
                else:
                    logger.debug(
                        f"[Chaser] {leg.leg_role} cancel not confirmed; order already missing or terminal"
                    )
            except Exception as e:
                logger.warning(f"[Chaser] Cancel failed for {leg.leg_role}: {e}")
                # Check if it filled in the meantime
                self._check_fill(leg)
                if leg.status == "FILLED":
                    return

        if leg.order_id:
            logger.warning(
                f"[Chaser] {leg.leg_role} cancel state unresolved for order {leg.order_id}, skip replace to avoid duplicate execution"
            )
            return

        if self._mark_filled_if_done(leg):
            return

        leg.order_id = ""
        leg.client_order_id = ""
        self._place_or_amend(leg, new_price)

    def _cancel_and_market(self, leg: LegOrder) -> None:
        """Cancel limit order and fill with market-like aggressive limit order."""
        # Cancel outstanding limit
        if leg.order_id:
            try:
                cancelled = self.client.cancel_order(
                    leg.symbol,
                    order_id=leg.order_id or None,
                    client_order_id=leg.client_order_id or None,
                )
            except Exception:
                cancelled = False

            # Check if it filled during cancel
            self._check_fill(leg)
            if leg.status == "FILLED" or self._mark_filled_if_done(leg):
                return
            if not cancelled and leg.order_id:
                logger.warning(
                    f"[Chaser] {leg.leg_role} cancel not confirmed for order {leg.order_id}, skip fallback to avoid duplicate execution"
                )
                return

        logger.info(
            f"[Chaser] {leg.leg_role} {leg.symbol} switching to market-like LIMIT fallback"
        )
        self._market_fallback(leg)

    def _market_fallback(self, leg: LegOrder) -> None:
        """Execute a ladder of aggressive IOC limit orders as last resort."""
        for step in range(self._market_fallback_steps):
            remaining_qty = self._remaining_qty(leg)
            if remaining_qty <= 1e-9:
                self._mark_filled_if_done(leg)
                return

            quote = self._get_quotes([leg]).get(leg.symbol)
            if not quote or quote.bid_price <= 0 or quote.ask_price <= 0:
                leg.status = "FAILED"
                logger.error(f"[Chaser] No valid quote for fallback on {leg.symbol}")
                return

            tick = self.cfg.tick_size_usdt
            if leg.side == "BUY":
                fallback_price = quote.ask_price + step * tick
            else:
                fallback_price = max(quote.bid_price - step * tick, tick)

            client_order_id = self._make_client_order_id(leg, f"M{step}")

            try:
                self._start_new_child_order(leg)
                result = self.client.place_order(
                    symbol=leg.symbol,
                    side=leg.side,
                    quantity=remaining_qty,
                    order_type="LIMIT",
                    price=fallback_price,
                    time_in_force="IOC",
                    client_order_id=client_order_id,
                )

                leg.order_id = result.order_id
                leg.client_order_id = client_order_id

                if result.status in ("FILLED", "PARTIALLY_FILLED"):
                    self._record_fill_progress(leg, result)
                    if self._mark_filled_if_done(leg):
                        logger.info(
                            f"[Chaser] {leg.leg_role} IOC fallback filled at {result.avg_price:.4f}"
                        )
                        return

                logger.warning(
                    f"[Chaser] {leg.leg_role} IOC fallback step {step + 1}/{self._market_fallback_steps} "
                    f"left {self._remaining_qty(leg):.4f}/{leg.quantity:.4f} unfilled"
                )
            except Exception as e:
                logger.warning(
                    f"[Chaser] IOC fallback step {step + 1}/{self._market_fallback_steps} failed "
                    f"for {leg.leg_role}: {e}"
                )

        leg.status = "FAILED"
        logger.error(
            f"[Chaser] {leg.leg_role} IOC fallback exhausted: {leg.filled_qty:.4f}/{leg.quantity:.4f} filled"
        )

    # ------------------------------------------------------------------
    # Quote helpers
    # ------------------------------------------------------------------

    def _get_quotes(
        self, legs: list[LegOrder], underlying: Optional[str] = None
    ) -> dict[str, OptionTicker]:
        """Fetch current quotes for all leg symbols."""
        now = time.monotonic()
        quotes: dict[str, OptionTicker] = {}
        missing: list[str] = []

        for leg in legs:
            if leg.status in ("FILLED", "FAILED"):
                continue
            if leg.symbol in quotes:
                continue
            cached = self._quote_cache.get(leg.symbol)
            if cached and cached[1] > now:
                quotes[leg.symbol] = cached[0]
                continue
            missing.append(leg.symbol)

        if missing and underlying and len(missing) > 1:
            try:
                batch_quotes = self.client.get_tickers_for_symbols(missing)
                for symbol, ticker in batch_quotes.items():
                    quotes[symbol] = ticker
                    self._quote_cache[symbol] = (ticker, now + self._quote_cache_ttl_sec)
                missing = [symbol for symbol in missing if symbol not in quotes]
            except Exception as e:
                logger.debug(f"[Chaser] Batch quote fetch failed for {underlying}: {e}")

        for symbol in missing:
            try:
                ticker = self.client.get_ticker(symbol)
                if ticker:
                    quotes[symbol] = ticker
                    self._quote_cache[symbol] = (ticker, now + self._quote_cache_ttl_sec)
            except Exception as e:
                logger.debug(f"[Chaser] Quote fetch failed for {symbol}: {e}")
        return quotes
