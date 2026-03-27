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

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from trader.binance_client import OrderResult, OptionTicker


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
        config: ChaserConfig | None = None,
    ):
        self.client = client
        self.cfg = config or ChaserConfig()

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

    def execute_legs(
        self,
        legs: list[LegOrder],
        underlying: str = "ETH",
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

        # Phase 1: Place initial limit orders for all legs
        self._place_initial_orders(legs, underlying)

        # Phase 2: Poll + amend loop
        self._chase_loop(legs, underlying, t0)

        return legs

    # ------------------------------------------------------------------
    # Phase 1: Initial placement
    # ------------------------------------------------------------------

    def _place_initial_orders(
        self,
        legs: list[LegOrder],
        underlying: str,
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
                continue

            price = self._compute_limit_price(leg, q, elapsed_ratio=0.0)
            self._place_or_amend(leg, price)

    # ------------------------------------------------------------------
    # Phase 2: Chase loop
    # ------------------------------------------------------------------

    def _chase_loop(
        self,
        legs: list[LegOrder],
        underlying: str,
        t0: float,
    ) -> None:
        """Poll unfilled orders and amend prices until all fill or timeout."""
        deadline = t0 + self.cfg.window_seconds
        market_time = t0 + self.cfg.window_seconds - self.cfg.market_fallback_sec

        while True:
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
                break

            # Check order statuses
            for leg in pending:
                self._check_fill(leg)

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

        try:
            self._start_new_child_order(leg)
            result = self.client.place_order(
                symbol=leg.symbol,
                side=leg.side,
                quantity=remaining_qty,
                order_type="LIMIT",
                price=price,
            )

            leg.order_id = result.order_id
            leg.current_price = price
            leg.attempts += 1

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
                    f"@ {price:.4f} for {remaining_qty:.4f} placed "
                    f"(attempt #{leg.attempts})"
                )

        except Exception as e:
            logger.error(f"[Chaser] Failed to place {leg.leg_role}: {e}")
            leg.status = "FAILED"

    def _check_fill(self, leg: LegOrder) -> None:
        """Query exchange for current order status."""
        if not leg.order_id or leg.status in ("FILLED", "FAILED"):
            return

        try:
            result = self.client.query_order(leg.symbol, leg.order_id)

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

        except Exception as e:
            logger.error(f"[Chaser] Query failed for {leg.leg_role}: {e}")

    def _cancel_and_replace(self, leg: LegOrder, new_price: float) -> None:
        """Cancel current order and place a new one at new_price."""
        if leg.order_id:
            try:
                self.client.cancel_order(leg.symbol, leg.order_id)
                self._check_fill(leg)
                logger.debug(
                    f"[Chaser] Cancelled {leg.leg_role} order {leg.order_id}"
                )
            except Exception as e:
                logger.warning(f"[Chaser] Cancel failed for {leg.leg_role}: {e}")
                # Check if it filled in the meantime
                self._check_fill(leg)
                if leg.status == "FILLED":
                    return

        if self._mark_filled_if_done(leg):
            return

        leg.order_id = ""
        self._place_or_amend(leg, new_price)

    def _cancel_and_market(self, leg: LegOrder) -> None:
        """Cancel limit order and fill with market-like aggressive limit order."""
        # Cancel outstanding limit
        if leg.order_id:
            try:
                self.client.cancel_order(leg.symbol, leg.order_id)
            except Exception:
                pass

            # Check if it filled during cancel
            self._check_fill(leg)
            if leg.status == "FILLED" or self._mark_filled_if_done(leg):
                return

        logger.info(
            f"[Chaser] {leg.leg_role} {leg.symbol} switching to market-like LIMIT fallback"
        )
        self._market_fallback(leg)

    def _market_fallback(self, leg: LegOrder) -> None:
        """Execute a market-like aggressive limit order as last resort."""
        remaining_qty = self._remaining_qty(leg)
        if remaining_qty <= 1e-9:
            self._mark_filled_if_done(leg)
            return

        try:
            self._start_new_child_order(leg)
            result = self.client.place_order(
                symbol=leg.symbol,
                side=leg.side,
                quantity=remaining_qty,
                order_type="MARKET",
            )

            if result.status in ("FILLED", "PARTIALLY_FILLED"):
                self._record_fill_progress(leg, result)
                leg.order_id = result.order_id
                if self._mark_filled_if_done(leg):
                    logger.info(
                        f"[Chaser] {leg.leg_role} MARKET filled at "
                        f"{result.avg_price:.4f}"
                    )
                else:
                    leg.status = "FAILED"
                    logger.error(
                        f"[Chaser] {leg.leg_role} MARKET only partially filled: "
                        f"{leg.filled_qty:.4f}/{leg.quantity:.4f}"
                    )
            else:
                leg.status = "FAILED"
                logger.error(
                    f"[Chaser] {leg.leg_role} MARKET order failed: "
                    f"{result.status}"
                )

        except Exception as e:
            logger.error(f"[Chaser] Market fallback failed for {leg.leg_role}: {e}")
            leg.status = "FAILED"

    # ------------------------------------------------------------------
    # Quote helpers
    # ------------------------------------------------------------------

    def _get_quotes(
        self, legs: list[LegOrder]
    ) -> dict[str, OptionTicker]:
        """Fetch current quotes for all leg symbols."""
        quotes: dict[str, OptionTicker] = {}
        for leg in legs:
            if leg.status in ("FILLED", "FAILED"):
                continue
            if leg.symbol in quotes:
                continue
            try:
                ticker = self.client.get_ticker(leg.symbol)
                if ticker:
                    quotes[leg.symbol] = ticker
            except Exception as e:
                logger.debug(f"[Chaser] Quote fetch failed for {leg.symbol}: {e}")
        return quotes
