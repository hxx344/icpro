"""Unit tests for the Limit Chaser engine (trader/limit_chaser.py)."""

from __future__ import annotations

import time
import unittest
from dataclasses import dataclass
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, PropertyMock

from trader.binance_client import (
    BinanceOptionsClient,
    OptionTicker,
    OrderResult,
)
from trader.config import ExchangeConfig
from trader.limit_chaser import ChaserConfig, LegOrder, LimitChaser


def _wire_submit_to_place(client: MagicMock) -> MagicMock:
    """Keep legacy tests working after chaser switched to submit_order()."""
    client.submit_order = client.place_order
    return client


def _make_ticker(
    symbol: str = "ETH-260321-2000-C",
    bid: float = 50.0,
    ask: float = 55.0,
    spot: float = 2000.0,
) -> OptionTicker:
    return OptionTicker(
        symbol=symbol,
        underlying="ETH",
        strike=2000.0,
        option_type="call",
        expiry=datetime(2026, 3, 21, 8, 0, tzinfo=timezone.utc),
        bid_price=bid,
        ask_price=ask,
        mark_price=(bid + ask) / 2,
        last_price=(bid + ask) / 2,
        underlying_price=spot,
        volume_24h=100.0,
        open_interest=50.0,
    )


def _make_fill(
    order_id: str = "ORD-1",
    symbol: str = "ETH-260321-2000-C",
    side: str = "BUY",
    price: float = 52.0,
    status: str = "FILLED",
    fee: float = 0.01,
    qty: float = 1.0,
) -> OrderResult:
    return OrderResult(
        order_id=order_id,
        symbol=symbol,
        side=side,
        quantity=qty,
        price=price,
        avg_price=price,
        status=status,
        fee=fee,
        raw={},
    )


class TestLegOrder(unittest.TestCase):
    """Test LegOrder data model."""

    def test_default_state(self):
        leg = LegOrder(
            leg_role="buy_put",
            symbol="ETH-260321-1800-P",
            side="BUY",
            quantity=1.0,
            strike=1800.0,
            option_type="put",
        )
        self.assertEqual(leg.status, "PENDING")
        self.assertEqual(leg.order_id, "")
        self.assertEqual(leg.attempts, 0)
        self.assertEqual(leg.filled_qty, 0.0)

    def test_side_normalization(self):
        leg = LegOrder(
            leg_role="sell_call",
            symbol="ETH-260321-2200-C",
            side="SELL",
            quantity=0.5,
            strike=2200.0,
            option_type="call",
        )
        self.assertEqual(leg.side, "SELL")


class TestChaserConfig(unittest.TestCase):
    """Test ChaserConfig defaults."""

    def test_defaults(self):
        cfg = ChaserConfig()
        self.assertEqual(cfg.window_seconds, 1800)
        self.assertEqual(cfg.poll_interval_sec, 60)
        self.assertEqual(cfg.tick_size_usdt, 5.0)
        self.assertEqual(cfg.market_fallback_sec, 60)
        self.assertEqual(cfg.max_amend_attempts, 180)

    def test_custom_values(self):
        cfg = ChaserConfig(window_seconds=900, poll_interval_sec=5)
        self.assertEqual(cfg.window_seconds, 900)
        self.assertEqual(cfg.poll_interval_sec, 5)


class TestLimitPriceComputation(unittest.TestCase):
    """Test _compute_limit_price at various elapsed ratios."""

    def setUp(self):
        self.client = MagicMock(spec=BinanceOptionsClient)
        self.cfg = ChaserConfig(tick_size_usdt=1.0)
        self.chaser = LimitChaser(self.client, self.cfg)

    def test_sell_at_start(self):
        """SELL at elapsed=0: should be near ask - tick."""
        leg = LegOrder("sell_call", "SYM", "SELL", 1.0, 2200.0, "call")
        q = _make_ticker(bid=50.0, ask=60.0)
        price = self.chaser._compute_limit_price(leg, q, elapsed_ratio=0.0)
        # Start: ask - tick = 60 - 1 = 59
        self.assertAlmostEqual(price, 59.0, places=1)

    def test_sell_at_end(self):
        """SELL at elapsed=1.0: should be at bid."""
        leg = LegOrder("sell_call", "SYM", "SELL", 1.0, 2200.0, "call")
        q = _make_ticker(bid=50.0, ask=60.0)
        price = self.chaser._compute_limit_price(leg, q, elapsed_ratio=1.0)
        # End: bid = 50
        self.assertAlmostEqual(price, 50.0, places=1)

    def test_sell_at_half(self):
        """SELL at elapsed=0.5: quadratic aggression = 0.25."""
        leg = LegOrder("sell_call", "SYM", "SELL", 1.0, 2200.0, "call")
        q = _make_ticker(bid=50.0, ask=60.0)
        price = self.chaser._compute_limit_price(leg, q, elapsed_ratio=0.5)
        # aggression = 0.25, start=59, end=50
        # price = 59 - 0.25 * (59 - 50) = 59 - 2.25 = 56.75 → rounded to 57
        self.assertTrue(55.0 <= price <= 59.0)

    def test_buy_at_start(self):
        """BUY at elapsed=0: should be near bid + tick."""
        leg = LegOrder("buy_put", "SYM", "BUY", 1.0, 1800.0, "put")
        q = _make_ticker(bid=50.0, ask=60.0)
        price = self.chaser._compute_limit_price(leg, q, elapsed_ratio=0.0)
        # Start: bid + tick = 50 + 1 = 51
        self.assertAlmostEqual(price, 51.0, places=1)

    def test_buy_at_end(self):
        """BUY at elapsed=1.0: should be at ask."""
        leg = LegOrder("buy_put", "SYM", "BUY", 1.0, 1800.0, "put")
        q = _make_ticker(bid=50.0, ask=60.0)
        price = self.chaser._compute_limit_price(leg, q, elapsed_ratio=1.0)
        # End: ask = 60
        self.assertAlmostEqual(price, 60.0, places=1)

    def test_sell_one_tick_spread_starts_passive(self):
        """1-tick spread 时 SELL 应先挂 ask，不直接贴 bid。"""
        leg = LegOrder("sell_call", "SYM", "SELL", 1.0, 2200.0, "call")
        q = _make_ticker(bid=50.0, ask=51.0)
        price = self.chaser._compute_limit_price(leg, q, elapsed_ratio=0.0)
        self.assertAlmostEqual(price, 51.0, places=1)

    def test_buy_one_tick_spread_starts_passive(self):
        """1-tick spread 时 BUY 应先挂 bid，不直接贴 ask。"""
        leg = LegOrder("buy_put", "SYM", "BUY", 1.0, 1800.0, "put")
        q = _make_ticker(bid=50.0, ask=51.0)
        price = self.chaser._compute_limit_price(leg, q, elapsed_ratio=0.0)
        self.assertAlmostEqual(price, 50.0, places=1)

    def test_buy_at_half(self):
        """BUY at elapsed=0.5: quadratic aggression = 0.25."""
        leg = LegOrder("buy_put", "SYM", "BUY", 1.0, 1800.0, "put")
        q = _make_ticker(bid=50.0, ask=60.0)
        price = self.chaser._compute_limit_price(leg, q, elapsed_ratio=0.5)
        # aggression = 0.25, start=51, end=60
        # price = 51 + 0.25 * (60 - 51) = 51 + 2.25 = 53.25 → rounded to 53
        self.assertTrue(51.0 <= price <= 56.0)

    def test_price_never_below_bid_for_sell(self):
        """SELL price should never go below bid."""
        leg = LegOrder("sell_call", "SYM", "SELL", 1.0, 2200.0, "call")
        q = _make_ticker(bid=50.0, ask=51.0)  # tight spread
        for ratio in [0.0, 0.5, 0.8, 1.0]:
            price = self.chaser._compute_limit_price(leg, q, ratio)
            self.assertGreaterEqual(price, 50.0,
                f"SELL price {price} < bid {50.0} at ratio {ratio}")

    def test_price_never_above_ask_for_buy(self):
        """BUY price should never go above ask."""
        leg = LegOrder("buy_put", "SYM", "BUY", 1.0, 1800.0, "put")
        q = _make_ticker(bid=50.0, ask=51.0)
        for ratio in [0.0, 0.5, 0.8, 1.0]:
            price = self.chaser._compute_limit_price(leg, q, ratio)
            self.assertLessEqual(price, 51.0,
                f"BUY price {price} > ask {51.0} at ratio {ratio}")

    def test_monotonic_drift_sell(self):
        """SELL price should decrease (drift toward bid) as time passes."""
        leg = LegOrder("sell_call", "SYM", "SELL", 1.0, 2200.0, "call")
        q = _make_ticker(bid=40.0, ask=60.0)  # wide spread
        prev = float("inf")
        for ratio in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            price = self.chaser._compute_limit_price(leg, q, ratio)
            self.assertLessEqual(price, prev,
                f"SELL not monotonically decreasing at {ratio}")
            prev = price

    def test_monotonic_drift_buy(self):
        """BUY price should increase (drift toward ask) as time passes."""
        leg = LegOrder("buy_put", "SYM", "BUY", 1.0, 1800.0, "put")
        q = _make_ticker(bid=40.0, ask=60.0)
        prev = 0.0
        for ratio in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            price = self.chaser._compute_limit_price(leg, q, ratio)
            self.assertGreaterEqual(price, prev,
                f"BUY not monotonically increasing at {ratio}")
            prev = price


class TestMarketFallback(unittest.TestCase):
    """Test market order fallback logic."""

    def setUp(self):
        self.client = _wire_submit_to_place(MagicMock(spec=BinanceOptionsClient))
        self.client.get_ticker.return_value = _make_ticker(bid=50.0, ask=55.0)
        self.chaser = LimitChaser(self.client, ChaserConfig())

    def test_market_fallback_fills(self):
        """Market fallback should mark leg as FILLED on success."""
        leg = LegOrder("sell_call", "SYM", "SELL", 1.0, 2200.0, "call")
        self.client.place_order.return_value = _make_fill(
            status="FILLED", price=50.0
        )

        self.chaser._market_fallback(leg)
        self.assertEqual(leg.status, "FILLED")
        self.assertEqual(leg.avg_price, 50.0)

    def test_market_fallback_fails(self):
        """Market fallback should mark leg as FAILED on API error."""
        leg = LegOrder("buy_put", "SYM", "BUY", 1.0, 1800.0, "put")
        self.client.place_order.side_effect = RuntimeError("API down")

        self.chaser._market_fallback(leg)
        self.assertEqual(leg.status, "FAILED")

    def test_market_fallback_unfilled_status(self):
        """Market fallback exhausting all IOC attempts should mark as FAILED."""
        leg = LegOrder("sell_put", "SYM", "SELL", 1.0, 1800.0, "put")
        self.client.place_order.return_value = _make_fill(status="EXPIRED", qty=0.0)

        self.chaser._market_fallback(leg)
        self.assertEqual(leg.status, "FAILED")
        self.assertEqual(self.client.place_order.call_count, 3)

    def test_market_fallback_ladders_remaining_quantity(self):
        """IOC fallback should retry only the remaining quantity at more aggressive prices."""
        leg = LegOrder("buy_put", "SYM", "BUY", 1.0, 1800.0, "put")
        self.client.place_order.side_effect = [
            _make_fill(status="PARTIALLY_FILLED", price=55.0, qty=0.4),
            _make_fill(status="FILLED", price=60.0, qty=0.6),
        ]

        self.chaser._market_fallback(leg)

        self.assertEqual(leg.status, "FILLED")
        self.assertAlmostEqual(leg.filled_qty, 1.0, places=6)
        first_call = self.client.place_order.call_args_list[0].kwargs
        second_call = self.client.place_order.call_args_list[1].kwargs
        self.assertAlmostEqual(first_call["quantity"], 1.0, places=6)
        self.assertAlmostEqual(first_call["price"], 55.0, places=6)
        self.assertAlmostEqual(second_call["quantity"], 0.6, places=6)
        self.assertAlmostEqual(second_call["price"], 60.0, places=6)


class TestPlaceOrAmend(unittest.TestCase):
    """Test initial limit order placement."""

    def setUp(self):
        self.client = _wire_submit_to_place(MagicMock(spec=BinanceOptionsClient))
        self.chaser = LimitChaser(self.client, ChaserConfig())

    def test_immediate_fill(self):
        """If limit order fills immediately, leg status = FILLED."""
        leg = LegOrder("sell_call", "SYM", "SELL", 1.0, 2200.0, "call")
        self.client.place_order.return_value = _make_fill(
            status="FILLED", price=55.0
        )

        self.chaser._place_or_amend(leg, 55.0)
        self.assertEqual(leg.status, "FILLED")
        self.assertEqual(leg.avg_price, 55.0)
        self.assertEqual(leg.attempts, 1)

    def test_order_pending(self):
        """New limit order → status = NEW, order_id set."""
        leg = LegOrder("buy_put", "SYM", "BUY", 1.0, 1800.0, "put")
        self.client.place_order.return_value = _make_fill(
            order_id="ORD-123", status="NEW", price=0.0
        )

        self.chaser._place_or_amend(leg, 45.0)
        self.assertEqual(leg.status, "NEW")
        self.assertEqual(leg.order_id, "ORD-123")
        self.assertEqual(leg.current_price, 45.0)
        self.assertEqual(leg.attempts, 1)
        self.assertTrue(self.client.place_order.call_args.kwargs["client_order_id"].startswith("LCL"))

    def test_submit_error_recovers_by_client_order_id(self):
        leg = LegOrder("buy_put", "SYM", "BUY", 1.0, 1800.0, "put")
        self.client.place_order.side_effect = RuntimeError("timeout")
        self.client.query_order.return_value = _make_fill(
            order_id="ORD-RECOVER", status="NEW", price=0.0, qty=0.0
        )

        self.chaser._place_or_amend(leg, 50.0)

        self.assertEqual(leg.status, "NEW")
        self.assertEqual(leg.order_id, "ORD-RECOVER")
        self.client.query_order.assert_called_once()
        self.assertEqual(
            self.client.query_order.call_args.kwargs["client_order_id"],
            leg.client_order_id,
        )

    def test_placement_error(self):
        """API error → leg status = FAILED."""
        leg = LegOrder("sell_put", "SYM", "SELL", 1.0, 1800.0, "put")
        self.client.place_order.side_effect = RuntimeError("timeout")
        self.client.query_order.side_effect = RuntimeError("not found")

        self.chaser._place_or_amend(leg, 50.0)
        self.assertEqual(leg.status, "FAILED")


class TestCheckFill(unittest.TestCase):
    """Test order status polling."""

    def setUp(self):
        self.client = _wire_submit_to_place(MagicMock(spec=BinanceOptionsClient))
        self.chaser = LimitChaser(self.client, ChaserConfig())

    def test_filled_order(self):
        leg = LegOrder("sell_call", "SYM", "SELL", 1.0, 2200.0, "call")
        leg.order_id = "ORD-1"
        leg.status = "NEW"

        self.client.query_order.return_value = _make_fill(
            status="FILLED", price=55.0, fee=0.02
        )

        self.chaser._check_fill(leg)
        self.assertEqual(leg.status, "FILLED")
        self.assertEqual(leg.avg_price, 55.0)
        self.assertEqual(leg.fee, 0.02)

    def test_still_pending(self):
        leg = LegOrder("buy_put", "SYM", "BUY", 1.0, 1800.0, "put")
        leg.order_id = "ORD-2"
        leg.status = "NEW"

        self.client.query_order.return_value = _make_fill(
            status="NEW", price=0.0
        )

        self.chaser._check_fill(leg)
        self.assertEqual(leg.status, "NEW")

    def test_cancelled_externally(self):
        leg = LegOrder("sell_put", "SYM", "SELL", 1.0, 1800.0, "put")
        leg.order_id = "ORD-3"
        leg.status = "NEW"

        self.client.query_order.return_value = _make_fill(status="CANCELLED", qty=0.0, fee=0.0)

        self.chaser._check_fill(leg)
        self.assertEqual(leg.status, "PENDING")
        self.assertEqual(leg.order_id, "")

    def test_partial_fill_tracks_cumulative_progress(self):
        leg = LegOrder("sell_put", "SYM", "SELL", 1.0, 1800.0, "put")
        leg.order_id = "ORD-4"
        leg.status = "NEW"

        self.client.query_order.return_value = _make_fill(
            order_id="ORD-4", status="PARTIALLY_FILLED", price=54.0, fee=0.01, qty=0.4
        )

        self.chaser._check_fill(leg)
        self.assertEqual(leg.status, "PARTIALLY_FILLED")
        self.assertAlmostEqual(leg.filled_qty, 0.4, places=6)
        self.assertAlmostEqual(leg.fee, 0.01, places=6)

    def test_skip_already_filled(self):
        """Should not query if already filled."""
        leg = LegOrder("sell_call", "SYM", "SELL", 1.0, 2200.0, "call")
        leg.status = "FILLED"
        leg.order_id = "ORD-X"

        self.chaser._check_fill(leg)
        self.client.query_order.assert_not_called()

    def test_skip_no_order_id(self):
        """Should not query if no order_id."""
        leg = LegOrder("sell_call", "SYM", "SELL", 1.0, 2200.0, "call")
        leg.status = "PENDING"
        leg.order_id = ""

        self.chaser._check_fill(leg)
        self.client.query_order.assert_not_called()


class TestCancelAndReplace(unittest.TestCase):
    """Test cancel-and-replace flow."""

    def setUp(self):
        self.client = _wire_submit_to_place(MagicMock(spec=BinanceOptionsClient))
        self.chaser = LimitChaser(self.client, ChaserConfig())

    def test_cancel_then_new_order(self):
        leg = LegOrder("sell_call", "SYM", "SELL", 1.0, 2200.0, "call")
        leg.order_id = "OLD-1"
        leg.status = "NEW"

        self.client.cancel_order.return_value = True
        self.client.place_order.return_value = _make_fill(
            order_id="NEW-1", status="NEW"
        )

        self.chaser._cancel_and_replace(leg, 53.0)

        self.client.cancel_order.assert_called_once_with(
            "SYM", order_id="OLD-1", client_order_id=None
        )
        self.assertEqual(leg.order_id, "NEW-1")
        self.assertEqual(leg.current_price, 53.0)

    def test_cancel_replace_uses_remaining_quantity_after_partial_fill(self):
        leg = LegOrder("sell_call", "SYM", "SELL", 1.0, 2200.0, "call")
        leg.order_id = "OLD-1"
        leg.status = "PARTIALLY_FILLED"
        leg.filled_qty = 0.4
        leg.current_order_filled_qty = 0.4
        leg.current_order_avg_price = 54.0

        self.client.cancel_order.return_value = True
        self.client.query_order.return_value = _make_fill(
            order_id="OLD-1", status="CANCELLED", price=54.0, fee=0.0, qty=0.4
        )
        self.client.place_order.return_value = _make_fill(
            order_id="NEW-1", status="NEW", qty=0.0
        )

        self.chaser._cancel_and_replace(leg, 53.0)

        self.client.place_order.assert_called_once()
        self.assertAlmostEqual(
            self.client.place_order.call_args.kwargs["quantity"],
            0.6,
            places=6,
        )
        self.assertAlmostEqual(leg.filled_qty, 0.4, places=6)

    def test_cancel_fails_but_filled(self):
        """If cancel fails because order already filled, detect it."""
        leg = LegOrder("sell_call", "SYM", "SELL", 1.0, 2200.0, "call")
        leg.order_id = "OLD-1"
        leg.status = "NEW"

        self.client.cancel_order.side_effect = RuntimeError("order not found")
        self.client.query_order.return_value = _make_fill(
            status="FILLED", price=54.0
        )

        self.chaser._cancel_and_replace(leg, 53.0)
        self.assertEqual(leg.status, "FILLED")
        self.client.place_order.assert_not_called()


class TestExecuteLegsSimulated(unittest.TestCase):
    """Test full execute_legs with simulated immediate fills."""

    def setUp(self):
        self.client = _wire_submit_to_place(MagicMock(spec=BinanceOptionsClient))
        # All orders fill immediately
        self.client.place_order.return_value = _make_fill(
            status="FILLED", price=50.0, fee=0.01
        )
        self.client.get_ticker.return_value = _make_ticker()

        self.cfg = ChaserConfig(
            window_seconds=5,
            poll_interval_sec=1,
            market_fallback_sec=2,
        )
        self.chaser = LimitChaser(self.client, self.cfg)

    def test_all_legs_fill_immediately(self):
        legs = [
            LegOrder("buy_put", "SYM1", "BUY", 1.0, 1800.0, "put"),
            LegOrder("buy_call", "SYM2", "BUY", 1.0, 2200.0, "call"),
            LegOrder("sell_put", "SYM3", "SELL", 1.0, 1800.0, "put"),
            LegOrder("sell_call", "SYM4", "SELL", 1.0, 2200.0, "call"),
        ]

        results = self.chaser.execute_legs(legs)

        self.assertEqual(len(results), 4)
        for r in results:
            self.assertEqual(r.status, "FILLED")
            self.assertEqual(r.avg_price, 50.0)

    def test_empty_legs(self):
        results = self.chaser.execute_legs([])
        self.assertEqual(results, [])


class TestCancelAndMarket(unittest.TestCase):
    """Test the cancel-and-market-fallback flow."""

    def setUp(self):
        self.client = _wire_submit_to_place(MagicMock(spec=BinanceOptionsClient))
        self.client.get_ticker.return_value = _make_ticker(bid=50.0, ask=55.0)
        self.chaser = LimitChaser(self.client, ChaserConfig())

    def test_cancel_then_market(self):
        leg = LegOrder("sell_call", "SYM", "SELL", 1.0, 2200.0, "call")
        leg.order_id = "LIMIT-1"
        leg.status = "NEW"

        # cancel succeeds, query shows not filled, market fills
        self.client.cancel_order.return_value = True
        self.client.query_order.return_value = _make_fill(status="CANCELLED", qty=0.0, fee=0.0)
        self.client.place_order.return_value = _make_fill(
            status="FILLED", price=49.0
        )

        self.chaser._cancel_and_market(leg)
        self.assertEqual(leg.status, "FILLED")
        self.assertEqual(leg.avg_price, 49.0)

    def test_market_fallback_uses_remaining_quantity(self):
        leg = LegOrder("sell_call", "SYM", "SELL", 1.0, 2200.0, "call")
        leg.status = "PARTIALLY_FILLED"
        leg.filled_qty = 0.25
        self.client.place_order.return_value = _make_fill(
            status="FILLED", price=49.0, qty=0.75
        )

        self.chaser._market_fallback(leg)

        self.client.place_order.assert_called_once()
        self.assertAlmostEqual(
            self.client.place_order.call_args.kwargs["quantity"],
            0.75,
            places=6,
        )
        self.assertEqual(leg.status, "FILLED")
        self.assertAlmostEqual(leg.filled_qty, 1.0, places=6)

    def test_cancel_then_market_with_partial_fill_accumulates_weighted_avg(self):
        leg = LegOrder("sell_call", "SYM", "SELL", 1.0, 2200.0, "call")
        leg.order_id = "LIMIT-1"
        leg.status = "NEW"

        self.client.cancel_order.return_value = True
        self.client.query_order.return_value = _make_fill(
            order_id="LIMIT-1", status="CANCELLED", price=52.0, fee=0.01, qty=0.4
        )
        self.client.place_order.return_value = _make_fill(
            order_id="MKT-1", status="FILLED", price=49.0, fee=0.02, qty=0.6
        )

        self.chaser._cancel_and_market(leg)

        self.assertEqual(leg.status, "FILLED")
        self.assertAlmostEqual(leg.filled_qty, 1.0, places=6)
        self.assertAlmostEqual(leg.avg_price, 50.2, places=6)
        self.assertAlmostEqual(leg.fee, 0.03, places=6)

    def test_already_filled_during_cancel(self):
        """If order fills during cancel, no market order needed."""
        leg = LegOrder("buy_put", "SYM", "BUY", 1.0, 1800.0, "put")
        leg.order_id = "LIMIT-2"
        leg.status = "NEW"

        self.client.cancel_order.return_value = True
        self.client.query_order.return_value = _make_fill(
            status="FILLED", price=51.0
        )

        self.chaser._cancel_and_market(leg)
        self.assertEqual(leg.status, "FILLED")
        self.assertEqual(leg.avg_price, 51.0)
        # place_order should NOT have been called for market
        self.client.place_order.assert_not_called()


class TestGetQuotes(unittest.TestCase):
    """Test quote fetching helper."""

    def setUp(self):
        self.client = _wire_submit_to_place(MagicMock(spec=BinanceOptionsClient))
        self.chaser = LimitChaser(self.client, ChaserConfig())

    def test_fetches_unique_symbols(self):
        self.client.get_ticker.return_value = _make_ticker()

        legs = [
            LegOrder("buy_put", "SYM1", "BUY", 1.0, 1800.0, "put"),
            LegOrder("sell_call", "SYM2", "SELL", 1.0, 2200.0, "call"),
        ]
        quotes = self.chaser._get_quotes(legs)
        self.assertEqual(len(quotes), 2)
        self.assertIn("SYM1", quotes)
        self.assertIn("SYM2", quotes)

    def test_uses_batch_fetch_when_underlying_given(self):
        self.client.get_tickers_for_symbols.return_value = {
            "SYM1": _make_ticker(symbol="SYM1"),
            "SYM2": _make_ticker(symbol="SYM2"),
        }

        legs = [
            LegOrder("buy_put", "SYM1", "BUY", 1.0, 1800.0, "put"),
            LegOrder("sell_call", "SYM2", "SELL", 1.0, 2200.0, "call"),
        ]
        quotes = self.chaser._get_quotes(legs, underlying="ETH")

        self.assertEqual(set(quotes.keys()), {"SYM1", "SYM2"})
        self.client.get_tickers_for_symbols.assert_called_once_with(["SYM1", "SYM2"])
        self.client.get_ticker.assert_not_called()

    def test_reuses_cached_quotes_within_ttl(self):
        self.client.get_ticker.return_value = _make_ticker(symbol="SYM1")
        legs = [LegOrder("buy_put", "SYM1", "BUY", 1.0, 1800.0, "put")]

        self.chaser._get_quotes(legs)
        self.chaser._get_quotes(legs)

        self.client.get_ticker.assert_called_once_with("SYM1")

    def test_skips_filled_legs(self):
        self.client.get_ticker.return_value = _make_ticker()

        legs = [
            LegOrder("buy_put", "SYM1", "BUY", 1.0, 1800.0, "put"),
            LegOrder("sell_call", "SYM2", "SELL", 1.0, 2200.0, "call"),
        ]
        legs[0].status = "FILLED"
        quotes = self.chaser._get_quotes(legs)
        self.assertEqual(len(quotes), 1)
        self.assertIn("SYM2", quotes)

    def test_handles_fetch_error(self):
        self.client.get_ticker.side_effect = RuntimeError("timeout")

        legs = [
            LegOrder("buy_put", "SYM1", "BUY", 1.0, 1800.0, "put"),
        ]
        quotes = self.chaser._get_quotes(legs)
        self.assertEqual(len(quotes), 0)


if __name__ == "__main__":
    unittest.main()
