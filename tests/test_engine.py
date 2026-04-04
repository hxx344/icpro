"""Tests for engine components – Account, Matcher, PositionManager."""

from datetime import datetime, timezone

import pytest

from options_backtest.data.models import Direction, Fill, OrderRequest, Position
from options_backtest.engine.account import Account
from options_backtest.engine.matcher import Matcher
from options_backtest.engine.position import PositionManager
from options_backtest.config import ExecutionConfig


NOW = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


class TestAccount:
    def test_initial_balance(self):
        acc = Account(initial_balance=2.0)
        assert acc.balance == 2.0

    def test_deposit_withdraw(self):
        acc = Account(initial_balance=1.0)
        acc.deposit(0.5)
        assert acc.balance == pytest.approx(1.5)
        acc.withdraw(0.3)
        assert acc.balance == pytest.approx(1.2)

    def test_fee(self):
        acc = Account(initial_balance=1.0)
        acc.pay_fee(0.001)
        assert acc.total_fee_paid == pytest.approx(0.001)
        assert acc.balance == pytest.approx(0.999)

    def test_equity(self):
        acc = Account(initial_balance=1.0)
        eq = acc.equity(unrealized_pnl=0.05)
        assert eq == pytest.approx(1.05)

    def test_equity_history(self):
        acc = Account(initial_balance=1.0)
        acc.record_equity(NOW, 0.01, 80000.0)
        acc.record_equity(NOW, -0.02, 81000.0)
        assert len(acc.equity_history) == 2
        # New 5-tuple: (timestamp, equity, balance, unrealized_pnl, underlying_price)
        assert len(acc.equity_history[0]) == 5
        assert acc.equity_history[0][4] == 80000.0


class TestMatcher:
    def test_fill_touch_price(self):
        matcher = Matcher(ExecutionConfig(slippage=0.0))
        order = OrderRequest(
            instrument_name="BTC-CALL",
            direction=Direction.LONG,
            quantity=1.0,
        )
        fill = matcher.execute(order, NOW, bid_price=0.05, ask_price=0.06, mark_price=0.055, underlying_price=80000)
        assert fill is not None
        assert fill.fill_price == pytest.approx(0.06)  # top-of-book ask for buys
        assert fill.fee > 0

    def test_slippage_buy_on_mark_fallback(self):
        matcher = Matcher(ExecutionConfig(slippage=0.001))
        order = OrderRequest(instrument_name="X", direction=Direction.LONG, quantity=1)
        fill = matcher.execute(order, NOW, None, None, 0.055, 80000)
        assert fill.fill_price == pytest.approx(0.055 + 0.001)

    def test_slippage_sell_on_mark_fallback(self):
        matcher = Matcher(ExecutionConfig(slippage=0.001))
        order = OrderRequest(instrument_name="X", direction=Direction.SHORT, quantity=1)
        fill = matcher.execute(order, NOW, None, None, 0.055, 80000)
        assert fill.fill_price == pytest.approx(0.055 - 0.001)

    def test_fee_cap(self):
        """Fee should not exceed 12.5 % of option price."""
        matcher = Matcher(ExecutionConfig(taker_fee=10.0, max_fee_pct=0.125, slippage=0.0))
        order = OrderRequest(instrument_name="X", direction=Direction.LONG, quantity=1)
        fill = matcher.execute(order, NOW, 0.01, 0.02, 0.015, 80000)
        # cap = fill_price × max_fee_pct
        assert fill.fee <= fill.fill_price * 0.125 + 1e-10


class TestPositionManager:
    def test_open_position(self):
        pm = PositionManager()
        fill = Fill(
            timestamp=NOW, instrument_name="OPT-1",
            direction=Direction.LONG, quantity=2.0,
            fill_price=0.05, fee=0.0001,
        )
        pm.apply_fill(fill)
        assert pm.has_position("OPT-1")
        assert pm.get("OPT-1").quantity == 2.0

    def test_close_position(self):
        pm = PositionManager()
        pm.apply_fill(Fill(timestamp=NOW, instrument_name="OPT-1",
                           direction=Direction.LONG, quantity=2.0,
                           fill_price=0.05, fee=0.0))
        # Close
        pm.apply_fill(Fill(timestamp=NOW, instrument_name="OPT-1",
                           direction=Direction.SHORT, quantity=2.0,
                           fill_price=0.07, fee=0.0))
        assert not pm.has_position("OPT-1")
        assert len(pm.closed_trades) == 1
        assert pm.closed_trades[0]["pnl"] == pytest.approx(0.02 * 2)

    def test_partial_close(self):
        pm = PositionManager()
        pm.apply_fill(Fill(timestamp=NOW, instrument_name="OPT-1",
                           direction=Direction.LONG, quantity=5.0,
                           fill_price=0.05, fee=0.0))
        pm.apply_fill(Fill(timestamp=NOW, instrument_name="OPT-1",
                           direction=Direction.SHORT, quantity=3.0,
                           fill_price=0.06, fee=0.0))
        assert pm.get("OPT-1").quantity == 2.0
        assert len(pm.closed_trades) == 1

    def test_settle_itm_call(self):
        pm = PositionManager()
        pm.apply_fill(Fill(timestamp=NOW, instrument_name="BTC-C",
                           direction=Direction.LONG, quantity=1.0,
                           fill_price=0.03, fee=0.0))
        # Settlement: BTC @ 90000, strike 80000 → intrinsic = 10000/90000
        # Entry cost (withdraw) already handled in _process_orders;
        # settle_expired only returns the intrinsic cash-flow.
        pnl = pm.settle_expired("BTC-C", 90000, 80000, "call", NOW)
        expected_intrinsic = 10000 / 90000
        assert pnl == pytest.approx(expected_intrinsic, abs=1e-6)
        assert not pm.has_position("BTC-C")

    def test_settle_otm_put(self):
        pm = PositionManager()
        pm.apply_fill(Fill(timestamp=NOW, instrument_name="BTC-P",
                           direction=Direction.LONG, quantity=1.0,
                           fill_price=0.02, fee=0.0))
        # Settlement: BTC @ 90000, strike 80000 → put OTM
        # No settlement cash-flow for OTM; entry cost already handled.
        pnl = pm.settle_expired("BTC-P", 90000, 80000, "put", NOW)
        assert pnl == pytest.approx(0.0)

    def test_mark_update(self):
        pm = PositionManager()
        pm.apply_fill(Fill(timestamp=NOW, instrument_name="OPT",
                           direction=Direction.LONG, quantity=1.0,
                           fill_price=0.05, fee=0.0))
        pm.update_marks({"OPT": 0.08})
        assert pm.get("OPT").unrealized_pnl == pytest.approx(0.03)
