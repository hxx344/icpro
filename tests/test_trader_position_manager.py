"""Tests for trader.position_manager — 仓位管理单元测试.

覆盖:
  - IronCondorPosition 数据模型 (max_profit, max_loss, legs)
  - CondorLeg 模型
  - PositionManager 初始化与恢复
  - open_iron_condor 流程 (4 腿下单)
  - 部分失败回滚 (_rollback_legs)
  - close_iron_condor 与 PnL 计算
  - close_all
  - get_unrealized_pnl
  - summary
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import pytest

from trader.binance_client import (
    BinanceOptionsClient,
    OrderResult,
    AccountInfo,
)
from trader.config import ExchangeConfig
from trader.limit_chaser import LegOrder
from trader.position_manager import (
    CondorLeg,
    IronCondorPosition,
    PositionManager,
)
from trader.storage import Storage


# ======================================================================
# Helpers
# ======================================================================


def _make_order_result(side="SELL", avg_price=0.05, fee=0.001, quantity=1.0, symbol="ETH-260321-2700-C") -> OrderResult:
    return OrderResult(
        order_id="ORD_001",
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=avg_price,
        avg_price=avg_price,
        status="FILLED",
        fee=fee,
        raw={},
    )


def _make_leg(
    symbol="ETH-260321-2700-C",
    side="SELL",
    option_type="call",
    strike=2700.0,
    entry_price=0.05,
) -> CondorLeg:
    return CondorLeg(
        symbol=symbol,
        side=side,
        option_type=option_type,
        strike=strike,
        quantity=1.0,
        entry_price=entry_price,
        trade_id=1,
        order_id="ORD_001",
    )


@pytest.fixture
def mock_client():
    client = MagicMock(spec=BinanceOptionsClient)
    client.place_order.return_value = _make_order_result()
    client.submit_order.side_effect = lambda **kwargs: _make_order_result(
        side=kwargs.get("side", "SELL"),
        quantity=float(kwargs.get("quantity", 1.0)),
        symbol=kwargs.get("symbol", "ETH-260321-2700-C"),
    )
    client.get_order_book.return_value = {
        "bids": [(100.0, 10.0)],
        "asks": [(105.0, 10.0)],
    }
    client.get_positions.return_value = []
    client.cancel_order.return_value = True
    return client


@pytest.fixture
def storage(tmp_path):
    s = Storage(str(tmp_path / "test_pm.db"))
    yield s
    s.close()


@pytest.fixture
def pos_mgr(mock_client, storage):
    return PositionManager(mock_client, storage)


# ======================================================================
# 1. Data Models
# ======================================================================


class TestCondorLeg:
    def test_fields(self):
        leg = _make_leg()
        assert leg.symbol == "ETH-260321-2700-C"
        assert leg.side == "SELL"
        assert leg.strike == 2700.0


class TestIronCondorPosition:
    def test_legs_returns_non_none(self):
        condor = IronCondorPosition(
            group_id="IC_001",
            entry_time=datetime.now(timezone.utc),
            underlying_price=2500.0,
            sell_call=_make_leg("SYM_SC", "SELL", "call", 2700),
            buy_call=_make_leg("SYM_BC", "BUY", "call", 2750),
            sell_put=None,
            buy_put=None,
        )
        assert len(condor.legs) == 2

    def test_legs_all_present(self):
        condor = IronCondorPosition(
            group_id="IC_001",
            entry_time=datetime.now(timezone.utc),
            underlying_price=2500.0,
            sell_call=_make_leg("SC", "SELL", "call", 2700),
            buy_call=_make_leg("BC", "BUY", "call", 2750),
            sell_put=_make_leg("SP", "SELL", "put", 2300),
            buy_put=_make_leg("BP", "BUY", "put", 2250),
        )
        assert len(condor.legs) == 4

    def test_max_profit(self):
        condor = IronCondorPosition(
            group_id="IC_001",
            entry_time=datetime.now(timezone.utc),
            underlying_price=2500.0,
            total_premium=0.08,
        )
        assert condor.max_profit == pytest.approx(0.08)

    def test_max_loss(self):
        condor = IronCondorPosition(
            group_id="IC_001",
            entry_time=datetime.now(timezone.utc),
            underlying_price=2500.0,
            sell_call=_make_leg("SC", "SELL", "call", 2700),
            buy_call=_make_leg("BC", "BUY", "call", 2750),
            sell_put=_make_leg("SP", "SELL", "put", 2300),
            buy_put=_make_leg("BP", "BUY", "put", 2250),
            total_premium=0.08,
        )
        # wing_width = max(2700-2750? no, put: 2300-2250=50, call: 2750-2700=50)
        # max_loss = 50 - 0.08
        assert condor.max_loss == pytest.approx(50 - 0.08)

    def test_max_loss_no_wings(self):
        condor = IronCondorPosition(
            group_id="IC_001",
            entry_time=datetime.now(timezone.utc),
            underlying_price=2500.0,
        )
        assert condor.max_loss == float("inf")


# ======================================================================
# 2. Open Iron Condor
# ======================================================================


class TestOpenIronCondor:
    @staticmethod
    def _make_filled_legs(legs: list[LegOrder], **kwargs) -> list[LegOrder]:
        """Simulate LimitChaser filling all legs."""
        for leg in legs:
            leg.status = "FILLED"
            leg.avg_price = 0.03
            leg.fee = 0.001
            leg.filled_qty = leg.quantity
            leg.order_id = "ORD_001"
        return legs

    def test_successful_open(self, pos_mgr, mock_client):
        """4 腿全部 FILLED → 返回 condor."""
        with patch.object(pos_mgr.chaser, "execute_legs", side_effect=self._make_filled_legs) as exec_mock:
            condor = pos_mgr.open_iron_condor(
                sell_call_symbol="ETH-260321-2700-C",
                buy_call_symbol="ETH-260321-2750-C",
                sell_put_symbol="ETH-260321-2300-P",
                buy_put_symbol="ETH-260321-2250-P",
                sell_call_strike=2700,
                buy_call_strike=2750,
                sell_put_strike=2300,
                buy_put_strike=2250,
                quantity=1.0,
                underlying_price=2500.0,
            )

        assert condor is not None
        assert len(condor.legs) == 4
        assert condor.is_open is True

    def test_market_open_splits_into_one_btc_batches(self, pos_mgr, mock_client):
        mock_client.get_positions.side_effect = [
            [
                {"symbol": "BTC-260321-100000-C", "side": "SHORT", "quantity": 1.0, "entryPrice": 0.05},
                {"symbol": "BTC-260321-80000-P", "side": "SHORT", "quantity": 1.0, "entryPrice": 0.05},
            ],
            [
                {"symbol": "BTC-260321-100000-C", "side": "SHORT", "quantity": 1.0, "entryPrice": 0.05},
                {"symbol": "BTC-260321-80000-P", "side": "SHORT", "quantity": 1.0, "entryPrice": 0.05},
            ],
            [
                {"symbol": "BTC-260321-100000-C", "side": "SHORT", "quantity": 1.0, "entryPrice": 0.05},
                {"symbol": "BTC-260321-80000-P", "side": "SHORT", "quantity": 1.0, "entryPrice": 0.05},
            ],
            [
                {"symbol": "BTC-260321-100000-C", "side": "SHORT", "quantity": 2.0, "entryPrice": 0.05},
                {"symbol": "BTC-260321-80000-P", "side": "SHORT", "quantity": 2.0, "entryPrice": 0.05},
            ],
        ]

        condor = pos_mgr.open_short_strangle(
            sell_call_symbol="BTC-260321-100000-C",
            sell_put_symbol="BTC-260321-80000-P",
            sell_call_strike=100000,
            sell_put_strike=80000,
            quantity=2.0,
            underlying_price=90000.0,
            execution_mode="market",
        )

        assert condor is not None
        assert mock_client.submit_order.call_count == 4
        quantities = [float(call.kwargs["quantity"]) for call in mock_client.submit_order.call_args_list]
        assert quantities == [1.0, 1.0, 1.0, 1.0]

    def test_market_open_does_not_upsize_sub_one_btc_order(self, pos_mgr, mock_client):
        mock_client.get_positions.side_effect = [[
            {"symbol": "BTC-260321-100000-C", "side": "SHORT", "quantity": 0.4, "entryPrice": 0.05},
            {"symbol": "BTC-260321-80000-P", "side": "SHORT", "quantity": 0.4, "entryPrice": 0.05},
        ]]

        condor = pos_mgr.open_short_strangle(
            sell_call_symbol="BTC-260321-100000-C",
            sell_put_symbol="BTC-260321-80000-P",
            sell_call_strike=100000,
            sell_put_strike=80000,
            quantity=0.4,
            underlying_price=90000.0,
            execution_mode="market",
        )

        assert condor is not None
        assert mock_client.submit_order.call_count == 2
        quantities = [float(call.kwargs["quantity"]) for call in mock_client.submit_order.call_args_list]
        assert quantities == [0.4, 0.4]

    def test_market_open_cancels_unresolved_order_and_supplements_gap(self, pos_mgr, mock_client):
        submit_results = [
            _make_order_result(side="SELL", quantity=1.0, symbol="BTC-260321-100000-C"),
            OrderResult(
                order_id="ORD_PUT_1",
                symbol="BTC-260321-80000-P",
                side="SELL",
                quantity=0.0,
                price=0.0,
                avg_price=0.0,
                status="NEW",
                fee=0.0,
                raw={},
            ),
            _make_order_result(side="SELL", quantity=0.6, symbol="BTC-260321-80000-P"),
        ]
        mock_client.submit_order.side_effect = submit_results
        mock_client.query_order.side_effect = [
            OrderResult("ORD_PUT_1", "BTC-260321-80000-P", "SELL", 0.0, 0.0, 0.0, "NEW", 0.0, {}),
            OrderResult("ORD_PUT_1", "BTC-260321-80000-P", "SELL", 0.0, 0.0, 0.0, "NEW", 0.0, {}),
            OrderResult("ORD_PUT_1", "BTC-260321-80000-P", "SELL", 0.0, 0.0, 0.0, "NEW", 0.0, {}),
            OrderResult("ORD_PUT_1", "BTC-260321-80000-P", "SELL", 0.0, 0.0, 0.0, "NEW", 0.0, {}),
        ]
        mock_client.get_positions.side_effect = [
            [
                {"symbol": "BTC-260321-100000-C", "side": "SHORT", "quantity": 1.0, "entryPrice": 0.05},
                {"symbol": "BTC-260321-80000-P", "side": "SHORT", "quantity": 0.4, "entryPrice": 0.05},
            ],
            [
                {"symbol": "BTC-260321-100000-C", "side": "SHORT", "quantity": 1.0, "entryPrice": 0.05},
                {"symbol": "BTC-260321-80000-P", "side": "SHORT", "quantity": 0.4, "entryPrice": 0.05},
            ],
            [
                {"symbol": "BTC-260321-100000-C", "side": "SHORT", "quantity": 1.0, "entryPrice": 0.05},
                {"symbol": "BTC-260321-80000-P", "side": "SHORT", "quantity": 0.4, "entryPrice": 0.05},
            ],
            [
                {"symbol": "BTC-260321-100000-C", "side": "SHORT", "quantity": 1.0, "entryPrice": 0.05},
                {"symbol": "BTC-260321-80000-P", "side": "SHORT", "quantity": 1.0, "entryPrice": 0.05},
            ],
        ]

        condor = pos_mgr.open_short_strangle(
            sell_call_symbol="BTC-260321-100000-C",
            sell_put_symbol="BTC-260321-80000-P",
            sell_call_strike=100000,
            sell_put_strike=80000,
            quantity=1.0,
            underlying_price=90000.0,
            execution_mode="market",
        )

        assert condor is not None
        assert [float(call.kwargs["quantity"]) for call in mock_client.submit_order.call_args_list] == [1.0, 1.0, 0.6]
        assert mock_client.cancel_order.call_count >= 1

    def test_failed_leg_triggers_rollback(self, pos_mgr, mock_client):
        """某腿挂单失败时应回滚已成交的腿."""
        def _partial_fill(legs, **kwargs):
            """前 2 腿 FILLED, 第 3 腿 FAILED."""
            for i, leg in enumerate(legs):
                if i < 2:
                    leg.status = "FILLED"
                    leg.avg_price = 0.05
                    leg.fee = 0.001
                    leg.filled_qty = leg.quantity
                    leg.order_id = f"ORD_{i}"
                else:
                    leg.status = "FAILED"
            return legs

        mock_client.submit_order.side_effect = lambda **kwargs: _make_order_result(
            side=kwargs.get("side", "SELL"),
            quantity=float(kwargs.get("quantity", 1.0)),
            symbol=kwargs.get("symbol", "ETH-260321-2700-C"),
        )

        with patch.object(pos_mgr.chaser, "execute_legs", side_effect=_partial_fill):
            condor = pos_mgr.open_iron_condor(
                sell_call_symbol="SC", buy_call_symbol="BC",
                sell_put_symbol="SP", buy_put_symbol="BP",
                sell_call_strike=2700, buy_call_strike=2750,
                sell_put_strike=2300, buy_put_strike=2250,
                quantity=1.0, underlying_price=2500.0,
            )

        assert condor is None
        # 2 filled legs should be rolled back via market orders
        assert mock_client.submit_order.call_count == 2

    def test_failed_leg_rollback_uses_actual_filled_qty(self, pos_mgr, mock_client):
        """回滚应按已成交量，而不是原始请求量。"""
        def _partial_fill(legs, **kwargs):
            for i, leg in enumerate(legs):
                if i < 2:
                    leg.status = "FILLED"
                    leg.avg_price = 0.05
                    leg.fee = 0.001
                    leg.filled_qty = 0.4
                    leg.order_id = f"ORD_{i}"
                else:
                    leg.status = "FAILED"
            return legs

        mock_client.submit_order.side_effect = lambda **kwargs: _make_order_result(
            side=kwargs.get("side", "SELL"),
            quantity=float(kwargs.get("quantity", 1.0)),
            symbol=kwargs.get("symbol", "ETH-260321-2700-C"),
        )

        with patch.object(pos_mgr.chaser, "execute_legs", side_effect=_partial_fill):
            condor = pos_mgr.open_iron_condor(
                sell_call_symbol="SC", buy_call_symbol="BC",
                sell_put_symbol="SP", buy_put_symbol="BP",
                sell_call_strike=2700, buy_call_strike=2750,
                sell_put_strike=2300, buy_put_strike=2250,
                quantity=1.0, underlying_price=2500.0,
            )

        assert condor is None
        quantities = [call.kwargs["quantity"] for call in mock_client.submit_order.call_args_list]
        assert quantities == [0.4, 0.4]

    def test_successful_open_records_actual_filled_qty(self, pos_mgr, storage):
        """入库与持仓数量应以实际成交量为准。"""
        def _filled_partial_size(legs, **kwargs):
            for leg in legs:
                leg.status = "FILLED"
                leg.avg_price = 0.03
                leg.fee = 0.001
                leg.filled_qty = 0.6
                leg.order_id = "ORD_001"
            return legs

        with patch.object(pos_mgr.chaser, "execute_legs", side_effect=_filled_partial_size):
            condor = pos_mgr.open_iron_condor(
                sell_call_symbol="ETH-260321-2700-C",
                buy_call_symbol="ETH-260321-2750-C",
                sell_put_symbol="ETH-260321-2300-P",
                buy_put_symbol="ETH-260321-2250-P",
                sell_call_strike=2700,
                buy_call_strike=2750,
                sell_put_strike=2300,
                buy_put_strike=2250,
                quantity=1.0,
                underlying_price=2500.0,
            )

        assert condor is not None
        assert all(leg.quantity == pytest.approx(0.6) for leg in condor.legs)
        trades = storage.get_open_trades(condor.group_id)
        assert len(trades) == 4
        assert all(t["quantity"] == pytest.approx(0.6) for t in trades)

    def test_exception_triggers_rollback(self, pos_mgr, mock_client):
        """下单抛异常也应回滚."""
        with patch.object(pos_mgr.chaser, "execute_legs", side_effect=Exception("Connection error")):
            condor = pos_mgr.open_iron_condor(
                sell_call_symbol="SC", buy_call_symbol="BC",
                sell_put_symbol="SP", buy_put_symbol="BP",
                sell_call_strike=2700, buy_call_strike=2750,
                sell_put_strike=2300, buy_put_strike=2250,
                quantity=1.0, underlying_price=2500.0,
            )

        assert condor is None


# ======================================================================
# 3. Close Iron Condor & PnL
# ======================================================================


class TestCloseIronCondor:
    def _setup_open_condor(self, pos_mgr, storage, qty: float = 1.0):
        """手动注入一个已开仓的 condor."""
        gid = "IC_TEST"

        # 模拟 4 笔 trade
        tid1 = storage.record_trade(gid, "SP", "SELL", qty, 0.03, meta={"leg_role": "sell_put"})
        tid2 = storage.record_trade(gid, "BP", "BUY", qty, 0.01, meta={"leg_role": "buy_put"})
        tid3 = storage.record_trade(gid, "SC", "SELL", qty, 0.04, meta={"leg_role": "sell_call"})
        tid4 = storage.record_trade(gid, "BC", "BUY", qty, 0.015, meta={"leg_role": "buy_call"})

        condor = IronCondorPosition(
            group_id=gid,
            entry_time=datetime.now(timezone.utc),
            underlying_price=2500.0,
            sell_put=CondorLeg("SP", "SELL", "put", 2300, qty, 0.03, tid1),
            buy_put=CondorLeg("BP", "BUY", "put", 2250, qty, 0.01, tid2),
            sell_call=CondorLeg("SC", "SELL", "call", 2700, qty, 0.04, tid3),
            buy_call=CondorLeg("BC", "BUY", "call", 2750, qty, 0.015, tid4),
            total_premium=0.045,  # (0.03+0.04) - (0.01+0.015)
        )
        pos_mgr.open_condors[gid] = condor
        return gid

    def test_close_all_legs(self, pos_mgr, mock_client, storage):
        gid = self._setup_open_condor(pos_mgr, storage)

        def _fill_close_legs(legs, **kwargs):
            for leg in legs:
                leg.status = "FILLED"
                leg.avg_price = 0.0
                leg.fee = 0.0
                leg.filled_qty = leg.quantity
                leg.order_id = "CLOSE_ORD"
            return legs

        with patch.object(pos_mgr.chaser, "execute_legs", side_effect=_fill_close_legs):
            pnl = pos_mgr.close_iron_condor(gid, reason="test")
        assert gid not in pos_mgr.open_condors

    def test_close_nonexistent(self, pos_mgr):
        pnl = pos_mgr.close_iron_condor("FAKE_ID")
        assert pnl == 0.0

    def test_close_all(self, pos_mgr, mock_client, storage):
        gid1 = self._setup_open_condor(pos_mgr, storage)

        def _fill_close_legs(legs, **kwargs):
            for leg in legs:
                leg.status = "FILLED"
                leg.avg_price = 0.0
                leg.fee = 0.0
                leg.filled_qty = leg.quantity
                leg.order_id = "CLOSE_ORD"
            return legs

        with patch.object(pos_mgr.chaser, "execute_legs", side_effect=_fill_close_legs):
            total = pos_mgr.close_all(reason="emergency")
        assert len(pos_mgr.open_condors) == 0

    def test_market_close_splits_into_one_btc_batches(self, pos_mgr, mock_client, storage):
        gid = self._setup_open_condor(pos_mgr, storage, qty=2.0)
        mock_client.get_positions.side_effect = [
            [
                {"symbol": "SP", "side": "SHORT", "quantity": 1.0, "entryPrice": 0.03},
                {"symbol": "BP", "side": "LONG", "quantity": 1.0, "entryPrice": 0.01},
                {"symbol": "SC", "side": "SHORT", "quantity": 1.0, "entryPrice": 0.04},
                {"symbol": "BC", "side": "LONG", "quantity": 1.0, "entryPrice": 0.015},
            ],
            [
                {"symbol": "SP", "side": "SHORT", "quantity": 1.0, "entryPrice": 0.03},
                {"symbol": "BP", "side": "LONG", "quantity": 1.0, "entryPrice": 0.01},
                {"symbol": "SC", "side": "SHORT", "quantity": 1.0, "entryPrice": 0.04},
                {"symbol": "BC", "side": "LONG", "quantity": 1.0, "entryPrice": 0.015},
            ],
            [
                {"symbol": "SP", "side": "SHORT", "quantity": 1.0, "entryPrice": 0.03},
                {"symbol": "BP", "side": "LONG", "quantity": 1.0, "entryPrice": 0.01},
                {"symbol": "SC", "side": "SHORT", "quantity": 1.0, "entryPrice": 0.04},
                {"symbol": "BC", "side": "LONG", "quantity": 1.0, "entryPrice": 0.015},
            ],
            [],
        ]

        pnl = pos_mgr.close_iron_condor(gid, reason="test_market", execution_mode="market")

        assert gid not in pos_mgr.open_condors
        assert mock_client.submit_order.call_count == 8
        quantities = [float(call.kwargs["quantity"]) for call in mock_client.submit_order.call_args_list]
        assert quantities == [1.0] * 8

    def test_market_close_cancels_unresolved_order_and_supplements_gap(self, pos_mgr, mock_client, storage):
        gid = self._setup_open_condor(pos_mgr, storage, qty=1.0)
        submit_results = [
            _make_order_result(side="BUY", quantity=1.0, symbol="SP"),
            OrderResult("ORD_BP_1", "BP", "SELL", 0.0, 0.0, 0.0, "NEW", 0.0, {}),
            _make_order_result(side="BUY", quantity=1.0, symbol="SC"),
            _make_order_result(side="SELL", quantity=1.0, symbol="BC"),
            _make_order_result(side="SELL", quantity=0.6, symbol="BP"),
        ]
        mock_client.submit_order.side_effect = submit_results
        mock_client.query_order.side_effect = [
            OrderResult("ORD_BP_1", "BP", "SELL", 0.0, 0.0, 0.0, "NEW", 0.0, {}),
            OrderResult("ORD_BP_1", "BP", "SELL", 0.0, 0.0, 0.0, "NEW", 0.0, {}),
            OrderResult("ORD_BP_1", "BP", "SELL", 0.0, 0.0, 0.0, "NEW", 0.0, {}),
            OrderResult("ORD_BP_1", "BP", "SELL", 0.0, 0.0, 0.0, "NEW", 0.0, {}),
        ]
        mock_client.get_positions.side_effect = [
            [
                {"symbol": "BP", "side": "LONG", "quantity": 0.4, "entryPrice": 0.01},
            ],
            [
                {"symbol": "BP", "side": "LONG", "quantity": 0.4, "entryPrice": 0.01},
            ],
            [
                {"symbol": "BP", "side": "LONG", "quantity": 0.4, "entryPrice": 0.01},
            ],
            [],
        ]

        pnl = pos_mgr.close_iron_condor(gid, reason="supplement_close", execution_mode="market")

        assert gid not in pos_mgr.open_condors
        assert [call.kwargs["symbol"] for call in mock_client.submit_order.call_args_list].count("BP") == 2
        assert any(float(call.kwargs["quantity"]) == 0.6 for call in mock_client.submit_order.call_args_list)
        assert mock_client.cancel_order.call_count >= 1

    def test_emergency_close_all_exchange_positions_splits_batches(self, pos_mgr, mock_client, storage):
        gid = self._setup_open_condor(pos_mgr, storage, qty=2.0)
        mock_client.get_positions.side_effect = [
            [
                {"symbol": "SP", "side": "SHORT", "quantity": 2.0, "entryPrice": 0.03},
                {"symbol": "BP", "side": "LONG", "quantity": 2.0, "entryPrice": 0.01},
                {"symbol": "SC", "side": "SHORT", "quantity": 2.0, "entryPrice": 0.04},
                {"symbol": "BC", "side": "LONG", "quantity": 2.0, "entryPrice": 0.015},
            ],
            [
                {"symbol": "SP", "side": "SHORT", "quantity": 2.0, "entryPrice": 0.03},
                {"symbol": "BP", "side": "LONG", "quantity": 2.0, "entryPrice": 0.01},
                {"symbol": "SC", "side": "SHORT", "quantity": 2.0, "entryPrice": 0.04},
                {"symbol": "BC", "side": "LONG", "quantity": 2.0, "entryPrice": 0.015},
            ],
            [
                {"symbol": "SP", "side": "SHORT", "quantity": 1.0, "entryPrice": 0.03},
                {"symbol": "BP", "side": "LONG", "quantity": 1.0, "entryPrice": 0.01},
                {"symbol": "SC", "side": "SHORT", "quantity": 1.0, "entryPrice": 0.04},
                {"symbol": "BC", "side": "LONG", "quantity": 1.0, "entryPrice": 0.015},
            ],
            [
                {"symbol": "SP", "side": "SHORT", "quantity": 1.0, "entryPrice": 0.03},
                {"symbol": "BP", "side": "LONG", "quantity": 1.0, "entryPrice": 0.01},
                {"symbol": "SC", "side": "SHORT", "quantity": 1.0, "entryPrice": 0.04},
                {"symbol": "BC", "side": "LONG", "quantity": 1.0, "entryPrice": 0.015},
            ],
            [
                {"symbol": "SP", "side": "SHORT", "quantity": 1.0, "entryPrice": 0.03},
                {"symbol": "BP", "side": "LONG", "quantity": 1.0, "entryPrice": 0.01},
                {"symbol": "SC", "side": "SHORT", "quantity": 1.0, "entryPrice": 0.04},
                {"symbol": "BC", "side": "LONG", "quantity": 1.0, "entryPrice": 0.015},
            ],
            [],
            [],
            [],
            [],
        ]

        total = pos_mgr.close_all_exchange_positions(reason="emergency_test")

        assert mock_client.submit_order.call_count == 8
        quantities = [float(call.kwargs["quantity"]) for call in mock_client.submit_order.call_args_list]
        assert quantities == [1.0] * 8


# ======================================================================
# 4. Unrealized PnL
# ======================================================================


class TestUnrealizedPnl:
    def test_zero_with_no_positions(self, pos_mgr):
        assert pos_mgr.get_unrealized_pnl({}) == 0.0

    def test_compute_upnl(self, pos_mgr, storage):
        gid = "IC_UPNL"
        tid = storage.record_trade(gid, "SC", "SELL", 1.0, 0.05)
        condor = IronCondorPosition(
            group_id=gid,
            entry_time=datetime.now(timezone.utc),
            underlying_price=2500.0,
            sell_call=CondorLeg("SC", "SELL", "call", 2700, 1.0, 0.05, tid),
        )
        pos_mgr.open_condors[gid] = condor

        # mark < entry → short leg profit
        upnl = pos_mgr.get_unrealized_pnl({"SC": 0.02})
        assert upnl == pytest.approx(0.03)  # (0.05 - 0.02) * 1.0

    def test_upnl_long_leg(self, pos_mgr, storage):
        gid = "IC_LONG"
        tid = storage.record_trade(gid, "BC", "BUY", 1.0, 0.01)
        condor = IronCondorPosition(
            group_id=gid,
            entry_time=datetime.now(timezone.utc),
            underlying_price=2500.0,
            buy_call=CondorLeg("BC", "BUY", "call", 2750, 1.0, 0.01, tid),
        )
        pos_mgr.open_condors[gid] = condor

        # mark > entry → long leg profit
        upnl = pos_mgr.get_unrealized_pnl({"BC": 0.03})
        assert upnl == pytest.approx(0.02)  # (0.03 - 0.01) * 1.0

    def test_upnl_missing_mark_uses_entry(self, pos_mgr, storage):
        """mark_prices 中没有该 symbol → 用 entry_price, PnL=0."""
        gid = "IC_MISS"
        tid = storage.record_trade(gid, "X", "SELL", 1.0, 0.05)
        condor = IronCondorPosition(
            group_id=gid,
            entry_time=datetime.now(timezone.utc),
            underlying_price=2500.0,
            sell_call=CondorLeg("X", "SELL", "call", 2700, 1.0, 0.05, tid),
        )
        pos_mgr.open_condors[gid] = condor

        upnl = pos_mgr.get_unrealized_pnl({})
        assert upnl == pytest.approx(0.0)


# ======================================================================
# 5. Summary & open_position_count
# ======================================================================


class TestSummary:
    def test_empty_summary(self, pos_mgr):
        assert pos_mgr.open_position_count == 0
        s = pos_mgr.summary()
        assert s["open_condors"] == 0

    def test_with_positions(self, pos_mgr, storage):
        gid = "IC_SUM"
        tid = storage.record_trade(gid, "SC", "SELL", 1.0, 0.05)
        condor = IronCondorPosition(
            group_id=gid,
            entry_time=datetime.now(timezone.utc),
            underlying_price=2500.0,
            sell_call=CondorLeg("SC", "SELL", "call", 2700, 1.0, 0.05, tid),
            total_premium=0.05,
        )
        pos_mgr.open_condors[gid] = condor

        s = pos_mgr.summary()
        assert s["open_condors"] == 1
        assert gid in s["condors"]
