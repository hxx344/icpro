"""Tests for preview sizing helpers and Binance formula helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from trader.binance_client import OptionTicker
from trader.order_preview import compute_option_order_preview
from trader.strategy import (
    _binance_option_otm_amount,
    estimate_binance_combo_open_margin_per_unit,
    estimate_binance_long_open_margin_per_unit,
    estimate_binance_option_fee,
    estimate_binance_short_open_margin_per_unit,
)


def _make_ticker(
    symbol: str,
    strike: float,
    option_type: str,
    *,
    bid: float,
    ask: float,
    mark: float,
    spot: float = 90000.0,
) -> OptionTicker:
    return OptionTicker(
        symbol=symbol,
        underlying="BTC",
        strike=strike,
        option_type=option_type,
        expiry=datetime.now(timezone.utc) + timedelta(days=7),
        bid_price=bid,
        ask_price=ask,
        mark_price=mark,
        last_price=mark,
        underlying_price=spot,
        volume_24h=1000.0,
        open_interest=500.0,
    )


class TestBinanceFormulaHelpers:
    def test_otm_amount_for_call_and_put(self):
        assert _binance_option_otm_amount(90000.0, 100000.0, "call") == 10000.0
        assert _binance_option_otm_amount(90000.0, 80000.0, "put") == 10000.0
        assert _binance_option_otm_amount(90000.0, 85000.0, "call") == 0.0
        assert _binance_option_otm_amount(90000.0, 95000.0, "put") == 0.0

    def test_fee_uses_smaller_of_index_component_and_10pct_order_price(self):
        fee = estimate_binance_option_fee(index_price=90000.0, order_price=100.0)
        assert fee == pytest.approx(10.0)

        fee = estimate_binance_option_fee(
            index_price=1000.0,
            order_price=500.0,
            contract_unit=1.0,
            transaction_fee_rate=0.0001,
        )
        assert fee == pytest.approx(0.1)

    def test_short_open_margin_matches_formula(self):
        margin = estimate_binance_short_open_margin_per_unit(
            index_price=90000.0,
            strike=100000.0,
            option_type="call",
            mark_price=110.0,
            order_price=100.0,
        )
        # max(9000, 13500 - 10000) + 110 - 100 + 10 = 9020
        assert margin == pytest.approx(9020.0)

    def test_long_open_margin_equals_order_plus_fee(self):
        margin = estimate_binance_long_open_margin_per_unit(
            index_price=90000.0,
            order_price=120.0,
        )
        assert margin == pytest.approx(132.0)

    def test_combo_margin_sums_short_and_long_legs(self):
        sell_call = _make_ticker("BTC-260411-100000-C", 100000.0, "call", bid=100.0, ask=110.0, mark=105.0)
        sell_put = _make_ticker("BTC-260411-80000-P", 80000.0, "put", bid=90.0, ask=100.0, mark=95.0)
        buy_call = _make_ticker("BTC-260411-105000-C", 105000.0, "call", bid=60.0, ask=70.0, mark=65.0)
        buy_put = _make_ticker("BTC-260411-75000-P", 75000.0, "put", bid=55.0, ask=65.0, mark=60.0)

        margin = estimate_binance_combo_open_margin_per_unit(
            index_price=90000.0,
            sell_call=sell_call,
            sell_put=sell_put,
            buy_call=buy_call,
            buy_put=buy_put,
        )
        expected = 9015.0 + 9014.0 + 77.0 + 71.5
        assert margin == pytest.approx(expected)


class TestOrderPreviewHelpers:
    def test_preview_strangle_uses_formula_margin_and_scaled_quantity(self):
        sell_call = _make_ticker("BTC-260411-100000-C", 100000.0, "call", bid=100.0, ask=110.0, mark=105.0)
        sell_put = _make_ticker("BTC-260411-80000-P", 80000.0, "put", bid=90.0, ask=100.0, mark=95.0)

        preview = compute_option_order_preview(
            mode="strangle",
            spot=90000.0,
            base_quantity=0.01,
            compound=True,
            equity=10000.0,
            available_balance=8000.0,
            max_capital_pct=0.30,
            otm_pct=0.10,
            wing_width_pct=0.05,
            sell_call=sell_call,
            sell_put=sell_put,
        )

        assert preview["quantity"] == pytest.approx(0.16, abs=0.01)
        assert preview["margin_per_unit"] == pytest.approx(18029.0)
        assert preview["margin_used"] == pytest.approx(preview["margin_per_unit"] * preview["quantity"])
        assert preview["break_even_upper"] == pytest.approx(100190.0)
        assert preview["break_even_lower"] == pytest.approx(79810.0)

    def test_preview_weekend_vol_caps_leverage_by_margin_budget(self):
        sell_call = _make_ticker("BTC-260411-95000-C", 95000.0, "call", bid=90.0, ask=100.0, mark=95.0, spot=87000.0)
        sell_put = _make_ticker("BTC-260411-79000-P", 79000.0, "put", bid=90.0, ask=100.0, mark=95.0, spot=87000.0)

        preview = compute_option_order_preview(
            mode="weekend_vol",
            spot=87000.0,
            base_quantity=0.1,
            compound=True,
            equity=10000.0,
            available_balance=8000.0,
            leverage=10.0,
            otm_pct=0.10,
            wing_width_pct=0.0,
            sell_call=sell_call,
            sell_put=sell_put,
        )

        assert preview["quantity"] == pytest.approx(0.4, abs=0.11)
        assert preview["equity_source"] == "实时权益(复利 10x)"

    def test_preview_iron_condor_returns_total_max_loss(self):
        sell_call = _make_ticker("BTC-260411-100000-C", 100000.0, "call", bid=100.0, ask=110.0, mark=105.0)
        sell_put = _make_ticker("BTC-260411-80000-P", 80000.0, "put", bid=90.0, ask=100.0, mark=95.0)
        buy_call = _make_ticker("BTC-260411-105000-C", 105000.0, "call", bid=60.0, ask=70.0, mark=65.0)
        buy_put = _make_ticker("BTC-260411-75000-P", 75000.0, "put", bid=55.0, ask=65.0, mark=60.0)

        preview = compute_option_order_preview(
            mode="iron_condor",
            spot=90000.0,
            base_quantity=0.5,
            compound=False,
            otm_pct=0.10,
            wing_width_pct=0.05,
            sell_call=sell_call,
            sell_put=sell_put,
            buy_call=buy_call,
            buy_put=buy_put,
        )

        assert preview["is_iron_condor"] is True
        assert preview["total_premium"] == pytest.approx(27.5)
        assert preview["total_max_loss"] == pytest.approx(2472.5)
        assert preview["margin_used"] == pytest.approx(preview["margin_per_unit"] * 0.5)
