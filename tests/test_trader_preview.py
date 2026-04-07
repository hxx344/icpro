"""Tests for preview sizing helpers and Bybit formula helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from trader.bybit_client import OptionTicker
from trader.config import ExchangeConfig
from trader.order_preview import compute_option_order_preview
from trader.strategy import (
    _bybit_option_otm_amount,
    estimate_bybit_combo_open_margin_per_unit,
    estimate_bybit_long_open_margin_per_unit,
    estimate_bybit_option_fee,
    estimate_bybit_short_open_margin_per_unit,
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


class TestBybitFormulaHelpers:
    def test_otm_amount_for_call_and_put(self):
        assert _bybit_option_otm_amount(90000.0, 100000.0, "call") == 10000.0
        assert _bybit_option_otm_amount(90000.0, 80000.0, "put") == 10000.0
        assert _bybit_option_otm_amount(90000.0, 85000.0, "call") == 0.0
        assert _bybit_option_otm_amount(90000.0, 95000.0, "put") == 0.0

    def test_fee_uses_smaller_of_index_component_and_7pct_order_price(self):
        fee = estimate_bybit_option_fee(index_price=90000.0, order_price=100.0)
        assert fee == pytest.approx(7.0)

        fee = estimate_bybit_option_fee(
            index_price=1000.0,
            order_price=500.0,
            contract_unit=1.0,
            transaction_fee_rate=0.0001,
        )
        assert fee == pytest.approx(0.1)

    def test_short_open_margin_matches_formula(self):
        margin = estimate_bybit_short_open_margin_per_unit(
            index_price=90000.0,
            strike=100000.0,
            option_type="call",
            mark_price=110.0,
            order_price=100.0,
        )
        assert margin == pytest.approx(8217.0)

    def test_long_open_margin_equals_order_plus_fee(self):
        margin = estimate_bybit_long_open_margin_per_unit(
            index_price=90000.0,
            order_price=120.0,
        )
        assert margin == pytest.approx(128.4)

    def test_combo_margin_sums_short_and_long_legs(self):
        sell_call = _make_ticker("BTC-11APR26-100000-C", 100000.0, "call", bid=100.0, ask=110.0, mark=105.0)
        sell_put = _make_ticker("BTC-11APR26-80000-P", 80000.0, "put", bid=90.0, ask=100.0, mark=95.0)
        buy_call = _make_ticker("BTC-11APR26-105000-C", 105000.0, "call", bid=60.0, ask=70.0, mark=65.0)
        buy_put = _make_ticker("BTC-11APR26-75000-P", 75000.0, "put", bid=55.0, ask=65.0, mark=60.0)

        margin = estimate_bybit_combo_open_margin_per_unit(
            index_price=90000.0,
            sell_call=sell_call,
            sell_put=sell_put,
            buy_call=buy_call,
            buy_put=buy_put,
            exchange_cfg=ExchangeConfig(),
        )
        assert margin > 0


class TestOrderPreviewHelpers:
    def test_preview_weekend_vol_without_wings_uses_fixed_quantity(self):
        sell_call = _make_ticker("BTC-11APR26-100000-C", 100000.0, "call", bid=100.0, ask=110.0, mark=105.0)
        sell_put = _make_ticker("BTC-11APR26-80000-P", 80000.0, "put", bid=90.0, ask=100.0, mark=95.0)

        preview = compute_option_order_preview(
            spot=90000.0,
            base_quantity=5.0,
            compound=False,
            equity=10000.0,
            available_balance=8000.0,
            leverage=3.0,
            sell_call=sell_call,
            sell_put=sell_put,
        )

        assert preview["quantity"] == pytest.approx(5.0)
        assert preview["margin_per_unit"] > 0
        assert preview["margin_used"] == pytest.approx(preview["margin_per_unit"] * preview["quantity"])
        assert preview["break_even_upper"] == pytest.approx(100190.0)
        assert preview["break_even_lower"] == pytest.approx(79810.0)

    def test_preview_equity_source_is_fixed_quantity(self):
        sell_call = _make_ticker("BTC-11APR26-95000-C", 95000.0, "call", bid=90.0, ask=100.0, mark=95.0, spot=87000.0)
        sell_put = _make_ticker("BTC-11APR26-79000-P", 79000.0, "put", bid=90.0, ask=100.0, mark=95.0, spot=87000.0)

        preview = compute_option_order_preview(
            spot=87000.0,
            base_quantity=2.0,
            compound=False,
            equity=10000.0,
            available_balance=8000.0,
            leverage=10.0,
            sell_call=sell_call,
            sell_put=sell_put,
        )

        assert preview["quantity"] == pytest.approx(2.0)
        assert preview["equity_source"] == "固定数量配置"

    def test_preview_winged_weekend_vol_returns_total_max_loss(self):
        sell_call = _make_ticker("BTC-11APR26-100000-C", 100000.0, "call", bid=100.0, ask=110.0, mark=105.0)
        sell_put = _make_ticker("BTC-11APR26-80000-P", 80000.0, "put", bid=90.0, ask=100.0, mark=95.0)
        buy_call = _make_ticker("BTC-11APR26-105000-C", 105000.0, "call", bid=60.0, ask=70.0, mark=65.0)
        buy_put = _make_ticker("BTC-11APR26-75000-P", 75000.0, "put", bid=55.0, ask=65.0, mark=60.0)

        preview = compute_option_order_preview(
            spot=90000.0,
            base_quantity=0.5,
            compound=False,
            sell_call=sell_call,
            sell_put=sell_put,
            buy_call=buy_call,
            buy_put=buy_put,
        )

        assert preview["is_iron_condor"] is True
        assert preview["total_premium"] == pytest.approx(27.5)
        assert preview["total_max_loss"] == pytest.approx(2472.5)
        assert preview["margin_used"] == pytest.approx(preview["margin_per_unit"] * 0.5)
