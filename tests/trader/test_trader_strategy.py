"""Tests for trader.strategy weekend_vol live strategy."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from trader.bybit_client import AccountInfo, BybitOptionsClient, OptionTicker
from trader.config import ExchangeConfig, StrategyConfig
from trader.position_manager import PositionManager
from trader.storage import Storage
from trader.strategy import WeekendVolStrategy, estimate_bybit_combo_open_margin_per_unit


def _make_ticker(
    symbol: str,
    strike: float,
    option_type: str,
    *,
    spot: float = 90000.0,
    bid: float = 100.0,
    ask: float = 110.0,
    mark: float = 105.0,
    expiry: datetime | None = None,
    delta: float = 0.0,
    mark_iv: float = 0.6,
) -> OptionTicker:
    return OptionTicker(
        symbol=symbol,
        underlying="BTC",
        strike=strike,
        option_type=option_type,
        expiry=expiry or (datetime.now(timezone.utc) + timedelta(days=2)),
        bid_price=bid,
        ask_price=ask,
        mark_price=mark,
        last_price=mark,
        underlying_price=spot,
        volume_24h=1000.0,
        open_interest=500.0,
        delta=delta,
        mark_iv=mark_iv,
    )


@pytest.fixture
def storage(tmp_path):
    s = Storage(str(tmp_path / "test_strategy.db"))
    yield s
    s.close()


@pytest.fixture
def client():
    c = MagicMock(spec=BybitOptionsClient)
    c.get_positions.return_value = []
    c.get_account.return_value = AccountInfo(
        total_balance=10000.0,
        available_balance=8000.0,
        unrealized_pnl=0.0,
        raw={},
    )
    c.get_spot_price.return_value = 90000.0
    c.get_realized_vol.return_value = 0.5
    c.get_mark_prices.return_value = {}
    c.enrich_greeks.return_value = None
    return c


@pytest.fixture
def pos_mgr(client, storage):
    return PositionManager(client, storage)


class TestWeekendVolStrategy:
    def test_should_enter_on_configured_day_and_time(self, client, pos_mgr, storage):
        cfg = StrategyConfig(underlying="BTC", entry_day="friday", entry_time_utc="16:00")
        strategy = WeekendVolStrategy(client, pos_mgr, storage, cfg)

        now = datetime(2026, 3, 27, 16, 10, tzinfo=timezone.utc)
        assert strategy._should_enter(now) is True

    def test_should_not_enter_after_10_minute_entry_window(self, client, pos_mgr, storage):
        cfg = StrategyConfig(underlying="BTC", entry_day="friday", entry_time_utc="16:00")
        strategy = WeekendVolStrategy(client, pos_mgr, storage, cfg)

        now = datetime(2026, 3, 27, 16, 11, tzinfo=timezone.utc)
        assert strategy._should_enter(now) is False

    def test_should_not_enter_when_risk_lock_active(self, client, pos_mgr, storage):
        cfg = StrategyConfig(underlying="BTC", entry_day="friday", entry_time_utc="16:00")
        strategy = WeekendVolStrategy(client, pos_mgr, storage, cfg)
        strategy._set_execution_risk_lock("test", event_type="position_open_partial", group_id="G1")

        now = datetime(2026, 3, 27, 16, 10, tzinfo=timezone.utc)
        assert strategy._should_enter(now) is False

    def test_compute_quantity_uses_fixed_quantity(self, client, pos_mgr, storage):
        cfg = StrategyConfig(underlying="BTC", quantity=5.0, leverage=6.0, compound=False)
        strategy = WeekendVolStrategy(client, pos_mgr, storage, cfg)

        sell_call = _make_ticker("BTC-29MAR26-96000-C", 96000.0, "call", bid=90.0, ask=100.0, mark=95.0)
        sell_put = _make_ticker("BTC-29MAR26-84000-P", 84000.0, "put", bid=95.0, ask=105.0, mark=100.0)

        qty = strategy._compute_quantity(90000.0, sell_call=sell_call, sell_put=sell_put)
        assert qty == pytest.approx(5.0)

    def test_margin_estimator_uses_exchange_config(self):
        sell_call = _make_ticker("BTC-29MAR26-96000-C", 96000.0, "call", bid=90.0, ask=100.0, mark=95.0)
        sell_put = _make_ticker("BTC-29MAR26-84000-P", 84000.0, "put", bid=95.0, ask=105.0, mark=100.0)
        exchange_cfg = ExchangeConfig()

        margin = estimate_bybit_combo_open_margin_per_unit(
            index_price=90000.0,
            sell_call=sell_call,
            sell_put=sell_put,
            exchange_cfg=exchange_cfg,
        )
        assert margin > 0

    def test_status_reports_weekend_vol_fields(self, client, pos_mgr, storage):
        cfg = StrategyConfig(underlying="BTC", target_delta=0.45, wing_delta=0.0)
        strategy = WeekendVolStrategy(client, pos_mgr, storage, cfg)

        status = strategy.status()
        assert status["mode"] == "weekend_vol"
        assert status["underlying"] == "BTC"
        assert status["target_delta"] == pytest.approx(0.45)
        assert status["wing_delta"] == pytest.approx(0.0)
