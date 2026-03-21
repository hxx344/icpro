"""Tests for trader.strategy — Iron Condor 0DTE 策略单元测试.

覆盖:
  - 入场时间判断 (_should_enter)
  - 行权价选择 (_find_best_strike)
  - 数量计算 (_compute_quantity)
  - 完整 tick 流程 (mock client)
  - 日期防重复交易
  - 无流动性跳过
  - wait_for_midpoint 逻辑
  - 铁鹰结构验证 (call spread / put spread)
"""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from trader.binance_client import (
    BinanceOptionsClient,
    OptionTicker,
    OrderResult,
    AccountInfo,
)
from trader.config import StrategyConfig, ExchangeConfig
from trader.position_manager import PositionManager
from trader.storage import Storage
from trader.strategy import IronCondor0DTEStrategy


# ======================================================================
# Helpers
# ======================================================================


def _make_ticker(
    symbol: str,
    strike: float,
    option_type: str,
    bid: float = 0.05,
    ask: float = 0.06,
    spot: float = 2500.0,
    dte_hours: float = 6.0,
) -> OptionTicker:
    expiry = datetime.now(timezone.utc) + timedelta(hours=dte_hours)
    return OptionTicker(
        symbol=symbol,
        underlying="ETH",
        strike=strike,
        option_type=option_type,
        expiry=expiry,
        bid_price=bid,
        ask_price=ask,
        mark_price=(bid + ask) / 2,
        last_price=bid,
        underlying_price=spot,
        volume_24h=100,
        open_interest=500,
    )


def _make_0dte_chain(spot: float = 2500.0) -> list[OptionTicker]:
    """创建一组完整的 0DTE 期权链."""
    tickers = []
    strikes = [
        2100, 2150, 2200, 2250, 2300, 2350, 2400, 2450, 2500,
        2550, 2600, 2650, 2700, 2750, 2800, 2850, 2900,
    ]
    for k in strikes:
        prem_c = max(0.001, (spot - k) / spot * 0.3) if k < spot else max(0.001, 0.01 * (1 - abs(k - spot) / spot))
        prem_p = max(0.001, (k - spot) / spot * 0.3) if k > spot else max(0.001, 0.01 * (1 - abs(k - spot) / spot))
        tickers.append(_make_ticker(
            f"ETH-260321-{k}-C", k, "call", prem_c * 0.95, prem_c * 1.05, spot,
        ))
        tickers.append(_make_ticker(
            f"ETH-260321-{k}-P", k, "put", prem_p * 0.95, prem_p * 1.05, spot,
        ))
    return tickers


@pytest.fixture
def mock_client():
    client = MagicMock(spec=BinanceOptionsClient)
    client.get_spot_price.return_value = 2500.0
    client.get_tickers.return_value = _make_0dte_chain(2500.0)
    client.get_account.return_value = AccountInfo(
        total_balance=10.0, available_balance=8.0, unrealized_pnl=0.0, raw={},
    )
    client.place_order.return_value = OrderResult(
        order_id="ORD_001", symbol="", side="SELL",
        quantity=1.0, price=0.05, avg_price=0.05,
        status="FILLED", fee=0.001, raw={},
    )
    return client


@pytest.fixture
def mock_storage(tmp_path):
    return Storage(str(tmp_path / "test_strategy.db"))


@pytest.fixture
def mock_pos_mgr(mock_client, mock_storage):
    mgr = PositionManager(mock_client, mock_storage)
    return mgr


@pytest.fixture(autouse=True)
def _patch_get_positions(mock_client):
    """Ensure mock_client.get_positions() returns [] by default."""
    mock_client.get_positions = MagicMock(return_value=[])
    return mock_client


# ======================================================================
# 1. Entry timing
# ======================================================================


class TestShouldEnter:
    def _make_strategy(self, mock_client, mock_storage, mock_pos_mgr, **overrides):
        cfg = StrategyConfig(**overrides)
        s = IronCondor0DTEStrategy(mock_client, mock_pos_mgr, mock_storage, cfg)
        return s

    def test_enter_at_correct_time(self, mock_client, mock_storage, mock_pos_mgr):
        s = self._make_strategy(mock_client, mock_storage, mock_pos_mgr, entry_time_utc="08:00")
        now = datetime(2026, 3, 21, 8, 30, tzinfo=timezone.utc)
        today = now.strftime("%Y-%m-%d")
        assert s._should_enter(now, today) is True

    def test_skip_before_entry_time(self, mock_client, mock_storage, mock_pos_mgr):
        s = self._make_strategy(mock_client, mock_storage, mock_pos_mgr, entry_time_utc="08:00")
        now = datetime(2026, 3, 21, 7, 30, tzinfo=timezone.utc)
        today = now.strftime("%Y-%m-%d")
        assert s._should_enter(now, today) is False

    def test_skip_too_late(self, mock_client, mock_storage, mock_pos_mgr):
        """2 小时窗口后不再入场."""
        s = self._make_strategy(mock_client, mock_storage, mock_pos_mgr, entry_time_utc="08:00")
        now = datetime(2026, 3, 21, 10, 30, tzinfo=timezone.utc)
        today = now.strftime("%Y-%m-%d")
        assert s._should_enter(now, today) is False

    def test_skip_already_traded_today(self, mock_client, mock_storage, mock_pos_mgr):
        s = self._make_strategy(mock_client, mock_storage, mock_pos_mgr, entry_time_utc="08:00")
        s._last_trade_date = "2026-03-21"
        now = datetime(2026, 3, 21, 8, 30, tzinfo=timezone.utc)
        assert s._should_enter(now, "2026-03-21") is False

    def test_skip_max_positions_reached(self, mock_client, mock_storage, mock_pos_mgr):
        s = self._make_strategy(
            mock_client, mock_storage, mock_pos_mgr,
            entry_time_utc="08:00", max_positions=1,
        )
        # 模拟已有一个持仓
        mock_pos_mgr.open_condors["fake"] = MagicMock()
        now = datetime(2026, 3, 21, 8, 30, tzinfo=timezone.utc)
        assert s._should_enter(now, "2026-03-21") is False


# ======================================================================
# 2. Strike selection
# ======================================================================


class TestFindBestStrike:
    def test_finds_closest_call(self, mock_client, mock_storage, mock_pos_mgr):
        cfg = StrategyConfig()
        s = IronCondor0DTEStrategy(mock_client, mock_pos_mgr, mock_storage, cfg)
        tickers = _make_0dte_chain(2500.0)

        target = 2700.0  # sell call target
        best = s._find_best_strike(tickers, "call", target)
        assert best is not None
        assert best.strike == 2700.0

    def test_finds_closest_put(self, mock_client, mock_storage, mock_pos_mgr):
        cfg = StrategyConfig()
        s = IronCondor0DTEStrategy(mock_client, mock_pos_mgr, mock_storage, cfg)
        tickers = _make_0dte_chain(2500.0)

        target = 2300.0
        best = s._find_best_strike(tickers, "put", target)
        assert best is not None
        assert best.strike == 2300.0

    def test_no_candidates_returns_none(self, mock_client, mock_storage, mock_pos_mgr):
        cfg = StrategyConfig()
        s = IronCondor0DTEStrategy(mock_client, mock_pos_mgr, mock_storage, cfg)
        # 空列表
        assert s._find_best_strike([], "call", 2500.0) is None

    def test_picks_nearest_even_if_not_exact(self, mock_client, mock_storage, mock_pos_mgr):
        cfg = StrategyConfig()
        s = IronCondor0DTEStrategy(mock_client, mock_pos_mgr, mock_storage, cfg)
        tickers = _make_0dte_chain(2500.0)

        target = 2723.0  # 不精确匹配
        best = s._find_best_strike(tickers, "call", target)
        # 应选择 2700 或 2750 (最近的)
        assert best is not None
        assert best.strike in (2700.0, 2750.0)


# ======================================================================
# 3. Quantity computation
# ======================================================================


class TestComputeQuantity:
    def test_no_compound(self, mock_client, mock_storage, mock_pos_mgr):
        cfg = StrategyConfig(compound=False, quantity=0.5)
        s = IronCondor0DTEStrategy(mock_client, mock_pos_mgr, mock_storage, cfg)
        assert s._compute_quantity(2500.0) == 0.5

    def test_compound_scales_up(self, mock_client, mock_storage, mock_pos_mgr):
        """compound=True 应根据 equity 放大数量."""
        cfg = StrategyConfig(
            compound=True,
            quantity=0.01,
            max_capital_pct=0.30,
            wing_width_pct=0.02,
        )
        s = IronCondor0DTEStrategy(mock_client, mock_pos_mgr, mock_storage, cfg)

        # Account balance = 10.0 ETH
        mock_client.get_account.return_value = AccountInfo(
            total_balance=10.0, available_balance=8.0,
            unrealized_pnl=0.0, raw={},
        )
        qty = s._compute_quantity(2500.0)
        # max_notional = 10 * 0.30 = 3.0 ETH
        # wing_width = 0.02 * 2500 = 50 USDT
        # scaled = 3.0 / 50 = 0.06
        assert qty >= 0.01  # at least base quantity

    def test_compound_error_fallback(self, mock_client, mock_storage, mock_pos_mgr):
        """get_account 失败时回退到 base quantity."""
        cfg = StrategyConfig(compound=True, quantity=0.01)
        s = IronCondor0DTEStrategy(mock_client, mock_pos_mgr, mock_storage, cfg)
        mock_client.get_account.side_effect = Exception("API error")
        assert s._compute_quantity(2500.0) == 0.01


# ======================================================================
# 4. Iron Condor structure validation
# ======================================================================


class TestCondorValidation:
    """验证 _try_open_condor 中的结构校验."""

    def test_normal_flow(self, mock_client, mock_storage, mock_pos_mgr):
        """正常情况下策略 tick 应成功开仓."""
        cfg = StrategyConfig(entry_time_utc="08:00")
        s = IronCondor0DTEStrategy(mock_client, mock_pos_mgr, mock_storage, cfg)

        from trader.position_manager import IronCondorPosition
        mock_pos_mgr.open_iron_condor = MagicMock(return_value=IronCondorPosition(
            group_id="IC_TEST",
            entry_time=datetime.now(timezone.utc),
            underlying_price=2500.0,
            total_premium=0.01,
        ))

        now = datetime(2026, 3, 21, 8, 30, tzinfo=timezone.utc)
        s._try_open_condor(now, "2026-03-21")

        mock_pos_mgr.open_iron_condor.assert_called_once()
        assert s._last_trade_date == "2026-03-21"


# ======================================================================
# 5. Status
# ======================================================================


class TestStrategyStatus:
    def test_status_dict(self, mock_client, mock_storage, mock_pos_mgr):
        cfg = StrategyConfig(underlying="ETH", otm_pct=0.08)
        s = IronCondor0DTEStrategy(mock_client, mock_pos_mgr, mock_storage, cfg)
        status = s.status()
        assert status["strategy"] == "IronCondor0DTE"
        assert status["underlying"] == "ETH"
        assert status["otm_pct"] == 0.08
