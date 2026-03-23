"""Unit tests for the trader package.

Tests cover:
  - Config: loading, defaults, env vars, YAML merge
  - Storage: CRUD for trades, equity snapshots, daily PnL, state
  - PositionManager: open/close condors, strangle, PnL tracking
  - Strategy: OptionSellingStrategy + WeekendVolStrategy entry logic
  - EquityTracker: snapshot, daily PnL, day rollover
  - LimitChaser: price computation, leg execution
  - Engine: init, start/stop lifecycle, status
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure trader package is importable
# ---------------------------------------------------------------------------
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


from trader.config import (
    ExchangeConfig,
    StrategyConfig,
    StorageConfig,
    MonitorConfig,
    ChaserConfig as ChaserCfgDataclass,
    TraderConfig,
    load_config,
    _merge_section,
)
from trader.storage import Storage
from trader.binance_client import (
    OptionTicker,
    OrderResult,
    AccountInfo,
    BinanceOptionsClient,
    _parse_symbol,
)
from trader.position_manager import (
    CondorLeg,
    IronCondorPosition,
    PositionManager,
)
from trader.limit_chaser import ChaserConfig, LegOrder, LimitChaser
from trader.equity import EquityTracker
from trader.strategy import OptionSellingStrategy, WeekendVolStrategy


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_db(tmp_path):
    """Return path to a temp SQLite database."""
    return str(tmp_path / "test_trader.db")


@pytest.fixture
def storage(tmp_db):
    """Create a Storage instance with a fresh temp DB."""
    s = Storage(tmp_db)
    yield s
    s.close()


@pytest.fixture
def default_config():
    """Build a TraderConfig with defaults + simulate_private=True."""
    cfg = TraderConfig()
    cfg.exchange.simulate_private = True
    cfg.exchange.testnet = True
    return cfg


def _make_ticker(
    symbol="BTC-260328-90000-C",
    underlying="BTC",
    strike=90000.0,
    option_type="call",
    expiry=None,
    bid_price=100.0,
    ask_price=120.0,
    mark_price=110.0,
    underlying_price=87000.0,
) -> OptionTicker:
    """Create a test OptionTicker."""
    if expiry is None:
        expiry = datetime(2026, 3, 28, 8, 0, 0, tzinfo=timezone.utc)
    return OptionTicker(
        symbol=symbol,
        underlying=underlying,
        strike=strike,
        option_type=option_type,
        expiry=expiry,
        bid_price=bid_price,
        ask_price=ask_price,
        mark_price=mark_price,
        last_price=mark_price,
        underlying_price=underlying_price,
        volume_24h=5000.0,
        open_interest=10000.0,
    )


class MockClient:
    """Mock Binance client for testing."""

    def __init__(self, spot=87000.0, equity=10000.0):
        self._spot = spot
        self._equity = equity
        self._tickers: list[OptionTicker] = []
        self._positions: list[dict] = []
        self._order_counter = 0

    def get_spot_price(self, underlying="BTC"):
        return self._spot

    def get_tickers(self, underlying="BTC"):
        return self._tickers

    def get_mark_prices(self, underlying="BTC"):
        return {t.symbol: t.mark_price for t in self._tickers}

    def enrich_greeks(self, tickers, underlying="BTC"):
        """Mock: no-op, tests set delta/mark_iv directly on tickers."""
        pass

    def get_account(self):
        return AccountInfo(
            total_balance=self._equity,
            available_balance=self._equity * 0.8,
            unrealized_pnl=0.0,
            raw={"simulated": True},
        )

    def get_positions(self, underlying="BTC"):
        return self._positions

    def place_order(self, symbol, side, quantity, order_type="MARKET",
                    price=None, reduce_only=False, client_order_id=None):
        self._order_counter += 1
        oid = f"TEST-{self._order_counter}"
        return OrderResult(
            order_id=oid,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price or 100.0,
            avg_price=price or 100.0,
            status="FILLED",
            fee=0.01,
            raw={"test": True},
        )

    def close_position(self, symbol, side, quantity):
        return self.place_order(symbol, "SELL" if side == "LONG" else "BUY", quantity)

    def get_ticker(self, symbol):
        for t in self._tickers:
            if t.symbol == symbol:
                return t
        return _make_ticker(symbol=symbol)

    def query_order(self, symbol, order_id):
        return OrderResult(
            order_id=order_id, symbol=symbol, side="",
            quantity=0.0, price=0.0, avg_price=100.0,
            status="FILLED", fee=0.01, raw={},
        )

    def cancel_order(self, symbol, order_id):
        return True


# ═══════════════════════════════════════════════════════════════════════
# Test: Config
# ═══════════════════════════════════════════════════════════════════════

class TestExchangeConfig:
    def test_defaults(self):
        cfg = ExchangeConfig()
        assert cfg.testnet is True
        assert cfg.account_currency == "USDT"
        assert cfg.simulate_private is False

    def test_post_init_binance_url(self):
        cfg = ExchangeConfig(testnet=False)
        assert "eapi.binance.com" in cfg.base_url

    def test_post_init_testnet_url(self):
        cfg = ExchangeConfig(testnet=True)
        assert "testnet" in cfg.base_url

    def test_env_var_override(self):
        with patch.dict(os.environ, {"BINANCE_API_KEY": "test_key", "BINANCE_API_SECRET": "test_secret"}):
            cfg = ExchangeConfig()
            assert cfg.api_key == "test_key"
            assert cfg.api_secret == "test_secret"


class TestStrategyConfig:
    def test_defaults(self):
        cfg = StrategyConfig()
        assert cfg.mode == "strangle"
        assert cfg.underlying == "ETH"
        assert cfg.target_delta == 0.40

    def test_weekend_vol_params(self):
        cfg = StrategyConfig(
            mode="weekend_vol",
            underlying="BTC",
            target_delta=0.40,
            wing_delta=0.05,
            leverage=3.0,
        )
        assert cfg.leverage == 3.0
        assert cfg.wing_delta == 0.05


class TestTraderConfig:
    def test_default_construction(self):
        cfg = TraderConfig()
        assert cfg.name == "Short Strangle 7DTE ±10%"
        assert isinstance(cfg.exchange, ExchangeConfig)
        assert isinstance(cfg.strategy, StrategyConfig)

    def test_merge_section(self):
        cfg = StrategyConfig()
        _merge_section(cfg, {"mode": "iron_condor", "underlying": "BTC"})
        assert cfg.mode == "iron_condor"
        assert cfg.underlying == "BTC"

    def test_load_config_default(self):
        cfg = load_config(None)
        assert isinstance(cfg, TraderConfig)

    def test_load_config_from_yaml(self, tmp_path):
        yaml_content = """
name: Test Config
exchange:
  testnet: true
  simulate_private: true
  account_currency: USDT
strategy:
  mode: weekend_vol
  underlying: BTC
  target_delta: 0.35
  leverage: 2.0
"""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml_content)

        cfg = load_config(str(config_file))
        assert cfg.name == "Test Config"
        assert cfg.strategy.mode == "weekend_vol"
        assert cfg.strategy.target_delta == 0.35
        assert cfg.strategy.leverage == 2.0

    def test_load_config_missing_file(self):
        cfg = load_config("/nonexistent/path.yaml")
        # Should return defaults
        assert isinstance(cfg, TraderConfig)


# ═══════════════════════════════════════════════════════════════════════
# Test: Storage
# ═══════════════════════════════════════════════════════════════════════

class TestStorage:
    def test_init_creates_db(self, tmp_db):
        s = Storage(tmp_db)
        assert Path(tmp_db).exists()
        s.close()

    def test_record_and_get_trade(self, storage):
        tid = storage.record_trade(
            trade_group="IC_001",
            symbol="BTC-260328-90000-C",
            side="SELL",
            quantity=0.1,
            price=150.0,
            fee=0.01,
            order_id="ORD_1",
            meta={"leg_role": "sell_call", "strike": 90000},
        )
        assert tid > 0

        trades = storage.get_open_trades("IC_001")
        assert len(trades) == 1
        assert trades[0]["symbol"] == "BTC-260328-90000-C"
        assert trades[0]["side"] == "SELL"
        assert trades[0]["quantity"] == 0.1

    def test_close_trade(self, storage):
        tid = storage.record_trade(
            trade_group="IC_002",
            symbol="BTC-260328-85000-P",
            side="SELL",
            quantity=0.1,
            price=100.0,
        )
        storage.close_trade(tid, close_price=50.0, pnl=5.0)

        open_trades = storage.get_open_trades("IC_002")
        assert len(open_trades) == 0

        all_trades = storage.get_all_trades()
        closed = [t for t in all_trades if t["id"] == tid]
        assert closed[0]["pnl"] == 5.0
        assert closed[0]["is_open"] == 0

    def test_equity_snapshot(self, storage):
        storage.record_equity_snapshot(
            total_equity=10000.0,
            available_balance=8000.0,
            unrealized_pnl=200.0,
            position_count=2,
            underlying_price=87000.0,
        )
        curve = storage.get_equity_curve()
        assert len(curve) == 1
        assert curve[0]["total_equity"] == 10000.0
        assert curve[0]["position_count"] == 2

    def test_daily_pnl(self, storage):
        storage.record_daily_pnl(
            date="2026-03-24",
            starting_equity=10000.0,
            ending_equity=10200.0,
            realized_pnl=200.0,
            trade_count=4,
        )
        records = storage.get_daily_pnl()
        assert len(records) == 1
        assert records[0]["date"] == "2026-03-24"
        assert records[0]["realized_pnl"] == 200.0

    def test_daily_pnl_upsert(self, storage):
        storage.record_daily_pnl("2026-03-24", 10000, 10100, 100)
        storage.record_daily_pnl("2026-03-24", 10000, 10200, 200)
        records = storage.get_daily_pnl()
        assert len(records) == 1
        assert records[0]["realized_pnl"] == 200.0

    def test_state_save_load(self, storage):
        storage.save_state("last_trade_date", "2026-03-20")
        val = storage.load_state("last_trade_date")
        assert val == "2026-03-20"

    def test_state_default(self, storage):
        val = storage.load_state("nonexistent_key", "default_val")
        assert val == "default_val"

    def test_state_overwrite(self, storage):
        storage.save_state("key", 1)
        storage.save_state("key", 2)
        assert storage.load_state("key") == 2

    def test_trade_stats(self, storage):
        # Create some trades
        t1 = storage.record_trade("G1", "SYM1", "SELL", 1, 100)
        storage.close_trade(t1, 50, 50.0)
        t2 = storage.record_trade("G1", "SYM2", "SELL", 1, 100)
        storage.close_trade(t2, 120, -20.0)
        t3 = storage.record_trade("G2", "SYM3", "SELL", 1, 100)  # open

        stats = storage.get_trade_stats()
        assert stats["total_trades"] == 3
        assert stats["open_trades"] == 1
        assert stats["closed_trades"] == 2
        assert stats["win_count"] == 1
        assert stats["loss_count"] == 1
        assert stats["total_pnl"] == pytest.approx(30.0)

    def test_get_open_trade_groups(self, storage):
        storage.record_trade("G1", "SYM1", "SELL", 1, 100)
        storage.record_trade("G2", "SYM2", "BUY", 1, 200)
        t3 = storage.record_trade("G3", "SYM3", "SELL", 1, 150)
        storage.close_trade(t3, 100, 50)

        groups = storage.get_open_trade_groups()
        assert set(groups) == {"G1", "G2"}


# ═══════════════════════════════════════════════════════════════════════
# Test: Binance Client symbol parsing
# ═══════════════════════════════════════════════════════════════════════

class TestParseSymbol:
    def test_parse_call(self):
        result = _parse_symbol("BTC-260328-90000-C")
        assert result is not None
        assert result["underlying"] == "BTC"
        assert result["strike"] == 90000.0
        assert result["option_type"] == "call"
        assert result["expiry"].year == 2026
        assert result["expiry"].month == 3
        assert result["expiry"].day == 28

    def test_parse_put(self):
        result = _parse_symbol("ETH-260401-3000-P")
        assert result is not None
        assert result["option_type"] == "put"
        assert result["strike"] == 3000.0

    def test_parse_invalid(self):
        assert _parse_symbol("INVALID") is None
        assert _parse_symbol("") is None


class TestOptionTicker:
    def test_mid_price(self):
        t = _make_ticker(bid_price=100, ask_price=120)
        assert t.mid_price == pytest.approx(110.0)

    def test_mid_price_no_bid(self):
        t = _make_ticker(bid_price=0, ask_price=120, mark_price=110)
        assert t.mid_price == 110.0

    def test_spread(self):
        t = _make_ticker(bid_price=100, ask_price=120)
        assert t.spread == pytest.approx(20.0)

    def test_dte_hours(self):
        future = datetime.now(timezone.utc) + timedelta(hours=48)
        t = _make_ticker(expiry=future)
        assert 47.9 < t.dte_hours < 48.1

    def test_moneyness(self):
        t = _make_ticker(strike=90000, underlying_price=87000)
        expected = (90000 / 87000 - 1.0) * 100
        assert t.moneyness_pct == pytest.approx(expected, rel=1e-4)


# ═══════════════════════════════════════════════════════════════════════
# Test: IronCondorPosition model
# ═══════════════════════════════════════════════════════════════════════

class TestIronCondorPosition:
    def test_legs_property(self):
        ic = IronCondorPosition(
            group_id="IC_001",
            entry_time=datetime.now(timezone.utc),
            underlying_price=87000,
        )
        ic.sell_call = CondorLeg("SYM1", "SELL", "call", 90000, 1, 100, 1)
        ic.sell_put = CondorLeg("SYM2", "SELL", "put", 85000, 1, 80, 2)
        assert len(ic.legs) == 2

    def test_max_profit(self):
        ic = IronCondorPosition(
            group_id="IC_001",
            entry_time=datetime.now(timezone.utc),
            underlying_price=87000,
            total_premium=150.0,
        )
        assert ic.max_profit == 150.0

    def test_max_loss_with_wings(self):
        ic = IronCondorPosition(
            group_id="IC_001",
            entry_time=datetime.now(timezone.utc),
            underlying_price=87000,
            total_premium=100.0,
        )
        ic.sell_call = CondorLeg("SYM1", "SELL", "call", 90000, 1, 150, 1)
        ic.buy_call = CondorLeg("SYM2", "BUY", "call", 95000, 1, 50, 2)
        ic.sell_put = CondorLeg("SYM3", "SELL", "put", 85000, 1, 120, 3)
        ic.buy_put = CondorLeg("SYM4", "BUY", "put", 80000, 1, 30, 4)
        # wing width = max(90000-95000, 85000-80000) ... but sell-buy:
        # call: buy - sell = 95000 - 90000 = 5000
        # put: sell - buy = 85000 - 80000 = 5000
        assert ic.max_loss == pytest.approx(5000 - 100)

    def test_max_loss_no_wings(self):
        ic = IronCondorPosition(
            group_id="IC_001",
            entry_time=datetime.now(timezone.utc),
            underlying_price=87000,
            total_premium=100.0,
        )
        ic.sell_call = CondorLeg("SYM1", "SELL", "call", 90000, 1, 150, 1)
        ic.sell_put = CondorLeg("SYM3", "SELL", "put", 85000, 1, 120, 3)
        # No wings → infinite max loss
        assert ic.max_loss == float("inf")


# ═══════════════════════════════════════════════════════════════════════
# Test: LimitChaser
# ═══════════════════════════════════════════════════════════════════════

class TestLimitChaser:
    def test_compute_limit_price_sell_initial(self):
        client = MockClient()
        chaser = LimitChaser(client, ChaserConfig(tick_size_usdt=1.0))

        leg = LegOrder("sell_call", "SYM1", "SELL", 1.0, 90000, "call")
        quote = _make_ticker(bid_price=100.0, ask_price=120.0)

        price = chaser._compute_limit_price(leg, quote, elapsed_ratio=0.0)
        # At t=0, SELL starts at ask - tick = 119
        assert price == pytest.approx(119.0)

    def test_compute_limit_price_buy_initial(self):
        client = MockClient()
        chaser = LimitChaser(client, ChaserConfig(tick_size_usdt=1.0))

        leg = LegOrder("buy_put", "SYM2", "BUY", 1.0, 80000, "put")
        quote = _make_ticker(bid_price=100.0, ask_price=120.0)

        price = chaser._compute_limit_price(leg, quote, elapsed_ratio=0.0)
        # At t=0, BUY starts at bid + tick = 101
        assert price == pytest.approx(101.0)

    def test_compute_limit_price_sell_at_deadline(self):
        client = MockClient()
        chaser = LimitChaser(client, ChaserConfig(tick_size_usdt=1.0))

        leg = LegOrder("sell_call", "SYM1", "SELL", 1.0, 90000, "call")
        quote = _make_ticker(bid_price=100.0, ask_price=120.0)

        price = chaser._compute_limit_price(leg, quote, elapsed_ratio=1.0)
        # At deadline, SELL should be at bid
        assert price == pytest.approx(100.0)

    def test_compute_limit_price_buy_at_deadline(self):
        client = MockClient()
        chaser = LimitChaser(client, ChaserConfig(tick_size_usdt=1.0))

        leg = LegOrder("buy_put", "SYM2", "BUY", 1.0, 80000, "put")
        quote = _make_ticker(bid_price=100.0, ask_price=120.0)

        price = chaser._compute_limit_price(leg, quote, elapsed_ratio=1.0)
        # At deadline, BUY should be at ask
        assert price == pytest.approx(120.0)

    def test_execute_legs_all_fill(self):
        client = MockClient()
        chaser = LimitChaser(client, ChaserConfig(
            window_seconds=2,
            poll_interval_sec=0.1,
            market_fallback_sec=1,
        ))

        legs = [
            LegOrder("sell_call", "SYM1", "SELL", 1.0, 90000, "call"),
            LegOrder("sell_put", "SYM2", "SELL", 1.0, 85000, "put"),
        ]

        results = chaser.execute_legs(legs)
        assert all(r.status == "FILLED" for r in results)
        assert results[0].order_id != ""


# ═══════════════════════════════════════════════════════════════════════
# Test: PositionManager
# ═══════════════════════════════════════════════════════════════════════

class TestPositionManager:
    def test_open_short_strangle(self, storage):
        client = MockClient()
        pm = PositionManager(client, storage)

        condor = pm.open_short_strangle(
            sell_call_symbol="BTC-260328-90000-C",
            sell_put_symbol="BTC-260328-85000-P",
            sell_call_strike=90000,
            sell_put_strike=85000,
            quantity=0.1,
            underlying_price=87000,
        )
        assert condor is not None
        assert condor.is_open
        assert len(condor.legs) == 2
        assert pm.open_position_count == 1

    def test_open_iron_condor(self, storage):
        client = MockClient()
        pm = PositionManager(client, storage)

        condor = pm.open_iron_condor(
            sell_call_symbol="BTC-260328-90000-C",
            buy_call_symbol="BTC-260328-95000-C",
            sell_put_symbol="BTC-260328-85000-P",
            buy_put_symbol="BTC-260328-80000-P",
            sell_call_strike=90000,
            buy_call_strike=95000,
            sell_put_strike=85000,
            buy_put_strike=80000,
            quantity=0.1,
            underlying_price=87000,
        )
        assert condor is not None
        assert condor.is_open
        assert len(condor.legs) == 4
        assert pm.open_position_count == 1

    def test_close_iron_condor(self, storage):
        client = MockClient()
        pm = PositionManager(client, storage)

        condor = pm.open_iron_condor(
            sell_call_symbol="BTC-260328-90000-C",
            buy_call_symbol="BTC-260328-95000-C",
            sell_put_symbol="BTC-260328-85000-P",
            buy_put_symbol="BTC-260328-80000-P",
            sell_call_strike=90000,
            buy_call_strike=95000,
            sell_put_strike=85000,
            buy_put_strike=80000,
            quantity=0.1,
            underlying_price=87000,
        )
        assert condor is not None

        pnl = pm.close_iron_condor(condor.group_id, reason="test")
        assert pm.open_position_count == 0
        # PnL is some value (depends on mock fills)
        assert isinstance(pnl, float)

    def test_close_all(self, storage):
        client = MockClient()
        pm = PositionManager(client, storage)

        pm.open_short_strangle(
            "SYM1", "SYM2", 90000, 85000, 0.1, 87000,
        )
        pm.open_short_strangle(
            "SYM3", "SYM4", 91000, 84000, 0.1, 87000,
        )
        assert pm.open_position_count == 2

        total = pm.close_all("test_close")
        assert pm.open_position_count == 0
        assert isinstance(total, float)

    def test_unrealized_pnl(self, storage):
        client = MockClient()
        pm = PositionManager(client, storage)

        condor = pm.open_short_strangle(
            "BTC-260328-90000-C", "BTC-260328-85000-P",
            90000, 85000, 1.0, 87000,
        )
        assert condor is not None

        # Compute unrealized PnL with different mark prices
        marks = {
            "BTC-260328-90000-C": 80.0,   # entry was 100, sold → profit
            "BTC-260328-85000-P": 120.0,   # entry was 100, sold → loss
        }
        upnl = pm.get_unrealized_pnl(marks)
        assert isinstance(upnl, float)

    def test_summary(self, storage):
        client = MockClient()
        pm = PositionManager(client, storage)

        pm.open_short_strangle("S1", "S2", 90000, 85000, 0.1, 87000)
        summary = pm.summary()
        assert summary["open_condors"] == 1
        assert "condors" in summary

    def test_load_open_positions(self, storage):
        """Test that positions are recovered from storage on init."""
        client = MockClient()
        pm = PositionManager(client, storage)

        condor = pm.open_short_strangle(
            "SYM1", "SYM2", 90000, 85000, 0.1, 87000,
        )
        assert condor is not None
        gid = condor.group_id

        # Create a new PositionManager that should recover positions
        pm2 = PositionManager(client, storage)
        assert pm2.open_position_count == 1
        assert gid in pm2.open_condors


# ═══════════════════════════════════════════════════════════════════════
# Test: EquityTracker
# ═══════════════════════════════════════════════════════════════════════

class TestEquityTracker:
    def test_take_snapshot_simulated(self, storage):
        client = MockClient()
        pm = PositionManager(client, storage)
        et = EquityTracker(client, pm, storage, underlying="BTC")

        result = et.take_snapshot()
        # Simulated account → skip recording
        assert result is None

    def test_on_day_start(self, storage):
        client = MockClient()
        pm = PositionManager(client, storage)
        et = EquityTracker(client, pm, storage, underlying="BTC")

        et.on_day_start(10000.0)
        assert et._day_start_equity == 10000.0
        assert et._current_date != ""

    def test_record_trade_pnl(self, storage):
        client = MockClient()
        pm = PositionManager(client, storage)
        et = EquityTracker(client, pm, storage)

        et.on_day_start(10000.0)
        et.record_trade_pnl(50.0, fee=1.0)
        et.record_trade_pnl(-20.0, fee=0.5)

        assert et._day_realized_pnl == pytest.approx(30.0)
        assert et._day_fees == pytest.approx(1.5)
        assert et._day_trade_count == 2

    def test_on_day_end(self, storage):
        client = MockClient()
        pm = PositionManager(client, storage)
        et = EquityTracker(client, pm, storage)

        et.on_day_start(10000.0)
        et.record_trade_pnl(100.0)
        et.on_day_end(10100.0)

        pnl_records = storage.get_daily_pnl()
        assert len(pnl_records) == 1
        assert pnl_records[0]["starting_equity"] == 10000.0
        assert pnl_records[0]["ending_equity"] == 10100.0

    def test_get_performance_stats(self, storage):
        client = MockClient()
        pm = PositionManager(client, storage)
        et = EquityTracker(client, pm, storage)

        stats = et.get_performance_stats()
        assert "total_trades" in stats
        assert "max_drawdown_pct" in stats
        assert "sharpe_ratio" in stats


# ═══════════════════════════════════════════════════════════════════════
# Test: OptionSellingStrategy
# ═══════════════════════════════════════════════════════════════════════

class TestOptionSellingStrategy:
    def test_init(self, storage):
        client = MockClient()
        pm = PositionManager(client, storage)
        cfg = StrategyConfig(mode="iron_condor", underlying="BTC")

        strategy = OptionSellingStrategy(client, pm, storage, cfg)
        assert strategy.cfg.mode == "iron_condor"

    def test_should_enter_wrong_time(self, storage):
        client = MockClient()
        pm = PositionManager(client, storage)
        cfg = StrategyConfig(entry_time_utc="16:00")

        strategy = OptionSellingStrategy(client, pm, storage, cfg)

        # 06:00 UTC → before entry time
        now = datetime(2026, 3, 24, 6, 0, 0, tzinfo=timezone.utc)
        today = now.strftime("%Y-%m-%d")
        assert strategy._should_enter(now, today) is False

    def test_should_enter_already_traded(self, storage):
        client = MockClient()
        pm = PositionManager(client, storage)
        cfg = StrategyConfig(entry_time_utc="08:00")

        strategy = OptionSellingStrategy(client, pm, storage, cfg)
        strategy._last_trade_date = "2026-03-24"

        now = datetime(2026, 3, 24, 9, 0, 0, tzinfo=timezone.utc)
        today = "2026-03-24"
        assert strategy._should_enter(now, today) is False

    def test_should_enter_max_positions(self, storage):
        client = MockClient()
        pm = PositionManager(client, storage)
        cfg = StrategyConfig(entry_time_utc="08:00", max_positions=1)

        strategy = OptionSellingStrategy(client, pm, storage, cfg)

        # Open a position
        pm.open_short_strangle("S1", "S2", 90000, 85000, 0.1, 87000)

        now = datetime(2026, 3, 24, 9, 0, 0, tzinfo=timezone.utc)
        today = "2026-03-24"
        assert strategy._should_enter(now, today) is False

    def test_should_enter_too_late(self, storage):
        client = MockClient()
        pm = PositionManager(client, storage)
        cfg = StrategyConfig(entry_time_utc="08:00")

        strategy = OptionSellingStrategy(client, pm, storage, cfg)

        # 11:00 UTC → 3 hours after entry (max is 2h)
        now = datetime(2026, 3, 24, 11, 0, 0, tzinfo=timezone.utc)
        today = "2026-03-24"
        assert strategy._should_enter(now, today) is False

    def test_find_best_strike(self, storage):
        client = MockClient()
        pm = PositionManager(client, storage)
        cfg = StrategyConfig()

        strategy = OptionSellingStrategy(client, pm, storage, cfg)

        tickers = [
            _make_ticker("S1", strike=85000, option_type="call"),
            _make_ticker("S2", strike=90000, option_type="call"),
            _make_ticker("S3", strike=95000, option_type="call"),
        ]

        best = strategy._find_best_strike(tickers, "call", 89000)
        assert best is not None
        assert best.strike == 90000

    def test_compute_quantity_base(self, storage):
        client = MockClient()
        pm = PositionManager(client, storage)
        cfg = StrategyConfig(quantity=0.01, compound=False)

        strategy = OptionSellingStrategy(client, pm, storage, cfg)
        qty = strategy._compute_quantity(87000)
        assert qty == 0.01

    def test_status(self, storage):
        client = MockClient()
        pm = PositionManager(client, storage)
        cfg = StrategyConfig(mode="iron_condor", underlying="BTC")

        strategy = OptionSellingStrategy(client, pm, storage, cfg)
        status = strategy.status()
        assert status["mode"] == "iron_condor"
        assert status["underlying"] == "BTC"


# ═══════════════════════════════════════════════════════════════════════
# Test: WeekendVolStrategy
# ═══════════════════════════════════════════════════════════════════════

class TestWeekendVolStrategy:
    def test_init(self, storage):
        client = MockClient()
        pm = PositionManager(client, storage)
        cfg = StrategyConfig(
            mode="weekend_vol",
            underlying="BTC",
            target_delta=0.40,
            wing_delta=0.05,
            leverage=3.0,
            entry_day="friday",
        )

        strategy = WeekendVolStrategy(client, pm, storage, cfg)
        assert strategy.cfg.target_delta == 0.40
        assert strategy.cfg.leverage == 3.0

    def test_should_enter_correct_day_and_time(self, storage):
        client = MockClient()
        pm = PositionManager(client, storage)
        cfg = StrategyConfig(
            mode="weekend_vol",
            entry_day="friday",
            entry_time_utc="16:00",
            max_positions=1,
        )

        strategy = WeekendVolStrategy(client, pm, storage, cfg)

        # Friday 16:30 UTC
        now = datetime(2026, 3, 27, 16, 30, 0, tzinfo=timezone.utc)
        assert now.weekday() == 4  # Friday
        assert strategy._should_enter(now) is True

    def test_should_enter_wrong_day(self, storage):
        client = MockClient()
        pm = PositionManager(client, storage)
        cfg = StrategyConfig(
            mode="weekend_vol",
            entry_day="friday",
            entry_time_utc="16:00",
        )

        strategy = WeekendVolStrategy(client, pm, storage, cfg)

        # Wednesday
        now = datetime(2026, 3, 25, 16, 30, 0, tzinfo=timezone.utc)
        assert now.weekday() == 2  # Wednesday
        assert strategy._should_enter(now) is False

    def test_should_enter_before_time(self, storage):
        client = MockClient()
        pm = PositionManager(client, storage)
        cfg = StrategyConfig(
            mode="weekend_vol",
            entry_day="friday",
            entry_time_utc="16:00",
        )

        strategy = WeekendVolStrategy(client, pm, storage, cfg)

        # Friday 10:00 → before 16:00
        now = datetime(2026, 3, 27, 10, 0, 0, tzinfo=timezone.utc)
        assert strategy._should_enter(now) is False

    def test_should_enter_already_traded_this_week(self, storage):
        client = MockClient()
        pm = PositionManager(client, storage)
        cfg = StrategyConfig(
            mode="weekend_vol",
            entry_day="friday",
            entry_time_utc="16:00",
            max_positions=1,
        )

        strategy = WeekendVolStrategy(client, pm, storage, cfg)
        now = datetime(2026, 3, 27, 17, 0, 0, tzinfo=timezone.utc)
        strategy._last_trade_week = now.strftime("%G-W%V")
        assert strategy._should_enter(now) is False

    def test_next_sunday_0800_from_friday(self):
        # Friday 16:00 → next Sunday 08:00
        friday = datetime(2026, 3, 27, 16, 0, 0, tzinfo=timezone.utc)
        sunday = WeekendVolStrategy._next_sunday_0800(friday)
        assert sunday.weekday() == 6  # Sunday
        assert sunday.hour == 8
        assert sunday.day == 29

    def test_next_sunday_0800_from_saturday(self):
        sat = datetime(2026, 3, 28, 10, 0, 0, tzinfo=timezone.utc)
        sunday = WeekendVolStrategy._next_sunday_0800(sat)
        assert sunday.weekday() == 6
        assert sunday.day == 29

    def test_next_sunday_0800_already_past(self):
        # Sunday 10:00 → already past 08:00, should get NEXT Sunday
        now = datetime(2026, 3, 29, 10, 0, 0, tzinfo=timezone.utc)
        next_sun = WeekendVolStrategy._next_sunday_0800(now)
        assert next_sun.day == 5  # April 5
        assert next_sun.weekday() == 6

    def test_compute_quantity_compound(self, storage):
        client = MockClient(spot=87000, equity=50000)
        pm = PositionManager(client, storage)
        cfg = StrategyConfig(
            mode="weekend_vol",
            quantity=0.1,
            leverage=3.0,
            compound=True,
        )

        strategy = WeekendVolStrategy(client, pm, storage, cfg)
        qty = strategy._compute_quantity(87000)
        # (50000 * 3) / 87000 ≈ 1.72, floor to 1.7
        assert qty >= 1.0
        assert qty == pytest.approx(1.7, abs=0.2)

    def test_compute_quantity_no_compound(self, storage):
        client = MockClient()
        pm = PositionManager(client, storage)
        cfg = StrategyConfig(
            mode="weekend_vol",
            quantity=0.5,
            compound=False,
        )

        strategy = WeekendVolStrategy(client, pm, storage, cfg)
        qty = strategy._compute_quantity(87000)
        assert qty == 0.5

    def test_check_settlement_expired(self, storage):
        client = MockClient()
        pm = PositionManager(client, storage)
        cfg = StrategyConfig(mode="weekend_vol")

        strategy = WeekendVolStrategy(client, pm, storage, cfg)

        # Create a condor with expired legs
        past_expiry = datetime(2026, 3, 22, 8, 0, 0, tzinfo=timezone.utc)
        condor = pm.open_short_strangle(
            "BTC-260322-90000-C", "BTC-260322-85000-P",
            90000, 85000, 0.1, 87000,
        )
        assert condor is not None
        assert pm.open_position_count == 1

        # Check settlement at a time after expiry
        now = datetime(2026, 3, 22, 9, 0, 0, tzinfo=timezone.utc)
        strategy._check_settlement(now)
        assert pm.open_position_count == 0

    def test_check_settlement_not_expired(self, storage):
        client = MockClient()
        pm = PositionManager(client, storage)
        cfg = StrategyConfig(mode="weekend_vol")

        strategy = WeekendVolStrategy(client, pm, storage, cfg)

        # Create a condor with future legs
        condor = pm.open_short_strangle(
            "BTC-260329-90000-C", "BTC-260329-85000-P",
            90000, 85000, 0.1, 87000,
        )
        assert pm.open_position_count == 1

        # Check at time before expiry
        now = datetime(2026, 3, 27, 16, 0, 0, tzinfo=timezone.utc)
        strategy._check_settlement(now)
        assert pm.open_position_count == 1  # Still open

    def test_status(self, storage):
        client = MockClient()
        pm = PositionManager(client, storage)
        cfg = StrategyConfig(
            mode="weekend_vol",
            underlying="BTC",
            target_delta=0.40,
            wing_delta=0.05,
            leverage=3.0,
        )

        strategy = WeekendVolStrategy(client, pm, storage, cfg)
        status = strategy.status()
        assert status["strategy"] == "WeekendVol"
        assert status["target_delta"] == 0.40
        assert status["wing_delta"] == 0.05
        assert status["leverage"] == 3.0

    def test_tick_no_entry_on_wrong_day(self, storage):
        """tick() on a non-Friday should not open positions."""
        client = MockClient()
        pm = PositionManager(client, storage)
        cfg = StrategyConfig(
            mode="weekend_vol",
            entry_day="friday",
            entry_time_utc="16:00",
        )

        strategy = WeekendVolStrategy(client, pm, storage, cfg)

        # Patch datetime to Wednesday
        with patch("trader.strategy.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 25, 17, 0, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            strategy.tick()

        assert pm.open_position_count == 0

    def test_find_by_delta_uses_exchange_delta(self, storage):
        """_find_by_delta should prefer exchange-provided delta over Black-76."""
        client = MockClient()
        pm = PositionManager(client, storage)
        cfg = StrategyConfig(
            mode="weekend_vol",
            target_delta=0.40,
            default_iv=0.80,
        )
        strategy = WeekendVolStrategy(client, pm, storage, cfg)

        # Create tickers with exchange delta set
        tickers = [
            _make_ticker("BTC-260329-90000-C", strike=90000, option_type="call"),
            _make_ticker("BTC-260329-95000-C", strike=95000, option_type="call"),
        ]
        tickers[0].delta = 0.42   # close to target 0.40
        tickers[1].delta = 0.25   # far from target

        result = strategy._find_by_delta(tickers, 87000.0, 0.005, 0.40, "call")
        assert result is not None
        assert result.strike == 90000  # picked the one with delta closer to 0.40

    def test_find_by_delta_fallback_to_black76(self, storage):
        """_find_by_delta should fall back to Black-76 when exchange delta is 0."""
        client = MockClient()
        pm = PositionManager(client, storage)
        cfg = StrategyConfig(
            mode="weekend_vol",
            target_delta=0.40,
            default_iv=0.80,
        )
        strategy = WeekendVolStrategy(client, pm, storage, cfg)

        # Create tickers WITHOUT exchange delta (delta=0.0)
        tickers = [
            _make_ticker("BTC-260329-85000-C", strike=85000, option_type="call"),
            _make_ticker("BTC-260329-95000-C", strike=95000, option_type="call"),
        ]
        # delta remains 0.0 → Black-76 fallback

        result = strategy._find_by_delta(tickers, 87000.0, 0.005, 0.40, "call")
        assert result is not None  # should still find a match


# ═══════════════════════════════════════════════════════════════════════
# Test: TradingEngine
# ═══════════════════════════════════════════════════════════════════════

class TestTradingEngine:
    def test_init(self, tmp_path):
        from trader.engine import TradingEngine

        cfg = TraderConfig()
        cfg.exchange.simulate_private = True
        cfg.storage.db_path = str(tmp_path / "engine_test.db")
        cfg.storage.log_dir = str(tmp_path / "logs")

        engine = TradingEngine(cfg)
        assert engine.is_running is False

    def test_start_stop(self, tmp_path):
        from trader.engine import TradingEngine

        cfg = TraderConfig()
        cfg.exchange.simulate_private = True
        cfg.storage.db_path = str(tmp_path / "engine_test.db")
        cfg.storage.log_dir = str(tmp_path / "logs")
        cfg.monitor.check_interval_sec = 1

        engine = TradingEngine(cfg)
        started = engine.start()
        assert started is True
        assert engine.is_running is True

        # Let it run briefly
        time.sleep(0.5)

        stopped = engine.stop(timeout=10)
        assert stopped is True
        assert engine.is_running is False

    def test_start_twice(self, tmp_path):
        from trader.engine import TradingEngine

        cfg = TraderConfig()
        cfg.exchange.simulate_private = True
        cfg.storage.db_path = str(tmp_path / "engine_test.db")
        cfg.storage.log_dir = str(tmp_path / "logs")
        cfg.monitor.check_interval_sec = 1

        engine = TradingEngine(cfg)
        engine.start()
        assert engine.start() is False  # Already running
        engine.stop()

    def test_status(self, tmp_path):
        from trader.engine import TradingEngine

        cfg = TraderConfig()
        cfg.exchange.simulate_private = True
        cfg.storage.db_path = str(tmp_path / "engine_test.db")
        cfg.storage.log_dir = str(tmp_path / "logs")

        engine = TradingEngine(cfg)
        status = engine.status()
        assert status["running"] is False
        assert "tick_count" in status

    def test_format_duration(self):
        from trader.engine import TradingEngine

        assert TradingEngine._format_duration(65) == "1m 5s"
        assert TradingEngine._format_duration(3665) == "1h 1m 5s"
        assert TradingEngine._format_duration(30) == "30s"


# ═══════════════════════════════════════════════════════════════════════
# Test: Engine singleton helpers
# ═══════════════════════════════════════════════════════════════════════

class TestEngineSingleton:
    def test_get_and_reset(self, tmp_path):
        from trader.engine import get_engine, reset_engine, _engine_instance

        cfg = TraderConfig()
        cfg.exchange.simulate_private = True
        cfg.storage.db_path = str(tmp_path / "singleton_test.db")
        cfg.storage.log_dir = str(tmp_path / "logs")

        engine = get_engine(cfg)
        assert engine is not None

        engine2 = get_engine(cfg)
        assert engine2 is engine  # same instance

        reset_engine()
        # After reset, no global instance
        from trader import engine as engine_mod
        assert engine_mod._engine_instance is None
