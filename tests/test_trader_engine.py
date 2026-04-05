"""Tests for trader.engine & trader.equity — 引擎与资产曲线单元测试.

覆盖:
  Engine:
    - 初始化与模块加载
    - start / stop 生命周期
    - singleton get_engine / reset_engine
    - status 字典
    - close_all_positions 紧急平仓

  EquityTracker:
    - take_snapshot (正常 + 模拟 + 错误)
    - on_day_start / on_day_end / record_trade_pnl
    - get_performance_stats
    - daily PnL 累计
"""

from __future__ import annotations

import time
import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from trader.binance_client import BinanceOptionsClient, AccountInfo
from trader.config import TraderConfig, ExchangeConfig, StrategyConfig, StorageConfig, MonitorConfig
from trader.engine import TradingEngine, get_engine, reset_engine
from trader.equity import EquityTracker
from trader.position_manager import PositionManager
from trader.storage import Storage


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def sim_config():
    """模拟配置 (simulate_private=True, 快速心跳)."""
    return TraderConfig(
        name="Test Engine",
        exchange=ExchangeConfig(
            api_key="", api_secret="",
            testnet=True, simulate_private=True,
        ),
        strategy=StrategyConfig(underlying="ETH"),
        storage=StorageConfig(db_path=":memory:", log_dir="./test_logs"),
        monitor=MonitorConfig(
            check_interval_sec=1,
            heartbeat_interval_sec=2,
            equity_snapshot_interval_sec=2,
        ),
    )


@pytest.fixture
def storage(tmp_path):
    s = Storage(str(tmp_path / "test_engine.db"))
    yield s
    s.close()


@pytest.fixture
def mock_client():
    client = MagicMock(spec=BinanceOptionsClient)
    client.get_account.return_value = AccountInfo(
        total_balance=10.0, available_balance=8.0,
        unrealized_pnl=0.0, raw={"simulated": True},
    )
    client.get_spot_price.return_value = 2500.0
    client.get_tickers.return_value = []
    client.get_mark_prices.return_value = {}
    return client


# ======================================================================
# Engine Tests
# ======================================================================


class TestTradingEngine:
    def test_initial_state(self, sim_config):
        engine = TradingEngine(sim_config)
        assert engine.is_running is False
        assert engine._tick_count == 0
        assert engine.status()["state"] == TradingEngine.STATE_STOPPED

    def test_status_not_running(self, sim_config):
        engine = TradingEngine(sim_config)
        s = engine.status()
        assert s["running"] is False
        assert s["state"] == TradingEngine.STATE_STOPPED
        assert s["uptime_sec"] == 0
        assert s["execution_metrics"] == {}
        assert s["recent_execution_events"] == []

    def test_status_includes_execution_metrics(self, sim_config):
        engine = TradingEngine(sim_config)
        storage_mock = MagicMock(spec=Storage)
        storage_mock.get_execution_metrics.return_value = {
            "open_attempts": 3,
            "success_rate_pct": 66.7,
        }
        storage_mock.get_execution_events.return_value = [
            {"event_type": "position_open_success", "group_id": "IC_TEST"}
        ]
        engine.storage = storage_mock

        s = engine.status()
        assert s["execution_metrics"]["open_attempts"] == 3
        assert s["recent_execution_events"][0]["event_type"] == "position_open_success"

    def test_start_stop(self, sim_config, tmp_path):
        """引擎启动后应在后台运行，stop 后退出."""
        sim_config.storage.db_path = str(tmp_path / "engine_ss.db")
        engine = TradingEngine(sim_config)

        started = engine.start()
        assert started is True
        assert engine.is_running is True

        # Let it tick once
        time.sleep(1.5)
        assert engine._tick_count >= 1

        stopped = engine.stop(timeout=5.0)
        assert stopped is True
        assert engine.is_running is False

    def test_double_start(self, sim_config, tmp_path):
        """重复 start 应返回 False."""
        sim_config.storage.db_path = str(tmp_path / "engine_ds.db")
        engine = TradingEngine(sim_config)

        engine.start()
        assert engine.start() is False

        engine.stop(timeout=5.0)

    def test_stop_when_not_running(self, sim_config):
        engine = TradingEngine(sim_config)
        assert engine.stop() is True

    def test_status_while_running(self, sim_config, tmp_path):
        sim_config.storage.db_path = str(tmp_path / "engine_sr.db")
        engine = TradingEngine(sim_config)
        engine.start()
        time.sleep(1.5)

        s = engine.status()
        assert s["running"] is True
        assert s["state"] == TradingEngine.STATE_RUNNING
        assert s["uptime_sec"] > 0
        assert s["tick_count"] >= 1

        engine.stop(timeout=5.0)

    def test_close_all_positions(self, sim_config, tmp_path):
        sim_config.storage.db_path = str(tmp_path / "engine_ca.db")
        engine = TradingEngine(sim_config)
        engine.start()
        time.sleep(0.5)

        pnl = engine.close_all_positions()
        assert pnl == 0.0  # no positions

        engine.stop(timeout=5.0)

    def test_format_duration(self):
        assert TradingEngine._format_duration(0) == "0s"
        assert TradingEngine._format_duration(65) == "1m 5s"
        assert TradingEngine._format_duration(3661) == "1h 1m 1s"

    def test_setup_logging_does_not_duplicate_handler(self, sim_config, tmp_path):
        sim_config.storage.log_dir = str(tmp_path / "logs")
        engine = TradingEngine(sim_config)

        with patch("trader.engine.logger.add", return_value=123) as add_mock:
            engine._setup_logging()
            engine._setup_logging()

        add_mock.assert_called_once()
        assert engine._log_handler_id == 123

    def test_cleanup_resources_closes_storage_and_removes_handler(self, sim_config):
        engine = TradingEngine(sim_config)
        storage_mock = MagicMock(spec=Storage)
        engine.storage = storage_mock
        engine._log_handler_id = 123

        with patch("trader.engine.logger.remove") as remove_mock:
            engine._cleanup_resources()

        storage_mock.close.assert_called_once()
        assert engine.storage is None
        assert engine._log_handler_id is None
        remove_mock.assert_called_once_with(123)

    def test_shutdown_cleans_resources_on_snapshot_error(self, sim_config):
        engine = TradingEngine(sim_config)
        engine.client = MagicMock(spec=BinanceOptionsClient)
        engine.client.get_account.return_value = AccountInfo(
            total_balance=10.0,
            available_balance=8.0,
            unrealized_pnl=0.5,
            raw={},
        )
        engine.equity_tracker = MagicMock(spec=EquityTracker)
        engine.equity_tracker.take_snapshot.side_effect = RuntimeError("snapshot fail")
        storage_mock = MagicMock(spec=Storage)
        engine.storage = storage_mock
        engine._log_handler_id = 321

        with patch("trader.engine.logger.remove") as remove_mock:
            engine._on_shutdown()

        storage_mock.close.assert_called_once()
        remove_mock.assert_called_once_with(321)
        assert engine.client is None
        assert engine.equity_tracker is None

    def test_start_init_failure_sets_error_state(self, sim_config):
        engine = TradingEngine(sim_config)

        with patch.object(engine, "_init_modules", side_effect=RuntimeError("boom")):
            assert engine.start() is False

        assert engine.status()["state"] == TradingEngine.STATE_ERROR
        assert "Init failed" in engine.status()["last_error"]

    def test_stop_timeout_sets_stuck_state(self, sim_config):
        engine = TradingEngine(sim_config)
        thread = MagicMock()
        thread.is_alive.return_value = True
        engine._thread = thread
        engine._state = TradingEngine.STATE_RUNNING

        assert engine.stop(timeout=0.01) is False
        assert engine.status()["state"] == TradingEngine.STATE_STUCK

    def test_run_loop_without_modules_sets_error_state(self, sim_config):
        engine = TradingEngine(sim_config)
        engine._run_loop()

        assert engine.status()["state"] == TradingEngine.STATE_ERROR
        assert engine.status()["last_error"] == "Engine modules are not initialized"


class TestEngineSingleton:
    def test_get_engine_creates_instance(self, sim_config, tmp_path):
        sim_config.storage.db_path = str(tmp_path / "singleton.db")
        reset_engine()  # clean state

        e1 = get_engine(sim_config)
        e2 = get_engine(sim_config)
        assert e1 is e2

        reset_engine()

    def test_reset_engine(self, sim_config, tmp_path):
        sim_config.storage.db_path = str(tmp_path / "singleton2.db")
        reset_engine()

        e1 = get_engine(sim_config)
        reset_engine()
        e2 = get_engine(sim_config)
        assert e1 is not e2

        reset_engine()


# ======================================================================
# EquityTracker Tests
# ======================================================================


class TestEquityTracker:
    @pytest.fixture
    def tracker(self, mock_client, storage):
        pos_mgr = MagicMock(spec=PositionManager)
        pos_mgr.open_position_count = 0
        pos_mgr.get_unrealized_pnl.return_value = 0.0
        return EquityTracker(
            client=mock_client,
            position_mgr=pos_mgr,
            storage=storage,
            underlying="ETH",
        )

    def test_take_snapshot_simulated_skips(self, tracker, mock_client):
        """模拟账户数据应跳过快照 (不写 DB)."""
        snap = tracker.take_snapshot()
        assert snap is None

    def test_take_snapshot_real(self, tracker, mock_client, storage):
        """真实账户数据应记录快照."""
        mock_client.get_account.return_value = AccountInfo(
            total_balance=10.0, available_balance=8.0,
            unrealized_pnl=0.5, raw={},  # not simulated
        )
        snap = tracker.take_snapshot()
        assert snap is not None
        assert snap["total_equity"] == pytest.approx(10.0)  # balance + upnl from pos_mgr (0)

        curve = storage.get_equity_curve()
        assert len(curve) == 1

    def test_on_day_start(self, tracker, storage):
        tracker.on_day_start(10.0)
        assert tracker._day_start_equity == 10.0
        assert tracker._day_realized_pnl == 0.0

    def test_record_trade_pnl(self, tracker):
        tracker.on_day_start(10.0)
        tracker.record_trade_pnl(0.5, fee=0.01)
        tracker.record_trade_pnl(0.3, fee=0.005)
        assert tracker._day_realized_pnl == pytest.approx(0.8)
        assert tracker._day_fees == pytest.approx(0.015)
        assert tracker._day_trade_count == 2

    def test_on_day_end_saves(self, tracker, mock_client, storage):
        mock_client.get_account.return_value = AccountInfo(
            total_balance=10.5, available_balance=9.0,
            unrealized_pnl=0.1, raw={},
        )
        tracker._current_date = "2026-03-20"
        tracker._day_start_equity = 10.0
        tracker._day_realized_pnl = 0.5

        tracker.on_day_end(10.5)

        pnl = storage.get_daily_pnl()
        assert len(pnl) == 1
        assert pnl[0]["date"] == "2026-03-20"
        assert pnl[0]["ending_equity"] == pytest.approx(10.5)

    def test_performance_stats_empty(self, tracker):
        stats = tracker.get_performance_stats()
        assert stats["total_trades"] == 0
        assert stats["max_drawdown_pct"] == 0.0
        assert stats["sharpe_ratio"] == 0

    def test_performance_stats_with_data(self, tracker, mock_client, storage):
        """写入若干天数据后统计应正确."""
        mock_client.get_account.return_value = AccountInfo(
            total_balance=10.0, available_balance=8.0,
            unrealized_pnl=0.0, raw={},
        )
        # 模拟 3 天
        storage.record_daily_pnl("2026-03-18", 10.0, 10.2, 0.2)
        storage.record_daily_pnl("2026-03-19", 10.2, 10.5, 0.3)
        storage.record_daily_pnl("2026-03-20", 10.5, 10.3, -0.2)

        stats = tracker.get_performance_stats()
        assert stats["total_days"] == 3
        assert stats["profitable_days"] == 2
        assert stats["loss_days"] == 1


# ======================================================================
# Integration: Engine + EquityTracker lifecycle
# ======================================================================


class TestEngineEquityIntegration:
    def test_engine_initializes_equity_tracker(self, sim_config, tmp_path):
        """引擎启动时应正确初始化 EquityTracker."""
        sim_config.storage.db_path = str(tmp_path / "integ.db")
        engine = TradingEngine(sim_config)
        engine.start()
        time.sleep(0.5)

        assert engine.equity_tracker is not None
        assert engine.client is not None
        assert engine.pos_mgr is not None
        assert engine.strategy is not None

        engine.stop(timeout=5.0)

    def test_engine_start_recovers_exchange_positions_into_status(self, sim_config, tmp_path):
        """引擎启动应把交易所已有仓位恢复到本地并反映到状态里。"""
        sim_config.storage.db_path = str(tmp_path / "recover.db")
        sim_config.strategy.underlying = "ETH"

        client = MagicMock(spec=BinanceOptionsClient)
        client.get_account.return_value = AccountInfo(
            total_balance=10.0,
            available_balance=8.0,
            unrealized_pnl=0.0,
            raw={"simulated": True},
        )
        client.get_spot_price.return_value = 2500.0
        client.get_tickers.return_value = []
        client.get_mark_prices.return_value = {}
        client.get_positions.return_value = [
            {"symbol": "ETH-260405-2700-C", "side": "SHORT", "quantity": -0.5, "entryPrice": 55.0, "unrealizedPnl": -2.0},
            {"symbol": "ETH-260405-2300-P", "side": "SHORT", "quantity": -0.5, "entryPrice": 50.0, "unrealizedPnl": 1.0},
        ]

        with patch("trader.engine.BinanceOptionsClient", return_value=client):
            engine = TradingEngine(sim_config)
            assert engine.start() is True
            time.sleep(0.5)

            status = engine.status()
            assert status["open_positions"] == 1
            assert engine.pos_mgr is not None
            assert engine.pos_mgr.open_position_count == 1
            assert engine.storage is not None
            open_trades = engine.storage.get_open_trades()
            assert len(open_trades) == 2
            assert {t["symbol"] for t in open_trades} == {"ETH-260405-2700-C", "ETH-260405-2300-P"}

            engine.stop(timeout=5.0)
