"""Tests for trader.storage — SQLite 持久化存储单元测试.

覆盖:
  - DB 初始化与表创建
  - trades CRUD (record, close, query)
  - equity_snapshots 记录与查询
  - daily_pnl upsert 与查询
  - strategy_state key-value 存储
  - trade_stats 聚合统计
  - 线程安全 (多线程写入)
  - 边界条件
"""

from __future__ import annotations

import json
import threading
import tempfile
from pathlib import Path

import pytest

from trader.storage import Storage


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def db(tmp_path):
    """每个测试独立的临时数据库."""
    storage = Storage(str(tmp_path / "test.db"))
    yield storage
    storage.close()


# ======================================================================
# 1. 初始化
# ======================================================================


class TestInit:
    def test_creates_db_file(self, tmp_path):
        db_path = tmp_path / "sub" / "deep" / "test.db"
        s = Storage(str(db_path))
        assert db_path.exists()
        s.close()

    def test_tables_exist(self, db):
        """验证所有 5 张表已创建."""
        conn = db._get_conn()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        names = {r["name"] for r in tables}
        assert "trades" in names
        assert "equity_snapshots" in names
        assert "daily_pnl" in names
        assert "strategy_state" in names

    def test_connection_pragmas(self, db):
        conn = db._get_conn()
        busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        synchronous = conn.execute("PRAGMA synchronous").fetchone()[0]

        assert busy_timeout == 5000
        assert synchronous == 1

    def test_migrates_legacy_schema_columns(self, tmp_path):
        db_path = tmp_path / "legacy.db"
        import sqlite3

        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                trade_group TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL
            );

            CREATE TABLE equity_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_equity REAL NOT NULL
            );

            CREATE TABLE daily_pnl (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL UNIQUE,
                starting_equity REAL NOT NULL,
                ending_equity REAL NOT NULL
            );

            CREATE TABLE strategy_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        conn.commit()
        conn.close()

        storage = Storage(str(db_path))
        migrated = storage._get_conn()

        trades_columns = {row["name"] for row in migrated.execute("PRAGMA table_info(trades)").fetchall()}
        equity_columns = {row["name"] for row in migrated.execute("PRAGMA table_info(equity_snapshots)").fetchall()}
        daily_columns = {row["name"] for row in migrated.execute("PRAGMA table_info(daily_pnl)").fetchall()}

        assert {"fee", "order_id", "pnl", "is_open", "close_timestamp", "close_price", "meta"} <= trades_columns
        assert {"available_balance", "unrealized_pnl", "position_count", "underlying_price", "meta"} <= equity_columns
        assert {"realized_pnl", "unrealized_pnl", "total_fees", "trade_count", "meta"} <= daily_columns

        storage.close()


# ======================================================================
# 2. Trades
# ======================================================================


class TestTrades:
    def test_record_and_query(self, db):
        tid = db.record_trade(
            trade_group="IC_001",
            symbol="ETH-260321-2000-C",
            side="SELL",
            quantity=1.0,
            price=0.05,
            fee=0.001,
            order_id="ORD123",
            meta={"leg_role": "sell_call", "strike": 2000},
        )
        assert tid > 0

        trades = db.get_open_trades("IC_001")
        assert len(trades) == 1
        assert trades[0]["symbol"] == "ETH-260321-2000-C"
        assert trades[0]["side"] == "SELL"
        assert trades[0]["is_open"] == 1

    def test_close_trade(self, db):
        tid = db.record_trade("IC_001", "ETH-260321-2000-C", "SELL", 1.0, 0.05)
        db.close_trade(tid, close_price=0.02, pnl=0.03)

        trades = db.get_open_trades("IC_001")
        assert len(trades) == 0  # 不再开放

        all_trades = db.get_all_trades()
        assert len(all_trades) == 1
        assert all_trades[0]["is_open"] == 0
        assert all_trades[0]["pnl"] == pytest.approx(0.03)

    def test_open_trade_groups(self, db):
        db.record_trade("IC_001", "SYM_A", "SELL", 1.0, 0.05)
        db.record_trade("IC_001", "SYM_B", "BUY", 1.0, 0.02)
        db.record_trade("IC_002", "SYM_C", "SELL", 1.0, 0.04)

        groups = db.get_open_trade_groups()
        assert set(groups) == {"IC_001", "IC_002"}

        # Close IC_001 trades
        trades = db.get_open_trades("IC_001")
        for t in trades:
            db.close_trade(t["id"], 0.01, 0.01)

        groups = db.get_open_trade_groups()
        assert groups == ["IC_002"]

    def test_get_all_trades_pagination(self, db):
        for i in range(10):
            db.record_trade(f"GRP_{i}", f"SYM_{i}", "SELL", 1.0, 0.01 * i)

        result = db.get_all_trades(limit=5)
        assert len(result) == 5

    def test_meta_json(self, db):
        meta = {"leg_role": "sell_call", "strike": 2000, "underlying_price": 2500}
        tid = db.record_trade("IC_001", "SYM", "SELL", 1.0, 0.05, meta=meta)

        trades = db.get_all_trades()
        stored_meta = json.loads(trades[0]["meta"])
        assert stored_meta["leg_role"] == "sell_call"
        assert stored_meta["strike"] == 2000


# ======================================================================
# 3. Equity Snapshots
# ======================================================================


class TestEquitySnapshots:
    def test_record_and_query(self, db):
        db.record_equity_snapshot(
            total_equity=10.0,
            available_balance=8.0,
            unrealized_pnl=0.5,
            position_count=2,
            underlying_price=2500.0,
        )
        curve = db.get_equity_curve()
        assert len(curve) == 1
        assert curve[0]["total_equity"] == pytest.approx(10.0)
        assert curve[0]["position_count"] == 2

    def test_multiple_snapshots_ordered(self, db):
        import time
        for i in range(3):
            db.record_equity_snapshot(10.0 + i, 8.0, 0.0, 0, 2500.0)
            time.sleep(0.01)  # ensure distinct timestamps

        curve = db.get_equity_curve()
        assert len(curve) == 3
        # 应按时间升序
        equities = [s["total_equity"] for s in curve]
        assert equities == sorted(equities)


# ======================================================================
# 4. Daily PnL
# ======================================================================


class TestDailyPnl:
    def test_record_and_query(self, db):
        db.record_daily_pnl(
            date="2026-03-20",
            starting_equity=10.0,
            ending_equity=10.5,
            realized_pnl=0.5,
            total_fees=0.01,
            trade_count=4,
        )
        pnl = db.get_daily_pnl()
        assert len(pnl) == 1
        assert pnl[0]["date"] == "2026-03-20"
        assert pnl[0]["ending_equity"] == pytest.approx(10.5)

    def test_upsert(self, db):
        """同一日期重复写入应更新而非报错."""
        db.record_daily_pnl("2026-03-20", 10.0, 10.5, 0.5)
        db.record_daily_pnl("2026-03-20", 10.0, 11.0, 1.0)  # 更新

        pnl = db.get_daily_pnl()
        assert len(pnl) == 1
        assert pnl[0]["ending_equity"] == pytest.approx(11.0)

    def test_date_range_filter(self, db):
        db.record_daily_pnl("2026-03-18", 10.0, 10.2, 0.2)
        db.record_daily_pnl("2026-03-19", 10.2, 10.5, 0.3)
        db.record_daily_pnl("2026-03-20", 10.5, 11.0, 0.5)

        result = db.get_daily_pnl(start_date="2026-03-19", end_date="2026-03-19")
        assert len(result) == 1
        assert result[0]["date"] == "2026-03-19"


# ======================================================================
# 5. Strategy State
# ======================================================================


class TestStrategyState:
    def test_save_and_load(self, db):
        db.save_state("last_trade_date", "2026-03-20")
        assert db.load_state("last_trade_date") == "2026-03-20"

    def test_load_default(self, db):
        assert db.load_state("nonexistent", "fallback") == "fallback"

    def test_load_none_default(self, db):
        assert db.load_state("missing") is None

    def test_overwrite(self, db):
        db.save_state("counter", 1)
        db.save_state("counter", 2)
        assert db.load_state("counter") == 2

    def test_complex_value(self, db):
        """存储复杂 JSON 值."""
        data = {"positions": [1, 2, 3], "nested": {"a": True}}
        db.save_state("complex", data)
        loaded = db.load_state("complex")
        assert loaded == data

    def test_float_value(self, db):
        db.save_state("equity", 10.12345)
        assert db.load_state("equity") == pytest.approx(10.12345)


# ======================================================================
# 6. Trade Stats
# ======================================================================


class TestTradeStats:
    def test_empty_stats(self, db):
        stats = db.get_trade_stats()
        assert stats["total_trades"] == 0
        assert stats["win_rate"] == 0.0

    def test_with_trades(self, db):
        # 4 trades: 2 wins, 1 loss, 1 open
        t1 = db.record_trade("G1", "S1", "SELL", 1.0, 0.05)
        t2 = db.record_trade("G1", "S2", "SELL", 1.0, 0.04)
        t3 = db.record_trade("G2", "S3", "SELL", 1.0, 0.03)
        t4 = db.record_trade("G3", "S4", "SELL", 1.0, 0.02)

        db.close_trade(t1, 0.02, 0.03)   # win
        db.close_trade(t2, 0.01, 0.03)   # win
        db.close_trade(t3, 0.05, -0.02)  # loss

        stats = db.get_trade_stats()
        assert stats["total_trades"] == 4
        assert stats["open_trades"] == 1
        assert stats["closed_trades"] == 3
        assert stats["win_count"] == 2
        assert stats["loss_count"] == 1
        assert stats["win_rate"] == pytest.approx(66.666, abs=0.1)
        assert stats["total_pnl"] == pytest.approx(0.04)


# ======================================================================
# 7. 线程安全
# ======================================================================


class TestThreadSafety:
    def test_concurrent_writes(self, tmp_path):
        """多线程并发写入不应丢失数据或 crash."""
        db = Storage(str(tmp_path / "concurrent.db"))
        errors = []
        N = 20

        def writer(thread_id):
            try:
                for i in range(N):
                    db.record_trade(
                        f"group_{thread_id}",
                        f"SYM_{thread_id}_{i}",
                        "SELL", 1.0, 0.01,
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"

        all_trades = db.get_all_trades(limit=1000)
        assert len(all_trades) == 4 * N
        db.close()


# ======================================================================
# 8. 边界条件
# ======================================================================


class TestEdgeCases:
    def test_close_already_closed(self, db):
        """二次关闭不应报错."""
        tid = db.record_trade("G1", "SYM", "SELL", 1.0, 0.05)
        db.close_trade(tid, 0.02, 0.03)
        db.close_trade(tid, 0.01, 0.04)  # 再次关闭

    def test_empty_meta(self, db):
        tid = db.record_trade("G1", "SYM", "SELL", 1.0, 0.05, meta=None)
        trades = db.get_all_trades()
        assert json.loads(trades[0]["meta"]) == {}

    def test_unicode_in_meta(self, db):
        """中文等 Unicode 字符存储正常."""
        meta = {"note": "铁鹰策略第一腿"}
        tid = db.record_trade("G1", "SYM", "SELL", 1.0, 0.05, meta=meta)
        trades = db.get_all_trades()
        loaded = json.loads(trades[0]["meta"])
        assert loaded["note"] == "铁鹰策略第一腿"
