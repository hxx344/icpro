"""SQLite persistent storage – 持久化存储.

Tables:
  - trades:          每笔成交记录
  - positions:       当前持仓快照
  - equity_snapshots: 资产曲线（定期快照）
  - daily_pnl:       每日损益汇总
  - strategy_state:  策略状态（跨重启恢复）
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger


class Storage:
    """SQLite-based persistent storage for the trader.

    All timestamps are stored as ISO-8601 UTC strings.
    Connections are per-thread to support multi-threaded access (e.g. Streamlit).
    """

    def __init__(self, db_path: str = "./data/trader.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    @contextmanager
    def _cursor(self):
        conn = self._get_conn()
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def close(self):
        conn = getattr(self._local, "conn", None)
        if conn:
            conn.close()
            self._local.conn = None

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                trade_group TEXT NOT NULL,        -- links legs of same condor
                symbol      TEXT NOT NULL,
                side        TEXT NOT NULL,          -- BUY / SELL
                quantity    REAL NOT NULL,
                price       REAL NOT NULL,
                fee         REAL DEFAULT 0,
                order_id    TEXT,
                pnl         REAL DEFAULT 0,          -- realized PnL (set on close)
                is_open     INTEGER DEFAULT 1,       -- 1=open, 0=closed
                close_timestamp TEXT,
                close_price REAL DEFAULT 0,
                meta        TEXT DEFAULT '{}'         -- JSON extra info
            );

            CREATE TABLE IF NOT EXISTS equity_snapshots (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                total_equity REAL NOT NULL,
                available_balance REAL NOT NULL,
                unrealized_pnl REAL DEFAULT 0,
                position_count INTEGER DEFAULT 0,
                underlying_price REAL DEFAULT 0,
                meta        TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS daily_pnl (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                date        TEXT NOT NULL UNIQUE,     -- YYYY-MM-DD
                starting_equity REAL NOT NULL,
                ending_equity REAL NOT NULL,
                realized_pnl REAL DEFAULT 0,
                unrealized_pnl REAL DEFAULT 0,
                total_fees  REAL DEFAULT 0,
                trade_count INTEGER DEFAULT 0,
                meta        TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS strategy_state (
                key         TEXT PRIMARY KEY,
                value       TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_trades_group ON trades(trade_group);
            CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
            CREATE INDEX IF NOT EXISTS idx_trades_open ON trades(is_open);
            CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity_snapshots(timestamp);
            CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_pnl(date);
        """)
        conn.commit()
        logger.info(f"Storage initialized: {self.db_path}")

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------

    def record_trade(
        self,
        trade_group: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        fee: float = 0.0,
        order_id: str = "",
        meta: dict | None = None,
    ) -> int:
        """Record a new trade (open leg).

        Returns the trade row ID.
        """
        ts = datetime.now(timezone.utc).isoformat()
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO trades (timestamp, trade_group, symbol, side,
                                    quantity, price, fee, order_id, meta)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (ts, trade_group, symbol, side, quantity, price, fee,
                  order_id, json.dumps(meta or {})))
            return cur.lastrowid

    def close_trade(
        self,
        trade_id: int,
        close_price: float,
        pnl: float,
    ) -> None:
        """Mark a trade as closed with realized PnL."""
        ts = datetime.now(timezone.utc).isoformat()
        with self._cursor() as cur:
            cur.execute("""
                UPDATE trades SET is_open = 0, close_timestamp = ?,
                                  close_price = ?, pnl = ?
                WHERE id = ?
            """, (ts, close_price, pnl, trade_id))

    def get_open_trades(self, trade_group: str | None = None) -> list[dict]:
        """Get all open trades, optionally filtered by group."""
        conn = self._get_conn()
        if trade_group:
            rows = conn.execute(
                "SELECT * FROM trades WHERE is_open = 1 AND trade_group = ?",
                (trade_group,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM trades WHERE is_open = 1"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_trades(
        self,
        limit: int = 500,
        offset: int = 0,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict]:
        """Get historical trades with pagination."""
        query = "SELECT * FROM trades"
        params: list[Any] = []
        conditions = []

        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        conn = self._get_conn()
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_open_trade_groups(self) -> list[str]:
        """Get distinct trade groups that still have open trades."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT DISTINCT trade_group FROM trades WHERE is_open = 1"
        ).fetchall()
        return [r["trade_group"] for r in rows]

    # ------------------------------------------------------------------
    # Equity snapshots
    # ------------------------------------------------------------------

    def record_equity_snapshot(
        self,
        total_equity: float,
        available_balance: float,
        unrealized_pnl: float = 0.0,
        position_count: int = 0,
        underlying_price: float = 0.0,
        meta: dict | None = None,
    ) -> None:
        """Record an equity snapshot for the equity curve."""
        ts = datetime.now(timezone.utc).isoformat()
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO equity_snapshots
                    (timestamp, total_equity, available_balance,
                     unrealized_pnl, position_count, underlying_price, meta)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (ts, total_equity, available_balance, unrealized_pnl,
                  position_count, underlying_price, json.dumps(meta or {})))

    def get_equity_curve(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict]:
        """Get equity curve data points."""
        query = "SELECT * FROM equity_snapshots"
        params: list[Any] = []
        conditions = []

        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY timestamp ASC"

        conn = self._get_conn()
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Daily PnL
    # ------------------------------------------------------------------

    def record_daily_pnl(
        self,
        date: str,
        starting_equity: float,
        ending_equity: float,
        realized_pnl: float = 0.0,
        unrealized_pnl: float = 0.0,
        total_fees: float = 0.0,
        trade_count: int = 0,
        meta: dict | None = None,
    ) -> None:
        """Upsert daily PnL summary."""
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO daily_pnl
                    (date, starting_equity, ending_equity, realized_pnl,
                     unrealized_pnl, total_fees, trade_count, meta)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date) DO UPDATE SET
                    ending_equity = excluded.ending_equity,
                    realized_pnl = excluded.realized_pnl,
                    unrealized_pnl = excluded.unrealized_pnl,
                    total_fees = excluded.total_fees,
                    trade_count = excluded.trade_count,
                    meta = excluded.meta
            """, (date, starting_equity, ending_equity, realized_pnl,
                  unrealized_pnl, total_fees, trade_count,
                  json.dumps(meta or {})))

    def get_daily_pnl(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict]:
        """Get daily PnL records."""
        query = "SELECT * FROM daily_pnl"
        params: list[Any] = []
        conditions = []

        if start_date:
            conditions.append("date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("date <= ?")
            params.append(end_date)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY date ASC"

        conn = self._get_conn()
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Strategy state (key-value for cross-restart recovery)
    # ------------------------------------------------------------------

    def save_state(self, key: str, value: Any) -> None:
        """Save a strategy state value (JSON-serializable)."""
        ts = datetime.now(timezone.utc).isoformat()
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO strategy_state (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
            """, (key, json.dumps(value), ts))

    def load_state(self, key: str, default: Any = None) -> Any:
        """Load a strategy state value."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT value FROM strategy_state WHERE key = ?", (key,)
        ).fetchone()
        if row:
            return json.loads(row["value"])
        return default

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------

    def get_trade_stats(self) -> dict:
        """Compute aggregate trade statistics."""
        conn = self._get_conn()

        total = conn.execute("SELECT COUNT(*) as cnt FROM trades").fetchone()["cnt"]
        closed = conn.execute(
            "SELECT COUNT(*) as cnt FROM trades WHERE is_open = 0"
        ).fetchone()["cnt"]
        open_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM trades WHERE is_open = 1"
        ).fetchone()["cnt"]

        pnl_row = conn.execute(
            "SELECT COALESCE(SUM(pnl), 0) as total_pnl, "
            "COALESCE(SUM(fee), 0) as total_fees "
            "FROM trades WHERE is_open = 0"
        ).fetchone()

        wins = conn.execute(
            "SELECT COUNT(*) as cnt FROM trades WHERE is_open = 0 AND pnl > 0"
        ).fetchone()["cnt"]
        losses = conn.execute(
            "SELECT COUNT(*) as cnt FROM trades WHERE is_open = 0 AND pnl < 0"
        ).fetchone()["cnt"]

        return {
            "total_trades": total,
            "open_trades": open_count,
            "closed_trades": closed,
            "total_pnl": pnl_row["total_pnl"],
            "total_fees": pnl_row["total_fees"],
            "win_count": wins,
            "loss_count": losses,
            "win_rate": (wins / closed * 100) if closed > 0 else 0.0,
        }
