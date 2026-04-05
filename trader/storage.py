"""SQLite persistent storage – 持久化存储.

Tables:
    - trades:            每笔成交记录
  - equity_snapshots: 资产曲线（定期快照）
    - daily_pnl:         每日损益汇总
    - strategy_state:    策略状态（跨重启恢复）
    - execution_events:  执行事件日志（下单/补腿/回滚/对账）
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
            conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
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

    def clear_all_data(self) -> None:
        """Delete all persisted historical data from the SQLite database."""
        conn = self._get_conn()
        logger.warning(f"Clearing all persisted storage data: {self.db_path}")
        conn.executescript("""
            DELETE FROM trades;
            DELETE FROM equity_snapshots;
            DELETE FROM daily_pnl;
            DELETE FROM strategy_state;
            DELETE FROM execution_events;
            DELETE FROM sqlite_sequence WHERE name IN ('trades', 'equity_snapshots', 'daily_pnl', 'execution_events');
        """)
        conn.commit()

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

            CREATE TABLE IF NOT EXISTS execution_events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT NOT NULL,
                event_type      TEXT NOT NULL,
                execution_state TEXT NOT NULL,
                severity        TEXT NOT NULL DEFAULT 'info',
                group_id        TEXT DEFAULT '',
                underlying      TEXT DEFAULT '',
                execution_mode  TEXT DEFAULT '',
                message         TEXT DEFAULT '',
                legs            TEXT DEFAULT '[]',
                meta            TEXT DEFAULT '{}'
            );
        """)
        self._ensure_schema_compatibility(conn)
        conn.executescript("""
            CREATE INDEX IF NOT EXISTS idx_trades_group ON trades(trade_group);
            CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
            CREATE INDEX IF NOT EXISTS idx_trades_open ON trades(is_open);
            CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity_snapshots(timestamp);
            CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_pnl(date);
            CREATE INDEX IF NOT EXISTS idx_exec_events_ts ON execution_events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_exec_events_group ON execution_events(group_id);
            CREATE INDEX IF NOT EXISTS idx_exec_events_type ON execution_events(event_type);
        """)
        conn.commit()
        logger.info(f"Storage initialized: {self.db_path}")

    def _ensure_schema_compatibility(self, conn: sqlite3.Connection) -> None:
        """Backfill columns for legacy databases created by older releases."""
        self._ensure_table_columns(
            conn,
            "trades",
            {
                "fee": "REAL DEFAULT 0",
                "order_id": "TEXT",
                "pnl": "REAL DEFAULT 0",
                "is_open": "INTEGER DEFAULT 1",
                "close_timestamp": "TEXT",
                "close_price": "REAL DEFAULT 0",
                "meta": "TEXT DEFAULT '{}'",
            },
        )
        self._ensure_table_columns(
            conn,
            "equity_snapshots",
            {
                "available_balance": "REAL NOT NULL DEFAULT 0",
                "unrealized_pnl": "REAL DEFAULT 0",
                "position_count": "INTEGER DEFAULT 0",
                "underlying_price": "REAL DEFAULT 0",
                "meta": "TEXT DEFAULT '{}'",
            },
        )
        self._ensure_table_columns(
            conn,
            "daily_pnl",
            {
                "realized_pnl": "REAL DEFAULT 0",
                "unrealized_pnl": "REAL DEFAULT 0",
                "total_fees": "REAL DEFAULT 0",
                "trade_count": "INTEGER DEFAULT 0",
                "meta": "TEXT DEFAULT '{}'",
            },
        )
        self._ensure_table_columns(
            conn,
            "execution_events",
            {
                "event_type": "TEXT NOT NULL DEFAULT ''",
                "execution_state": "TEXT NOT NULL DEFAULT ''",
                "severity": "TEXT NOT NULL DEFAULT 'info'",
                "group_id": "TEXT DEFAULT ''",
                "underlying": "TEXT DEFAULT ''",
                "execution_mode": "TEXT DEFAULT ''",
                "message": "TEXT DEFAULT ''",
                "legs": "TEXT DEFAULT '[]'",
                "meta": "TEXT DEFAULT '{}'",
            },
        )

    def _ensure_table_columns(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        required_columns: dict[str, str],
    ) -> None:
        existing_columns = {
            row["name"] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        for column_name, column_sql in required_columns.items():
            if column_name in existing_columns:
                continue
            logger.info(f"Migrating legacy SQLite schema: adding {table_name}.{column_name}")
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}")

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
            return int(cur.lastrowid or 0)

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

    def delete_open_trades(self, symbol_prefix: str | None = None) -> int:
        """Delete open trades, optionally filtered by symbol prefix."""
        with self._cursor() as cur:
            if symbol_prefix:
                cur.execute(
                    "DELETE FROM trades WHERE is_open = 1 AND symbol LIKE ?",
                    (f"{symbol_prefix}%",),
                )
            else:
                cur.execute("DELETE FROM trades WHERE is_open = 1")
            return int(cur.rowcount or 0)

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

    def get_equity_curve_stats(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, float | int]:
        """Get aggregated equity curve stats without loading all rows into Python."""
        conditions: list[str] = []
        params: list[Any] = []

        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date)

        where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            WITH filtered AS (
                SELECT timestamp, total_equity
                FROM equity_snapshots
                {where_clause}
            ),
            numbered AS (
                SELECT
                    timestamp,
                    total_equity,
                    ROW_NUMBER() OVER (ORDER BY timestamp ASC) AS rn,
                    COUNT(*) OVER () AS cnt,
                    MAX(total_equity) OVER (
                        ORDER BY timestamp ASC
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS running_peak
                FROM filtered
            )
            SELECT
                COALESCE(MAX(cnt), 0) AS point_count,
                COALESCE(MAX(CASE WHEN rn = 1 THEN total_equity END), 0) AS first_equity,
                COALESCE(MAX(CASE WHEN rn = cnt THEN total_equity END), 0) AS last_equity,
                COALESCE(
                    MAX(
                        CASE
                            WHEN running_peak > 0 THEN (running_peak - total_equity) * 100.0 / running_peak
                            ELSE 0
                        END
                    ),
                    0
                ) AS max_drawdown_pct
            FROM numbered
        """

        conn = self._get_conn()
        row = conn.execute(query, params).fetchone()
        if row is None:
            return {
                "point_count": 0,
                "first_equity": 0.0,
                "last_equity": 0.0,
                "max_drawdown_pct": 0.0,
            }
        return {
            "point_count": int(row["point_count"] or 0),
            "first_equity": float(row["first_equity"] or 0.0),
            "last_equity": float(row["last_equity"] or 0.0),
            "max_drawdown_pct": float(row["max_drawdown_pct"] or 0.0),
        }

    def get_equity_curve_sampled(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        max_points: int = 1500,
    ) -> list[dict]:
        """Get sampled equity curve rows for fast chart rendering."""
        stats = self.get_equity_curve_stats(start_date, end_date)
        point_count = int(stats.get("point_count") or 0)
        if point_count <= 0:
            return []

        max_points = max(int(max_points or 0), 1)
        if point_count <= max_points:
            return self.get_equity_curve(start_date, end_date)

        step = max(1, (point_count + max_points - 1) // max_points)
        conditions: list[str] = []
        params: list[Any] = []

        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date)

        where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            WITH filtered AS (
                SELECT
                    id,
                    timestamp,
                    total_equity,
                    available_balance,
                    unrealized_pnl,
                    position_count,
                    underlying_price,
                    meta
                FROM equity_snapshots
                {where_clause}
            ),
            numbered AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (ORDER BY timestamp ASC) AS rn,
                    COUNT(*) OVER () AS cnt
                FROM filtered
            )
            SELECT *
            FROM numbered
            WHERE rn = 1 OR rn = cnt OR ((rn - 1) % ?) = 0
            ORDER BY timestamp ASC
        """

        conn = self._get_conn()
        rows = conn.execute(query, [*params, step]).fetchall()
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
    # Execution events
    # ------------------------------------------------------------------

    def record_execution_event(
        self,
        event_type: str,
        execution_state: str,
        severity: str = "info",
        group_id: str = "",
        underlying: str = "",
        execution_mode: str = "",
        message: str = "",
        legs: list[dict[str, Any]] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> int:
        ts = datetime.now(timezone.utc).isoformat()
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO execution_events
                    (timestamp, event_type, execution_state, severity, group_id,
                     underlying, execution_mode, message, legs, meta)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    event_type,
                    execution_state,
                    severity,
                    group_id,
                    underlying,
                    execution_mode,
                    message,
                    json.dumps(legs or []),
                    json.dumps(meta or {}),
                ),
            )
            return int(cur.lastrowid or 0)

    def get_execution_events(
        self,
        limit: int = 50,
        start_time: str | None = None,
        group_id: str | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM execution_events"
        params: list[Any] = []
        conditions: list[str] = []

        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time)
        if group_id:
            conditions.append("group_id = ?")
            params.append(group_id)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY timestamp DESC"
        if limit > 0:
            query += " LIMIT ?"
            params.append(limit)

        conn = self._get_conn()
        rows = conn.execute(query, params).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["legs"] = json.loads(item.get("legs") or "[]")
            item["meta"] = json.loads(item.get("meta") or "{}")
            result.append(item)
        return result

    def get_execution_metrics(self, lookback_hours: int = 168) -> dict[str, Any]:
        start_time = None
        if lookback_hours > 0:
            start_time = datetime.now(timezone.utc).timestamp() - lookback_hours * 3600
            start_time = datetime.fromtimestamp(start_time, tz=timezone.utc).isoformat()

        events = self.get_execution_events(limit=0, start_time=start_time)
        starts = [e for e in events if e.get("event_type") == "position_open_start"]
        successes = [e for e in events if e.get("event_type") == "position_open_success"]
        partials = [e for e in events if e.get("event_type") == "position_open_partial"]
        failures = [
            e for e in events
            if e.get("event_type") in {"position_open_failed", "position_open_error"}
        ]
        reconcile_adjustments = [e for e in events if e.get("event_type") == "exchange_reconcile_adjustment"]
        risk_alerts = [
            e for e in events
            if str(e.get("execution_state") or "") in {"partial_exposure", "partial_exposure_timeout"}
            or str(e.get("severity") or "") in {"warning", "error"}
        ]

        started_at: dict[str, str] = {}
        terminal_events: dict[str, dict[str, Any]] = {}
        for event in sorted(events, key=lambda item: str(item.get("timestamp") or "")):
            group_id = str(event.get("group_id") or "")
            if not group_id:
                continue
            if event.get("event_type") == "position_open_start" and group_id not in started_at:
                started_at[group_id] = str(event.get("timestamp") or "")
            if event.get("event_type") in {"position_open_success", "position_open_partial", "position_open_failed", "position_open_error"}:
                terminal_events[group_id] = event

        durations: list[float] = []
        for group_id, start_ts in started_at.items():
            terminal = terminal_events.get(group_id)
            if terminal is None:
                continue
            try:
                start_dt = datetime.fromisoformat(start_ts)
                end_dt = datetime.fromisoformat(str(terminal.get("timestamp") or ""))
            except Exception:
                continue
            durations.append(max((end_dt - start_dt).total_seconds(), 0.0))

        attempts = len(starts)
        return {
            "lookback_hours": lookback_hours,
            "open_attempts": attempts,
            "open_successes": len(successes),
            "open_partials": len(partials),
            "open_failures": len(failures),
            "reconcile_adjustments": len(reconcile_adjustments),
            "risk_alerts": len(risk_alerts),
            "success_rate_pct": (len(successes) / attempts * 100.0) if attempts > 0 else 0.0,
            "partial_rate_pct": (len(partials) / attempts * 100.0) if attempts > 0 else 0.0,
            "failure_rate_pct": (len(failures) / attempts * 100.0) if attempts > 0 else 0.0,
            "avg_open_duration_sec": (sum(durations) / len(durations)) if durations else 0.0,
            "latest_event_at": str(events[0].get("timestamp") or "") if events else "",
        }

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
