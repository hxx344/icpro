"""Trading Engine – 后台交易引擎线程包装.

将 TraderApp 包装为可在后台线程运行的引擎, 供 Dashboard 或其他调用方使用.
支持 start / stop / 状态查询, 线程安全.
"""

from __future__ import annotations

import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from trader.binance_client import BinanceOptionsClient
from trader.config import TraderConfig, load_config
from trader.equity import EquityTracker
from trader.limit_chaser import ChaserConfig as _ChaserConfig
from trader.position_manager import PositionManager
from trader.storage import Storage
from trader.strategy import OptionSellingStrategy, WeekendVolStrategy


class TradingEngine:
    """Thread-safe trading engine that runs the strategy loop in background.

    Usage::

        engine = TradingEngine(config)
        engine.start()      # 启动后台线程
        engine.is_running    # True
        engine.status()      # {"running": True, "uptime": 123.4, ...}
        engine.stop()        # 优雅停止
    """

    def __init__(self, config: TraderConfig):
        self.cfg = config
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # State
        self._running = False
        self._start_time: Optional[float] = None
        self._last_tick_time: Optional[float] = None
        self._tick_count = 0
        self._last_error: Optional[str] = None
        self._error_count = 0

        # Exchange client (Binance USD margin)
        self.client: Optional[BinanceOptionsClient] = None
        self.pos_mgr: Optional[PositionManager] = None
        self.strategy: Optional[Any] = None                   # OptionSelling or WeekendVol
        self.equity_tracker: Optional[EquityTracker] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        """Start the trading engine in a background thread.

        Returns True if started successfully, False if already running.
        """
        with self._lock:
            if self.is_running:
                logger.warning("Engine already running, ignoring start()")
                return False

            # Reset state
            self._stop_event.clear()
            self._last_error = None
            self._error_count = 0
            self._tick_count = 0

            # Initialize modules
            try:
                self._init_modules()
            except Exception as e:
                self._last_error = f"Init failed: {e}"
                logger.error(f"Engine init failed: {e}")
                return False

            # Start thread
            self._thread = threading.Thread(
                target=self._run_loop,
                name="TradingEngine",
                daemon=True,
            )
            self._running = True
            self._start_time = time.time()
            self._thread.start()

            logger.info("Trading engine started in background thread")
            return True

    def stop(self, timeout: float = 30.0) -> bool:
        """Stop the trading engine gracefully.

        Returns True if stopped successfully.
        """
        with self._lock:
            if not self.is_running:
                logger.info("Engine not running, nothing to stop")
                return True

            logger.info("Stopping trading engine...")
            self._stop_event.set()

        # Wait outside the lock
        if self._thread:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.error("Engine thread did not stop within timeout")
                return False

        with self._lock:
            self._running = False
            logger.info("Trading engine stopped")

        return True

    def status(self) -> dict:
        """Get engine status as a dict."""
        uptime = time.time() - self._start_time if self._start_time and self.is_running else 0
        last_tick_ago = (
            time.time() - self._last_tick_time if self._last_tick_time else None
        )

        return {
            "running": self.is_running,
            "uptime_sec": round(uptime, 1),
            "uptime_str": self._format_duration(uptime) if uptime > 0 else "-",
            "tick_count": self._tick_count,
            "last_tick_ago_sec": round(last_tick_ago, 1) if last_tick_ago else None,
            "last_error": self._last_error,
            "error_count": self._error_count,
            "check_interval": self.cfg.monitor.check_interval_sec,
            "open_positions": (
                self.pos_mgr.open_position_count if self.pos_mgr else 0
            ),
        }

    def close_all_positions(self) -> float:
        """Emergency close all positions. Returns total PnL."""
        if not self.pos_mgr:
            return 0.0
        logger.warning("[Engine] Manual close-all requested")
        return self.pos_mgr.close_all(reason="manual_close_all")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _init_modules(self) -> None:
        """Initialize all trading modules."""
        logger.info(f"Initializing engine: {self.cfg.name}")

        # Setup logging
        self._setup_logging()

        self.storage = Storage(self.cfg.storage.db_path)

        # Binance USD margin client
        strategy_mode = getattr(self.cfg.strategy, "mode", "strangle")

        self.client = BinanceOptionsClient(self.cfg.exchange)
        logger.info(
            f"Binance client: "
            f"{'TESTNET' if self.cfg.exchange.testnet else 'PRODUCTION'}"
        )

        self.pos_mgr = PositionManager(
            self.client,
            self.storage,
            chaser_config=_ChaserConfig(
                window_seconds=self.cfg.chaser.window_seconds,
                poll_interval_sec=self.cfg.chaser.poll_interval_sec,
                tick_size_usdt=self.cfg.chaser.tick_size_usdt,
                market_fallback_sec=self.cfg.chaser.market_fallback_sec,
                max_amend_attempts=self.cfg.chaser.max_amend_attempts,
            ),
        )

        # Select strategy
        if strategy_mode == "weekend_vol":
            self.strategy = WeekendVolStrategy(
                client=self.client,
                position_mgr=self.pos_mgr,
                storage=self.storage,
                config=self.cfg.strategy,
            )
        else:
            self.strategy = OptionSellingStrategy(
                client=self.client,
                position_mgr=self.pos_mgr,
                storage=self.storage,
                config=self.cfg.strategy,
            )

        self.equity_tracker = EquityTracker(
            client=self.client,
            position_mgr=self.pos_mgr,
            storage=self.storage,
            underlying=self.cfg.strategy.underlying,
        )

        logger.info(f"Strategy: {strategy_mode} | All modules initialized")

    def _setup_logging(self) -> None:
        """Configure loguru for the engine."""
        log_dir = Path(self.cfg.storage.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Don't remove existing handlers (Streamlit may be using stderr)
        # Just add file handler if not already added
        logger.add(
            str(log_dir / "trader_{time:YYYYMMDD}.log"),
            level="DEBUG",
            rotation=self.cfg.storage.log_rotation,
            retention=self.cfg.storage.log_retention,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
                   "{name}:{function}:{line} | {message}",
            encoding="utf-8",
            enqueue=True,  # Thread-safe
        )

    def _run_loop(self) -> None:
        """Main trading loop – runs in background thread."""
        logger.info("Engine loop started")

        if not self.client or not self.strategy or not self.equity_tracker:
            self._last_error = "Engine modules are not initialized"
            logger.error(self._last_error)
            return

        # Init day tracking
        self._init_day()

        last_snapshot = 0.0
        last_heartbeat = 0.0

        while not self._stop_event.is_set():
            now = time.time()

            # --- Strategy tick ---
            try:
                self._check_day_rollover()
                self.strategy.tick()
                self._tick_count += 1
                self._last_tick_time = now
            except Exception as e:
                self._error_count += 1
                self._last_error = f"Tick error: {e}"
                logger.error(f"Strategy tick error: {e}")

            # --- Equity snapshot ---
            if now - last_snapshot >= self.cfg.monitor.equity_snapshot_interval_sec:
                try:
                    self.equity_tracker.take_snapshot()
                    last_snapshot = now
                except Exception as e:
                    logger.error(f"Equity snapshot error: {e}")

            # --- Heartbeat ---
            if now - last_heartbeat >= self.cfg.monitor.heartbeat_interval_sec:
                self._heartbeat()
                last_heartbeat = now

            # Sleep with interruptible wait
            self._stop_event.wait(timeout=self.cfg.monitor.check_interval_sec)

        # --- Shutdown ---
        self._on_shutdown()
        logger.info("Engine loop exited")

    def _init_day(self) -> None:
        if not self.client or not self.equity_tracker:
            return
        try:
            account = self.client.get_account()
            if account.raw.get("simulated"):
                return
            equity = account.total_balance + account.unrealized_pnl
            self.equity_tracker.on_day_start(equity)
        except Exception as e:
            logger.warning(f"Could not init day equity: {e}")

    def _check_day_rollover(self) -> None:
        if not self.client or not self.equity_tracker:
            return
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if (
            today != self.equity_tracker._current_date
            and self.equity_tracker._current_date
        ):
            try:
                account = self.client.get_account()
                if account.raw.get("simulated"):
                    return
                equity = account.total_balance + account.unrealized_pnl
                self.equity_tracker.on_day_end(equity)
                self.equity_tracker.on_day_start(equity)
            except Exception as e:
                logger.warning(f"Day rollover error: {e}")

    def _heartbeat(self) -> None:
        if not self.client:
            return
        now_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
        pos_count = self.pos_mgr.open_position_count if self.pos_mgr else 0
        try:
            account = self.client.get_account()
            if account.raw.get("simulated"):
                balance_str = "simulated"
                upnl_str = "simulated"
            else:
                balance_str = f"{account.total_balance:.4f}"
                upnl_str = f"{account.unrealized_pnl:.4f}"
        except Exception:
            balance_str = "error"
            upnl_str = "error"

        logger.info(
            f"[Heartbeat] {now_str} UTC | "
            f"balance={balance_str} | upnl={upnl_str} | "
            f"positions={pos_count} | ticks={self._tick_count}"
        )

    def _on_shutdown(self) -> None:
        """Cleanup on loop exit."""
        if not self.client or not self.equity_tracker:
            return
        try:
            account = self.client.get_account()
            if account.raw.get("simulated"):
                return
            equity = account.total_balance + account.unrealized_pnl
            self.equity_tracker.on_day_end(equity)
        except Exception:
            pass

        try:
            self.equity_tracker.take_snapshot()
        except Exception:
            pass

    @staticmethod
    def _format_duration(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h}h {m}m {s}s"
        elif m > 0:
            return f"{m}m {s}s"
        return f"{s}s"


# ---------------------------------------------------------------------------
# Singleton helper for Streamlit (survives reruns)
# ---------------------------------------------------------------------------

_engine_instance: Optional[TradingEngine] = None
_engine_lock = threading.Lock()


def get_engine(config: TraderConfig) -> TradingEngine:
    """Get or create the global TradingEngine singleton.

    Uses module-level storage so the instance survives Streamlit reruns.
    """
    global _engine_instance
    with _engine_lock:
        if _engine_instance is None:
            _engine_instance = TradingEngine(config)
        return _engine_instance


def reset_engine() -> None:
    """Stop and discard the current engine instance."""
    global _engine_instance
    with _engine_lock:
        if _engine_instance is not None:
            if _engine_instance.is_running:
                _engine_instance.stop()
            _engine_instance = None
