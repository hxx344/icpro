"""Main trader entry point — 主程序入口.

启动交易系统:
  python -m trader.main run -c configs/trader/weekend_vol_btc.yaml

功能:
  - 加载配置
  - 初始化各模块 (API client, storage, position manager, strategy, equity tracker)
  - 主循环: 定时策略 tick + 仓位监控 + 资产快照
  - 信号处理 (Ctrl+C 优雅退出)
  - 命令行查询 (status, equity, trades, stats)
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from trader.binance_client import BinanceOptionsClient
from trader.config import TraderConfig, load_config
from trader.equity import EquityTracker
from trader.position_manager import PositionManager
from trader.storage import Storage
from trader.strategy import IronCondor0DTEStrategy, WeekendVolStrategy


DEFAULT_CONFIG_PATH = os.environ.get(
    "TRADER_CONFIG_PATH",
    "configs/trader/weekend_vol_btc.yaml",
)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(cfg: TraderConfig) -> None:
    """Configure loguru logger with file + console output."""
    log_dir = Path(cfg.storage.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Console: colored, INFO level
    logger.add(
        sys.stderr,
        level=cfg.storage.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        colorize=True,
    )

    # File: detailed, rotated daily
    logger.add(
        str(log_dir / "trader_{time:YYYYMMDD}.log"),
        level="DEBUG",
        rotation=cfg.storage.log_rotation,
        retention=cfg.storage.log_retention,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
               "{name}:{function}:{line} | {message}",
        encoding="utf-8",
    )

    # Trades log: separate file for trade records
    logger.add(
        str(log_dir / "trades_{time:YYYYMMDD}.log"),
        level="INFO",
        rotation=cfg.storage.log_rotation,
        retention=cfg.storage.log_retention,
        filter=lambda record: "[Trade]" in record["message"],
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Trader application
# ---------------------------------------------------------------------------

class TraderApp:
    """Main trading application orchestrator.

    Composes all modules and runs the main loop.
    """

    def __init__(self, config: TraderConfig):
        self.cfg = config
        self._running = False
        self._last_snapshot_time = 0.0
        self._last_heartbeat_time = 0.0

        # Initialize modules
        logger.info(f"Initializing trader: {config.name}")

        self.storage = Storage(config.storage.db_path)

        # Binance USD margin client
        strategy_mode = getattr(config.strategy, "mode", "strangle")

        self.client = BinanceOptionsClient(config.exchange)
        logger.info(
            f"Binance client: {'TESTNET' if config.exchange.testnet else 'PRODUCTION'} "
            f"base={config.exchange.base_url}"
        )

        self.pos_mgr = PositionManager(self.client, self.storage)

        # Select strategy based on mode
        if strategy_mode == "weekend_vol":
            self.strategy = WeekendVolStrategy(
                client=self.client,
                position_mgr=self.pos_mgr,
                storage=self.storage,
                config=config.strategy,
            )
        else:
            self.strategy = IronCondor0DTEStrategy(
                client=self.client,
                position_mgr=self.pos_mgr,
                storage=self.storage,
                config=config.strategy,
            )

        self.equity_tracker = EquityTracker(
            client=self.client,
            position_mgr=self.pos_mgr,
            storage=self.storage,
            underlying=config.strategy.underlying,
        )

        logger.info(f"Strategy: {strategy_mode} | Exchange: Binance")
        logger.info("All modules initialized")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the main trading loop.

        Runs until Ctrl+C or SIGTERM.
        """
        self._running = True
        logger.info("=" * 60)
        logger.info(f"  {self.cfg.name}")
        logger.info(f"  Underlying:    {self.cfg.strategy.underlying}")

        mode = getattr(self.cfg.strategy, "mode", "strangle")
        if mode == "weekend_vol":
            logger.info(f"  Mode:          Weekend Vol Selling")
            logger.info(f"  Target delta:  {self.cfg.strategy.target_delta}")
            logger.info(f"  Wing delta:    {self.cfg.strategy.wing_delta}")
            logger.info(f"  Leverage:      {self.cfg.strategy.leverage}x")
            logger.info(f"  Entry:         {self.cfg.strategy.entry_day} {self.cfg.strategy.entry_time_utc} UTC")
        else:
            logger.info(f"  OTM:           ±{self.cfg.strategy.otm_pct*100:.0f}%")
            logger.info(f"  Wing width:    {self.cfg.strategy.wing_width_pct*100:.0f}%")
            logger.info(f"  Entry time:    {self.cfg.strategy.entry_time_utc} UTC")

        logger.info(f"  DB:            {self.cfg.storage.db_path}")
        logger.info(f"  Check interval:{self.cfg.monitor.check_interval_sec}s")
        logger.info("=" * 60)
        logger.info("Trader started. Press Ctrl+C to stop.")

        # Record day start
        self._init_day()

        try:
            while self._running:
                self._tick()
                time.sleep(self.cfg.monitor.check_interval_sec)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self._shutdown()

    def _tick(self) -> None:
        """Single iteration of the main loop."""
        now = time.time()

        # Day rollover check
        self._check_day_rollover()

        # Strategy tick
        try:
            self.strategy.tick()
        except Exception as e:
            logger.error(f"Strategy tick error: {e}")

        # Equity snapshot
        if now - self._last_snapshot_time >= self.cfg.monitor.equity_snapshot_interval_sec:
            try:
                self.equity_tracker.take_snapshot()
                self._last_snapshot_time = now
            except Exception as e:
                logger.error(f"Equity snapshot error: {e}")

        # Heartbeat
        if now - self._last_heartbeat_time >= self.cfg.monitor.heartbeat_interval_sec:
            self._heartbeat()
            self._last_heartbeat_time = now

    def _init_day(self) -> None:
        """Initialize day tracking."""
        try:
            account = self.client.get_account()
            equity = account.total_balance + account.unrealized_pnl
            self.equity_tracker.on_day_start(equity)
        except Exception as e:
            logger.warning(f"Could not init day equity: {e}")

    def _check_day_rollover(self) -> None:
        """Detect UTC day change and trigger day start/end."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self.equity_tracker._current_date and self.equity_tracker._current_date:
            # Day changed — record end of previous day
            try:
                account = self.client.get_account()
                equity = account.total_balance + account.unrealized_pnl
                self.equity_tracker.on_day_end(equity)
                self.equity_tracker.on_day_start(equity)
            except Exception as e:
                logger.warning(f"Day rollover error: {e}")

    def _heartbeat(self) -> None:
        """Log periodic heartbeat with status summary."""
        now = datetime.now(timezone.utc)
        pos_count = self.pos_mgr.open_position_count

        try:
            account = self.client.get_account()
            balance = account.total_balance
            upnl = account.unrealized_pnl
        except Exception:
            balance = -1
            upnl = 0

        rv_str = "-"
        basket_str = "-"
        try:
            strategy_status = self.strategy.status()
            rv_val = strategy_status.get("entry_realized_vol_current")
            basket_val = strategy_status.get("basket_pnl_pct")
            if isinstance(rv_val, (int, float)):
                rv_str = f"{rv_val:.2%}"
            if isinstance(basket_val, (int, float)):
                basket_str = f"{basket_val:.1f}%"
        except Exception:
            pass

        logger.info(
            f"[Heartbeat] {now.strftime('%H:%M:%S')} UTC | "
            f"balance={balance:.4f} | upnl={upnl:.4f} | "
            f"positions={pos_count} | rv24={rv_str} | basket={basket_str}"
        )

    def _shutdown(self) -> None:
        """Graceful shutdown: save state, close connections."""
        logger.info("Shutting down trader...")
        self._running = False

        # Save daily PnL
        try:
            account = self.client.get_account()
            equity = account.total_balance + account.unrealized_pnl
            self.equity_tracker.on_day_end(equity)
        except Exception:
            pass

        # Final equity snapshot
        try:
            self.equity_tracker.take_snapshot()
        except Exception:
            pass

        # Print summary
        try:
            self.equity_tracker.print_summary()
        except Exception:
            pass

        # Close storage
        self.storage.close()
        logger.info("Trader stopped.")

    def stop(self) -> None:
        """Request graceful stop."""
        self._running = False

    # ------------------------------------------------------------------
    # CLI query methods
    # ------------------------------------------------------------------

    def print_status(self) -> None:
        """Print current strategy and position status."""
        status = self.strategy.status()
        import json
        logger.info("Current status:\n" + json.dumps(status, indent=2, default=str))

    def print_trades(self, limit: int = 20) -> None:
        """Print recent trades."""
        trades = self.storage.get_all_trades(limit=limit)
        logger.info(f"Recent {len(trades)} trades:")
        for t in trades:
            status = "OPEN" if t["is_open"] else "CLOSED"
            logger.info(
                f"  [{status}] {t['timestamp'][:19]} | {t['side']} {t['symbol']} "
                f"qty={t['quantity']} @ {t['price']:.4f} | "
                f"pnl={t['pnl']:.4f} | group={t['trade_group']}"
            )

    def print_equity_curve(self, last_n: int = 24) -> None:
        """Print recent equity snapshots."""
        curve = self.storage.get_equity_curve()
        recent = curve[-last_n:] if len(curve) > last_n else curve
        logger.info(f"Equity curve (last {len(recent)} snapshots):")
        for snap in recent:
            logger.info(
                f"  {snap['timestamp'][:19]} | "
                f"equity={snap['total_equity']:.4f} | "
                f"upnl={snap['unrealized_pnl']:.4f} | "
                f"positions={snap['position_count']}"
            )


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_run(args) -> None:
    """Run the trader."""
    cfg = load_config(args.config)
    _setup_logging(cfg)

    app = TraderApp(cfg)

    # Signal handler for graceful shutdown
    def _signal_handler(sig, frame):
        logger.info(f"Signal {sig} received, stopping...")
        app.stop()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    app.run()


def cmd_status(args) -> None:
    """Show current status."""
    cfg = load_config(args.config)
    _setup_logging(cfg)

    app = TraderApp(cfg)
    app.print_status()
    app.storage.close()


def cmd_trades(args) -> None:
    """Show recent trades."""
    cfg = load_config(args.config)
    _setup_logging(cfg)

    app = TraderApp(cfg)
    app.print_trades(limit=args.limit)
    app.storage.close()


def cmd_equity(args) -> None:
    """Show equity curve."""
    cfg = load_config(args.config)
    _setup_logging(cfg)

    app = TraderApp(cfg)
    app.print_equity_curve(last_n=args.last)
    app.storage.close()


def cmd_stats(args) -> None:
    """Show performance statistics."""
    cfg = load_config(args.config)
    _setup_logging(cfg)

    app = TraderApp(cfg)
    app.equity_tracker.print_summary()
    app.storage.close()


def cmd_close_all(args) -> None:
    """Emergency: close all positions."""
    cfg = load_config(args.config)
    _setup_logging(cfg)

    app = TraderApp(cfg)
    logger.warning("CLOSING ALL POSITIONS")
    pnl = app.pos_mgr.close_all(reason="manual_close_all")
    logger.info(f"All positions closed. Total PnL: {pnl:.4f}")
    app.storage.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="期权交易程序 – Weekend Vol / Iron Condor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  run         启动交易程序 (主循环)
  status      查看当前策略和持仓状态
  trades      查看最近成交记录
  equity      查看资产曲线
  stats       查看绩效统计
  close-all   紧急平仓所有持仓

Examples:
  python -m trader.main run --config configs/trader/weekend_vol_btc.yaml
  python -m trader.main status
  python -m trader.main trades --limit 50
  python -m trader.main stats
        """,
    )

    parser.add_argument(
        "--config", "-c",
        default=DEFAULT_CONFIG_PATH,
        help=f"配置文件路径 (默认: {DEFAULT_CONFIG_PATH})",
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # run
    sp_run = subparsers.add_parser("run", help="启动交易程序")
    sp_run.set_defaults(func=cmd_run)

    # status
    sp_status = subparsers.add_parser("status", help="查看状态")
    sp_status.set_defaults(func=cmd_status)

    # trades
    sp_trades = subparsers.add_parser("trades", help="查看成交记录")
    sp_trades.add_argument("--limit", "-n", type=int, default=20, help="显示条数")
    sp_trades.set_defaults(func=cmd_trades)

    # equity
    sp_equity = subparsers.add_parser("equity", help="查看资产曲线")
    sp_equity.add_argument("--last", "-n", type=int, default=24, help="最近N条记录")
    sp_equity.set_defaults(func=cmd_equity)

    # stats
    sp_stats = subparsers.add_parser("stats", help="查看绩效统计")
    sp_stats.set_defaults(func=cmd_stats)

    # close-all
    sp_close = subparsers.add_parser("close-all", help="紧急平仓")
    sp_close.set_defaults(func=cmd_close_all)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
