"""Equity curve & PnL tracker – 资产曲线与损益记录.

Periodically snapshots account equity, computes daily PnL,
and provides reporting utilities (equity chart, drawdown, stats).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from loguru import logger

from trader.position_manager import PositionManager
from trader.storage import Storage


class EquityTracker:
    """Tracks equity curve, daily PnL, and generates reports.

    Responsibilities:
    - Take periodic equity snapshots → SQLite
    - Compute daily starting/ending equity
    - Track realized + unrealized PnL
    - Generate summary statistics
    """

    def __init__(
        self,
        client: Any,
        position_mgr: PositionManager,
        storage: Storage,
        underlying: str = "ETH",
    ):
        self.client = client
        self.pos_mgr = position_mgr
        self.storage = storage
        self.underlying = underlying

        # State
        self._current_date: str = ""
        self._day_start_equity: float = storage.load_state("day_start_equity", 0.0)
        self._day_realized_pnl: float = storage.load_state("day_realized_pnl", 0.0)
        self._day_fees: float = storage.load_state("day_fees", 0.0)
        self._day_trade_count: int = storage.load_state("day_trade_count", 0)

    # ------------------------------------------------------------------
    # Periodic snapshot (called by scheduler)
    # ------------------------------------------------------------------

    def take_snapshot(self) -> dict | None:
        """Capture current equity state and persist.

        Returns snapshot dict or None on failure.
        """
        try:
            account = self.client.get_account()
        except Exception as e:
            logger.error(f"Failed to get account for snapshot: {e}")
            return None

        # Skip recording if account data is simulated (no real credentials)
        if account.raw.get("simulated"):
            logger.debug("Skipping equity snapshot – using simulated account data")
            return None

        # Get mark prices for unrealized PnL
        try:
            mark_prices = self.client.get_mark_prices(self.underlying)
            upnl = self.pos_mgr.get_unrealized_pnl(mark_prices)
        except Exception:
            upnl = account.unrealized_pnl

        spot = self.client.get_spot_price(self.underlying)

        total_equity = account.total_balance + upnl
        pos_count = self.pos_mgr.open_position_count

        self.storage.record_equity_snapshot(
            total_equity=total_equity,
            available_balance=account.available_balance,
            unrealized_pnl=upnl,
            position_count=pos_count,
            underlying_price=spot,
        )

        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_equity": total_equity,
            "available_balance": account.available_balance,
            "unrealized_pnl": upnl,
            "position_count": pos_count,
            "underlying_price": spot,
        }

        logger.debug(
            f"Equity snapshot: equity={total_equity:.4f} "
            f"upnl={upnl:.4f} positions={pos_count}"
        )

        return snapshot

    # ------------------------------------------------------------------
    # Daily PnL tracking
    # ------------------------------------------------------------------

    def on_day_start(self, equity: float) -> None:
        """Record start-of-day equity for PnL tracking."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._current_date:
            # Save previous day if exists
            if self._current_date:
                self._save_daily_pnl()

            self._current_date = today
            self._day_start_equity = equity
            self._day_realized_pnl = 0.0
            self._day_fees = 0.0
            self._day_trade_count = 0

            self.storage.save_state("day_start_equity", equity)
            self.storage.save_state("day_realized_pnl", 0.0)
            self.storage.save_state("day_fees", 0.0)
            self.storage.save_state("day_trade_count", 0)

            logger.info(f"Day start: {today}, equity={equity:.4f}")

    def record_trade_pnl(self, pnl: float, fee: float = 0.0) -> None:
        """Accumulate intraday realized PnL."""
        self._day_realized_pnl += pnl
        self._day_fees += fee
        self._day_trade_count += 1

        self.storage.save_state("day_realized_pnl", self._day_realized_pnl)
        self.storage.save_state("day_fees", self._day_fees)
        self.storage.save_state("day_trade_count", self._day_trade_count)

    def on_day_end(self, equity: float) -> None:
        """Record end-of-day equity and save daily PnL."""
        try:
            account = self.client.get_account()
            upnl = account.unrealized_pnl
        except Exception:
            upnl = 0.0

        self._save_daily_pnl(ending_equity=equity, unrealized_pnl=upnl)

    def _save_daily_pnl(
        self,
        ending_equity: float | None = None,
        unrealized_pnl: float = 0.0,
    ) -> None:
        """Persist daily PnL record."""
        if not self._current_date:
            return

        if ending_equity is None:
            try:
                account = self.client.get_account()
                ending_equity = account.total_balance
            except Exception:
                ending_equity = self._day_start_equity

        self.storage.record_daily_pnl(
            date=self._current_date,
            starting_equity=self._day_start_equity,
            ending_equity=ending_equity,
            realized_pnl=self._day_realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_fees=self._day_fees,
            trade_count=self._day_trade_count,
        )

        logger.info(
            f"Daily PnL saved: {self._current_date} "
            f"start={self._day_start_equity:.4f} end={ending_equity:.4f} "
            f"rpnl={self._day_realized_pnl:.4f} fees={self._day_fees:.4f}"
        )

    # ------------------------------------------------------------------
    # Reporting / statistics
    # ------------------------------------------------------------------

    def get_equity_curve(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict]:
        """Get equity curve data for charting."""
        return self.storage.get_equity_curve(start_date, end_date)

    def get_daily_pnl_history(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict]:
        """Get daily PnL records."""
        return self.storage.get_daily_pnl(start_date, end_date)

    def get_performance_stats(self) -> dict:
        """Compute comprehensive performance statistics."""
        trade_stats = self.storage.get_trade_stats()
        daily_pnl = self.storage.get_daily_pnl()
        equity_curve = self.storage.get_equity_curve()

        # Drawdown calculation
        peak = 0.0
        max_dd = 0.0
        for snap in equity_curve:
            eq = snap["total_equity"]
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        # Daily returns
        daily_returns = []
        for d in daily_pnl:
            if d["starting_equity"] > 0:
                ret = (d["ending_equity"] - d["starting_equity"]) / d["starting_equity"]
                daily_returns.append(ret)

        avg_daily = sum(daily_returns) / len(daily_returns) if daily_returns else 0
        profitable_days = sum(1 for r in daily_returns if r > 0)
        loss_days = sum(1 for r in daily_returns if r < 0)

        # Sharpe ratio (annualized, assuming 365 trading days)
        if len(daily_returns) > 1:
            import math
            mean_ret = avg_daily
            std_ret = math.sqrt(
                sum((r - mean_ret) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
            )
            sharpe = (mean_ret / std_ret * math.sqrt(365)) if std_ret > 0 else 0
        else:
            sharpe = 0

        return {
            **trade_stats,
            "max_drawdown_pct": max_dd * 100,
            "total_days": len(daily_pnl),
            "profitable_days": profitable_days,
            "loss_days": loss_days,
            "avg_daily_return_pct": avg_daily * 100,
            "sharpe_ratio": sharpe,
            "equity_snapshots": len(equity_curve),
        }

    def print_summary(self) -> None:
        """Print a formatted performance summary to logger."""
        stats = self.get_performance_stats()

        logger.info("=" * 60)
        logger.info("Performance Summary")
        logger.info("=" * 60)
        logger.info(f"  Total trades:       {stats['total_trades']}")
        logger.info(f"  Open trades:        {stats['open_trades']}")
        logger.info(f"  Closed trades:      {stats['closed_trades']}")
        logger.info(f"  Win rate:           {stats['win_rate']:.1f}%")
        logger.info(f"  Total PnL:          {stats['total_pnl']:.4f} USD")
        logger.info(f"  Total fees:         {stats['total_fees']:.4f} USD")
        logger.info(f"  Max drawdown:       {stats['max_drawdown_pct']:.2f}%")
        logger.info(f"  Avg daily return:   {stats['avg_daily_return_pct']:.4f}%")
        logger.info(f"  Sharpe ratio:       {stats['sharpe_ratio']:.2f}")
        logger.info(f"  Profitable days:    {stats['profitable_days']}")
        logger.info(f"  Loss days:          {stats['loss_days']}")
        logger.info("=" * 60)
