"""Iron Condor 0DTE strategy – 铁鹰 0DTE 策略.

每日在指定时间 (默认 UTC 08:00) 执行:
1. 获取当日到期 (0DTE) 的期权链
2. 选择 ±8% OTM 的行权价做为短腿 (sell call / sell put)
3. 选择更远 OTM 的行权价做为长腿保护 (buy call / buy put)
4. 下单形成铁鹰组合
5. 持续监控并持有到结算

策略逻辑:
  - 做空波动率: 收取双边权利金
  - 铁鹰结构: 限制最大亏损
  - 0DTE: 每日结算, 快速 theta 衰减
"""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
from typing import Any

from loguru import logger

from trader.binance_client import BinanceOptionsClient, OptionTicker, _parse_symbol
from trader.config import StrategyConfig
from trader.position_manager import PositionManager
from trader.storage import Storage


class IronCondor0DTEStrategy:
    """Daily 0DTE Iron Condor strategy for Binance options.

    Workflow per day:
    1. At entry_time_utc: scan for 0DTE options
    2. Find best strikes for ±otm_pct short legs
    3. Find wing strikes at ±(otm_pct + wing_width_pct)
    4. Execute 4-leg iron condor
    5. Hold to settlement
    """

    def __init__(
        self,
        client: BinanceOptionsClient,
        position_mgr: PositionManager,
        storage: Storage,
        config: StrategyConfig,
    ):
        self.client = client
        self.pos_mgr = position_mgr
        self.storage = storage
        self.cfg = config

        # Recovery: last trade date from storage
        self._last_trade_date: str = self.storage.load_state(
            "last_trade_date", ""
        )
        self._day_start_equity: float = self.storage.load_state(
            "day_start_equity", 0.0
        )

    # ------------------------------------------------------------------
    # Main entry: called periodically by the scheduler
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Single strategy tick – called every check interval.

        Handles: entry, position status logging.
        """
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")

        # 1. Check if it's time to open a new position
        if self._should_enter(now, today):
            self._try_open_condor(now, today)

        # 2. Log position status (hold to settlement, no early close)
        if self.pos_mgr.open_position_count > 0:
            self._log_positions()

    # ------------------------------------------------------------------
    # Entry logic
    # ------------------------------------------------------------------

    def _should_enter(self, now: datetime, today: str) -> bool:
        """Determine if we should open a new position now."""
        # Already traded today?
        if self._last_trade_date == today:
            return False

        # Max positions reached?
        if self.pos_mgr.open_position_count >= self.cfg.max_positions:
            return False

        # Exchange position guard: skip if positions already exist on exchange
        try:
            exchange_positions = self.client.get_positions(
                self.cfg.underlying.upper()
            )
            if exchange_positions:
                symbols = [p["symbol"] for p in exchange_positions]
                logger.warning(
                    f"[Strategy] Exchange has {len(exchange_positions)} "
                    f"existing position(s): {symbols} – skipping entry"
                )
                return False
        except Exception as e:
            logger.error(f"[Strategy] Position check failed: {e} – skipping entry")
            return False

        # Parse entry time (HH:MM)
        entry_parts = self.cfg.entry_time_utc.split(":")
        entry_minutes = int(entry_parts[0]) * 60 + int(entry_parts[1])
        now_minutes = now.hour * 60 + now.minute

        # Is it entry time?
        if now_minutes < entry_minutes:
            return False

        # Don't enter too late (2 hours after entry time max)
        if now_minutes > entry_minutes + 120:
            return False

        return True

    def _try_open_condor(self, now: datetime, today: str) -> None:
        """Attempt to open a new iron condor position."""
        ul = self.cfg.underlying.upper()
        logger.info(f"[Strategy] Scanning 0DTE options for {ul}...")

        # Get all tickers
        try:
            tickers = self.client.get_tickers(ul)
        except Exception as e:
            logger.error(f"Failed to fetch tickers: {e}")
            return

        if not tickers:
            logger.warning("No tickers available")
            return

        # Get spot price
        spot = self.client.get_spot_price(ul)
        if spot <= 0:
            # Try from tickers
            prices = [t.underlying_price for t in tickers if t.underlying_price > 0]
            spot = prices[0] if prices else 0
        if spot <= 0:
            logger.error("Cannot determine spot price")
            return

        logger.info(f"[Strategy] {ul} spot = {spot:.2f}")

        # Filter for 0DTE options (expiring today, DTE < 24h)
        today_tickers = [
            t for t in tickers
            if t.dte_hours > 0 and t.dte_hours < 24
        ]

        if not today_tickers:
            logger.warning("No 0DTE options available today")
            return

        # --- Wait-for-midpoint check ---
        if self.cfg.wait_for_midpoint:
            all_strikes = sorted({t.strike for t in today_tickers})
            if len(all_strikes) >= 2:
                # Find the two strikes that bracket the spot price
                lower = [k for k in all_strikes if k <= spot]
                upper = [k for k in all_strikes if k > spot]
                if lower and upper:
                    k_low, k_high = lower[-1], upper[0]
                    midpoint = (k_low + k_high) / 2.0
                    gap = k_high - k_low
                    # Allow 10% tolerance of the gap width
                    tolerance = gap * 0.10
                    if abs(spot - midpoint) > tolerance:
                        logger.info(
                            f"[Strategy] wait_for_midpoint: spot={spot:.2f} "
                            f"midpoint={midpoint:.2f} (K={k_low}/{k_high}), "
                            f"delta={abs(spot-midpoint):.2f} > tol={tolerance:.2f} – skip"
                        )
                        return
                    logger.info(
                        f"[Strategy] wait_for_midpoint: spot={spot:.2f} ≈ "
                        f"midpoint={midpoint:.2f} ✓"
                    )

        # Find strike targets
        short_call_target = spot * (1 + self.cfg.otm_pct)
        short_put_target = spot * (1 - self.cfg.otm_pct)
        long_call_target = spot * (1 + self.cfg.otm_pct + self.cfg.wing_width_pct)
        long_put_target = spot * (1 - self.cfg.otm_pct - self.cfg.wing_width_pct)

        logger.info(
            f"[Strategy] Strike targets: "
            f"sell_call≈{short_call_target:.0f} buy_call≈{long_call_target:.0f} "
            f"sell_put≈{short_put_target:.0f} buy_put≈{long_put_target:.0f}"
        )

        # Select best instruments
        sell_call = self._find_best_strike(today_tickers, "call", short_call_target)
        buy_call = self._find_best_strike(today_tickers, "call", long_call_target)
        sell_put = self._find_best_strike(today_tickers, "put", short_put_target)
        buy_put = self._find_best_strike(today_tickers, "put", long_put_target)

        if not all([sell_call, buy_call, sell_put, buy_put]):
            logger.warning("Could not find all 4 legs for iron condor")
            return

        assert sell_call is not None
        assert buy_call is not None
        assert sell_put is not None
        assert buy_put is not None

        # Validate: call strikes must be sell < buy, put strikes must be sell > buy
        if sell_call.strike >= buy_call.strike:
            logger.warning(
                f"Invalid call spread: sell K={sell_call.strike} >= buy K={buy_call.strike}"
            )
            return
        if sell_put.strike <= buy_put.strike:
            logger.warning(
                f"Invalid put spread: sell K={sell_put.strike} <= buy K={buy_put.strike}"
            )
            return

        # Validate liquidity (bid > 0 for short legs)
        if sell_call.bid_price <= 0 or sell_put.bid_price <= 0:
            logger.warning("Short legs have no bid – skipping (no liquidity)")
            return

        # Determine quantity
        quantity = self._compute_quantity(spot)
        if quantity <= 0:
            logger.warning("Computed quantity <= 0, skipping")
            return

        logger.info(
            f"[Strategy] Opening Iron Condor:\n"
            f"  Sell Call: {sell_call.symbol} K={sell_call.strike} "
            f"bid={sell_call.bid_price:.4f}\n"
            f"  Buy Call:  {buy_call.symbol} K={buy_call.strike} "
            f"ask={buy_call.ask_price:.4f}\n"
            f"  Sell Put:  {sell_put.symbol} K={sell_put.strike} "
            f"bid={sell_put.bid_price:.4f}\n"
            f"  Buy Put:   {buy_put.symbol} K={buy_put.strike} "
            f"ask={buy_put.ask_price:.4f}\n"
            f"  Qty: {quantity}"
        )

        # Execute
        condor = self.pos_mgr.open_iron_condor(
            sell_call_symbol=sell_call.symbol,
            buy_call_symbol=buy_call.symbol,
            sell_put_symbol=sell_put.symbol,
            buy_put_symbol=buy_put.symbol,
            sell_call_strike=sell_call.strike,
            buy_call_strike=buy_call.strike,
            sell_put_strike=sell_put.strike,
            buy_put_strike=buy_put.strike,
            quantity=quantity,
            underlying_price=spot,
        )

        if condor:
            self._last_trade_date = today
            self.storage.save_state("last_trade_date", today)
            logger.info(
                f"[Strategy] Iron Condor opened: {condor.group_id} "
                f"premium={condor.total_premium:.4f}"
            )
        else:
            logger.error("[Strategy] Failed to open Iron Condor")

    def _find_best_strike(
        self,
        tickers: list[OptionTicker],
        option_type: str,
        target_strike: float,
    ) -> OptionTicker | None:
        """Find the ticker with strike closest to target."""
        candidates = [
            t for t in tickers if t.option_type == option_type
        ]
        if not candidates:
            return None

        # Sort by distance to target
        candidates.sort(key=lambda t: abs(t.strike - target_strike))
        return candidates[0]

    def _compute_quantity(self, spot: float) -> float:
        """Compute order quantity based on account equity and config.

        If compound=True, scale quantity to current equity, capped by
        max_capital_pct.

        Binance European Options use portfolio margin. For an Iron Condor
        (4 legs), the margin requirement ≈ one side's spread width × qty
        (the two vertical spreads share margin since they can't both lose).
        We use a conservative estimate: margin ≈ wing_width_pct × spot × qty.
        Additionally each leg incurs premium outflow for the longs and inflow
        for the shorts, but margin is the binding constraint.

        quantity (base) is the default / minimum.
        max_capital_pct is the safety cap — scaled_qty will not exceed it.
        """
        quantity = self.cfg.quantity

        if self.cfg.compound:
            try:
                account = self.client.get_account()
                equity = account.total_balance
                if equity > 0 and spot > 0:
                    # Max capital we're willing to allocate
                    max_notional = equity * self.cfg.max_capital_pct
                    # Margin per Iron Condor ≈ spread width of one side
                    # (both sides share margin, so we don't double-count)
                    wing_width = self.cfg.wing_width_pct * spot
                    if wing_width > 0:
                        # How many Iron Condors can we afford?
                        max_qty = max_notional / wing_width
                        # Round down to step size, cap at max_qty
                        scaled_qty = math.floor(max_qty * 100) / 100
                        quantity = min(scaled_qty, max(quantity, 0.01))
                        # If base quantity exceeds cap, warn and clamp
                        if self.cfg.quantity > scaled_qty:
                            logger.warning(
                                f"Base quantity {self.cfg.quantity} exceeds "
                                f"capital cap {scaled_qty:.2f} "
                                f"(equity={equity:.2f}, "
                                f"max_capital_pct={self.cfg.max_capital_pct})"
                            )
                            quantity = scaled_qty
                        else:
                            # Scale up from base, but never exceed cap
                            quantity = min(scaled_qty, max(self.cfg.quantity, 0.01))

                    logger.info(
                        f"[Quantity] equity={equity:.2f}, spot={spot:.2f}, "
                        f"wing_width={wing_width:.2f}, "
                        f"max_notional={max_notional:.2f}, "
                        f"scaled_qty={quantity:.2f}"
                    )
            except Exception as e:
                logger.warning(f"Could not scale quantity: {e}")

        return quantity

    # ------------------------------------------------------------------
    # Position monitoring
    # ------------------------------------------------------------------

    def _log_positions(self) -> None:
        """Log status of open positions (hold to settlement)."""
        now = datetime.now(timezone.utc)
        for gid, condor in self.pos_mgr.open_condors.items():
            if not condor.is_open:
                continue
            for leg in condor.legs:
                parsed = _parse_symbol(leg.symbol)
                if parsed:
                    time_to_expiry = (parsed["expiry"] - now).total_seconds()
                    hours = time_to_expiry / 3600
                    if hours > 0:
                        logger.debug(
                            f"[Strategy] {gid} -> settlement in {hours:.1f}h"
                        )
                    break

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Return current strategy status."""
        return {
            "strategy": "IronCondor0DTE",
            "underlying": self.cfg.underlying,
            "otm_pct": self.cfg.otm_pct,
            "wing_width_pct": self.cfg.wing_width_pct,
            "last_trade_date": self._last_trade_date,
            "open_positions": self.pos_mgr.open_position_count,
            "max_positions": self.cfg.max_positions,
            "positions": self.pos_mgr.summary(),
        }
