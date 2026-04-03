"""Options selling strategy – 期权卖出策略.

包含两个策略类:

1. OptionSellingStrategy
   - 支持 strangle (裸卖双卖) / iron_condor (铁鹰)
   - 每日执行, OTM% 选择行权价, 适用于 Binance

2. WeekendVolStrategy
   - 周末波动率卖出铁鹰策略
   - 每周五 16:00 UTC 入场, 持有至周日 08:00 UTC 结算
   - Delta 选择行权价 (short delta=0.40, wing delta=0.05)
   - 3x 杠杆, USD 保证金复利
   - 适用于 Binance BTC
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


class OptionSellingStrategy:
    """Options selling strategy for Binance options.

    Supports two modes:
    - strangle: 2-leg naked sell (sell call + sell put)
    - iron_condor: 4-leg iron condor (sell + buy on each side)

    Workflow per day:
    1. At entry_time_utc: scan for options within target DTE range
    2. Find best strikes for ±otm_pct short legs
    3. (iron_condor) Find wing strikes at ±(otm_pct + wing_width_pct)
    4. Execute position
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
        """Attempt to open a new position (strangle or iron condor)."""
        ul = self.cfg.underlying.upper()
        mode = getattr(self.cfg, "mode", "iron_condor")
        target_dte_days = getattr(self.cfg, "target_dte_days", 0)
        dte_window = getattr(self.cfg, "dte_window_hours", 48)

        mode_label = "Short Strangle" if mode == "strangle" else "Iron Condor"
        logger.info(f"[Strategy] Scanning {mode_label} DTE≈{target_dte_days}d options for {ul}...")

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
            prices = [t.underlying_price for t in tickers if t.underlying_price > 0]
            spot = prices[0] if prices else 0
        if spot <= 0:
            logger.error("Cannot determine spot price")
            return

        logger.info(f"[Strategy] {ul} spot = {spot:.2f}")

        # Filter by DTE range
        target_dte_hours = target_dte_days * 24
        dte_min = max(0.1, target_dte_hours - dte_window)
        dte_max = target_dte_hours + dte_window

        filtered_tickers = [
            t for t in tickers
            if t.dte_hours > dte_min and t.dte_hours < dte_max
        ]

        if not filtered_tickers:
            logger.warning(
                f"No options with DTE in [{dte_min:.0f}h, {dte_max:.0f}h] "
                f"(target={target_dte_days}d ±{dte_window}h)"
            )
            return

        # --- Wait-for-midpoint check ---
        if self.cfg.wait_for_midpoint:
            all_strikes = sorted({t.strike for t in filtered_tickers})
            if len(all_strikes) >= 2:
                lower = [k for k in all_strikes if k <= spot]
                upper = [k for k in all_strikes if k > spot]
                if lower and upper:
                    k_low, k_high = lower[-1], upper[0]
                    midpoint = (k_low + k_high) / 2.0
                    gap = k_high - k_low
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

        # Select best instruments for short legs
        sell_call = self._find_best_strike(filtered_tickers, "call", short_call_target)
        sell_put = self._find_best_strike(filtered_tickers, "put", short_put_target)

        if not sell_call or not sell_put:
            logger.warning("Could not find both short legs")
            return

        # Validate liquidity (bid > 0 for short legs)
        if sell_call.bid_price <= 0 or sell_put.bid_price <= 0:
            logger.warning("Short legs have no bid – skipping (no liquidity)")
            return

        # ---- Mode: Strangle (2-leg naked sell) ----
        if mode == "strangle":
            # Determine quantity
            quantity = self._compute_quantity(spot)
            if quantity <= 0:
                logger.warning("Computed quantity <= 0, skipping")
                return

            logger.info(
                f"[Strategy] Opening Short Strangle:\n"
                f"  Sell Call: {sell_call.symbol} K={sell_call.strike} "
                f"bid={sell_call.bid_price:.4f}\n"
                f"  Sell Put:  {sell_put.symbol} K={sell_put.strike} "
                f"bid={sell_put.bid_price:.4f}\n"
                f"  Qty: {quantity}  DTE: {sell_call.dte_hours:.0f}h"
            )

            condor = self.pos_mgr.open_short_strangle(
                sell_call_symbol=sell_call.symbol,
                sell_put_symbol=sell_put.symbol,
                sell_call_strike=sell_call.strike,
                sell_put_strike=sell_put.strike,
                quantity=quantity,
                underlying_price=spot,
                execution_mode="market",
            )

            if condor:
                self._last_trade_date = today
                self.storage.save_state("last_trade_date", today)
                logger.info(
                    f"[Strategy] Short Strangle opened: {condor.group_id} "
                    f"premium={condor.total_premium:.4f}"
                )
            else:
                logger.error("[Strategy] Failed to open Short Strangle")
            return

        # ---- Mode: Iron Condor (4-leg) ----
        long_call_target = spot * (1 + self.cfg.otm_pct + self.cfg.wing_width_pct)
        long_put_target = spot * (1 - self.cfg.otm_pct - self.cfg.wing_width_pct)

        logger.info(
            f"[Strategy] Strike targets: "
            f"sell_call≈{short_call_target:.0f} buy_call≈{long_call_target:.0f} "
            f"sell_put≈{short_put_target:.0f} buy_put≈{long_put_target:.0f}"
        )

        # Select instruments for wing legs
        buy_call = self._find_best_strike(filtered_tickers, "call", long_call_target)
        buy_put = self._find_best_strike(filtered_tickers, "put", long_put_target)

        if not all([buy_call, buy_put]):
            logger.warning("Could not find wing legs for iron condor")
            return

        assert buy_call is not None
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
            execution_mode="market",
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

        compound=False → 直接用 quantity (base)
        compound=True  → 按权益放大, quantity 是下限, max_capital_pct 是上限

        保证金估算:
        - strangle (裸卖): margin ≈ otm_pct × spot × qty × 2
        - iron_condor: margin ≈ wing_width × spot × qty × 2
        """
        quantity = self.cfg.quantity
        mode = getattr(self.cfg, "mode", "iron_condor")

        if self.cfg.compound:
            try:
                account = self.client.get_account()
                equity = account.total_balance
                if equity > 0 and spot > 0:
                    max_notional = equity * self.cfg.max_capital_pct

                    if mode == "strangle":
                        # 裸卖: 保守用 otm_pct 作为保证金估算基础
                        margin_per_unit = self.cfg.otm_pct * spot * 2
                    else:
                        # 铁鹰: 用翼宽
                        wing_width = self.cfg.wing_width_pct * spot
                        margin_per_unit = wing_width * 2 if wing_width > 0 else self.cfg.otm_pct * spot * 2

                    if margin_per_unit > 0:
                        scaled_qty = math.floor(
                            max_notional / margin_per_unit * 100
                        ) / 100
                        quantity = max(self.cfg.quantity, scaled_qty)

                    logger.info(
                        f"[Quantity] equity={equity:.2f}, spot={spot:.2f}, "
                        f"mode={mode}, "
                        f"margin_per_unit={margin_per_unit:.2f}, "
                        f"max_notional={max_notional:.2f}, "
                        f"scaled_qty={scaled_qty:.2f}, "
                        f"final_qty={quantity:.2f}"
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
        mode = getattr(self.cfg, "mode", "iron_condor")
        target_dte = getattr(self.cfg, "target_dte_days", 0)
        return {
            "strategy": "ShortStrangle" if mode == "strangle" else "IronCondor",
            "mode": mode,
            "target_dte_days": target_dte,
            "underlying": self.cfg.underlying,
            "otm_pct": self.cfg.otm_pct,
            "wing_width_pct": self.cfg.wing_width_pct,
            "last_trade_date": self._last_trade_date,
            "open_positions": self.pos_mgr.open_position_count,
            "max_positions": self.cfg.max_positions,
            "positions": self.pos_mgr.summary(),
        }


# Backward-compatible alias
IronCondor0DTEStrategy = OptionSellingStrategy


# ======================================================================
# Weekend Volatility Selling Strategy (新策略 – 替代旧策略)
# ======================================================================

class WeekendVolStrategy:
    """Weekend volatility selling strategy on Binance.

    Strategy logic (mirrors backtest_weekend_vol.py):
    ─────────────────────────────────────────────────
    1. Every **Friday at entry_time_utc** (default 16:00 UTC):
       a. Fetch BTC option chain for the coming **Sunday 08:00 UTC** expiry
       b. Compute Black-76 delta for each strike using mark_iv or default_iv
       c. Select short call: closest strike to |delta| = target_delta
       d. Select short put:  closest strike to |delta| = target_delta
         e. Select wing call: closest strike to |delta| = wing_delta (optional protection)
         f. Select wing put: closest strike to |delta| = wing_delta (optional protection)
       g. Compute quantity:  (balance × leverage) / spot, compounding
         h. Execute short strangle / iron condor via PositionManager

     2. During holding period:
         a. Monitor basket PnL using option mark prices
         b. If basket PnL% <= -stop_loss_pct, close all legs early
         c. Otherwise hold to settlement

    Parameters (from StrategyConfig):
    ──────────────────────────────────
    - target_delta : 0.40 (|δ| for short legs)
    - wing_delta   : 0.05 (|δ| for protection legs, 0 → strangle)
    - max_delta_diff : 0.20 (actual |δ| vs target |δ| max deviation)
    - leverage     : 4.0
    - entry_day    : "friday"
    - entry_time_utc : "18:00"
    - underlying   : "BTC"
    - compound     : True
    """

    DAY_MAP = {
        "monday": 0, "tuesday": 1, "wednesday": 2,
        "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
    }

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

        # Recovery: last trade week identifier to avoid duplicate entry
        self._last_trade_week: str = self.storage.load_state(
            "wv_last_trade_week", ""
        )
        self._last_order_attempt_at: str = ""
        self._last_order_status: str = "idle"
        self._last_order_message: str = ""
        self._last_order_group_id: str = ""
        self._last_order_exchange_positions_checked: bool = False
        self._last_order_exchange_positions_underlying: str = ""
        self._last_order_exchange_positions: list[dict[str, Any]] = []
        self._last_order_exchange_positions_error: str = ""

    # ------------------------------------------------------------------
    # Main entry: called periodically by engine / main loop
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Single strategy tick."""
        now = datetime.now(timezone.utc)

        # 1. Check settlement of open positions
        self._check_settlement(now)

        # 2. Check live stop management
        self._manage_open_positions(now)

        # 3. Check if it's time to open a new position
        if self._should_enter(now):
            self._try_open_iron_condor(now)

    # ------------------------------------------------------------------
    # Entry logic
    # ------------------------------------------------------------------

    def _should_enter(self, now: datetime) -> bool:
        """Determine if we should open a new position now."""
        # Correct day of week?
        target_day = self.DAY_MAP.get(self.cfg.entry_day.lower(), 4)
        if now.weekday() != target_day:
            return False

        # Parse entry time
        parts = self.cfg.entry_time_utc.split(":")
        entry_min = int(parts[0]) * 60 + int(parts[1])
        now_min = now.hour * 60 + now.minute

        # Before entry time?
        if now_min < entry_min:
            return False

        # Too late? (max 2 hours after entry)
        if now_min > entry_min + 120:
            return False

        # Already traded this week?
        week_id = now.strftime("%G-W%V")  # ISO week
        if self._last_trade_week == week_id:
            return False

        # Max positions?
        if self.pos_mgr.open_position_count >= self.cfg.max_positions:
            return False

        # Exchange position guard
        try:
            exchange_positions = self.client.get_positions(self.cfg.underlying.upper())
            if exchange_positions:
                symbols = [p["symbol"] for p in exchange_positions]
                logger.warning(
                    f"[WeekendVol] Exchange has {len(exchange_positions)} "
                    f"existing position(s): {symbols} – skipping"
                )
                return False
        except Exception as e:
            logger.error(f"[WeekendVol] Position check failed: {e} – skipping")
            return False

        return True

    # ------------------------------------------------------------------
    # Core: open weekend volatility position via delta-based strike selection
    # ------------------------------------------------------------------

    def _try_open_iron_condor(self, now: datetime) -> None:
        """Fetch option chain, select strikes by delta, execute combo position."""
        ul = self.cfg.underlying.upper()
        self._mark_order_attempt("opening", f"开始扫描 {ul} 周末合约")
        logger.info(f"[WeekendVol] Scanning {ul} weekend options for target combo...")

        rv24 = self._get_entry_realized_vol(ul)
        if not self._passes_entry_rv_filter(rv24):
            return

        # --- Get tickers ---
        try:
            tickers = self.client.get_tickers(ul)
        except Exception as e:
            logger.error(f"[WeekendVol] Failed to fetch tickers: {e}")
            return

        if not tickers:
            logger.warning("[WeekendVol] No tickers available")
            return

        # --- Spot price ---
        spot = self.client.get_spot_price(ul)
        if spot <= 0:
            prices = [t.underlying_price for t in tickers if t.underlying_price > 0]
            spot = prices[0] if prices else 0
        if spot <= 0:
            logger.error("[WeekendVol] Cannot determine spot price")
            return

        logger.info(f"[WeekendVol] {ul} spot = {spot:.2f}")

        # --- Filter for coming Sunday 08:00 UTC expiry ---
        sunday_expiry = self._next_sunday_0800(now)
        tolerance_hours = 2.0  # allow ±2h tolerance for expiry matching

        weekend_tickers = [
            t for t in tickers
            if abs((t.expiry - sunday_expiry).total_seconds()) < tolerance_hours * 3600
        ]

        if not weekend_tickers:
            logger.warning(
                f"[WeekendVol] No options expiring around {sunday_expiry.isoformat()} "
                f"(found {len(tickers)} total tickers)"
            )
            return

        logger.info(
            f"[WeekendVol] Found {len(weekend_tickers)} tickers for Sunday expiry "
            f"{sunday_expiry.strftime('%Y-%m-%d %H:%M')} UTC"
        )

        # --- Fetch Greeks from exchange ---
        try:
            self.client.enrich_greeks(weekend_tickers, ul)
        except Exception as e:
            logger.warning(f"[WeekendVol] Failed to fetch greeks, will use Black-76 fallback: {e}")

        calls = [t for t in weekend_tickers if t.option_type == "call"]
        puts = [t for t in weekend_tickers if t.option_type == "put"]

        T_years = max((sunday_expiry - now).total_seconds() / (365.25 * 86400), 1e-6)

        # --- Select strikes by delta ---
        sell_call = self._find_by_delta(calls, spot, T_years, self.cfg.target_delta, "call")
        sell_put = self._find_by_delta(puts, spot, T_years, self.cfg.target_delta, "put")

        if not sell_call or not sell_put:
            logger.warning("[WeekendVol] Could not find both short legs by delta")
            return

        # Validate liquidity
        if sell_call.bid_price <= 0 or sell_put.bid_price <= 0:
            logger.warning("[WeekendVol] Short legs have no bid – skipping")
            return

        # --- Wing legs ---
        buy_call = None
        buy_put = None
        if self.cfg.wing_delta > 0:
            buy_call = self._find_by_delta(calls, spot, T_years, self.cfg.wing_delta, "call")
            buy_put = self._find_by_delta(puts, spot, T_years, self.cfg.wing_delta, "put")

            if not buy_call or not buy_put:
                logger.warning("[WeekendVol] Could not find wing legs by delta")
                return

            # Validate: call sell < buy, put sell > buy
            if sell_call.strike >= buy_call.strike:
                logger.warning(
                    f"[WeekendVol] Invalid call spread: sell K={sell_call.strike} "
                    f">= buy K={buy_call.strike}"
                )
                return
            if sell_put.strike <= buy_put.strike:
                logger.warning(
                    f"[WeekendVol] Invalid put spread: sell K={sell_put.strike} "
                    f"<= buy K={buy_put.strike}"
                )
                return

        # --- Compute quantity ---
        quantity = self._compute_quantity(spot)
        if quantity <= 0:
            logger.warning("[WeekendVol] Computed quantity <= 0, skipping")
            return

        # --- Execute ---
        week_id = now.strftime("%G-W%V")

        if self.cfg.wing_delta > 0 and buy_call and buy_put:
            logger.info(
                f"[WeekendVol] Opening Iron Condor:\n"
                f"  Sell Call: {sell_call.symbol} K={sell_call.strike}\n"
                f"  Buy Call:  {buy_call.symbol} K={buy_call.strike}\n"
                f"  Sell Put:  {sell_put.symbol} K={sell_put.strike}\n"
                f"  Buy Put:   {buy_put.symbol} K={buy_put.strike}\n"
                f"  Qty: {quantity:.3f}  Leverage: {self.cfg.leverage}x  "
                f"T={T_years*365.25:.1f}d"
            )
            try:
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
                    execution_mode="market",
                )
            except Exception as e:
                self._record_live_order_failure(
                    underlying=ul,
                    message=f"实盘开仓异常: {e}",
                    status="error",
                )
                raise
        else:
            # No wings → short strangle
            logger.info(
                f"[WeekendVol] Opening Short Strangle:\n"
                f"  Sell Call: {sell_call.symbol} K={sell_call.strike}\n"
                f"  Sell Put:  {sell_put.symbol} K={sell_put.strike}\n"
                f"  Qty: {quantity:.3f}  Leverage: {self.cfg.leverage}x  "
                f"T={T_years*365.25:.1f}d"
            )
            try:
                condor = self.pos_mgr.open_short_strangle(
                    sell_call_symbol=sell_call.symbol,
                    sell_put_symbol=sell_put.symbol,
                    sell_call_strike=sell_call.strike,
                    sell_put_strike=sell_put.strike,
                    quantity=quantity,
                    underlying_price=spot,
                    execution_mode="market",
                )
            except Exception as e:
                self._record_live_order_failure(
                    underlying=ul,
                    message=f"实盘开仓异常: {e}",
                    status="error",
                )
                raise

        if condor:
            self._last_trade_week = week_id
            self.storage.save_state("wv_last_trade_week", week_id)
            self._mark_order_success(
                group_id=condor.group_id,
                message=f"实盘开仓成功: {condor.group_id}",
            )
            logger.info(
                f"[WeekendVol] Position opened: {condor.group_id} "
                f"premium={condor.total_premium:.6f}"
            )
        else:
            self._record_live_order_failure(
                underlying=ul,
                message="实盘开仓失败：至少有一条腿未成交，或成交后本地记账失败",
                status="failed",
            )
            logger.error("[WeekendVol] Failed to open position")

    def _mark_order_attempt(self, status: str, message: str, group_id: str = "") -> None:
        self._last_order_attempt_at = datetime.now(timezone.utc).isoformat()
        self._last_order_status = status
        self._last_order_message = message
        self._last_order_group_id = group_id
        self._last_order_exchange_positions_checked = False
        self._last_order_exchange_positions_underlying = ""
        self._last_order_exchange_positions = []
        self._last_order_exchange_positions_error = ""

    def _mark_order_success(self, group_id: str, message: str) -> None:
        self._last_order_attempt_at = datetime.now(timezone.utc).isoformat()
        self._last_order_status = "success"
        self._last_order_message = message
        self._last_order_group_id = group_id
        self._last_order_exchange_positions_checked = False
        self._last_order_exchange_positions_underlying = ""
        self._last_order_exchange_positions = []
        self._last_order_exchange_positions_error = ""

    def _record_live_order_failure(self, underlying: str, message: str, status: str) -> None:
        self._last_order_attempt_at = datetime.now(timezone.utc).isoformat()
        self._last_order_status = status
        self._last_order_message = message
        self._last_order_group_id = ""
        self._last_order_exchange_positions_checked = True
        self._last_order_exchange_positions_underlying = underlying.upper()
        try:
            positions = self.client.get_positions(underlying.upper())
            self._last_order_exchange_positions = positions
            self._last_order_exchange_positions_error = ""
            if positions:
                logger.error(
                    f"[WeekendVol] Post-failure exchange position check found {len(positions)} live position(s): "
                    f"{[p.get('symbol') for p in positions]}"
                )
            else:
                logger.info(f"[WeekendVol] Post-failure exchange position check: no live {underlying.upper()} positions")
        except Exception as e:
            self._last_order_exchange_positions = []
            self._last_order_exchange_positions_error = str(e)
            logger.error(f"[WeekendVol] Post-failure exchange position check failed: {e}")

    def _manage_open_positions(self, now: datetime) -> None:
        if self.pos_mgr.open_position_count <= 0 or self.cfg.stop_loss_pct <= 0:
            return

        try:
            mark_prices = self.client.get_mark_prices(self.cfg.underlying.upper())
            basket_pnl_pct = self._basket_pnl_pct(mark_prices)
        except Exception as e:
            logger.warning(f"[WeekendVol] Failed to evaluate live stop: {e}")
            return

        if basket_pnl_pct is None:
            return

        if basket_pnl_pct <= -float(self.cfg.stop_loss_pct):
            logger.warning(
                f"[WeekendVol] Stop loss triggered: basket pnl {basket_pnl_pct:.1f}% "
                f"<= -{self.cfg.stop_loss_pct:.1f}%"
            )
            self.pos_mgr.close_all(
                reason=f"weekend_vol_stop_loss_{self.cfg.stop_loss_pct:.0f}",
                execution_mode="market",
            )

    def _get_entry_realized_vol(self, underlying: str) -> float | None:
        if self.cfg.entry_realized_vol_lookback_hours <= 1 or self.cfg.entry_realized_vol_max <= 0:
            return None
        return self._compute_realized_vol(underlying, log_result=True)

    def _compute_realized_vol(self, underlying: str, log_result: bool = False) -> float | None:
        try:
            rv = self.client.get_realized_vol(underlying, self.cfg.entry_realized_vol_lookback_hours)
        except Exception as e:
            logger.warning(f"[WeekendVol] Failed to compute RV filter input: {e}")
            return None
        if rv is not None and log_result:
            logger.info(
                f"[WeekendVol] RV{self.cfg.entry_realized_vol_lookback_hours} = {rv:.2%}"
            )
        return rv

    def _passes_entry_rv_filter(self, realized_vol: float | None) -> bool:
        if self.cfg.entry_realized_vol_lookback_hours <= 1 or self.cfg.entry_realized_vol_max <= 0:
            return True
        if realized_vol is None:
            logger.warning("[WeekendVol] RV filter enabled but realized vol unavailable – skipping entry")
            return False
        if realized_vol > self.cfg.entry_realized_vol_max:
            logger.info(
                f"[WeekendVol] Skip entry: RV{self.cfg.entry_realized_vol_lookback_hours} "
                f"{realized_vol:.2%} > {self.cfg.entry_realized_vol_max:.2%}"
            )
            return False
        return True

    def _basket_pnl_pct(self, mark_prices: dict[str, float]) -> float | None:
        entry_credit = 0.0
        close_cost = 0.0

        for condor in self.pos_mgr.open_condors.values():
            if not condor.is_open:
                continue
            for leg in condor.legs:
                sign = 1.0 if leg.side == "SELL" else -1.0
                mark = float(mark_prices.get(leg.symbol, leg.entry_price) or leg.entry_price)
                entry_credit += float(leg.entry_price) * float(leg.quantity) * sign
                close_cost += mark * float(leg.quantity) * sign

        if entry_credit <= 0:
            return None

        pnl = entry_credit - close_cost
        return pnl / entry_credit * 100.0

    def _current_basket_pnl_pct(self) -> float | None:
        if self.pos_mgr.open_position_count <= 0:
            return None
        try:
            mark_prices = self.client.get_mark_prices(self.cfg.underlying.upper())
        except Exception as e:
            logger.warning(f"[WeekendVol] Failed to fetch mark prices for status: {e}")
            return None
        return self._basket_pnl_pct(mark_prices)

    # ------------------------------------------------------------------
    # Delta-based strike selection
    # ------------------------------------------------------------------

    def _find_by_delta(
        self,
        tickers: list,
        spot: float,
        T_years: float,
        target_abs_delta: float,
        option_type: str,
    ) -> Any:
        """Find the ticker whose |delta| is closest to target.

        Prefers exchange-provided delta from /eapi/v1/mark.
        Falls back to Black-76 calculation when delta is not available.
        """
        best = None
        best_diff = float("inf")
        best_delta = 0.0
        used_exchange = False

        for t in tickers:
            # Prefer exchange delta
            if t.delta != 0.0:
                abs_d = abs(t.delta)
            else:
                # Fallback: compute via Black-76
                from options_backtest.pricing.black76 import delta as bs_delta
                iv = t.mark_iv or self.cfg.default_iv
                if iv <= 0:
                    iv = self.cfg.default_iv
                try:
                    d = bs_delta(spot, t.strike, T_years, sigma=iv,
                                 option_type=option_type, r=0.0)
                except (ValueError, ZeroDivisionError):
                    continue
                abs_d = abs(d)

            diff = abs(abs_d - target_abs_delta)
            if diff < best_diff:
                best_diff = diff
                best = t
                best_delta = abs_d
                used_exchange = t.delta != 0.0

        if best:
            source = "exchange" if used_exchange else "Black-76"
            if best_diff > self.cfg.max_delta_diff:
                if used_exchange:
                    logger.warning(
                        f"[WeekendVol] Reject {option_type} strike {best.symbol}: "
                        f"|δ|={best_delta:.4f} target={target_abs_delta:.4f} "
                        f"diff={best_diff:.4f} > max_delta_diff={self.cfg.max_delta_diff:.4f}"
                    )
                    return None
                logger.warning(
                    f"[WeekendVol] Accepting closest {option_type} strike via Black-76 fallback: "
                    f"{best.symbol} |δ|={best_delta:.4f} target={target_abs_delta:.4f} "
                    f"diff={best_diff:.4f} > max_delta_diff={self.cfg.max_delta_diff:.4f}"
                )
            logger.info(
                f"[WeekendVol] {option_type} strike selection: "
                f"K={best.strike} |δ|={best_delta:.4f} "
                f"(target={target_abs_delta}, diff={best_diff:.4f}) source={source} "
                f"IV={best.mark_iv:.1%} {best.symbol}"
            )

        return best

    # ------------------------------------------------------------------
    # Quantity computation (3x leverage, compounding)
    # ------------------------------------------------------------------

    def _compute_quantity(self, spot: float) -> float:
        """Compute position size with leverage + compounding.

        USD margin mode:
          qty_asset = (account_equity_usd × leverage) / spot
          Rounded down to 0.1 underlying increments.
        """
        base_qty = self.cfg.quantity
        ul = self.cfg.underlying.upper()

        if self.cfg.compound:
            try:
                account = self.client.get_account()
                equity = account.total_balance
                if equity > 0 and spot > 0:
                    qty = (equity * self.cfg.leverage) / spot
                    # Round down to 0.1 underlying units
                    qty = math.floor(qty * 10) / 10
                    base_qty = max(self.cfg.quantity, qty)
                    logger.info(
                        f"[WeekendVol] qty: equity={equity:.2f} USD, "
                        f"leverage={self.cfg.leverage}x, spot={spot:.2f}, "
                        f"raw={equity * self.cfg.leverage / spot:.4f}, "
                        f"final={base_qty:.1f} {ul}"
                    )
            except Exception as e:
                logger.warning(f"[WeekendVol] Could not compute compound qty: {e}")

        return base_qty

    # ------------------------------------------------------------------
    # Settlement check
    # ------------------------------------------------------------------

    def _check_settlement(self, now: datetime) -> None:
        """Check if any open position has expired and record settlement.

        Binance auto-settles expired options. We detect expiry from the
        symbol and mark the position as closed.
        """
        for gid in list(self.pos_mgr.open_condors.keys()):
            condor = self.pos_mgr.open_condors[gid]
            if not condor.is_open:
                continue

            # Check expiry from any leg
            for leg in condor.legs:
                parsed = _parse_symbol(leg.symbol)
                if parsed and parsed["expiry"] <= now:
                    underlying = str(parsed.get("underlying") or self.cfg.underlying).upper()
                    settlement_spot = self.client.get_spot_price(underlying)
                    if settlement_spot <= 0:
                        settlement_spot = condor.underlying_price
                    logger.info(
                        f"[WeekendVol] Position {gid} expired at "
                        f"{parsed['expiry'].isoformat()} – marking settled "
                        f"with {underlying} spot={settlement_spot:.4f}"
                    )

                    total_settlement_pnl = 0.0
                    for leg2 in condor.legs:
                        settlement_price = self._settlement_price(leg2, settlement_spot)
                        pnl = self._settlement_pnl(leg2, settlement_price)
                        self.storage.close_trade(
                            trade_id=leg2.trade_id,
                            close_price=settlement_price,
                            pnl=pnl,
                        )
                        total_settlement_pnl += pnl
                        logger.info(
                            f"[WeekendVol] Settled {leg2.side} {leg2.option_type} {leg2.symbol}: "
                            f"entry={leg2.entry_price:.4f} settle={settlement_price:.4f} pnl={pnl:.4f}"
                        )
                    logger.info(f"[WeekendVol] Position {gid} settlement pnl={total_settlement_pnl:.4f}")
                    condor.is_open = False
                    del self.pos_mgr.open_condors[gid]
                    break

    @staticmethod
    def _settlement_price(leg, settlement_spot: float) -> float:
        if settlement_spot <= 0:
            return 0.0
        if leg.option_type == "call":
            return max(settlement_spot - leg.strike, 0.0)
        return max(leg.strike - settlement_spot, 0.0)

    @staticmethod
    def _settlement_pnl(leg, settlement_price: float) -> float:
        if leg.side == "SELL":
            return (leg.entry_price - settlement_price) * leg.quantity
        return (settlement_price - leg.entry_price) * leg.quantity

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _next_sunday_0800(now: datetime) -> datetime:
        """Find the next Sunday 08:00 UTC from now (or this Sunday if today is Fri/Sat)."""
        days_until_sunday = (6 - now.weekday()) % 7
        if days_until_sunday == 0 and now.hour >= 8:
            days_until_sunday = 7  # already past this Sunday 08:00
        sunday = now.replace(hour=8, minute=0, second=0, microsecond=0) + timedelta(days=days_until_sunday)
        return sunday

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Return current strategy status."""
        rv24 = self._compute_realized_vol(self.cfg.underlying.upper(), log_result=False)
        basket_pnl_pct = self._current_basket_pnl_pct()
        return {
            "strategy": "WeekendVol",
            "mode": "weekend_vol",
            "underlying": self.cfg.underlying,
            "target_delta": self.cfg.target_delta,
            "wing_delta": self.cfg.wing_delta,
            "max_delta_diff": self.cfg.max_delta_diff,
            "leverage": self.cfg.leverage,
            "entry_day": self.cfg.entry_day,
            "entry_time_utc": self.cfg.entry_time_utc,
            "entry_realized_vol_lookback_hours": self.cfg.entry_realized_vol_lookback_hours,
            "entry_realized_vol_max": self.cfg.entry_realized_vol_max,
            "entry_realized_vol_current": rv24,
            "stop_loss_pct": self.cfg.stop_loss_pct,
            "basket_pnl_pct": basket_pnl_pct,
            "last_order_attempt_at": self._last_order_attempt_at,
            "last_order_status": self._last_order_status,
            "last_order_message": self._last_order_message,
            "last_order_group_id": self._last_order_group_id,
            "last_order_exchange_positions_checked": self._last_order_exchange_positions_checked,
            "last_order_exchange_positions_underlying": self._last_order_exchange_positions_underlying,
            "last_order_exchange_positions": self._last_order_exchange_positions,
            "last_order_exchange_positions_error": self._last_order_exchange_positions_error,
            "last_trade_week": self._last_trade_week,
            "open_positions": self.pos_mgr.open_position_count,
            "max_positions": self.cfg.max_positions,
            "positions": self.pos_mgr.summary(),
        }
