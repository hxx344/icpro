"""Short Strangle strategy.

Sell OTM call + OTM put to collect premium. Manage position by
take‑profit, stop‑loss, and near‑expiry rolling.

Supports both long-dated (14-45 DTE) and 0DTE modes via parameters.

Drawdown-reduction features (all optional, disabled by default):
  - vol_filter:  skip trading when N-hour realized vol exceeds threshold
  - dd_scaling:  reduce position size when equity drawdown exceeds threshold
  - adaptive_otm: widen OTM distance when vol is elevated
  - intraday_sl:  close all if intraday unrealized loss exceeds X% of equity
  - equity_ma:    only trade when equity > N-day moving average of equity
  - progressive_dd: linearly scale down position as drawdown deepens
  - single_leg:  sell only call/put when trend detected (instead of both)
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from options_backtest.strategy.base import BaseStrategy


class ShortStrangleStrategy(BaseStrategy):
    """Sell an OTM call and OTM put (strangle)."""

    name = "ShortStrangle"

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(params)
        self.target_delta: float = self.params.get("target_delta", 0.25)
        self.min_days_to_expiry: float = self.params.get("min_days_to_expiry", 14)
        self.max_days_to_expiry: float = self.params.get("max_days_to_expiry", 45)
        self.roll_days_before_expiry: float = self.params.get("roll_days_before_expiry", 3)
        self.take_profit_pct: float = self.params.get("take_profit_pct", 50)
        self.stop_loss_pct: float = self.params.get("stop_loss_pct", 200)
        self.quantity: float = self.params.get("quantity", 1.0)
        self.max_positions: int = self.params.get("max_positions", 2)  # pairs

        # OTM offset: fraction of underlying price (0.05 = 5% OTM)
        self.otm_pct: float = self.params.get("otm_pct", 0.10)

        # Daily rolling mode (for 0DTE)
        self.roll_daily: bool = self.params.get("roll_daily", False)
        self.entry_hour: int = self.params.get("entry_hour", 8)
        self.compound: bool = self.params.get("compound", False)

        # --- Drawdown-reduction parameters ---
        # Volatility filter: skip if N-hour realized vol (annualized) > threshold
        self.vol_lookback: int = self.params.get("vol_lookback", 0)  # hours; 0 = disabled
        self.vol_threshold: float = self.params.get("vol_threshold", 999.0)  # annualized vol

        # Drawdown-based position scaling (step function)
        self.dd_reduce_threshold: float = self.params.get("dd_reduce_threshold", 0.0)  # 0 = disabled
        self.dd_reduce_factor: float = self.params.get("dd_reduce_factor", 0.5)

        # Progressive drawdown scaling (linear ramp: 1.0 at dd_start → dd_min_scale at dd_full)
        self.dd_start: float = self.params.get("dd_start", 0.0)  # 0 = disabled
        self.dd_full: float = self.params.get("dd_full", 0.25)
        self.dd_min_scale: float = self.params.get("dd_min_scale", 0.2)

        # Adaptive OTM: otm = max(otm_pct, recent_vol * vol_otm_mult)
        self.adaptive_otm: bool = self.params.get("adaptive_otm", False)
        self.vol_otm_mult: float = self.params.get("vol_otm_mult", 1.0)

        # Cooldown: skip N days after a loss day
        self.cooldown_days: int = self.params.get("cooldown_days", 0)

        # Intraday stop-loss: close all if day's unrealized loss > X% of equity at day start
        self.intraday_sl_pct: float = self.params.get("intraday_sl_pct", 0.0)  # 0 = disabled

        # Equity curve filter: only trade when equity > N-day SMA of equity
        self.equity_ma_days: int = self.params.get("equity_ma_days", 0)  # 0 = disabled

        # Protective wings (iron condor): buy further OTM options to cap loss
        self.hedge_otm_pct: float = self.params.get("hedge_otm_pct", 0.0)  # 0 = disabled

        # IV-conditional wing skip: skip protective wings when ATM IV < threshold (%)
        # e.g. skip_wings_below_iv: 40 means skip buying wings when ATM IV < 40%
        self.skip_wings_below_iv: float = self.params.get("skip_wings_below_iv", 0.0)  # 0 = disabled

        # Single-leg mode: sell only put (or call) when N-hour trend exceeds threshold
        self.trend_filter: bool = self.params.get("trend_filter", False)
        self.trend_lookback: int = self.params.get("trend_lookback", 24)  # hours
        self.trend_threshold: float = self.params.get("trend_threshold", 0.03)  # 3% move = trending

        # Skip weekends: do not open new positions on Saturday/Sunday (UTC)
        self.skip_weekends: bool = self.params.get("skip_weekends", False)

        # Wait-for-midpoint: delay entry until spot is near strike midpoint
        self.wait_for_midpoint: bool = self.params.get("wait_for_midpoint", False)
        self.midpoint_tolerance_pct: float = self.params.get("midpoint_tolerance_pct", 0.10)  # 10% of gap
        self.midpoint_max_wait_hours: int = self.params.get("midpoint_max_wait_hours", 2)

        self._last_trade_date: str = ""

        # Track entry premium to compute TP / SL
        self._pair_entries: dict[str, float] = {}

        # Internal state for vol / dd tracking
        self._price_history: deque = deque(maxlen=max(self.vol_lookback + 1, self.trend_lookback + 1, 50))
        self._equity_peak: float = 0.0
        self._last_loss_date: str = ""
        self._prev_day_equity: float = 0.0
        self._prev_date: str = ""

        # Intraday SL state
        self._day_start_equity: float = 0.0
        self._intraday_stopped: bool = False

        # Equity MA tracking
        self._daily_equity_history: list[float] = []

    def on_step(self, context) -> None:
        positions = context.positions
        price = context.underlying_price

        # Track price history for vol computation
        self._price_history.append(price)

        # Track equity peak for drawdown scaling
        equity = context.account.equity(
            sum(p.unrealized_pnl for p in positions.values())
        )
        if equity > self._equity_peak:
            self._equity_peak = equity

        # ----- Daily rolling mode (0DTE style) -----
        if self.roll_daily:
            current_date = context.current_time.strftime("%Y-%m-%d")
            current_hour = context.current_time.hour
            is_new_day = current_date != self._last_trade_date

            # --- New day bookkeeping ---
            if current_date != self._prev_date:
                # Record previous day equity for cooldown
                if self._prev_date and self._prev_day_equity > 0:
                    if equity < self._prev_day_equity:
                        self._last_loss_date = self._prev_date
                self._prev_day_equity = equity
                self._prev_date = current_date
                # Record daily equity for equity MA
                self._daily_equity_history.append(equity)
                # Reset intraday SL state
                self._day_start_equity = equity
                self._intraday_stopped = False

            # --- Intraday stop-loss: close all if unrealised loss > threshold ---
            if self.intraday_sl_pct > 0 and not self._intraday_stopped:
                if self._day_start_equity > 0:
                    day_loss = (self._day_start_equity - equity) / self._day_start_equity
                    if day_loss >= self.intraday_sl_pct:
                        for name in list(positions.keys()):
                            context.close(name)
                        self._intraday_stopped = True
                        return

            # If already stopped for the day, do nothing
            if self._intraday_stopped:
                return

            # Manage TP / SL even in roll_daily mode
            self._manage_existing(context)

            # --- Skip weekends (Saturday=5, Sunday=6 in UTC) ---
            if self.skip_weekends:
                dow = context.current_time.weekday()
                if dow >= 5:  # Saturday or Sunday
                    return

            # --- Entry window logic ---
            if self.wait_for_midpoint:
                # Allow entry from entry_hour to entry_hour + max_wait_hours
                if current_hour < self.entry_hour or current_hour > self.entry_hour + self.midpoint_max_wait_hours:
                    return
                if not is_new_day and self._last_trade_date == current_date:
                    return  # already traded today
            else:
                # Original: only open at entry_hour on a new day
                if not is_new_day or current_hour != self.entry_hour:
                    return

            # --- Cooldown check ---
            if self.cooldown_days > 0 and self._last_loss_date:
                from datetime import datetime
                loss_dt = datetime.strptime(self._last_loss_date, "%Y-%m-%d")
                curr_dt = datetime.strptime(current_date, "%Y-%m-%d")
                if (curr_dt - loss_dt).days <= self.cooldown_days:
                    self._last_trade_date = current_date
                    return

            # --- Volatility filter ---
            if self.vol_lookback > 0 and not self._vol_ok():
                for name in list(positions.keys()):
                    context.close(name)
                self._last_trade_date = current_date
                return

            # --- Equity curve MA filter ---
            if self.equity_ma_days > 0 and not self._equity_above_ma():
                for name in list(positions.keys()):
                    context.close(name)
                self._last_trade_date = current_date
                return

            # Close all remaining legs before re-entering
            for name in list(positions.keys()):
                context.close(name)

            chain = context.option_chain
            if chain.empty:
                return

            # --- Wait-for-midpoint check ---
            if self.wait_for_midpoint and current_hour < self.entry_hour + self.midpoint_max_wait_hours:
                # Not yet at forced-entry time, check midpoint condition
                import numpy as np
                F = context.underlying_price
                all_strikes = np.unique(chain["strike_price"].values)
                all_strikes.sort()
                lower = all_strikes[all_strikes <= F]
                upper = all_strikes[all_strikes > F]
                if len(lower) > 0 and len(upper) > 0:
                    k_low, k_high = lower[-1], upper[0]
                    midpoint = (k_low + k_high) / 2.0
                    gap = k_high - k_low
                    tolerance = gap * self.midpoint_tolerance_pct
                    if abs(F - midpoint) > tolerance:
                        self.log(
                            f"Midpoint wait: spot={F:.2f} mid={midpoint:.2f} "
                            f"(K={k_low:.0f}/{k_high:.0f}) delta={abs(F-midpoint):.2f} "
                            f"> tol={tolerance:.2f} – skip"
                        )
                        return
                    self.log(f"Midpoint OK: spot={F:.2f} ≈ midpoint={midpoint:.2f}")
            elif self.wait_for_midpoint:
                self.log(f"Midpoint timeout at hour {current_hour}, forcing entry")

            # --- Trend filter: decide which legs to sell ---
            legs = "both"  # default: sell both call + put
            if self.trend_filter:
                legs = self._get_trend_legs()

            self._open_strangle(context, chain, legs=legs)
            self._last_trade_date = current_date
            return

        # ----- Original mode: manage existing positions -----
        self._manage_existing(context)

        # ----- Open new strangle if slot available -----
        # Count pairs: each strangle = 2 legs
        n_pairs = len(positions) // 2
        if n_pairs >= self.max_positions:
            return

        chain = context.option_chain
        if chain.empty:
            return
        self._open_strangle(context, chain)

    # ------------------------------------------------------------------

    def _vol_ok(self) -> bool:
        """Return True if recent realized vol is below threshold."""
        if len(self._price_history) < 2:
            return True
        prices = list(self._price_history)
        n = min(self.vol_lookback, len(prices) - 1)
        if n < 2:
            return True
        recent = prices[-n - 1:]
        log_returns = [math.log(recent[i + 1] / recent[i]) for i in range(len(recent) - 1) if recent[i] > 0]
        if len(log_returns) < 2:
            return True
        mean_r = sum(log_returns) / len(log_returns)
        var_r = sum((r - mean_r) ** 2 for r in log_returns) / (len(log_returns) - 1)
        hourly_vol = math.sqrt(var_r)
        annualized_vol = hourly_vol * math.sqrt(8760)
        return annualized_vol <= self.vol_threshold

    def _get_effective_otm(self) -> float:
        """Return OTM pct, optionally widened based on recent vol."""
        if not self.adaptive_otm or len(self._price_history) < 2:
            return self.otm_pct
        # Compute recent hourly vol
        prices = list(self._price_history)
        n = min(self.vol_lookback if self.vol_lookback > 0 else 24, len(prices) - 1)
        if n < 2:
            return self.otm_pct
        recent = prices[-n - 1:]
        log_returns = [math.log(recent[i + 1] / recent[i]) for i in range(len(recent) - 1) if recent[i] > 0]
        if len(log_returns) < 2:
            return self.otm_pct
        mean_r = sum(log_returns) / len(log_returns)
        var_r = sum((r - mean_r) ** 2 for r in log_returns) / (len(log_returns) - 1)
        daily_vol = math.sqrt(var_r) * math.sqrt(24)  # ~daily vol
        # OTM = max(base_otm, daily_vol * mult)
        adaptive = daily_vol * self.vol_otm_mult
        return max(self.otm_pct, adaptive)

    def _get_dd_scale(self, equity: float) -> float:
        """Return position scale factor based on drawdown.

        Supports both step-function (dd_reduce_threshold) and progressive (dd_start).
        """
        # Progressive mode takes priority
        if self.dd_start > 0 and self._equity_peak > 0:
            dd = (self._equity_peak - equity) / self._equity_peak
            if dd <= self.dd_start:
                return 1.0
            if dd >= self.dd_full:
                return self.dd_min_scale
            # Linear interpolation
            t = (dd - self.dd_start) / (self.dd_full - self.dd_start)
            return 1.0 - t * (1.0 - self.dd_min_scale)

        # Step-function mode
        if self.dd_reduce_threshold <= 0 or self._equity_peak <= 0:
            return 1.0
        dd = (self._equity_peak - equity) / self._equity_peak
        if dd >= self.dd_reduce_threshold:
            return self.dd_reduce_factor
        return 1.0

    def _equity_above_ma(self) -> bool:
        """Return True if current equity > N-day SMA of equity."""
        n = self.equity_ma_days
        if n <= 0 or len(self._daily_equity_history) < n:
            return True  # not enough data, allow trading
        recent = self._daily_equity_history[-n:]
        ma = sum(recent) / len(recent)
        return self._daily_equity_history[-1] >= ma

    def _get_trend_legs(self) -> str:
        """Decide which legs to sell based on recent price trend.

        Returns: 'both', 'put_only', 'call_only'
        """
        n = self.trend_lookback
        if len(self._price_history) < n + 1:
            return "both"
        prices = list(self._price_history)
        old_price = prices[-n - 1]
        new_price = prices[-1]
        if old_price <= 0:
            return "both"
        change = (new_price - old_price) / old_price
        if change > self.trend_threshold:
            return "put_only"   # uptrend → only sell put (safe side)
        if change < -self.trend_threshold:
            return "call_only"  # downtrend → only sell call (safe side)
        return "both"

    def _compute_atm_iv(self, chain, underlying_price: float) -> float:
        """Compute ATM implied volatility from option chain mark prices.

        Picks the nearest-to-ATM call, uses Black-76 IV solver to back out IV.
        Returns IV as percentage (e.g. 60.0 for 60%).  Returns 999.0 on failure.
        """
        import numpy as np

        try:
            from options_backtest.pricing.iv_solver import implied_volatility_btc
        except ImportError:
            return 999.0

        F = underlying_price
        strike_arr = chain["strike_price"].values
        opt_type_arr = chain["option_type"].values
        mark_arr = chain["mark_price"].values
        dte_arr = chain["days_to_expiry"].values

        # Use the same DTE filter as _open_strangle
        dte_mask = (dte_arr >= self.min_days_to_expiry) & (dte_arr <= self.max_days_to_expiry)
        is_call = np.char.startswith(opt_type_arr.astype(str), "c")
        atm_mask = dte_mask & is_call & (mark_arr > 0)
        idx = np.flatnonzero(atm_mask)
        if len(idx) == 0:
            return 999.0

        # Pick the call closest to ATM
        dists = np.abs(strike_arr[idx] - F)
        best = idx[np.argmin(dists)]

        K = float(strike_arr[best])
        T = float(dte_arr[best]) / 365.0
        mark_coin = float(mark_arr[best])

        if T <= 1e-8 or mark_coin <= 0 or K <= 0:
            return 999.0

        try:
            iv = implied_volatility_btc(mark_coin, F, K, T, option_type="call", r=0.0)
            if np.isfinite(iv) and iv > 0:
                return iv * 100.0  # decimal → percentage
        except Exception:
            pass
        return 999.0

    def _manage_existing(self, context) -> None:
        """Check take‑profit, stop‑loss, near‑expiry for open positions.

        Uses context.get_instrument_dte() instead of building the full
        option chain – this is the most impactful optimisation since
        _manage_existing is called every timestep.
        """
        positions = context.positions

        for name, pos in list(positions.items()):
            if pos.direction.value != "short":
                continue

            # Near‑expiry roll — use fast DTE lookup (no chain needed)
            dte = context.get_instrument_dte(name)
            if dte <= self.roll_days_before_expiry:
                self.log(f"Rolling {name}, DTE={dte:.1f}")
                context.close(name)
                continue

            # Take‑profit / stop‑loss on individual leg
            if pos.entry_price > 0:
                # For a short position: profit when mark < entry, loss when mark > entry
                pnl_pct = (pos.entry_price - pos.current_mark_price) / pos.entry_price * 100
                if pnl_pct >= self.take_profit_pct:
                    self.log(f"TP on {name} ({pnl_pct:.1f}%)")
                    context.close(name)
                    continue
                if pnl_pct <= -self.stop_loss_pct:
                    self.log(f"SL on {name} ({pnl_pct:.1f}%)")
                    context.close(name)

    def _open_strangle(self, context, chain, legs: str = "both") -> None:
        """Select and sell an OTM call + OTM put (or single leg).

        Args:
            legs: 'both' (default), 'call_only', or 'put_only'

        Uses numpy array operations directly to minimise pandas overhead.
        """
        import numpy as np

        F = context.underlying_price

        # --- Fast numpy filtering (avoid pandas boolean indexing) ---
        dte_arr = chain["days_to_expiry"].values
        strike_arr = chain["strike_price"].values
        opt_type_arr = chain["option_type"].values
        names_arr = chain["instrument_name"].values
        exp_arr = chain["expiration_date"].values

        # Filter by DTE
        dte_mask = (dte_arr >= self.min_days_to_expiry) & (dte_arr <= self.max_days_to_expiry)
        elig_idx = np.flatnonzero(dte_mask)
        if len(elig_idx) == 0:
            return

        # Pick the closest-to-target expiry
        target_dte = (self.min_days_to_expiry + self.max_days_to_expiry) / 2
        dte_dist = np.abs(dte_arr[elig_idx] - target_dte)
        best_i = elig_idx[np.argmin(dte_dist)]
        best_expiry = exp_arr[best_i]

        # Narrow to that expiry
        exp_mask = dte_mask & (exp_arr == best_expiry)
        exp_idx = np.flatnonzero(exp_mask)
        if len(exp_idx) == 0:
            return

        exp_strikes = strike_arr[exp_idx]
        exp_types = opt_type_arr[exp_idx]
        exp_names = names_arr[exp_idx]
        exp_dte = dte_arr[exp_idx]

        # Split calls / puts
        is_call = np.char.startswith(exp_types.astype(str), 'c')
        is_put = np.char.startswith(exp_types.astype(str), 'p')

        call_mask = is_call & (exp_strikes > F)
        put_mask = is_put & (exp_strikes < F)
        call_local_idx = np.flatnonzero(call_mask)
        put_local_idx = np.flatnonzero(put_mask)

        if len(call_local_idx) == 0 or len(put_local_idx) == 0:
            return

        # Pick nearest to target offset
        target_offset = self._get_effective_otm()
        call_offsets = (exp_strikes[call_local_idx] - F) / F
        call_dist = np.abs(call_offsets - target_offset)
        best_call_li = call_local_idx[np.argmin(call_dist)]

        put_offsets = (F - exp_strikes[put_local_idx]) / F
        put_dist = np.abs(put_offsets - target_offset)
        best_put_li = put_local_idx[np.argmin(put_dist)]

        call_name = str(exp_names[best_call_li])
        put_name = str(exp_names[best_put_li])

        # Skip instruments that already have open positions
        if call_name in context.positions or put_name in context.positions:
            return

        call_strike = exp_strikes[best_call_li]
        put_strike = exp_strikes[best_put_li]
        call_dte = exp_dte[best_call_li]

        # Compound sizing
        if self.compound:
            equity = context.account.equity(
                sum(p.unrealized_pnl for p in context.positions.values())
            )
            sell_qty = equity * self.quantity / context.account.initial_balance
            # Apply drawdown scaling
            dd_scale = self._get_dd_scale(equity)
            sell_qty = sell_qty * dd_scale
            sell_qty = max(sell_qty, 0.01)
        else:
            sell_qty = self.quantity

        self.log(
            f"Opening Short Strangle: "
            f"Call {call_name} (K={call_strike}), "
            f"Put {put_name} (K={put_strike}), "
            f"DTE={call_dte:.1f}, qty={sell_qty:.4f}, "
            f"otm={target_offset*100:.2f}%, legs={legs}"
        )

        if legs in ("both", "call_only"):
            context.sell(call_name, sell_qty)
        if legs in ("both", "put_only"):
            context.sell(put_name, sell_qty)

        # --- Protective wings (iron condor) ---
        if self.hedge_otm_pct > 0:
            # IV-conditional wing skip: if ATM IV < threshold, skip wings
            buy_wings = True
            if self.skip_wings_below_iv > 0:
                atm_iv = self._compute_atm_iv(chain, F)
                if atm_iv < self.skip_wings_below_iv:
                    buy_wings = False
                    self.log(f"  Skip wings: ATM IV={atm_iv:.1f}% < threshold {self.skip_wings_below_iv:.0f}%")
                else:
                    self.log(f"  Wings ON: ATM IV={atm_iv:.1f}% >= threshold {self.skip_wings_below_iv:.0f}%")

            if buy_wings:
                hedge_offset = self.hedge_otm_pct
                # Long call: further OTM than the short call
                if legs in ("both", "call_only"):
                    farther_calls = call_local_idx[exp_strikes[call_local_idx] > call_strike]
                    if len(farther_calls) > 0:
                        fc_offsets = (exp_strikes[farther_calls] - F) / F
                        fc_dist = np.abs(fc_offsets - hedge_offset)
                        lc_li = farther_calls[np.argmin(fc_dist)]
                        long_call_name = str(exp_names[lc_li])
                        if long_call_name not in context.positions:
                            context.buy(long_call_name, sell_qty)
                            self.log(f"  Hedge: Buy Call {long_call_name} (K={exp_strikes[lc_li]})")
                # Long put: further OTM than the short put
                if legs in ("both", "put_only"):
                    farther_puts = put_local_idx[exp_strikes[put_local_idx] < put_strike]
                    if len(farther_puts) > 0:
                        fp_offsets = (F - exp_strikes[farther_puts]) / F
                        fp_dist = np.abs(fp_offsets - hedge_offset)
                        lp_li = farther_puts[np.argmin(fp_dist)]
                        long_put_name = str(exp_names[lp_li])
                        if long_put_name not in context.positions:
                            context.buy(long_put_name, sell_qty)
                            self.log(f"  Hedge: Buy Put {long_put_name} (K={exp_strikes[lp_li]})")
