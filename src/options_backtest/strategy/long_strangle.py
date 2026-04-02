"""Long Strangle / Straddle strategy (双买策略).

Buy an OTM call + OTM put, either in daily 0DTE mode or multi-day hold mode.

Key properties:
- Loss is bounded by premium paid (no unlimited risk).
- Profits when underlying moves beyond breakeven in either direction.
- Take-profit during the day is *critical*: lock gamma profits before
  theta decay eats them back by expiry.
- Works best in high-volatility environments.

Parameters:
    otm_pct          : OTM distance (0.02 = 2%). Use 0 for nearest-ATM.
    take_profit_pct  : Close ALL legs when combined PnL reaches X% of entry cost.
                       E.g. 100 = combined value doubled.  0 = disabled.
    stop_loss_pct    : Close if combined value drops below X% of entry.
                       E.g. 80 = close if 80% of premium lost. 0 = disabled.
    entry_hour       : UTC hour to buy (8 = Deribit settlement time).
    roll_daily       : True for daily 0DTE mode; False for multi-day hold mode.
    entry_weekdays   : Optional list of weekdays allowed for entry (Mon=0 ... Sun=6).
    compound         : Scale quantity to equity.
    skip_weekends    : True = Mon-Fri only.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np

from options_backtest.strategy.base import BaseStrategy


class LongStrangleStrategy(BaseStrategy):
    """Buy an OTM call and OTM put (long strangle / straddle)."""

    name = "LongStrangle"

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(params)

        # Strike selection
        self.otm_pct: float = self.params.get("otm_pct", 0.02)

        # DTE window
        self.min_days_to_expiry: float = self.params.get("min_days_to_expiry", 0.0)
        self.max_days_to_expiry: float = self.params.get("max_days_to_expiry", 1.5)
        self.roll_days_before_expiry: float = self.params.get("roll_days_before_expiry", 0)

        # TP / SL on *combined* position (percentage of entry cost)
        self.take_profit_pct: float = self.params.get("take_profit_pct", 100)
        self.stop_loss_pct: float = self.params.get("stop_loss_pct", 0)  # 0 = disabled

        # Sizing
        self.quantity: float = self.params.get("quantity", 1.0)
        self.max_positions: int = self.params.get("max_positions", 1)
        self.compound: bool = self.params.get("compound", False)

        # Daily rolling
        self.roll_daily: bool = self.params.get("roll_daily", False)
        self.entry_hour: int = self.params.get("entry_hour", 8)
        self.entry_weekdays: list[int] = [
            int(d) for d in self.params.get("entry_weekdays", [])
            if 0 <= int(d) <= 6
        ]

        # Weekend filter
        self.skip_weekends: bool = self.params.get("skip_weekends", False)

        # Volatility filter: only buy when realized vol > threshold
        self.vol_lookback: int = self.params.get("vol_lookback", 0)
        self.vol_threshold: float = self.params.get("vol_threshold", 0.0)

        # Internal state
        self._last_trade_date: str = ""
        self._price_history: deque = deque(maxlen=max(self.vol_lookback + 1, 50))

        # Combined-pair tracking: date → total entry cost (call_entry + put_entry)
        # We monitor TP/SL on the combined value of all open legs.
        self._pair_entry_cost: float = 0.0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def on_step(self, context) -> None:
        positions = context.positions
        price = context.underlying_price
        self._price_history.append(price)

        current_date = context.current_time.strftime("%Y-%m-%d")
        current_hour = context.current_time.hour
        current_dow = context.current_time.weekday()
        is_new_day = current_date != self._last_trade_date

        # --- Combined TP / SL check on open long positions ---
        if positions and self._pair_entry_cost > 0:
            total_current = sum(
                p.current_mark_price * p.quantity
                for p in positions.values()
                if p.direction.value == "long"
            )
            combined_pnl_pct = (total_current - self._pair_entry_cost) / self._pair_entry_cost * 100

            # Take-profit
            if self.take_profit_pct > 0 and combined_pnl_pct >= self.take_profit_pct:
                self.log(
                    f"Combined TP: PnL={combined_pnl_pct:+.1f}% >= {self.take_profit_pct}% "
                    f"(value ${total_current:.2f} vs entry ${self._pair_entry_cost:.2f})"
                )
                for name in list(positions.keys()):
                    context.close(name)
                self._pair_entry_cost = 0.0
                return

            # Stop-loss (salvage remaining premium)
            if self.stop_loss_pct > 0 and combined_pnl_pct <= -self.stop_loss_pct:
                self.log(
                    f"Combined SL: PnL={combined_pnl_pct:+.1f}% <= -{self.stop_loss_pct}% "
                    f"(value ${total_current:.2f} vs entry ${self._pair_entry_cost:.2f})"
                )
                for name in list(positions.keys()):
                    context.close(name)
                self._pair_entry_cost = 0.0
                return

        if not positions:
            self._pair_entry_cost = 0.0

        # In multi-day mode, hold existing positions until TP/SL or expiry.
        if positions and not self.roll_daily:
            return

        # --- Entry logic ---
        if self.roll_daily:
            should_enter = is_new_day and current_hour == self.entry_hour
        else:
            should_enter = is_new_day and current_hour == self.entry_hour and not positions

        if not should_enter:
            return

        # --- Skip weekends ---
        if self.skip_weekends:
            if current_dow >= 5:
                self._last_trade_date = current_date
                return

        # --- Optional weekday filter ---
        if self.entry_weekdays and current_dow not in self.entry_weekdays:
            self._last_trade_date = current_date
            return

        if not self.roll_daily and positions:
            self._last_trade_date = current_date
            return

        # --- Volatility filter: only buy when vol is HIGH ---
        if self.vol_lookback > 0 and self.vol_threshold > 0:
            if not self._vol_high_enough():
                return

        # --- Close any remaining positions before re-entering ---
        for name in list(positions.keys()):
            context.close(name)
        self._pair_entry_cost = 0.0

        # --- Open new long strangle ---
        chain = context.option_chain
        if chain.empty:
            self._last_trade_date = current_date
            return

        self._open_long_strangle(context, chain)
        self._last_trade_date = current_date

    # ------------------------------------------------------------------
    # Open long strangle
    # ------------------------------------------------------------------

    def _open_long_strangle(self, context, chain) -> None:
        """Buy an OTM call + OTM put."""
        F = context.underlying_price

        dte_arr = chain["days_to_expiry"].values
        strike_arr = chain["strike_price"].values
        opt_type_arr = chain["option_type"].values
        names_arr = chain["instrument_name"].values
        exp_arr = chain["expiration_date"].values
        mark_arr = chain["mark_price"].values

        # Filter by DTE
        dte_mask = (dte_arr >= self.min_days_to_expiry) & (dte_arr <= self.max_days_to_expiry)
        elig_idx = np.flatnonzero(dte_mask)
        if len(elig_idx) == 0:
            return

        # Pick closest-to-target expiry
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
        exp_marks = mark_arr[exp_idx]

        # Split calls / puts (OTM)
        is_call = np.char.startswith(exp_types.astype(str), "c")
        is_put = np.char.startswith(exp_types.astype(str), "p")

        call_mask = is_call & (exp_strikes > F)
        put_mask = is_put & (exp_strikes < F)
        call_idx = np.flatnonzero(call_mask)
        put_idx = np.flatnonzero(put_mask)

        if len(call_idx) == 0 or len(put_idx) == 0:
            return

        # Pick strikes closest to target OTM offset
        target_offset = self.otm_pct

        if target_offset <= 0.001:
            # Near-ATM: pick nearest OTM strike
            call_dist = exp_strikes[call_idx] - F
            best_call_li = call_idx[np.argmin(call_dist)]
            put_dist = F - exp_strikes[put_idx]
            best_put_li = put_idx[np.argmin(put_dist)]
        else:
            call_offsets = (exp_strikes[call_idx] - F) / F
            call_dist = np.abs(call_offsets - target_offset)
            best_call_li = call_idx[np.argmin(call_dist)]

            put_offsets = (F - exp_strikes[put_idx]) / F
            put_dist = np.abs(put_offsets - target_offset)
            best_put_li = put_idx[np.argmin(put_dist)]

        call_name = str(exp_names[best_call_li])
        put_name = str(exp_names[best_put_li])
        call_strike = exp_strikes[best_call_li]
        put_strike = exp_strikes[best_put_li]
        call_dte = exp_dte[best_call_li]
        call_mark = exp_marks[best_call_li]
        put_mark = exp_marks[best_put_li]

        # Compound sizing
        if self.compound:
            equity = context.account.equity(
                sum(p.unrealized_pnl for p in context.positions.values())
            )
            buy_qty = equity * self.quantity / context.account.initial_balance
            buy_qty = max(buy_qty, 0.01)
        else:
            buy_qty = self.quantity

        context.buy(call_name, buy_qty)
        context.buy(put_name, buy_qty)

        # Track combined entry cost from actual fill prices (USD base)
        # Position.entry_price is in the same unit as current_mark_price
        self._pair_entry_cost = 0.0
        positions = context.positions
        for leg_name in [call_name, put_name]:
            if leg_name in positions:
                p = positions[leg_name]
                self._pair_entry_cost += p.entry_price * p.quantity

        self.log(
            f"Opening Long Strangle: "
            f"Call {call_name} (K={call_strike}), "
            f"Put {put_name} (K={put_strike}), "
            f"DTE={call_dte:.1f}, qty={buy_qty:.4f}, "
            f"otm={target_offset*100:.2f}%, "
            f"cost=${self._pair_entry_cost:.2f}"
        )

    # ------------------------------------------------------------------
    # Volatility filter
    # ------------------------------------------------------------------

    def _vol_high_enough(self) -> bool:
        """Return True if recent realized vol exceeds threshold.

        For a long strangle buyer, we WANT high vol (opposite of seller).
        """
        if len(self._price_history) < 2:
            return True
        prices = list(self._price_history)
        n = min(self.vol_lookback, len(prices) - 1)
        if n < 2:
            return True
        recent = prices[-n - 1:]
        log_returns = [
            math.log(recent[i + 1] / recent[i])
            for i in range(len(recent) - 1)
            if recent[i] > 0
        ]
        if len(log_returns) < 2:
            return True
        mean_r = sum(log_returns) / len(log_returns)
        var_r = sum((r - mean_r) ** 2 for r in log_returns) / (len(log_returns) - 1)
        hourly_vol = math.sqrt(var_r)
        annualized_vol = hourly_vol * math.sqrt(8760)
        return annualized_vol >= self.vol_threshold
