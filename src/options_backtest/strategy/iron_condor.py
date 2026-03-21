"""Iron Condor strategy.

Sell an OTM call spread and an OTM put spread (same expiry):
  - short OTM call + long farther OTM call
  - short OTM put  + long farther OTM put
"""

from __future__ import annotations

from typing import Any

from options_backtest.strategy.base import BaseStrategy


class IronCondorStrategy(BaseStrategy):
    """Sell a call spread and a put spread with the same expiry."""

    name = "IronCondor"

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(params)
        self.target_delta: float = self.params.get("target_delta", 0.25)
        self.min_days_to_expiry: int = self.params.get("min_days_to_expiry", 14)
        self.max_days_to_expiry: int = self.params.get("max_days_to_expiry", 45)
        self.roll_days_before_expiry: int = self.params.get("roll_days_before_expiry", 1)
        self.take_profit_pct: float = self.params.get("take_profit_pct", 90)
        self.stop_loss_pct: float = self.params.get("stop_loss_pct", 200)
        self.quantity: float = self.params.get("quantity", 1.0)
        self.max_positions: int = self.params.get("max_positions", 1)  # condor groups

        self.short_otm_pct: float = self.params.get("short_otm_pct", 0.10)
        self.long_otm_pct: float = self.params.get("long_otm_pct", 0.15)

    def on_step(self, context) -> None:
        positions = context.positions

        self._manage_existing(context)

        n_groups = len(positions) // 4
        if n_groups >= self.max_positions:
            return

        chain = context.option_chain
        if chain.empty:
            return

        self._open_condor(context, chain)

    def _manage_existing(self, context) -> None:
        positions = context.positions

        if not positions:
            return

        dtes = [context.get_instrument_dte(name) for name in positions]
        min_dte = min(dtes) if dtes else 999.0
        if min_dte <= self.roll_days_before_expiry:
            self.log(f"Rolling condor (near expiry), min DTE={min_dte:.1f}")
            context.close_all()
            return

        entry_base = sum(max(pos.entry_price, 0.0) * pos.quantity for pos in positions.values())
        if entry_base <= 0:
            return

        total_unrealized = sum(pos.unrealized_pnl for pos in positions.values())
        pnl_pct = total_unrealized / entry_base * 100.0

        if pnl_pct >= self.take_profit_pct:
            self.log(f"Take profit on condor basket ({pnl_pct:.1f}%)")
            context.close_all()
            return

        if pnl_pct <= -self.stop_loss_pct:
            self.log(f"Stop loss on condor basket ({pnl_pct:.1f}%)")
            context.close_all()
            return

    def _open_condor(self, context, chain) -> None:
        import numpy as np

        F = context.underlying_price

        dte_arr = chain["days_to_expiry"].values
        strike_arr = chain["strike_price"].values
        opt_type_arr = chain["option_type"].values
        names_arr = chain["instrument_name"].values
        exp_arr = chain["expiration_date"].values

        dte_mask = (dte_arr >= self.min_days_to_expiry) & (dte_arr <= self.max_days_to_expiry)
        elig_idx = np.flatnonzero(dte_mask)
        if len(elig_idx) == 0:
            return

        target_dte = (self.min_days_to_expiry + self.max_days_to_expiry) / 2
        dte_dist = np.abs(dte_arr[elig_idx] - target_dte)
        best_i = elig_idx[np.argmin(dte_dist)]
        best_expiry = exp_arr[best_i]

        exp_mask = dte_mask & (exp_arr == best_expiry)
        exp_idx = np.flatnonzero(exp_mask)
        if len(exp_idx) == 0:
            return

        exp_strikes = strike_arr[exp_idx]
        exp_types = opt_type_arr[exp_idx]
        exp_names = names_arr[exp_idx]
        exp_dte = dte_arr[exp_idx]

        is_call = np.char.startswith(exp_types.astype(str), "c")
        is_put = np.char.startswith(exp_types.astype(str), "p")

        call_idx = np.flatnonzero(is_call & (exp_strikes > F))
        put_idx = np.flatnonzero(is_put & (exp_strikes < F))
        if len(call_idx) < 2 or len(put_idx) < 2:
            return

        call_offsets = (exp_strikes[call_idx] - F) / F
        put_offsets = (F - exp_strikes[put_idx]) / F

        short_call_local = call_idx[np.argmin(np.abs(call_offsets - self.short_otm_pct))]
        short_put_local = put_idx[np.argmin(np.abs(put_offsets - self.short_otm_pct))]

        farther_calls = call_idx[exp_strikes[call_idx] > exp_strikes[short_call_local]]
        farther_puts = put_idx[exp_strikes[put_idx] < exp_strikes[short_put_local]]
        if len(farther_calls) == 0 or len(farther_puts) == 0:
            return

        long_call_offsets = (exp_strikes[farther_calls] - F) / F
        long_put_offsets = (F - exp_strikes[farther_puts]) / F

        long_call_local = farther_calls[np.argmin(np.abs(long_call_offsets - self.long_otm_pct))]
        long_put_local = farther_puts[np.argmin(np.abs(long_put_offsets - self.long_otm_pct))]

        short_call = str(exp_names[short_call_local])
        short_put = str(exp_names[short_put_local])
        long_call = str(exp_names[long_call_local])
        long_put = str(exp_names[long_put_local])

        legs = {short_call, short_put, long_call, long_put}
        if len(legs) < 4:
            return

        if any(leg in context.positions for leg in legs):
            return

        self.log(
            "Opening Iron Condor: "
            f"Short Call {short_call}, Long Call {long_call}, "
            f"Short Put {short_put}, Long Put {long_put}, "
            f"DTE={float(exp_dte[short_call_local]):.1f}"
        )

        context.sell(short_call, self.quantity)
        context.sell(short_put, self.quantity)
        context.buy(long_call, self.quantity)
        context.buy(long_put, self.quantity)
