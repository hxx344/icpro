"""Long Call strategy.

Simple directional strategy: buy an OTM call option when no position exists,
hold until take‑profit, stop‑loss, or near‑expiry roll.
"""

from __future__ import annotations

from typing import Any

from options_backtest.strategy.base import BaseStrategy


class LongCallStrategy(BaseStrategy):
    """Buy a call option and hold to take‑profit / stop‑loss / near‑expiry."""

    name = "LongCall"

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(params)
        # Parameters (with defaults)
        self.target_delta: float = self.params.get("target_delta", 0.40)
        self.min_days_to_expiry: int = self.params.get("min_days_to_expiry", 14)
        self.max_days_to_expiry: int = self.params.get("max_days_to_expiry", 45)
        self.roll_days_before_expiry: int = self.params.get("roll_days_before_expiry", 3)
        self.take_profit_pct: float = self.params.get("take_profit_pct", 100)  # 100 % gain
        self.stop_loss_pct: float = self.params.get("stop_loss_pct", 50)      # 50 % loss
        self.quantity: float = self.params.get("quantity", 1.0)

    def on_step(self, context) -> None:
        positions = context.positions
        chain = context.option_chain

        if chain.empty:
            return

        # Filter calls only
        calls = chain[chain["option_type"].str.lower().str.startswith("c")].copy()
        if calls.empty:
            return

        # ----- Manage existing position -----
        for name, pos in list(positions.items()):
            if pos.direction.value != "long":
                continue

            # Check take‑profit
            if pos.entry_price > 0:
                gain_pct = (pos.current_mark_price - pos.entry_price) / pos.entry_price * 100
                if gain_pct >= self.take_profit_pct:
                    self.log(f"Take profit on {name} ({gain_pct:.1f}%)")
                    context.close(name)
                    continue

                # Check stop‑loss
                if gain_pct <= -self.stop_loss_pct:
                    self.log(f"Stop loss on {name} ({gain_pct:.1f}%)")
                    context.close(name)
                    continue

            # Check near‑expiry → close and re‑enter (roll)
            match = chain[chain["instrument_name"] == name]
            if not match.empty:
                dte = match.iloc[0].get("days_to_expiry", 999)
                if dte <= self.roll_days_before_expiry:
                    self.log(f"Rolling {name}, DTE={dte:.1f}")
                    context.close(name)
                    # Will open new in the same step below

        # ----- Open new position if none -----
        if len(positions) > 0:
            return  # only hold one position at a time

        # Select calls with appropriate DTE & closest delta to target
        candidates = calls[
            (calls["days_to_expiry"] >= self.min_days_to_expiry)
            & (calls["days_to_expiry"] <= self.max_days_to_expiry)
        ].copy()

        if candidates.empty:
            return

        # Pick the strike closest to target moneyness (approximation for delta)
        F = context.underlying_price
        # target_strike ≈ F × (1 + some OTM offset based on target delta)
        # For simplicity, pick the strike where (strike - F) / F is closest to a heuristic
        candidates["moneyness"] = (candidates["strike_price"] - F) / F
        # target moneyness for a ~0.40 delta call ≈ slightly OTM → moneyness ~ 0.02‑0.10
        target_moneyness = 0.05  # rough heuristic
        candidates["distance"] = (candidates["moneyness"] - target_moneyness).abs()
        best = candidates.sort_values("distance").iloc[0]

        instrument = best["instrument_name"]
        self.log(
            f"Opening Long Call: {instrument}, "
            f"strike={best['strike_price']}, DTE={best['days_to_expiry']:.1f}"
        )
        context.buy(instrument, self.quantity)
