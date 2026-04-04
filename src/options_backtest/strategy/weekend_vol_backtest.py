"""Backtest-only WeekendVol variants.

Keeps live trading logic untouched while allowing backtest-specific sizing
semantics such as fixed USD notional when `compound` is disabled.
"""

from __future__ import annotations

import math
from typing import Any

from options_backtest.strategy.weekend_vol import WeekendVolStrategy


class WeekendVolBacktestStrategy(WeekendVolStrategy):
    """Backtest-only WeekendVol strategy.

    Extra params:
    - `fixed_notional_usd`: if > 0 and `compound=False`, size each entry as
      `fixed_notional_usd * leverage / spot` instead of using a fixed BTC qty.
    """

    name = "WeekendVolBacktest"

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(params)
        self.fixed_notional_usd: float = float(self.params.get("fixed_notional_usd", 0.0) or 0.0)

    def _compute_quantity(self, context, spot: float) -> float:
        if not self.compound and self.fixed_notional_usd > 0 and spot > 0:
            raw_qty = self.fixed_notional_usd * self.leverage / spot
            step = self.quantity_step if self.quantity_step > 0 else 0.0
            if step > 0:
                raw_qty = math.floor(raw_qty / step) * step
            return max(raw_qty, 0.0)
        return super()._compute_quantity(context, spot)