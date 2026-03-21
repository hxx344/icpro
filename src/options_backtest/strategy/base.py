"""Base strategy class – all strategies inherit from this."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from options_backtest.data.models import Fill
    from options_backtest.engine.backtest import StrategyContext


class BaseStrategy:
    """Abstract base for all option strategies.

    Sub‑classes **must** override :meth:`on_step`; all other callbacks
    are optional.
    """

    name: str = "BaseStrategy"

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params or {}

    # ------------------------------------------------------------------
    # Lifecycle callbacks
    # ------------------------------------------------------------------

    def initialize(self, context: "StrategyContext") -> None:
        """Called once before the first time step.

        Use this to read params, set internal state, etc.
        """

    def on_step(self, context: "StrategyContext") -> None:
        """Called every time step with the latest market snapshot.

        **Must** be implemented by sub‑classes.
        """
        raise NotImplementedError

    def on_fill(self, context: "StrategyContext", fill: "Fill") -> None:
        """Called after an order is filled."""

    def on_expiry(self, context: "StrategyContext", expired: list[str]) -> None:
        """Called when positions are settled at expiry."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def log(self, msg: str) -> None:
        logger.info(f"[{self.name}] {msg}")
