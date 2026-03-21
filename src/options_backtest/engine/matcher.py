"""Order matching / execution simulator.

MVP: mid‑price fill with fixed slippage.
Supports both coin-margined (Deribit) and USD-margined (Binance) fee models.
"""

from __future__ import annotations

from datetime import datetime

from options_backtest.config import ExecutionConfig
from options_backtest.data.models import Direction, Fill, OrderRequest


class Matcher:
    """Simulates order execution against historical quotes."""

    def __init__(self, config: ExecutionConfig | None = None, *, margin_usd: bool = False):
        self.cfg = config or ExecutionConfig()
        self._margin_usd = margin_usd

    def execute(
        self,
        order: OrderRequest,
        timestamp: datetime,
        bid_price: float | None,
        ask_price: float | None,
        mark_price: float,
        underlying_price: float,
    ) -> Fill | None:
        """Try to fill *order* and return a Fill, or ``None`` if unfillable.

        Prices are in coin (Deribit) or USD (Binance) depending on margin mode.
        """
        # Determine execution price
        if bid_price is not None and ask_price is not None and bid_price > 0 and ask_price > 0:
            mid = (bid_price + ask_price) / 2
        else:
            mid = mark_price

        if mid <= 0:
            return None

        # Apply slippage
        if order.direction == Direction.LONG:
            fill_price = mid + self.cfg.slippage
        else:
            fill_price = max(mid - self.cfg.slippage, 0.0001)

        # Compute fee (Deribit model)
        fee = self._compute_fee(fill_price, order.quantity, underlying_price)

        return Fill(
            timestamp=timestamp,
            instrument_name=order.instrument_name,
            direction=order.direction,
            quantity=order.quantity,
            fill_price=fill_price,
            fee=fee,
            underlying_price=underlying_price,
        )

    def _compute_fee(
        self, price_btc: float, quantity: float, underlying_price: float
    ) -> float:
        """Compute trading fee.

        Coin margin (Deribit): fixed fee per contract (0.03% of underlying), capped.
        USD margin (Binance): percentage of premium (taker_fee × premium × qty).
        """
        if self._margin_usd:
            # Binance: fee = taker_rate × premium × quantity
            fee = self.cfg.taker_fee * price_btc * quantity
            return max(fee, 0.0)

        # Deribit: 0.03 % of underlying value (in BTC, underlying = 1 BTC per contract)
        fee_per_contract = self.cfg.taker_fee  # 0.0003 BTC

        # Floor
        fee_per_contract = max(fee_per_contract, self.cfg.min_fee)

        # Cap: 12.5 % of option price
        cap = price_btc * self.cfg.max_fee_pct
        fee_per_contract = min(fee_per_contract, cap)

        return fee_per_contract * quantity

    def compute_delivery_fee(
        self, settlement_pnl_btc: float, quantity: float
    ) -> float:
        """Delivery fee for ITM options at expiry: 0.015 % of underlying."""
        if settlement_pnl_btc <= 0:
            return 0.0
        return self.cfg.delivery_fee * quantity
