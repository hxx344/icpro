"""Order matching / execution simulator.

Uses top-of-book bid/ask fills when available.
Falls back to mark-price fill with fixed slippage when the book is unavailable.
Supports both coin-margined (Deribit) and USD-margined fee models.
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

        Prices are in coin (Deribit) or USD depending on margin mode.
        """
        require_touch_quote = bool(getattr(self.cfg, "require_touch_quote", False))

        if require_touch_quote:
            if order.direction == Direction.LONG:
                if ask_price is None or ask_price <= 0:
                    return None
                fill_price = ask_price
            else:
                if bid_price is None or bid_price <= 0:
                    return None
                fill_price = bid_price
        else:
            # Determine execution price.
            # When a valid top-of-book exists, use the touch price directly:
            #   - LONG  -> ask
            #   - SHORT -> bid
            # This is more conservative than midpoint execution and better matches
            # immediate marketable fills in the hourly snapshot backtest.
            if bid_price is not None and ask_price is not None and bid_price > 0 and ask_price > 0:
                if order.direction == Direction.LONG:
                    fill_price = ask_price
                else:
                    fill_price = bid_price
            else:
                mid = mark_price
                if mid <= 0:
                    return None

                # Apply slippage only when we have to fall back to mark pricing.
                if order.direction == Direction.LONG:
                    fill_price = mid + self.cfg.slippage
                else:
                    fill_price = max(mid - self.cfg.slippage, 0.0001)

        if fill_price <= 0:
            return None

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

        Deribit model (both margin modes):
          fee_per_contract = taker_fee × underlying_price  (coin: taker_fee alone)
          Capped at max_fee_pct × option_price.

        Coin margin: prices in BTC → fee in BTC.
        USD margin:  prices in USD → fee in USD.
        """
        if self._margin_usd:
            # Deribit-style fee expressed in USD:
            #   base = taker_fee (BTC per contract) × underlying_price (USD/BTC)
            fee_per_contract = self.cfg.taker_fee * underlying_price
            fee_per_contract = max(fee_per_contract, self.cfg.min_fee * underlying_price)
            # Cap: max_fee_pct of option premium (in USD)
            cap = price_btc * self.cfg.max_fee_pct
            if cap > 0:
                fee_per_contract = min(fee_per_contract, cap)
            return fee_per_contract * quantity

        # Deribit coin margin: 0.03 % of underlying value (in BTC, underlying = 1 BTC per contract)
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
