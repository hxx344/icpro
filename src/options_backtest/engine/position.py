"""Position tracking and management."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from loguru import logger

from options_backtest.data.models import Direction, Fill, Position


class PositionManager:
    """Manages all open option positions."""

    def __init__(self):
        self._positions: dict[str, Position] = {}
        self._closed_trades: list[dict] = []

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def positions(self) -> dict[str, Position]:
        return self._positions

    @property
    def closed_trades(self) -> list[dict]:
        return self._closed_trades

    def get(self, instrument_name: str) -> Optional[Position]:
        return self._positions.get(instrument_name)

    def has_position(self, instrument_name: str) -> bool:
        return instrument_name in self._positions

    @property
    def total_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self._positions.values())

    # ------------------------------------------------------------------
    # Open / update position from a fill
    # ------------------------------------------------------------------

    def apply_fill(self, fill: Fill) -> None:
        """Apply a Fill event: open new or modify existing position."""
        name = fill.instrument_name
        existing = self._positions.get(name)

        if existing is None:
            # New position
            self._positions[name] = Position(
                instrument_name=name,
                direction=fill.direction,
                quantity=fill.quantity,
                entry_price=fill.fill_price,
                entry_time=fill.timestamp,
                current_mark_price=fill.fill_price,
            )
            logger.debug(f"Opened {fill.direction.value} {fill.quantity} × {name} @ {fill.fill_price:.6f}")
            return

        # Same direction → add to position (average price)
        if existing.direction == fill.direction:
            total_qty = existing.quantity + fill.quantity
            existing.entry_price = (
                (existing.entry_price * existing.quantity + fill.fill_price * fill.quantity)
                / total_qty
            )
            existing.quantity = total_qty
            logger.debug(f"Added {fill.quantity} to {name}, total={total_qty}")
            return

        # Opposite direction → reduce or close
        if fill.quantity >= existing.quantity:
            # Close (possibly partial remaining opens new opposite)
            close_qty = existing.quantity
            realised = (fill.fill_price - existing.entry_price) * close_qty * existing.direction_sign
            self._record_close(existing, fill, close_qty, realised)

            remainder = fill.quantity - close_qty
            del self._positions[name]

            if remainder > 0:
                # Open new opposite position with remainder
                self._positions[name] = Position(
                    instrument_name=name,
                    direction=fill.direction,
                    quantity=remainder,
                    entry_price=fill.fill_price,
                    entry_time=fill.timestamp,
                    current_mark_price=fill.fill_price,
                )
        else:
            # Partial close
            realised = (fill.fill_price - existing.entry_price) * fill.quantity * existing.direction_sign
            self._record_close(existing, fill, fill.quantity, realised)
            existing.quantity -= fill.quantity

    # ------------------------------------------------------------------
    # Mark‑to‑market
    # ------------------------------------------------------------------

    def update_marks(self, mark_prices: dict[str, float]) -> None:
        """Update mark prices and unrealised PnL for all positions."""
        for name, pos in self._positions.items():
            if name in mark_prices:
                pos.update_mark(mark_prices[name])

    # ------------------------------------------------------------------
    # Expiry settlement
    # ------------------------------------------------------------------

    def settle_expired(
        self,
        instrument_name: str,
        settlement_price_usd: float,
        strike_price: float,
        option_type: str,
        timestamp: datetime,
        delivery_fee_per_qty: float = 0.0,
        delivery_fee_max_pct: float = 0.0,
        *,
        margin_usd: bool = False,
    ) -> float:
        """Settle an expired position, returning realised PnL.

        Coin margin (Deribit):
          Call intrinsic = max(0, S − K) / S
          Put  intrinsic = max(0, K − S) / S
        USD margin (Binance):
          Call intrinsic = max(0, S − K)
          Put  intrinsic = max(0, K − S)

        where S = settlement_price_usd.
        """
        pos = self._positions.get(instrument_name)
        if pos is None:
            return 0.0

        S = settlement_price_usd
        K = strike_price

        if margin_usd:
            # USD margin: intrinsic in USD
            if option_type.lower().startswith("c"):
                intrinsic = max(0.0, S - K)
            else:
                intrinsic = max(0.0, K - S)
        else:
            # Coin margin: intrinsic in coin
            if option_type.lower().startswith("c"):
                intrinsic = max(0.0, (S - K) / S) if S > 0 else 0.0
            else:
                intrinsic = max(0.0, (K - S) / S) if S > 0 else 0.0

        is_itm = intrinsic > 0
        if is_itm:
            # Long receives intrinsic minus entry cost; short is opposite
            pnl = (intrinsic - pos.entry_price) * pos.quantity * pos.direction_sign
            fee = delivery_fee_per_qty * pos.quantity
            # Cap delivery fee at delivery_fee_max_pct of option value (intrinsic)
            if delivery_fee_max_pct > 0:
                cap = intrinsic * delivery_fee_max_pct * pos.quantity
                fee = min(fee, cap)
        else:
            # OTM: buyer loses premium, seller keeps premium
            pnl = -pos.entry_price * pos.quantity * pos.direction_sign
            fee = 0.0

        pnl -= fee

        self._record_close_settlement(pos, timestamp, intrinsic, pnl, fee)
        del self._positions[instrument_name]

        # Derive underlying symbol from instrument name (e.g. ETH-28MAR25-3200-C → ETH)
        underlying = instrument_name.split("-")[0] if "-" in instrument_name else "coin"
        logger.debug(
            f"Settled {instrument_name}: {'ITM' if is_itm else 'OTM'}, "
            f"intrinsic={intrinsic:.6f}, pnl={pnl:.6f} {underlying}"
        )
        return pnl

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _record_close(self, pos: Position, fill: Fill, qty: float, pnl: float) -> None:
        self._closed_trades.append({
            "instrument_name": pos.instrument_name,
            "direction": pos.direction.value,
            "quantity": qty,
            "entry_price": pos.entry_price,
            "exit_price": fill.fill_price,
            "entry_time": pos.entry_time,
            "exit_time": fill.timestamp,
            "pnl": pnl,
            "fee": fill.fee,
            "close_type": "trade",
        })
        underlying = pos.instrument_name.split("-")[0] if "-" in pos.instrument_name else "coin"
        logger.debug(f"Closed {qty} × {pos.instrument_name}, pnl={pnl:.6f} {underlying}")

    def _record_close_settlement(
        self, pos: Position, timestamp: datetime,
        settlement_value: float, pnl: float, fee: float,
    ) -> None:
        self._closed_trades.append({
            "instrument_name": pos.instrument_name,
            "direction": pos.direction.value,
            "quantity": pos.quantity,
            "entry_price": pos.entry_price,
            "exit_price": settlement_value,
            "entry_time": pos.entry_time,
            "exit_time": timestamp,
            "pnl": pnl,
            "fee": fee,
            "close_type": "settlement",
        })
