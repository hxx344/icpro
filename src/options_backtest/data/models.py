"""Pydantic data models for Deribit options data."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OptionType(str, Enum):
    """Option type."""
    CALL = "call"
    PUT = "put"


class Direction(str, Enum):
    """Trade / position direction."""
    LONG = "long"
    SHORT = "short"


class Underlying(str, Enum):
    """Supported underlyings."""
    BTC = "BTC"
    ETH = "ETH"


# ---------------------------------------------------------------------------
# Market data models
# ---------------------------------------------------------------------------

class OptionInstrument(BaseModel):
    """Static information about an option contract."""

    instrument_name: str = Field(..., description="e.g. BTC-26MAR26-80000-C")
    underlying: Underlying
    strike_price: float
    option_type: OptionType
    expiration_date: datetime = Field(..., description="Expiry in UTC")
    creation_date: Optional[datetime] = None
    contract_size: float = 1.0
    tick_size: float = 0.0001
    is_active: bool = True

    @property
    def is_call(self) -> bool:
        return self.option_type == OptionType.CALL

    @property
    def is_put(self) -> bool:
        return self.option_type == OptionType.PUT


class OptionMarketData(BaseModel):
    """A single market‐data snapshot for one option contract at one point in time."""

    timestamp: datetime
    instrument_name: str
    underlying_price: float = Field(..., description="Index price in USD")
    mark_price: float = Field(..., description="Mark price in BTC/ETH")
    mark_iv: float = Field(0.0, description="Mark implied volatility (%)")

    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    last_price: Optional[float] = None
    volume_24h: float = 0.0
    open_interest: float = 0.0

    # Greeks (as reported by Deribit or computed locally)
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None


class UnderlyingOHLCV(BaseModel):
    """OHLCV bar for the underlying asset."""

    timestamp: datetime
    underlying: Underlying
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class SettlementRecord(BaseModel):
    """Settlement / delivery record for an expiry date."""

    expiration_date: datetime
    underlying: Underlying
    delivery_price: float = Field(..., description="Settlement price in USD")


# ---------------------------------------------------------------------------
# Engine models
# ---------------------------------------------------------------------------

class Position(BaseModel):
    """A single option position held by the engine."""

    instrument_name: str
    underlying: Underlying = Underlying.BTC
    strike_price: float = 0.0
    option_type: OptionType = OptionType.CALL
    expiration_date: datetime = Field(default_factory=datetime.utcnow)
    direction: Direction = Direction.LONG
    quantity: float = 0.0
    entry_price: float = 0.0  # avg entry price in BTC
    entry_time: Optional[datetime] = None
    current_mark_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    @property
    def direction_sign(self) -> int:
        """Return +1 for long, -1 for short."""
        return 1 if self.direction == Direction.LONG else -1

    def update_mark(self, mark_price: float) -> None:
        """Recompute unrealized PnL based on latest mark price."""
        self.current_mark_price = mark_price
        # PnL in BTC = (mark - entry) × qty × direction
        self.unrealized_pnl = (
            (mark_price - self.entry_price) * self.quantity * self.direction_sign
        )


class OrderRequest(BaseModel):
    """An order submitted by a strategy."""

    instrument_name: str
    direction: Direction
    quantity: float
    order_type: str = "market"  # "market" only in MVP
    limit_price: Optional[float] = None


class Fill(BaseModel):
    """A fill / execution report returned by the matcher."""

    timestamp: datetime
    instrument_name: str
    direction: Direction
    quantity: float
    fill_price: float  # price in BTC after slippage
    fee: float = 0.0  # fee in BTC
    underlying_price: float = 0.0  # index price at fill time
