"""Position manager – 仓位管理.

Tracks open iron condor positions, maps exchange positions to
local trade records, and provides aggregated risk views.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from loguru import logger

from trader.binance_client import (
    OrderResult,
    OptionTicker,
)
from trader.limit_chaser import ChaserConfig, LegOrder, LimitChaser
from trader.storage import Storage


# ---------------------------------------------------------------------------
# Iron Condor position model
# ---------------------------------------------------------------------------

@dataclass
class CondorLeg:
    """Single leg of an iron condor."""
    symbol: str
    side: str              # "SELL" or "BUY"
    option_type: str       # "call" / "put"
    strike: float
    quantity: float
    entry_price: float
    trade_id: int          # DB row id in trades table
    order_id: str = ""


@dataclass
class IronCondorPosition:
    """Complete iron condor position (4 legs).

    Legs:
      - sell_put:  short OTM put  (collect premium)
      - buy_put:   long further OTM put  (protection)
      - sell_call: short OTM call (collect premium)
      - buy_call:  long further OTM call (protection)
    """
    group_id: str               # unique group identifier
    entry_time: datetime
    underlying_price: float     # spot at entry
    sell_put: CondorLeg | None = None
    buy_put: CondorLeg | None = None
    sell_call: CondorLeg | None = None
    buy_call: CondorLeg | None = None
    is_open: bool = True
    total_premium: float = 0.0  # net premium collected

    @property
    def legs(self) -> list[CondorLeg]:
        return [l for l in [self.sell_put, self.buy_put,
                            self.sell_call, self.buy_call] if l is not None]

    @property
    def max_profit(self) -> float:
        """Max profit = net premium collected."""
        return self.total_premium

    @property
    def max_loss(self) -> float:
        """Max loss = wing width - net premium (per unit).

        For an iron condor: max_loss = (short_strike - long_strike) - premium.
        """
        if self.sell_put and self.buy_put:
            put_width = self.sell_put.strike - self.buy_put.strike
        else:
            put_width = float("inf")

        if self.sell_call and self.buy_call:
            call_width = self.buy_call.strike - self.sell_call.strike
        else:
            call_width = float("inf")

        width = max(put_width, call_width)
        if width == float("inf"):
            return float("inf")
        return width - self.total_premium


# ---------------------------------------------------------------------------
# Position Manager
# ---------------------------------------------------------------------------

class PositionManager:
    """Manages iron condor positions with exchange integration.

    Responsibilities:
    - Open new iron condor positions (4-leg order)
    - Track all open positions
    - Monitor PnL and trigger TP/SL
    - Close positions (4-leg close)
    - Sync with exchange positions
    - Record all trades to persistent storage
    """

    def __init__(
        self,
        client: Any,
        storage: Storage,
        chaser_config: ChaserConfig | None = None,
    ):
        self.client = client
        self.storage = storage
        self.chaser = LimitChaser(client, chaser_config)
        self.open_condors: dict[str, IronCondorPosition] = {}  # group_id -> position
        self._load_open_positions()

    def _load_open_positions(self) -> None:
        """Recover open positions from storage on restart."""
        open_groups = self.storage.get_open_trade_groups()
        for group_id in open_groups:
            trades = self.storage.get_open_trades(group_id)
            if not trades:
                continue

            condor = IronCondorPosition(
                group_id=group_id,
                entry_time=datetime.fromisoformat(trades[0]["timestamp"]),
                underlying_price=0.0,
                is_open=True,
            )

            total_premium = 0.0
            for t in trades:
                import json
                meta = json.loads(t.get("meta", "{}"))
                leg = CondorLeg(
                    symbol=t["symbol"],
                    side=t["side"],
                    option_type=meta.get("option_type", ""),
                    strike=meta.get("strike", 0),
                    quantity=t["quantity"],
                    entry_price=t["price"],
                    trade_id=t["id"],
                    order_id=t.get("order_id", ""),
                )

                leg_role = meta.get("leg_role", "")
                if leg_role == "sell_put":
                    condor.sell_put = leg
                    total_premium += leg.entry_price * leg.quantity
                elif leg_role == "buy_put":
                    condor.buy_put = leg
                    total_premium -= leg.entry_price * leg.quantity
                elif leg_role == "sell_call":
                    condor.sell_call = leg
                    total_premium += leg.entry_price * leg.quantity
                elif leg_role == "buy_call":
                    condor.buy_call = leg
                    total_premium -= leg.entry_price * leg.quantity

                condor.underlying_price = meta.get("underlying_price",
                                                    condor.underlying_price)

            condor.total_premium = total_premium
            self.open_condors[group_id] = condor

        if self.open_condors:
            logger.info(f"Recovered {len(self.open_condors)} open condor position(s)")

    @staticmethod
    def _infer_underlying(symbol: str) -> str:
        return str(symbol).split("-", 1)[0].upper() if symbol else "ETH"

    # ------------------------------------------------------------------
    # Open a new short strangle (2-leg naked sell)
    # ------------------------------------------------------------------

    def open_short_strangle(
        self,
        sell_call_symbol: str,
        sell_put_symbol: str,
        sell_call_strike: float,
        sell_put_strike: float,
        quantity: float,
        underlying_price: float,
    ) -> IronCondorPosition | None:
        """Open a naked short strangle via limit-order chaser (2 legs).

        Returns the IronCondorPosition (with buy legs = None) if all legs
        fill, or None on failure.
        """
        group_id = f"SS_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        logger.info(
            f"Opening Short Strangle {group_id} (limit chaser): "
            f"sell_put={sell_put_strike} "
            f"sell_call={sell_call_strike} "
            f"qty={quantity} spot={underlying_price}"
        )

        leg_orders = [
            LegOrder(
                leg_role="sell_put", symbol=sell_put_symbol, side="SELL",
                quantity=quantity, strike=sell_put_strike, option_type="put",
                client_order_prefix=f"{group_id}:sell_put",
            ),
            LegOrder(
                leg_role="sell_call", symbol=sell_call_symbol, side="SELL",
                quantity=quantity, strike=sell_call_strike, option_type="call",
                client_order_prefix=f"{group_id}:sell_call",
            ),
        ]
        underlying = self._infer_underlying(sell_call_symbol or sell_put_symbol)

        try:
            results = self.chaser.execute_legs(leg_orders, underlying=underlying)
        except Exception as e:
            logger.error(f"Short Strangle {group_id}: execute_legs exception: {e}")
            return None

        failed = [r for r in results if r.status != "FILLED"]
        if failed:
            logger.error(
                f"Short Strangle {group_id}: {len(failed)} leg(s) failed to fill – "
                + ", ".join(f"{r.leg_role}={r.status}" for r in failed)
            )
            filled_results = [
                (r.leg_role, OrderResult(
                    order_id=r.order_id, symbol=r.symbol, side=r.side,
                    quantity=r.filled_qty, price=r.avg_price,
                    avg_price=r.avg_price, status="FILLED",
                    fee=r.fee, raw={},
                ))
                for r in results if r.status == "FILLED"
            ]
            self._rollback_legs(filled_results)
            return None

        condor = IronCondorPosition(
            group_id=group_id,
            entry_time=datetime.now(timezone.utc),
            underlying_price=underlying_price,
        )

        total_premium = 0.0
        for leg_order in results:
            leg_qty = leg_order.filled_qty or quantity
            meta = {
                "leg_role": leg_order.leg_role,
                "option_type": leg_order.option_type,
                "strike": leg_order.strike,
                "underlying_price": underlying_price,
                "group_id": group_id,
            }
            trade_id = self.storage.record_trade(
                trade_group=group_id,
                symbol=leg_order.symbol,
                side=leg_order.side,
                quantity=leg_qty,
                price=leg_order.avg_price,
                fee=leg_order.fee,
                order_id=leg_order.order_id,
                meta=meta,
            )

            leg = CondorLeg(
                symbol=leg_order.symbol,
                side=leg_order.side,
                option_type=leg_order.option_type,
                strike=leg_order.strike,
                quantity=leg_qty,
                entry_price=leg_order.avg_price,
                trade_id=trade_id,
                order_id=leg_order.order_id,
            )
            setattr(condor, leg_order.leg_role, leg)
            total_premium += leg_order.avg_price * leg_qty

        condor.total_premium = total_premium
        self.open_condors[group_id] = condor

        logger.info(
            f"Short Strangle {group_id} opened via limit chaser. "
            f"Net premium: {total_premium:.4f} USD"
        )

        return condor

    # ------------------------------------------------------------------
    # Open a new iron condor
    # ------------------------------------------------------------------

    def open_iron_condor(
        self,
        sell_call_symbol: str,
        buy_call_symbol: str,
        sell_put_symbol: str,
        buy_put_symbol: str,
        sell_call_strike: float,
        buy_call_strike: float,
        sell_put_strike: float,
        buy_put_strike: float,
        quantity: float,
        underlying_price: float,
        status_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> IronCondorPosition | None:
        """Open a full iron condor via limit-order chaser.

        All 4 legs are submitted as adaptive limit orders that drift
        toward market price over a 30-minute window.  If a leg cannot
        fill within the window it falls back to a market order.

        Returns the IronCondorPosition if all legs fill, or None on failure.
        """
        group_id = f"IC_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        logger.info(
            f"Opening Iron Condor {group_id} (limit chaser): "
            f"sell_put={sell_put_strike} buy_put={buy_put_strike} "
            f"sell_call={sell_call_strike} buy_call={buy_call_strike} "
            f"qty={quantity} spot={underlying_price}"
        )

        # Build LegOrder list – protection legs first, then short legs
        leg_orders = [
            LegOrder(
                leg_role="buy_put", symbol=buy_put_symbol, side="BUY",
                quantity=quantity, strike=buy_put_strike, option_type="put",
                client_order_prefix=f"{group_id}:buy_put",
            ),
            LegOrder(
                leg_role="buy_call", symbol=buy_call_symbol, side="BUY",
                quantity=quantity, strike=buy_call_strike, option_type="call",
                client_order_prefix=f"{group_id}:buy_call",
            ),
            LegOrder(
                leg_role="sell_put", symbol=sell_put_symbol, side="SELL",
                quantity=quantity, strike=sell_put_strike, option_type="put",
                client_order_prefix=f"{group_id}:sell_put",
            ),
            LegOrder(
                leg_role="sell_call", symbol=sell_call_symbol, side="SELL",
                quantity=quantity, strike=sell_call_strike, option_type="call",
                client_order_prefix=f"{group_id}:sell_call",
            ),
        ]
        underlying = self._infer_underlying(sell_call_symbol or sell_put_symbol)

        if status_callback is not None:
            status_callback({
                "event": "position_open_start",
                "message": f"开始提交 Iron Condor {group_id}",
                "group_id": group_id,
                "legs": [
                    {"leg_role": leg.leg_role, "symbol": leg.symbol, "side": leg.side, "quantity": leg.quantity, "status": leg.status}
                    for leg in leg_orders
                ],
                "total_legs": len(leg_orders),
            })

        # Execute via limit chaser (blocks up to window_seconds)
        try:
            results = self.chaser.execute_legs(
                leg_orders,
                underlying=underlying,
                status_callback=status_callback,
            )
        except Exception as e:
            logger.error(f"Iron Condor {group_id}: execute_legs exception: {e}")
            if status_callback is not None:
                status_callback({
                    "event": "position_open_error",
                    "message": f"追单异常: {e}",
                    "group_id": group_id,
                })
            return None

        # Check for failures
        failed = [r for r in results if r.status != "FILLED"]
        if failed:
            logger.error(
                f"Iron Condor {group_id}: {len(failed)} leg(s) failed to fill – "
                + ", ".join(f"{r.leg_role}={r.status}" for r in failed)
            )
            # Rollback filled legs
            filled_results = [
                (r.leg_role, OrderResult(
                    order_id=r.order_id, symbol=r.symbol, side=r.side,
                    quantity=r.filled_qty, price=r.avg_price,
                    avg_price=r.avg_price, status="FILLED",
                    fee=r.fee, raw={},
                ))
                for r in results if r.status == "FILLED"
            ]
            self._rollback_legs(filled_results)
            if status_callback is not None:
                status_callback({
                    "event": "position_open_failed",
                    "message": "至少有一条腿未成交，已触发回滚",
                    "group_id": group_id,
                    "failed_legs": [r.leg_role for r in failed],
                    "legs": [
                        {
                            "leg_role": r.leg_role,
                            "symbol": r.symbol,
                            "side": r.side,
                            "quantity": r.quantity,
                            "filled_qty": r.filled_qty,
                            "status": r.status,
                            "avg_price": r.avg_price,
                        }
                        for r in results
                    ],
                })
            return None

        # All filled – build condor position
        condor = IronCondorPosition(
            group_id=group_id,
            entry_time=datetime.now(timezone.utc),
            underlying_price=underlying_price,
        )

        total_premium = 0.0

        for leg_order in results:
            leg_qty = leg_order.filled_qty or quantity
            meta = {
                "leg_role": leg_order.leg_role,
                "option_type": leg_order.option_type,
                "strike": leg_order.strike,
                "underlying_price": underlying_price,
                "group_id": group_id,
            }
            trade_id = self.storage.record_trade(
                trade_group=group_id,
                symbol=leg_order.symbol,
                side=leg_order.side,
                quantity=leg_qty,
                price=leg_order.avg_price,
                fee=leg_order.fee,
                order_id=leg_order.order_id,
                meta=meta,
            )

            leg = CondorLeg(
                symbol=leg_order.symbol,
                side=leg_order.side,
                option_type=leg_order.option_type,
                strike=leg_order.strike,
                quantity=leg_qty,
                entry_price=leg_order.avg_price,
                trade_id=trade_id,
                order_id=leg_order.order_id,
            )
            setattr(condor, leg_order.leg_role, leg)

            if leg_order.side == "SELL":
                total_premium += leg_order.avg_price * leg_qty
            else:
                total_premium -= leg_order.avg_price * leg_qty

        condor.total_premium = total_premium
        self.open_condors[group_id] = condor

        logger.info(
            f"Iron Condor {group_id} opened via limit chaser. "
            f"Net premium: {total_premium:.4f} USD"
        )

        if status_callback is not None:
            status_callback({
                "event": "position_open_success",
                "message": f"Iron Condor {group_id} 已全部成交",
                "group_id": group_id,
                "net_premium": total_premium,
                "legs": [
                    {
                        "leg_role": r.leg_role,
                        "symbol": r.symbol,
                        "side": r.side,
                        "quantity": r.quantity,
                        "filled_qty": r.filled_qty,
                        "status": r.status,
                        "avg_price": r.avg_price,
                    }
                    for r in results
                ],
            })

        return condor

    def _rollback_legs(
        self,
        filled_legs: list[tuple[str, OrderResult]],
    ) -> None:
        """Attempt to close already-filled legs on partial failure."""
        logger.warning(f"Rolling back {len(filled_legs)} filled leg(s)")
        for leg_role, result in filled_legs:
            try:
                close_side = "SELL" if result.side == "BUY" else "BUY"
                self.client.place_order(
                    symbol=result.symbol,
                    side=close_side,
                    quantity=result.quantity,
                    order_type="MARKET",
                    reduce_only=True,
                )
                logger.info(f"Rolled back {leg_role}")
            except Exception as e:
                logger.error(f"Failed to rollback {leg_role}: {e}")

    # ------------------------------------------------------------------
    # Close an iron condor
    # ------------------------------------------------------------------

    def close_iron_condor(self, group_id: str, reason: str = "") -> float:
        """Close all legs of an iron condor via limit chaser.

        Returns total realized PnL.
        """
        condor = self.open_condors.get(group_id)
        if not condor or not condor.is_open:
            logger.warning(f"Condor {group_id} not found or already closed")
            return 0.0

        logger.info(f"Closing Iron Condor {group_id}, reason: {reason}")

        # Build close legs
        close_legs: list[LegOrder] = []
        for leg in condor.legs:
            close_side = "SELL" if leg.side == "BUY" else "BUY"
            close_legs.append(LegOrder(
                leg_role=f"close_{leg.option_type}_{leg.side.lower()}",
                symbol=leg.symbol,
                side=close_side,
                quantity=leg.quantity,
                strike=leg.strike,
                option_type=leg.option_type,
            ))

        # Execute close via limit chaser
        underlying = self._infer_underlying(condor.legs[0].symbol) if condor.legs else "ETH"
        results = self.chaser.execute_legs(close_legs, underlying=underlying)

        total_pnl = 0.0
        for leg, close_result in zip(condor.legs, results):
            if close_result.status != "FILLED":
                logger.error(
                    f"Failed to close {leg.symbol}: {close_result.status}"
                )
                continue

            # Compute PnL for this leg
            if leg.side == "SELL":
                leg_pnl = (leg.entry_price - close_result.avg_price) * leg.quantity
            else:
                leg_pnl = (close_result.avg_price - leg.entry_price) * leg.quantity

            leg_pnl -= close_result.fee
            total_pnl += leg_pnl

            # Update storage
            self.storage.close_trade(
                trade_id=leg.trade_id,
                close_price=close_result.avg_price,
                pnl=leg_pnl,
            )

            logger.info(
                f"  Closed {leg.side} {leg.option_type} K={leg.strike}: "
                f"entry={leg.entry_price:.4f} close={close_result.avg_price:.4f} "
                f"pnl={leg_pnl:.4f}"
            )

        condor.is_open = False
        del self.open_condors[group_id]

        logger.info(
            f"Iron Condor {group_id} closed. "
            f"Total PnL: {total_pnl:.4f} USD. Reason: {reason}"
        )

        return total_pnl

    def close_all(self, reason: str = "close_all") -> float:
        """Close all open iron condors."""
        total = 0.0
        for gid in list(self.open_condors.keys()):
            total += self.close_iron_condor(gid, reason=reason)
        return total

    # ------------------------------------------------------------------
    # Position monitoring
    # ------------------------------------------------------------------

    def get_unrealized_pnl(self, mark_prices: dict[str, float]) -> float:
        """Compute total unrealized PnL across all open condors.

        Parameters
        ----------
        mark_prices : dict of symbol -> current mark price.
        """
        total_upnl = 0.0
        for condor in self.open_condors.values():
            for leg in condor.legs:
                mark = mark_prices.get(leg.symbol, leg.entry_price)
                if leg.side == "SELL":
                    leg_upnl = (leg.entry_price - mark) * leg.quantity
                else:
                    leg_upnl = (mark - leg.entry_price) * leg.quantity
                total_upnl += leg_upnl
        return total_upnl

    @property
    def open_position_count(self) -> int:
        return len(self.open_condors)

    def summary(self, mark_prices: dict[str, float] | None = None) -> dict:
        """Return summary of current position state."""
        upnl = self.get_unrealized_pnl(mark_prices or {})
        return {
            "open_condors": self.open_position_count,
            "unrealized_pnl": upnl,
            "condors": {
                gid: {
                    "entry_time": c.entry_time.isoformat(),
                    "underlying_price": c.underlying_price,
                    "total_premium": c.total_premium,
                    "legs": len(c.legs),
                }
                for gid, c in self.open_condors.items()
            },
        }
