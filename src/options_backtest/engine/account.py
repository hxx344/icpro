"""Account management – tracks balance, equity, and fees."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Account:
    """Simple coin‑denominated trading account (BTC / ETH / etc)."""

    initial_balance: float = 1.0
    balance: float = 0.0  # cash (realised)
    total_fee_paid: float = 0.0
    _equity_history: list[tuple] = field(default_factory=list, repr=False)

    def __post_init__(self):
        if self.balance == 0.0:
            self.balance = self.initial_balance

    # ------------------------------------------------------------------

    def deposit(self, amount: float) -> None:
        self.balance += amount

    def withdraw(self, amount: float) -> None:
        self.balance -= amount

    def pay_fee(self, fee: float) -> None:
        self.balance -= fee
        self.total_fee_paid += fee

    def equity(self, unrealized_pnl: float = 0.0) -> float:
        """Current equity = balance + unrealised PnL."""
        return self.balance + unrealized_pnl

    def record_equity(self, timestamp, unrealized_pnl: float = 0.0, underlying_price: float = 0.0) -> None:
        """Append a snapshot to the equity history.

        The tuple stored is:
        (timestamp, equity_coin, balance_coin, unrealized_pnl_coin, underlying_price_usd)
        """
        eq = self.equity(unrealized_pnl)
        self._equity_history.append((timestamp, eq, self.balance, unrealized_pnl, underlying_price))

    @property
    def equity_history(self) -> list[tuple]:
        return self._equity_history
