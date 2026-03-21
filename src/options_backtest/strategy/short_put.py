"""Short Put (卖看跌) strategy.

每天滚动卖出最近到期、最接近当前价格的看跌期权，不设止盈止损。
等价于 cash-secured put：假设持有等值 USD 保证金。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from options_backtest.strategy.base import BaseStrategy


class ShortPutStrategy(BaseStrategy):
    """每日滚动卖出最近到期 ATM 看跌期权（short put）。

    策略逻辑
    --------
    1. 每个交易日（整点判断是否跨日）检查是否持仓。
    2. 如果无持仓或跨日需要滚仓，则：
       a. 平掉所有已有 short put 仓位。
       b. 从 option chain 中筛选出**最近到期**的看跌期权。
       c. 在该到期日中，选择**行权价最接近当前标的价格**的合约。
       d. 卖出该合约。
    3. 到期自动由引擎交割，不做止盈止损。

    Notes
    -----
    - Deribit 是币本位保证金：卖 put 亏损时从 BTC 余额扣除。
    - Put intrinsic = max(0, K-S) / S (BTC 计价)，价格下跌时亏损。
    - 与 covered call 互补：CC 在上涨时亏损，SP 在下跌时亏损。
    """

    name = "ShortPut"

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(params)
        self.quantity: float = self.params.get("quantity", 1.0)
        self.min_dte: float = self.params.get("min_dte", 0.0)
        self.roll_daily: bool = self.params.get("roll_daily", True)
        self.compound: bool = self.params.get("compound", False)
        self._last_trade_date: str = ""

    # ------------------------------------------------------------------

    def on_step(self, context) -> None:
        current_date = context.current_time.strftime("%Y-%m-%d")
        is_new_day = current_date != self._last_trade_date

        if not is_new_day:
            return

        positions = context.positions

        if not self.roll_daily and len(positions) > 0:
            return

        chain = context.option_chain
        if chain.empty:
            return

        # ---------- 筛选最近到期的 ATM put ----------
        put_name = self._select_nearest_atm_put(chain, context.underlying_price)
        if put_name is None:
            return

        # 如果选出的合约与当前持仓相同，不动
        if put_name in positions:
            self._last_trade_date = current_date
            return

        # ---------- 平掉旧仓位 ----------
        for name, pos in list(positions.items()):
            if pos.direction.value == "short":
                self.log(f"平仓滚动: {name}")
                context.close(name)

        # ---------- 卖出新合约 ----------
        if self.compound:
            equity = context.account.equity(
                sum(p.unrealized_pnl for p in context.positions.values())
            )
            sell_qty = equity * self.quantity / context.account.initial_balance
            sell_qty = max(sell_qty, 0.01)
        else:
            sell_qty = self.quantity

        self.log(f"卖出 Put: {put_name} x{sell_qty:.4f}")
        context.sell(put_name, sell_qty)
        self._last_trade_date = current_date

    # ------------------------------------------------------------------

    def _select_nearest_atm_put(self, chain, underlying_price: float) -> str | None:
        """从 option chain 中选出最近到期、行权价最接近标的价的 put。"""
        opt_type_arr = chain["option_type"].values
        dte_arr = chain["days_to_expiry"].values.astype(float)
        strike_arr = chain["strike_price"].values.astype(float)
        names_arr = chain["instrument_name"].values

        # 1. 筛选 put
        is_put = np.char.startswith(opt_type_arr.astype(str), "p")
        put_idx = np.flatnonzero(is_put)
        if len(put_idx) == 0:
            return None

        # 2. 过滤 DTE
        dte_filtered = dte_arr[put_idx]
        valid_dte_mask = dte_filtered > self.min_dte
        put_idx = put_idx[valid_dte_mask]
        if len(put_idx) == 0:
            return None

        # 3. 最小 DTE（最近到期）
        dte_puts = dte_arr[put_idx]
        min_dte = np.min(dte_puts)

        # 4. 最近到期中最接近 ATM 的行权价
        nearest_exp_mask = np.abs(dte_puts - min_dte) < 0.001
        nearest_idx = put_idx[nearest_exp_mask]
        if len(nearest_idx) == 0:
            return None

        strikes = strike_arr[nearest_idx]
        dist = np.abs(strikes - underlying_price)
        best_local = np.argmin(dist)

        return str(names_arr[nearest_idx[best_local]])
