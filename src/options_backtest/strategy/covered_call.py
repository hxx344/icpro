"""Covered Call (备兑看涨) strategy.

每天滚动卖出最近到期、最接近当前价格的看涨期权，不设止盈止损。
假设持有等量现货作为备兑（covered），到期自动交割或滚仓。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from options_backtest.strategy.base import BaseStrategy


class CoveredCallStrategy(BaseStrategy):
    """每日滚动卖出最近到期 ATM 看涨期权（covered call）。

    策略逻辑
    --------
    1. 每个交易日（整点判断是否跨日）检查是否持仓。
    2. 如果无持仓或跨日需要滚仓，则：
       a. 平掉所有已有 short call 仓位。
       b. 从 option chain 中筛选出**最近到期**的看涨期权。
       c. 在该到期日中，选择**行权价最接近当前标的价格**的合约。
       d. 卖出该合约。
    3. 到期自动由引擎交割，不做止盈止损。
    """

    name = "CoveredCall"

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(params)
        self.quantity: float = self.params.get("quantity", 1.0)
        # 最小 DTE 过滤：避免卖出当天即将到期（DTE < min）的合约
        self.min_dte: float = self.params.get("min_dte", 0.0)
        # 是否每天滚仓（True=每天平旧开新，False=到期才换）
        self.roll_daily: bool = self.params.get("roll_daily", True)
        # 复利模式：卖出数量按帐户权益动态调整
        self.compound: bool = self.params.get("compound", False)
        # 内部状态：上一次开仓日期（用于判断跨日）
        self._last_trade_date: str = ""

    # ------------------------------------------------------------------

    def on_step(self, context) -> None:
        # 只在每天第一个 bar 执行（按日期判断跨日）
        current_date = context.current_time.strftime("%Y-%m-%d")
        is_new_day = current_date != self._last_trade_date

        if not is_new_day:
            return  # 同一天内不再操作

        positions = context.positions

        # 如果不是每日滚仓模式，只要有持仓就跳过
        if not self.roll_daily and len(positions) > 0:
            return

        chain = context.option_chain
        if chain.empty:
            return

        # ---------- 筛选最近到期的 ATM call ----------
        call_name = self._select_nearest_atm_call(chain, context.underlying_price)
        if call_name is None:
            return

        # 如果选出的合约与当前持仓相同，不动
        if call_name in positions:
            self._last_trade_date = current_date
            return

        # ---------- 平掉旧仓位 ----------
        for name, pos in list(positions.items()):
            if pos.direction.value == "short":
                self.log(f"平仓滚动: {name}")
                context.close(name)

        # ---------- 卖出新合约 ----------
        # 复利模式：按当前权益动态调整卖出数量
        if self.compound:
            equity = context.account.equity(
                sum(p.unrealized_pnl for p in context.positions.values())
            )
            sell_qty = equity * self.quantity / context.account.initial_balance
            sell_qty = max(sell_qty, 0.01)  # 最小 0.01
        else:
            sell_qty = self.quantity

        self.log(f"卖出备兑 Call: {call_name} x{sell_qty:.4f}")
        context.sell(call_name, sell_qty)
        self._last_trade_date = current_date

    # ------------------------------------------------------------------

    def _select_nearest_atm_call(self, chain, underlying_price: float) -> str | None:
        """从 option chain 中选出最近到期、行权价最接近标的价的 call。

        使用 numpy 向量化操作以提高性能。
        """
        opt_type_arr = chain["option_type"].values
        dte_arr = chain["days_to_expiry"].values.astype(float)
        strike_arr = chain["strike_price"].values.astype(float)
        names_arr = chain["instrument_name"].values
        exp_arr = chain["expiration_date"].values

        # 1. 筛选 call
        is_call = np.char.startswith(opt_type_arr.astype(str), "c")
        call_idx = np.flatnonzero(is_call)
        if len(call_idx) == 0:
            return None

        # 2. 进一步过滤 DTE
        dte_filtered = dte_arr[call_idx]
        valid_dte_mask = dte_filtered > self.min_dte
        call_idx = call_idx[valid_dte_mask]
        if len(call_idx) == 0:
            return None

        # 3. 找到最小 DTE（最近到期）
        dte_calls = dte_arr[call_idx]
        min_dte = np.min(dte_calls)

        # 4. 在最近到期的合约中找最接近 ATM 的行权价（容差比较避免浮点精度问题）
        nearest_exp_mask = np.abs(dte_calls - min_dte) < 0.001
        nearest_idx = call_idx[nearest_exp_mask]
        if len(nearest_idx) == 0:
            return None

        strikes = strike_arr[nearest_idx]
        dist = np.abs(strikes - underlying_price)
        best_local = np.argmin(dist)

        return str(names_arr[nearest_idx[best_local]])
