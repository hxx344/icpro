"""Naked Call selling strategy (USD margin simulation).

每天在指定 `entry_hour` 卖出最近到期的 Call（不持现货）。
支持 ATM / OTM 行权价选择、日内止损/止盈、波动率过滤等风控参数。
适用于模拟用 USD 保证金裸卖 Call（回测中账户仍以 ETH 单位保存，但初始余额由 USD 折算）。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from options_backtest.strategy.base import BaseStrategy


class NakedCallStrategy(BaseStrategy):
    """每日卖出最近到期 Call（裸卖），含多种风控优化。"""

    name = "NakedCall"

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(params)
        self.quantity: float = self.params.get("quantity", 1.0)
        self.min_dte: float = self.params.get("min_dte", 0.0)
        self.roll_daily: bool = self.params.get("roll_daily", True)
        self.compound: bool = self.params.get("compound", False)
        self.entry_hour: int = self.params.get("entry_hour", 8)

        # --- Protective call (bear call spread) ---
        self.buy_protective_call: bool = self.params.get("buy_protective_call", False)
        self.hedge_strike_pct: float = self.params.get("hedge_strike_pct", 1.10)
        self.hedge_quantity_ratio: float = self.params.get("hedge_quantity_ratio", 1.0)

        # --- OTM offset: sell OTM instead of ATM ---
        # strike_offset_pct = 0.0  → ATM (legacy behaviour)
        # strike_offset_pct = 0.02 → sell call at strike ≈ price × 1.02 (2% OTM)
        self.strike_offset_pct: float = self.params.get("strike_offset_pct", 0.0)

        # --- Intraday stop-loss ---
        # stop_loss_pct: if the option mark price rises above entry × (1 + stop_loss_pct),
        # close position immediately (caps worst-case loss)
        # e.g. stop_loss_pct = 2.0 → stop-loss when mark rises 200% vs entry premium
        self.stop_loss_pct: float = self.params.get("stop_loss_pct", 0.0)

        # --- Intraday take-profit ---
        # take_profit_pct: if the option mark price falls below entry × (1 - take_profit_pct/100),
        # close early and lock in profits
        # e.g. take_profit_pct = 50 → close when 50% of premium captured
        self.take_profit_pct: float = self.params.get("take_profit_pct", 0.0)

        # --- Volatility filter ---
        # vol_lookback_hours: number of hourly bars to compute recent realized volatility
        # max_vol_pct: skip entry if recent realized vol (annualized) > max_vol_pct
        self.vol_lookback_hours: int = self.params.get("vol_lookback_hours", 0)
        self.max_vol_pct: float = self.params.get("max_vol_pct", 0.0)

        # --- Dynamic OTM: scale strike offset by recent vol ---
        # If dynamic_otm = True, actual offset = strike_offset_pct × (realized_vol / base_vol)
        # This sells further OTM when vol is high, closer when vol is low.
        self.dynamic_otm: bool = self.params.get("dynamic_otm", False)
        self.base_vol: float = self.params.get("base_vol", 0.60)  # annualised baseline IV

        # --- Max loss per trade (coin-denominated) ---
        # If unrealized loss on the short exceeds this fraction of account equity, close.
        # e.g. max_loss_equity_pct = 0.05 → close if loss > 5% of equity
        self.max_loss_equity_pct: float = self.params.get("max_loss_equity_pct", 0.0)

        # Internal state
        self._last_trade_day: str = ""
        self._price_history: list[float] = []  # for vol computation

    def on_step(self, context) -> None:
        current_hour = context.current_time.hour
        day_id = context.current_time.strftime("%Y-%m-%d")

        # Track price history for volatility filter
        self._price_history.append(context.underlying_price)

        positions = context.positions

        # --- Intraday risk management on existing short positions ---
        if positions:
            self._manage_risk(context)

        # 仅在 entry_hour 之后且尚未当日交易时开仓
        if not (current_hour >= self.entry_hour and day_id != self._last_trade_day):
            return

        # 如果不是每日滚仓并且已有持仓则跳过
        if not self.roll_daily and len(positions) > 0:
            return

        # --- Volatility filter: skip entry during extreme vol ---
        if self.vol_lookback_hours > 0 and self.max_vol_pct > 0:
            recent_vol = self._compute_recent_vol()
            if recent_vol is not None and recent_vol > self.max_vol_pct:
                self.log(f"跳过开仓: 近期波动率 {recent_vol:.1f}% > 阈值 {self.max_vol_pct:.1f}%")
                self._last_trade_day = day_id
                return

        chain = context.option_chain
        if chain.empty:
            return

        # Determine effective OTM offset (static or dynamic)
        offset = self._effective_offset()

        call_name = self._select_call(chain, context.underlying_price, offset)
        if call_name is None:
            return

        # 如果已经持有该空头则跳过
        if call_name in positions:
            self._last_trade_day = day_id
            return

        # 平掉旧仓位
        for name in list(positions.keys()):
            self.log(f"平仓滚动: {name}")
            context.close(name)

        # 计算下单数量（复利模式按权益比例）
        if self.compound:
            equity = context.account.equity(sum(p.unrealized_pnl for p in context.positions.values()))
            sell_qty = equity * self.quantity / context.account.initial_balance
            sell_qty = max(sell_qty, 0.01)
        else:
            sell_qty = self.quantity

        self.log(f"卖出裸 Call: {call_name} x{sell_qty:.4f}")
        context.sell(call_name, sell_qty)

        # 可选：同时买入更远行权价的 Call 作为保护
        if self.buy_protective_call:
            prot = self._select_protective_call(call_name, chain, context.underlying_price)
            if prot is not None:
                buy_qty = sell_qty * self.hedge_quantity_ratio
                self.log(f"买入保护性 Call: {prot} x{buy_qty:.4f}")
                context.buy(prot, buy_qty)

        self._last_trade_day = day_id

    # ------------------------------------------------------------------
    # Risk management
    # ------------------------------------------------------------------

    def _manage_risk(self, context) -> None:
        """Intraday stop-loss / take-profit / max-loss checks on open positions."""
        for name, pos in list(context.positions.items()):
            if pos.direction.value != "short":
                continue
            if pos.entry_price <= 0:
                continue

            # Stop-loss: option mark price rose too much vs entry premium
            if self.stop_loss_pct > 0:
                # For short: loss occurs when mark > entry
                price_rise = (pos.current_mark_price - pos.entry_price) / pos.entry_price
                if price_rise >= self.stop_loss_pct:
                    self.log(f"止损平仓 {name}: mark涨幅 {price_rise:.0%} >= {self.stop_loss_pct:.0%}")
                    context.close(name)
                    continue

            # Take-profit: option mark price fell significantly (premium decay captured)
            if self.take_profit_pct > 0:
                pnl_pct = (pos.entry_price - pos.current_mark_price) / pos.entry_price * 100
                if pnl_pct >= self.take_profit_pct:
                    self.log(f"止盈平仓 {name}: 已捕获 {pnl_pct:.1f}% 权利金")
                    context.close(name)
                    continue

            # Max loss as fraction of equity
            if self.max_loss_equity_pct > 0:
                equity = context.account.equity(
                    sum(p.unrealized_pnl for p in context.positions.values())
                )
                if equity > 0 and abs(pos.unrealized_pnl) / equity > self.max_loss_equity_pct:
                    self.log(f"权益止损 {name}: 亏损占权益 {abs(pos.unrealized_pnl)/equity:.1%}")
                    context.close(name)
                    continue

    # ------------------------------------------------------------------
    # Volume / volatility helpers
    # ------------------------------------------------------------------

    def _compute_recent_vol(self) -> float | None:
        """Compute annualised realised vol from recent hourly prices."""
        n = self.vol_lookback_hours
        if len(self._price_history) < n + 1:
            return None
        prices = self._price_history[-(n + 1):]
        log_returns = np.diff(np.log(prices))
        hourly_std = float(np.std(log_returns))
        ann_vol = hourly_std * np.sqrt(8760) * 100  # annualised, in %
        return ann_vol

    def _effective_offset(self) -> float:
        """Return strike offset, optionally scaled by recent vol."""
        base_offset = self.strike_offset_pct
        if not self.dynamic_otm or base_offset == 0:
            return base_offset
        recent_vol = self._compute_recent_vol()
        if recent_vol is None:
            return base_offset
        # Scale: offset × (recent_vol / base_vol)
        # Higher vol → sell further OTM; lower vol → sell closer to ATM
        scale = (recent_vol / 100.0) / self.base_vol if self.base_vol > 0 else 1.0
        return base_offset * max(scale, 0.5)  # floor at 50% of base offset

    # ------------------------------------------------------------------
    # Strike selection
    # ------------------------------------------------------------------

    def _select_call(self, chain, underlying_price: float, offset: float = 0.0) -> str | None:
        """Select a call to sell. If offset > 0, target OTM strike (price × (1+offset))."""
        opt_type_arr = chain["option_type"].values
        dte_arr = chain["days_to_expiry"].values.astype(float)
        strike_arr = chain["strike_price"].values.astype(float)
        names_arr = chain["instrument_name"].values

        is_call = np.char.startswith(opt_type_arr.astype(str), "c")
        call_idx = np.flatnonzero(is_call)
        if len(call_idx) == 0:
            return None

        dte_filtered = dte_arr[call_idx]
        valid_dte_mask = dte_filtered > self.min_dte
        call_idx = call_idx[valid_dte_mask]
        if len(call_idx) == 0:
            return None

        dte_calls = dte_arr[call_idx]
        min_dte = np.min(dte_calls)

        nearest_exp_mask = np.abs(dte_calls - min_dte) < 0.001
        nearest_idx = call_idx[nearest_exp_mask]
        if len(nearest_idx) == 0:
            return None

        strikes = strike_arr[nearest_idx]
        target_strike = underlying_price * (1.0 + offset)

        if offset > 0:
            # OTM mode: pick the first strike >= target; if none, use highest available
            otm_mask = strikes >= target_strike
            if otm_mask.any():
                # Among those >= target, pick closest to target
                otm_idx = np.flatnonzero(otm_mask)
                best = otm_idx[np.argmin(strikes[otm_idx] - target_strike)]
            else:
                best = np.argmax(strikes)
        else:
            # ATM mode: pick closest to underlying price
            dist = np.abs(strikes - underlying_price)
            best = np.argmin(dist)

        return str(names_arr[nearest_idx[best]])

    def _select_protective_call(self, sold_instrument: str, chain, underlying_price: float) -> str | None:
        """选择与卖出合约同到期、行权价更高且接近目标的保护性 Call。

        目标行权价 = 卖出行权价 * self.hedge_strike_pct
        """
        names = chain["instrument_name"].values
        strikes = chain["strike_price"].values.astype(float)
        opt_types = chain["option_type"].values
        dtes = chain["days_to_expiry"].values.astype(float)

        # 找到被卖出的合约信息
        mask = names == sold_instrument
        if not mask.any():
            return None
        sold_idx = int(np.flatnonzero(mask)[0])
        sold_strike = float(strikes[sold_idx])
        sold_dte = float(dtes[sold_idx])

        # 筛选同为 Call、同到期且行权价更高的合约
        is_call = np.char.startswith(opt_types.astype(str), "c")
        same_exp_mask = np.abs(dtes - sold_dte) < 0.001
        candidate_mask = is_call & same_exp_mask & (strikes > sold_strike)
        cand_idx = np.flatnonzero(candidate_mask)
        if len(cand_idx) == 0:
            return None

        target = sold_strike * float(self.hedge_strike_pct)
        cand_strikes = strikes[cand_idx]
        # 找到第一个大于等于目标的最小行权价，否则选择最大行权价
        ge_mask = cand_strikes >= target
        if ge_mask.any():
            sel_local = np.flatnonzero(ge_mask)[0]
            sel_idx = cand_idx[sel_local]
        else:
            sel_idx = int(cand_idx[-1])

        return str(names[sel_idx])
