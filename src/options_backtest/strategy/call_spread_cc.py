"""Call Spread Covered Call (备兑看涨价差) strategy.

每天滚动构建 call credit spread：
  - 卖出 ATM call（获取权利金）
  - 买入 OTM call（cap 住上行风险）
假设持有等量现货作为备兑。

与普通 Covered Call 相比，买入高行权价 call 降低了极端上涨时的损失，
代价是权利金收入减少（spread width 之间的差额）。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from options_backtest.strategy.base import BaseStrategy


class CallSpreadCCStrategy(BaseStrategy):
    """每日滚动 call credit spread + 现货备兑。

    策略逻辑
    --------
    1. 每个交易日检查是否需要开仓/滚仓。
    2. 智能滚仓：只在标的价格偏离当前 short strike 超过阈值时才滚仓,
       避免长 DTE 合约下无意义的高频交易和手续费消耗。
    3. 从 option chain 中筛选最近到期的 call：
       a. 卖出行权价最接近当前标的价格的 ATM call。
       b. 买入行权价高于 ATM 一定偏移（spread_width_pct）的 OTM call。
    4. 到期自动由引擎交割。

    参数
    ----
    quantity : float
        每次卖出的合约数量（默认 1.0）。
    min_dte : float
        最小剩余到期天数过滤（默认 0.0 = 允许 0-DTE）。
    roll_daily : bool
        是否每天滚仓（默认 True）。
    compound : bool
        复利模式：按权益动态调整数量（默认 False）。
    spread_width_pct : float
        long call 行权价相对 ATM 的上偏百分比（默认 0.05 = 5%）。
        例如 BTC=100000, ATM strike=100000, long strike≈105000。
    roll_threshold_pct : float
        滚仓阈值：标的价格偏离当前 short strike 超过此比例才触发滚仓
        (默认 0.0 = 每次 ATM strike 变动即滚)。
        建议长 DTE 设 0.05~0.10, 短 DTE 设 0.0。
    max_dte : float
        最大剩余到期天数过滤（默认 0.0 = 不限制）。
        可用于优先选择较近到期日的合约。
    entry_hour : int
        每日开仓的 UTC 小时（默认 8 = 08:00 UTC，即 Deribit 交割后立即开仓）。
        设为 0 则在每日 00:00 UTC 开仓。
    """

    name = "CallSpreadCC"

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(params)
        self.quantity: float = self.params.get("quantity", 1.0)
        self.min_dte: float = self.params.get("min_dte", 0.0)
        self.roll_daily: bool = self.params.get("roll_daily", True)
        self.compound: bool = self.params.get("compound", False)
        self.spread_width_pct: float = self.params.get("spread_width_pct", 0.05)
        self.roll_threshold_pct: float = self.params.get("roll_threshold_pct", 0.0)
        self.max_dte: float = self.params.get("max_dte", 0.0)
        self.entry_hour: int = self.params.get("entry_hour", 8)
        self._last_trade_date: str = ""
        self._current_short_strike: float = 0.0  # 当前 short leg 的行权价

    # ------------------------------------------------------------------

    def on_step(self, context) -> None:
        current_hour = context.current_time.hour
        # 使用 entry_hour 后的日期标识（避免同一天重复操作）
        if self.entry_hour > 0:
            # 以 entry_hour 为界划分"交易日"：
            # 例如 entry_hour=8, 则 08:00 UTC 3/15 ~ 07:59 UTC 3/16 算同一交易日
            trade_day_id = context.current_time.strftime("%Y-%m-%d") + ("" if current_hour >= self.entry_hour else "_prev")
        else:
            trade_day_id = context.current_time.strftime("%Y-%m-%d")

        is_entry_time = (trade_day_id != self._last_trade_date and
                         current_hour >= self.entry_hour)

        if not is_entry_time:
            return

        positions = context.positions

        # 非每日滚仓模式下，有持仓就跳过
        if not self.roll_daily and len(positions) > 0:
            return

        chain = context.option_chain
        if chain.empty:
            return

        # ---------- 智能滚仓判断 ----------
        # 如果已有持仓且设置了 roll_threshold_pct, 检查是否需要滚仓
        if len(positions) > 0 and self.roll_threshold_pct > 0 and self._current_short_strike > 0:
            price_deviation = abs(context.underlying_price - self._current_short_strike) / self._current_short_strike
            if price_deviation < self.roll_threshold_pct:
                # 标的价格偏离不够大，不需要滚仓
                self._last_trade_date = trade_day_id
                return
            else:
                self.log(
                    f"触发滚仓: 标的 {context.underlying_price:.0f} 偏离 "
                    f"short strike {self._current_short_strike:.0f} "
                    f"达 {price_deviation:.1%} > 阈值 {self.roll_threshold_pct:.1%}"
                )

        # ---------- 选择 short call (ATM) 和 long call (OTM) ----------
        short_name, long_name = self._select_spread(chain, context.underlying_price)
        if short_name is None:
            return

        # 如果两条腿都已在持仓中（相同合约），不动
        if short_name in positions and (long_name is None or long_name in positions):
            self._last_trade_date = trade_day_id
            return

        # ---------- 平掉旧仓位 ----------
        for name in list(positions.keys()):
            self.log(f"平仓滚动: {name}")
            context.close(name)

        # ---------- 计算下单数量 ----------
        if self.compound:
            equity = context.account.equity(
                sum(p.unrealized_pnl for p in context.positions.values())
            )
            sell_qty = equity * self.quantity / context.account.initial_balance
            sell_qty = max(sell_qty, 0.01)
        else:
            sell_qty = self.quantity

        # ---------- 构建 call credit spread ----------
        # Leg 1: 卖出 ATM call（收取权利金）
        self.log(f"卖出 ATM Call: {short_name} x{sell_qty:.4f}")
        context.sell(short_name, sell_qty)

        # Leg 2: 买入 OTM call（保护上行）
        if long_name is not None:
            self.log(f"买入 OTM Call: {long_name} x{sell_qty:.4f}")
            context.buy(long_name, sell_qty)

        # 记录当前 short strike 用于滚仓判断
        parsed = short_name.split("-")
        if len(parsed) >= 3:
            try:
                self._current_short_strike = float(parsed[2])
            except ValueError:
                pass

        self._last_trade_date = trade_day_id

    # ------------------------------------------------------------------

    def _select_spread(
        self, chain, underlying_price: float
    ) -> tuple[str | None, str | None]:
        """从 option chain 中选出 short ATM call + long OTM call。

        Returns
        -------
        (short_call_name, long_call_name)
            如果找不到合适合约则返回 (None, None)。
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
            return None, None

        # 2. DTE 过滤
        dte_filtered = dte_arr[call_idx]
        valid_dte_mask = dte_filtered > self.min_dte
        if self.max_dte > 0:
            valid_dte_mask &= dte_filtered <= self.max_dte
        call_idx = call_idx[valid_dte_mask]
        if len(call_idx) == 0:
            return None, None

        # 3. 找最小 DTE（最近到期）
        dte_calls = dte_arr[call_idx]
        min_dte = np.min(dte_calls)

        # 4. 在最近到期的合约中筛选
        nearest_exp_mask = np.abs(dte_calls - min_dte) < 0.001
        nearest_idx = call_idx[nearest_exp_mask]
        if len(nearest_idx) == 0:
            return None, None

        strikes = strike_arr[nearest_idx]
        names = names_arr[nearest_idx]

        # 5. Short leg: 最接近 ATM 的行权价
        dist_atm = np.abs(strikes - underlying_price)
        short_local = np.argmin(dist_atm)
        short_name = str(names[short_local])
        short_strike = strikes[short_local]

        # 6. Long leg: 行权价 ≥ short_strike * (1 + spread_width_pct)
        target_long_strike = short_strike * (1.0 + self.spread_width_pct)
        higher_mask = strikes >= target_long_strike
        if not np.any(higher_mask):
            # 找不到足够 OTM 的 call → 只做 short leg（退化为普通 CC）
            self.log(f"未找到 OTM call (目标 ≥{target_long_strike:.0f}), 退化为普通 CC")
            return short_name, None

        higher_strikes = strikes[higher_mask]
        higher_names = names[higher_mask]
        # 选最接近目标行权价的（最低行权价的 OTM call）
        long_local = np.argmin(np.abs(higher_strikes - target_long_strike))
        long_name = str(higher_names[long_local])

        return short_name, long_name
