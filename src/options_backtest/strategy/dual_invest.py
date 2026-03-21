"""OKX Dual Investment (双币赢) strategy simulation.

模拟 OKX 双币赢每日滚动操作：
- 持有 BTC 时执行「高卖」（Sell High），等价于卖 covered call
- 持有 USDT 时执行「低买」（Buy Low），等价于卖 cash-secured put
- 到期被行权时资产自动转换并切换模式，继续滚动
- 收益率为固定 APR（模拟平台报价，内含平台利差）
"""

from __future__ import annotations

from typing import Any

from options_backtest.strategy.base import BaseStrategy


class DualInvestStrategy(BaseStrategy):
    """OKX 双币赢每日滚动模拟。

    策略逻辑
    --------
    1. 初始持有 BTC，以「高卖（Sell High）」模式入场。
    2. 每天选取最接近当前价格的行权价档位作为 target price。
    3. 到期结算（每日一次，新日第一根 bar）：

       - **高卖模式**：若结算价 ≥ strike → BTC 转换为 USDT（行权）；
         否则保留 BTC。
       - **低买模式**：若结算价 ≤ strike → USDT 转换为 BTC（行权）；
         否则保留 USDT。

    4. 无论是否行权，均获得固定日收益（APR / 365），计入本金。
    5. 被行权后自动切换模式，继续滚动。

    Parameters
    ----------
    apr : float
        年化收益率，3.60 = 360%（默认）。
    strike_step : float
        行权价档位间距（USDT），默认 500。

    Notes
    -----
    - 双币赢的手续费/利差已隐含在 APR 中，因此 execution 费用应设为 0。
    - 策略不通过引擎的订单/持仓系统交易，而是直接操作 account.balance，
      因此 total_trades / win_rate 等指标为 0，仅看权益曲线和收益率。
    - 在 USDT 模式下，每根 bar 会将 USDT 持仓按当前价折算为等值 BTC
      更新 balance，保证权益曲线正确反映 USD 价值变化。
    """

    name = "DualInvest"

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(params)
        self.apr: float = self.params.get("apr", 3.60)        # 360%
        self.strike_step: float = self.params.get("strike_step", 500.0)
        self.daily_mult: float = 1 + self.apr / 365

        # ---- internal state ----
        self._mode: str = "btc"          # "btc" = 高卖, "usdt" = 低买
        self._btc_qty: float = 0.0
        self._usdt_qty: float = 0.0
        self._strike: float = 0.0
        self._last_date: str = ""
        self._initialized: bool = False

        # statistics
        self._conversions: int = 0
        self._days_in_btc: int = 0
        self._days_in_usdt: int = 0

    # ------------------------------------------------------------------

    def on_step(self, context) -> None:
        current_date = context.current_time.strftime("%Y-%m-%d")
        is_new_day = current_date != self._last_date

        if not is_new_day:
            # 非跨日 bar：仅更新 USDT 模式下的 BTC 等值余额
            if self._mode == "usdt":
                price = context.underlying_price
                if price > 0:
                    context.account.balance = self._usdt_qty / price
            return

        price = context.underlying_price

        # ---------- 首次初始化 ----------
        if not self._initialized:
            self._btc_qty = context.account.balance
            self._strike = self._nearest_strike(price)
            self._last_date = current_date
            self._initialized = True
            self.log(
                f"初始: {self._btc_qty:.4f} BTC @ ${price:.0f}, "
                f"高卖 strike=${self._strike:.0f}"
            )
            return

        # ---------- 每日结算 ----------
        strike = self._strike

        if self._mode == "btc":
            self._days_in_btc += 1
            if price >= strike:
                # 行权：BTC → USDT
                # OKX 公式：获得 BTC数量 × strike × (1 + APR/365)
                self._usdt_qty = self._btc_qty * strike * self.daily_mult
                self._btc_qty = 0.0
                self._mode = "usdt"
                context.account.balance = self._usdt_qty / price
                self._conversions += 1
                self.log(
                    f"[Day {self._days_in_btc + self._days_in_usdt}] "
                    f"高卖行权: BTC→${self._usdt_qty:,.2f} USDT "
                    f"@ strike={strike:.0f}, 现价={price:.0f}"
                )
            else:
                # 未行权：保留 BTC + 日收益
                # OKX 公式：取回 BTC数量 × (1 + APR/365)
                self._btc_qty *= self.daily_mult
                context.account.balance = self._btc_qty
                self.log(
                    f"[Day {self._days_in_btc + self._days_in_usdt}] "
                    f"高卖保留: {self._btc_qty:.6f} BTC "
                    f"(+{(self.daily_mult - 1) * 100:.3f}%), "
                    f"strike={strike:.0f}, 现价={price:.0f}"
                )
        else:
            # ---- USDT 模式（低买） ----
            self._days_in_usdt += 1
            if price <= strike:
                # 行权：USDT → BTC
                # OKX 公式：获得 (USDT数量 / strike) × (1 + APR/365)
                self._btc_qty = (self._usdt_qty / strike) * self.daily_mult
                self._usdt_qty = 0.0
                self._mode = "btc"
                context.account.balance = self._btc_qty
                self._conversions += 1
                self.log(
                    f"[Day {self._days_in_btc + self._days_in_usdt}] "
                    f"低买行权: USDT→{self._btc_qty:.6f} BTC "
                    f"@ strike={strike:.0f}, 现价={price:.0f}"
                )
            else:
                # 未行权：保留 USDT + 日收益
                self._usdt_qty *= self.daily_mult
                context.account.balance = self._usdt_qty / price
                self.log(
                    f"[Day {self._days_in_btc + self._days_in_usdt}] "
                    f"低买保留: ${self._usdt_qty:,.2f} USDT "
                    f"(+{(self.daily_mult - 1) * 100:.3f}%), "
                    f"strike={strike:.0f}, 现价={price:.0f}"
                )

        # ---------- 设定下一日 strike ----------
        self._strike = self._nearest_strike(price)
        self._last_date = current_date

    # ------------------------------------------------------------------

    def _nearest_strike(self, price: float) -> float:
        """Round price to nearest strike step (e.g. nearest $500)."""
        return round(price / self.strike_step) * self.strike_step
