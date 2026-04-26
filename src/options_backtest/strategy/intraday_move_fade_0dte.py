"""0DTE intraday move fade strategy.

逻辑：
- 记录 UTC 当日 00:00 的 BTC 价格作为日内起点；
- 到指定 `entry_hour` 时，若日内单边涨跌幅绝对值超过 `move_threshold_pct`；
- 则卖出当日 08:00 到期、与波动方向相反的一张 OTM 期权：
  - 当日上涨 → 卖 OTM Call
  - 当日下跌 → 卖 OTM Put
- 持有到当日到期结算，不做提前止盈止损。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from options_backtest.strategy.base import BaseStrategy


class IntradayMoveFade0DTEStrategy(BaseStrategy):
    """Sell same-day OTM option opposite to intraday move direction."""

    name = "IntradayMoveFade0DTE"

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(params)
        self.move_threshold_pct: float = float(self.params.get("move_threshold_pct", 0.01))
        self.entry_hour: int = int(self.params.get("entry_hour", 6))
        self.otm_pct: float = float(self.params.get("otm_pct", 0.01))
        self.quantity: float = float(self.params.get("quantity", 1.0))
        self.compound: bool = bool(self.params.get("compound", False))
        self.min_mark_price: float = float(self.params.get("min_mark_price", 0.0))
        self.expiry_hour_utc: int = int(self.params.get("expiry_hour_utc", 8))
        self.max_hours_to_expiry: float = float(self.params.get("max_hours_to_expiry", 24.0))
        self.entry_realized_vol_lookback_hours: int = int(self.params.get("entry_realized_vol_lookback_hours", 0))
        self.entry_realized_vol_max: float = float(self.params.get("entry_realized_vol_max", 0.0))
        self.take_profit_pct: float = float(self.params.get("take_profit_pct", 0.0))
        self.stop_loss_pct: float = float(self.params.get("stop_loss_pct", 0.0))
        self.underlying_move_stop_pct: float = float(self.params.get("underlying_move_stop_pct", 0.0))
        self.max_loss_equity_pct: float = float(self.params.get("max_loss_equity_pct", 0.0))

        if self.move_threshold_pct > 1.0:
            self.move_threshold_pct /= 100.0
        if self.otm_pct > 1.0:
            self.otm_pct /= 100.0
        if self.underlying_move_stop_pct > 1.0:
            self.underlying_move_stop_pct /= 100.0
        if self.max_loss_equity_pct > 1.0:
            self.max_loss_equity_pct /= 100.0

        self._current_day: str = ""
        self._day_open_price: float | None = None
        self._last_trade_day: str = ""
        self._entry_diagnostics: list[dict[str, Any]] = []
        self._exit_diagnostics: list[dict[str, Any]] = []
        self._active_entry_spot: float | None = None

    def on_step(self, context) -> None:
        now = pd.Timestamp(context.current_time)
        day_id = now.strftime("%Y-%m-%d")

        if day_id != self._current_day:
            self._current_day = day_id
            self._day_open_price = float(context.underlying_price)

        if context.positions:
            self._maybe_manage_open_position(context, now)
            return
        self._active_entry_spot = None
        if day_id == self._last_trade_day:
            return
        if now.hour != self.entry_hour:
            return

        open_price = float(self._day_open_price or 0.0)
        spot = float(context.underlying_price or 0.0)
        if open_price <= 0 or spot <= 0:
            self._last_trade_day = day_id
            return

        move_pct = spot / open_price - 1.0
        if abs(move_pct) < self.move_threshold_pct:
            self._last_trade_day = day_id
            return

        realized_vol = None
        if self.entry_realized_vol_lookback_hours > 1 and self.entry_realized_vol_max > 0:
            realized_vol = self._recent_realized_vol(context, now, self.entry_realized_vol_lookback_hours)
            if realized_vol is None:
                self._last_trade_day = day_id
                return
            if realized_vol > self.entry_realized_vol_max:
                self.log(
                    f"Skip fade entry: {self.entry_realized_vol_lookback_hours}h RV {realized_vol:.2%} "
                    f"> {self.entry_realized_vol_max:.2%}"
                )
                self._last_trade_day = day_id
                return

        option_type = "call" if move_pct > 0 else "put"
        chain = context.option_chain
        if chain is None or getattr(chain, "empty", True):
            self._last_trade_day = day_id
            return

        selected = self._select_same_day_otm_option(chain, now, spot, option_type)
        if selected is None:
            self._last_trade_day = day_id
            return

        qty = self._compute_quantity(context)
        if qty <= 0:
            self._last_trade_day = day_id
            return

        instrument_name = str(selected["instrument_name"])
        self.log(
            f"Sell 0DTE {option_type.upper()} fade: {instrument_name} qty={qty:.4f} "
            f"move={move_pct:.2%} open={open_price:.2f} spot={spot:.2f}"
        )
        context.sell(instrument_name, qty)
        self._entry_diagnostics.append(
            {
                "entry_time": now,
                "day_open_price": open_price,
                "spot": spot,
                "move_pct": float(move_pct),
                "move_threshold_pct": float(self.move_threshold_pct),
                "entry_hour": int(self.entry_hour),
                "otm_pct": float(self.otm_pct),
                "option_type": option_type,
                "rv_24h": None if realized_vol is None else float(realized_vol),
                "instrument_name": instrument_name,
                "strike_price": float(selected.get("strike_price", np.nan)),
                "expiration_date": pd.Timestamp(selected.get("expiration_date")),
                "mark_price": float(selected.get("mark_price", np.nan)),
                "quantity": float(qty),
            }
        )
        self._active_entry_spot = spot
        self._last_trade_day = day_id

    def _maybe_manage_open_position(self, context, now: pd.Timestamp) -> None:
        total_unrealized = float(sum(p.unrealized_pnl for p in context.positions.values()))
        equity = float(context.account.equity(total_unrealized))
        basket_pnl_pct = self._basket_pnl_pct(context)

        if self.underlying_move_stop_pct > 0 and self._active_entry_spot and self._active_entry_spot > 0:
            spot_move = abs(float(context.underlying_price) / self._active_entry_spot - 1.0)
            if spot_move >= self.underlying_move_stop_pct:
                self.log(
                    f"Close fade basket: spot move {spot_move:.2%} >= {self.underlying_move_stop_pct:.2%} from entry"
                )
                self._queue_exit_diagnostic("underlying_move_stop", now, basket_pnl_pct, context, total_unrealized)
                context.close_all()
                return

        if self.max_loss_equity_pct > 0 and equity > 0:
            loss_ratio = max(0.0, -total_unrealized) / equity
            if loss_ratio >= self.max_loss_equity_pct:
                self.log(
                    f"Close fade basket: unrealized loss/equity {loss_ratio:.2%} >= {self.max_loss_equity_pct:.2%}"
                )
                self._queue_exit_diagnostic("max_loss_equity", now, basket_pnl_pct, context, total_unrealized)
                context.close_all()
                return

        if basket_pnl_pct is None:
            return

        if self.take_profit_pct > 0 and basket_pnl_pct >= self.take_profit_pct:
            self.log(
                f"Take profit on fade basket: {basket_pnl_pct:.1f}% >= {self.take_profit_pct:.1f}%"
            )
            self._queue_exit_diagnostic("take_profit", now, basket_pnl_pct, context, total_unrealized)
            context.close_all()
            return

        if self.stop_loss_pct > 0 and basket_pnl_pct <= -self.stop_loss_pct:
            self.log(
                f"Stop loss on fade basket: {basket_pnl_pct:.1f}% <= -{self.stop_loss_pct:.1f}%"
            )
            self._queue_exit_diagnostic("stop_loss", now, basket_pnl_pct, context, total_unrealized)
            context.close_all()

    def _basket_pnl_pct(self, context) -> float | None:
        entry_credit = 0.0
        close_cost = 0.0
        for pos in context.positions.values():
            sign = 1.0 if pos.direction.value == "short" else -1.0
            entry_credit += float(pos.entry_price) * float(pos.quantity) * sign
            close_cost += float(pos.current_mark_price) * float(pos.quantity) * sign

        if not np.isfinite(entry_credit) or entry_credit <= 0:
            return None
        if not np.isfinite(close_cost):
            return None
        pnl = entry_credit - close_cost
        return pnl / entry_credit * 100.0

    def _queue_exit_diagnostic(self, reason: str, now: pd.Timestamp, basket_pnl_pct: float | None, context, total_unrealized: float) -> None:
        self._exit_diagnostics.append(
            {
                "time": now,
                "reason": reason,
                "basket_pnl_pct": None if basket_pnl_pct is None else float(basket_pnl_pct),
                "spot": float(context.underlying_price or np.nan),
                "total_unrealized": float(total_unrealized),
                "entry_spot": None if self._active_entry_spot is None else float(self._active_entry_spot),
                "positions": list(context.positions.keys()),
            }
        )

    def _recent_realized_vol(self, context, now: pd.Timestamp, lookback_hours: int) -> float | None:
        window = self._get_underlying_window(context, now, lookback_hours)
        if window is None or len(window) < 3:
            return None
        prices = np.asarray(window, dtype=float)
        prices = prices[np.isfinite(prices) & (prices > 0)]
        if len(prices) < 3:
            return None
        log_returns = np.diff(np.log(prices))
        if len(log_returns) < 2:
            return None
        std = float(np.std(log_returns, ddof=1))
        if not np.isfinite(std):
            return None
        return std * math.sqrt(24.0 * 365.25)

    def _get_underlying_window(self, context, now: pd.Timestamp, lookback_hours: int) -> np.ndarray | None:
        engine = getattr(context, "_engine", None)
        if engine is None:
            return None
        udf = getattr(engine, "_underlying_df", None)
        if udf is None or getattr(udf, "empty", True):
            return None
        ts_arr = udf["timestamp"].to_numpy(dtype="datetime64[ns]").astype("int64")
        close_arr = udf["close"].to_numpy(dtype=float)
        end_ns = int(now.value)
        start_ns = end_ns - int(lookback_hours) * 3_600_000_000_000
        left = int(np.searchsorted(ts_arr, start_ns, side="left"))
        right = int(np.searchsorted(ts_arr, end_ns, side="right"))
        if right - left < 2:
            return None
        return close_arr[left:right]

    def _compute_quantity(self, context) -> float:
        if not self.compound:
            return max(0.0, float(self.quantity))
        equity = context.account.equity(sum(p.unrealized_pnl for p in context.positions.values()))
        base = float(context.account.initial_balance or 0.0)
        if base <= 0:
            return max(0.0, float(self.quantity))
        qty = max(0.0, equity * float(self.quantity) / base)
        return max(qty, 0.01) if qty > 0 else 0.0

    def _select_same_day_otm_option(self, chain, now: pd.Timestamp, spot: float, option_type: str):
        if hasattr(chain, "_to_dataframe"):
            df = chain._to_dataframe().copy()
        else:
            df = pd.DataFrame(chain).copy()
        if df.empty:
            return None
        if "expiration_date" not in df.columns or "strike_price" not in df.columns or "option_type" not in df.columns:
            return None

        df["expiration_date"] = pd.to_datetime(df["expiration_date"], utc=True, errors="coerce")
        df["strike_price"] = pd.to_numeric(df["strike_price"], errors="coerce")
        if "mark_price" in df.columns:
            df["mark_price"] = pd.to_numeric(df["mark_price"], errors="coerce")
        else:
            df["mark_price"] = np.nan
        if "bid_price" in df.columns:
            df["bid_price"] = pd.to_numeric(df["bid_price"], errors="coerce")
        else:
            df["bid_price"] = np.nan
        if "ask_price" in df.columns:
            df["ask_price"] = pd.to_numeric(df["ask_price"], errors="coerce")
        else:
            df["ask_price"] = np.nan

        hours_to_expiry = (df["expiration_date"] - now).dt.total_seconds() / 3600.0
        valid_expiry = (
            df["expiration_date"].notna()
            & (df["expiration_date"] > now)
            & (df["expiration_date"].dt.hour == self.expiry_hour_utc)
            & (hours_to_expiry <= self.max_hours_to_expiry)
            & df["strike_price"].notna()
        )
        df = df.loc[valid_expiry].copy()
        if df.empty:
            return None

        side_mask = df["option_type"].astype(str).str.lower().str.startswith(option_type[0])
        df = df.loc[side_mask].copy()
        if df.empty:
            return None

        expiry = df["expiration_date"].min()
        df = df.loc[df["expiration_date"] == expiry].copy()
        if df.empty:
            return None

        ref_price = df[["mark_price", "bid_price", "ask_price"]].max(axis=1, skipna=True)
        if self.min_mark_price > 0:
            df = df.loc[ref_price >= self.min_mark_price].copy()
            if df.empty:
                return None

        if option_type == "call":
            target_strike = spot * (1.0 + self.otm_pct)
            candidates = df.loc[df["strike_price"] >= max(spot, target_strike)].copy()
            if candidates.empty:
                candidates = df.loc[df["strike_price"] >= spot].copy()
        else:
            target_strike = spot * (1.0 - self.otm_pct)
            candidates = df.loc[df["strike_price"] <= min(spot, target_strike)].copy()
            if candidates.empty:
                candidates = df.loc[df["strike_price"] <= spot].copy()
        if candidates.empty:
            candidates = df.copy()

        candidates["target_dist"] = (candidates["strike_price"] - target_strike).abs()
        candidates = candidates.sort_values(["target_dist", "strike_price", "instrument_name"], ascending=[True, True, True])
        return candidates.iloc[0]

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            "intraday_move_fade_entries": list(self._entry_diagnostics),
            "intraday_move_fade_exits": list(self._exit_diagnostics),
        }