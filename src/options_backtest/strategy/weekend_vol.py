"""Weekend volatility selling strategy for hourly option snapshots."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from options_backtest.pricing.black76 import delta as bs_delta
from options_backtest.strategy.base import BaseStrategy


class WeekendVolStrategy(BaseStrategy):
    """Sell a delta-selected weekend strangle / iron condor."""

    name = "WeekendVol"

    DAY_MAP = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }

    EXPIRE_MAP = {
        "saturday": (5, 8, 16),
        "sunday": (6, 8, 40),
        "monday": (0, 8, 64),
    }

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(params)
        self.target_delta: float = float(self.params.get("target_delta", 0.40))
        self.target_call_delta: float = float(self.params.get("target_call_delta", self.target_delta))
        self.target_put_delta: float = float(self.params.get("target_put_delta", self.target_delta))
        self.wing_delta: float = float(self.params.get("wing_delta", 0.05))
        self.default_iv: float = float(self.params.get("default_iv", 0.60))
        self.entry_day: str = str(self.params.get("entry_day", "friday")).lower()
        self.entry_time_utc: str = str(self.params.get("entry_time_utc", "16:00"))
        self.expire_day: str = str(self.params.get("expire_day", "sunday")).lower()
        self.close_day: str = str(self.params.get("close_day", self.expire_day)).lower()
        self.close_time_utc: str = str(self.params.get("close_time_utc", "08:00"))
        self.early_profit_close_day: str = str(self.params.get("early_profit_close_day", "")).lower()
        self.early_profit_close_time_utc: str = str(self.params.get("early_profit_close_time_utc", ""))
        self.early_profit_take_profit_pct: float = float(self.params.get("early_profit_take_profit_pct", 0.0))
        self.early_profit_close_fraction: float = float(self.params.get("early_profit_close_fraction", 1.0))
        self.scheduled_reduce_day: str = str(self.params.get("scheduled_reduce_day", "")).lower()
        self.scheduled_reduce_time_utc: str = str(self.params.get("scheduled_reduce_time_utc", ""))
        self.scheduled_reduce_fraction: float = float(self.params.get("scheduled_reduce_fraction", 0.0))
        self.secondary_profit_close_day: str = str(self.params.get("secondary_profit_close_day", "")).lower()
        self.secondary_profit_close_time_utc: str = str(self.params.get("secondary_profit_close_time_utc", ""))
        self.secondary_profit_take_profit_pct: float = float(self.params.get("secondary_profit_take_profit_pct", 0.0))
        self.secondary_profit_close_fraction: float = float(self.params.get("secondary_profit_close_fraction", 1.0))
        self.expiry_selection: str = str(self.params.get("expiry_selection", "exact")).lower()
        self.leverage: float = float(self.params.get("leverage", 1.0))
        self.quantity: float = float(self.params.get("quantity", 0.1))
        self.quantity_step: float = float(self.params.get("quantity_step", 0.1))
        self.max_positions: int = int(self.params.get("max_positions", 1))
        self.compound: bool = bool(self.params.get("compound", True))
        self.max_delta_diff: float = float(self.params.get("max_delta_diff", 0.15))
        self.max_delta_diff_lower: float = float(self.params.get("max_delta_diff_lower", self.max_delta_diff))
        self.max_delta_diff_upper: float = float(self.params.get("max_delta_diff_upper", self.max_delta_diff))
        self.expiry_tolerance_hours: float = float(self.params.get("expiry_tolerance_hours", 12.0))
        self.take_profit_pct: float = float(self.params.get("take_profit_pct", 0.0))
        self.stop_loss_pct: float = float(self.params.get("stop_loss_pct", 0.0))
        self.max_loss_equity_pct: float = float(self.params.get("max_loss_equity_pct", 0.0))
        self.underlying_move_stop_pct: float = float(self.params.get("underlying_move_stop_pct", 0.0))
        self.stop_loss_underlying_move_pct: float = float(self.params.get("stop_loss_underlying_move_pct", 0.0))
        self.trail_profit_activation_pct: float = float(self.params.get("trail_profit_activation_pct", 0.0))
        self.trail_profit_giveback_pct: float = float(self.params.get("trail_profit_giveback_pct", 0.0))
        self.trail_profit_close_fraction: float = float(self.params.get("trail_profit_close_fraction", 1.0))
        self.drawdown_reduce_threshold: float = float(self.params.get("drawdown_reduce_threshold", 0.0))
        self.drawdown_reduce_factor: float = float(self.params.get("drawdown_reduce_factor", 1.0))
        self.loss_cooldown_days: int = int(self.params.get("loss_cooldown_days", 0))
        self.entry_abs_move_lookback_hours: int = int(self.params.get("entry_abs_move_lookback_hours", 0))
        self.entry_abs_move_max_pct: float = float(self.params.get("entry_abs_move_max_pct", 0.0))
        self.entry_realized_vol_lookback_hours: int = int(self.params.get("entry_realized_vol_lookback_hours", 0))
        self.entry_realized_vol_max: float = float(self.params.get("entry_realized_vol_max", 0.0))
        self.entry_realized_vol_size_threshold: float = float(self.params.get("entry_realized_vol_size_threshold", 0.0))
        self.entry_realized_vol_size_factor: float = float(self.params.get("entry_realized_vol_size_factor", 1.0))
        self.initial_entry_fraction: float = float(self.params.get("initial_entry_fraction", 1.0))
        self.secondary_entry_time_utc: str = str(self.params.get("secondary_entry_time_utc", ""))
        self.secondary_entry_fraction: float = float(self.params.get("secondary_entry_fraction", 0.0))
        self.secondary_entry_max_abs_move_pct: float = float(self.params.get("secondary_entry_max_abs_move_pct", 0.0))
        self.conditional_wing_rv_threshold: float = float(self.params.get("conditional_wing_rv_threshold", 0.0))
        self.conditional_wing_delta: float = float(self.params.get("conditional_wing_delta", 0.0))
        self.directional_adjust_day: str = str(self.params.get("directional_adjust_day", "")).lower()
        self.directional_adjust_time_utc: str = str(self.params.get("directional_adjust_time_utc", ""))
        self.directional_adjust_move_pct: float = float(self.params.get("directional_adjust_move_pct", 0.0))
        self.directional_adjust_close_fraction: float = float(self.params.get("directional_adjust_close_fraction", 1.0))
        self.reactive_hedge_move_pct: float = float(self.params.get("reactive_hedge_move_pct", 0.0))
        self.reactive_hedge_delta: float = float(self.params.get("reactive_hedge_delta", 0.0))
        self.reactive_hedge_fraction: float = float(self.params.get("reactive_hedge_fraction", 0.0))
        self._last_trade_week: str = ""
        self._entry_tranche_week: str = ""
        self._entry_tranches_done: int = 0
        self._active_entry_spot: float | None = None
        self._active_trade_entry_equity: float | None = None
        self._active_basket_pnl_peak_pct: float = 0.0
        self._last_scheduled_reduce_week: str = ""
        self._last_early_profit_action_week: str = ""
        self._last_secondary_profit_action_week: str = ""
        self._last_directional_adjust_week: str = ""
        self._last_reactive_hedge_week: str = ""
        self._last_trail_profit_week: str = ""
        self._equity_peak: float = 0.0
        self._last_loss_close_time: pd.Timestamp | None = None
        self._had_positions_prev: bool = False
        self._entry_diagnostics: list[dict[str, Any]] = []
        self._exit_diagnostics: list[dict[str, Any]] = []
        self._pending_exit_diagnostic: dict[str, Any] | None = None
        self._active_entry_time: pd.Timestamp | None = None

        hh, mm = self.entry_time_utc.split(":", 1)
        self.entry_hour = int(hh)
        self.entry_minute = int(mm)

        if self.secondary_entry_time_utc:
            se_hh, se_mm = self.secondary_entry_time_utc.split(":", 1)
            self.secondary_entry_hour = int(se_hh)
            self.secondary_entry_minute = int(se_mm)
        else:
            self.secondary_entry_hour = -1
            self.secondary_entry_minute = -1

        close_hh, close_mm = self.close_time_utc.split(":", 1)
        self.close_hour = int(close_hh)
        self.close_minute = int(close_mm)

        if self.early_profit_close_time_utc:
            ep_hh, ep_mm = self.early_profit_close_time_utc.split(":", 1)
            self.early_profit_close_hour = int(ep_hh)
            self.early_profit_close_minute = int(ep_mm)
        else:
            self.early_profit_close_hour = -1
            self.early_profit_close_minute = -1

        if self.scheduled_reduce_time_utc:
            sr_hh, sr_mm = self.scheduled_reduce_time_utc.split(":", 1)
            self.scheduled_reduce_hour = int(sr_hh)
            self.scheduled_reduce_minute = int(sr_mm)
        else:
            self.scheduled_reduce_hour = -1
            self.scheduled_reduce_minute = -1

        if self.secondary_profit_close_time_utc:
            sp_hh, sp_mm = self.secondary_profit_close_time_utc.split(":", 1)
            self.secondary_profit_close_hour = int(sp_hh)
            self.secondary_profit_close_minute = int(sp_mm)
        else:
            self.secondary_profit_close_hour = -1
            self.secondary_profit_close_minute = -1

        if self.directional_adjust_time_utc:
            da_hh, da_mm = self.directional_adjust_time_utc.split(":", 1)
            self.directional_adjust_hour = int(da_hh)
            self.directional_adjust_minute = int(da_mm)
        else:
            self.directional_adjust_hour = -1
            self.directional_adjust_minute = -1

        expire_cfg = self.EXPIRE_MAP.get(self.expire_day, self.EXPIRE_MAP["sunday"])
        self.expire_weekday = int(expire_cfg[0])
        self.expire_hour = int(expire_cfg[1])
        self.target_hold_hours = float(expire_cfg[2])

        if self.underlying_move_stop_pct > 1.0:
            self.underlying_move_stop_pct /= 100.0
        if self.stop_loss_underlying_move_pct > 1.0:
            self.stop_loss_underlying_move_pct /= 100.0

    def _normalize_iv(self, value: Any) -> float:
        try:
            iv = float(value)
        except Exception:
            iv = self.default_iv
        if not np.isfinite(iv) or iv <= 0:
            iv = self.default_iv
        if iv > 3.0:
            iv /= 100.0
        return float(iv)

    def _quote_reference_price(self, row: pd.Series) -> float:
        bid = pd.to_numeric(pd.Series([row.get("bid_price")]), errors="coerce").iloc[0]
        ask = pd.to_numeric(pd.Series([row.get("ask_price")]), errors="coerce").iloc[0]
        mark = pd.to_numeric(pd.Series([row.get("mark_price")]), errors="coerce").iloc[0]
        if np.isfinite(bid) and np.isfinite(ask) and float(bid) > 0 and float(ask) > 0:
            return float((float(bid) + float(ask)) / 2.0)
        if np.isfinite(mark) and float(mark) > 0:
            return float(mark)
        return 0.0

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            "weekend_vol_entries": list(self._entry_diagnostics),
            "weekend_vol_exits": list(self._exit_diagnostics),
        }

    def on_fill(self, context, fill) -> None:
        if self._pending_exit_diagnostic is None:
            return
        if context.positions:
            return
        exit_diag = dict(self._pending_exit_diagnostic)
        exit_diag["exit_time"] = pd.Timestamp(fill.timestamp)
        self._exit_diagnostics.append(exit_diag)
        self._pending_exit_diagnostic = None
        self._active_entry_time = None

    def _queue_exit_diagnostic(
        self,
        reason: str,
        now: pd.Timestamp,
        basket_pnl_pct: float | None,
        context,
        total_unrealized: float | None = None,
    ) -> None:
        self._pending_exit_diagnostic = {
            "entry_time": None if self._active_entry_time is None else pd.Timestamp(self._active_entry_time),
            "trigger_time": pd.Timestamp(now),
            "reason": str(reason),
            "basket_pnl_pct": None if basket_pnl_pct is None else float(basket_pnl_pct),
            "spot": float(context.underlying_price),
            "position_count": int(len(context.positions)),
            "entry_spot": None if self._active_entry_spot is None else float(self._active_entry_spot),
            "account_balance": float(context.account.balance),
            "account_equity": float(context.account.equity(0.0 if total_unrealized is None else total_unrealized)),
            "stop_loss_pct": float(self.stop_loss_pct),
        }

    def on_step(self, context) -> None:
        now = pd.Timestamp(context.current_time)
        if now.tzinfo is None:
            now = now.tz_localize("UTC")
        else:
            now = now.tz_convert("UTC")

        current_equity = float(context.account.equity(sum(p.unrealized_pnl for p in context.positions.values())))
        week_id = now.strftime("%G-W%V")
        if self._entry_tranche_week != week_id:
            self._entry_tranche_week = week_id
            self._entry_tranches_done = 0

        if current_equity > self._equity_peak:
            self._equity_peak = current_equity

        if self._had_positions_prev and not context.positions and self._active_trade_entry_equity is not None:
            if current_equity < self._active_trade_entry_equity:
                self._last_loss_close_time = now
            self._active_trade_entry_equity = None

        if not context.positions:
            self._active_entry_spot = None
            self._active_basket_pnl_peak_pct = 0.0

        if context.positions:
            if self._manage_open_position(context, now):
                self._had_positions_prev = bool(context.positions)
                return

        if context.positions and self._should_exit(now):
            self.log(f"Close weekend basket at scheduled time {now.isoformat()}")
            self._queue_exit_diagnostic("scheduled_close", now, None, context)
            context.close_all()
            self._had_positions_prev = bool(context.positions)
            return

        secondary_fraction = self._secondary_entry_fraction_for_now(context, now)
        if context.positions and secondary_fraction > 0:
            self._open_entry_tranche(context, now, secondary_fraction)
            self._had_positions_prev = bool(context.positions)
            return

        group_count = max(1, 4 if self.wing_delta > 0 else 2)
        if len(context.positions) // group_count >= self.max_positions:
            self._had_positions_prev = bool(context.positions)
            return
        if context.positions:
            self._had_positions_prev = bool(context.positions)
            return
        if not self._should_enter(now):
            self._had_positions_prev = bool(context.positions)
            return
        if not self._passes_entry_filters(context, now):
            self._had_positions_prev = bool(context.positions)
            return

        initial_fraction = self._initial_entry_fraction()
        if initial_fraction <= 0:
            self._had_positions_prev = bool(context.positions)
            return
        self._open_entry_tranche(context, now, initial_fraction)
        self._had_positions_prev = bool(context.positions)
        return

    def _should_enter(self, now: pd.Timestamp) -> bool:
        if now.weekday() != self.DAY_MAP.get(self.entry_day, 4):
            return False
        if now.hour != self.entry_hour or now.minute != self.entry_minute:
            return False
        if self.loss_cooldown_days > 0 and self._last_loss_close_time is not None:
            if now < self._last_loss_close_time + pd.Timedelta(days=self.loss_cooldown_days):
                return False
        week_id = now.strftime("%G-W%V")
        if self._last_trade_week == week_id:
            return False
        return True

    def _should_exit(self, now: pd.Timestamp) -> bool:
        target_day = self.DAY_MAP.get(self.close_day, self.expire_weekday)
        if now.weekday() != target_day:
            return False
        return now.hour == self.close_hour and now.minute == self.close_minute

    def _target_close_ts(self, now: pd.Timestamp) -> pd.Timestamp:
        target_day = self.DAY_MAP.get(self.close_day, self.expire_weekday)
        candidate = now.normalize() + pd.Timedelta(hours=self.close_hour, minutes=self.close_minute)
        days_ahead = (target_day - now.weekday()) % 7
        candidate = candidate + pd.Timedelta(days=days_ahead)
        if candidate <= now:
            candidate = candidate + pd.Timedelta(days=7)
        return candidate

    def _filter_target_expiry(self, chain, now: pd.Timestamp) -> pd.DataFrame:
        exp_arr = pd.to_datetime(chain["expiration_date"].values, utc=True, errors="coerce")
        if len(exp_arr) == 0:
            return pd.DataFrame()

        exp_arr_ns = exp_arr.astype("int64", copy=False)
        valid_mask = exp_arr.notna() & (exp_arr_ns > now.value)
        if not np.any(valid_mask):
            return pd.DataFrame()

        target_close = self._target_close_ts(now)
        if self.expiry_selection == "first_after_close":
            after_close = valid_mask & (exp_arr_ns >= target_close.value)
            if not np.any(after_close):
                return pd.DataFrame()
            future_expiries = exp_arr[after_close]
            best_expiry = future_expiries.min()
            return chain[exp_arr == best_expiry].copy()

        diff_hours = (exp_arr_ns - target_close.value) / 3_600_000_000_000.0
        mask = valid_mask & (np.abs(diff_hours) <= self.expiry_tolerance_hours)
        if not np.any(mask):
            return pd.DataFrame()

        df = chain[mask].copy()
        if df.empty:
            return df

        expiry_col = pd.to_datetime(df["expiration_date"], utc=True, errors="coerce")
        expiry_col_ns = expiry_col.astype("int64", copy=False)
        diff_hours = (expiry_col_ns - target_close.value) / 3_600_000_000_000.0
        best_expiry = expiry_col.iloc[int(np.argmin(np.abs(diff_hours)))]
        return df.loc[expiry_col == best_expiry].copy()

    def _select_leg(
        self,
        df: pd.DataFrame,
        spot: float,
        now: pd.Timestamp,
        option_type: str,
        target_abs_delta: float,
    ) -> pd.Series | None:
        if df.empty:
            return None

        opt_mask = df["option_type"].astype(str).str.lower().str.startswith(option_type[0])
        side_df = df.loc[opt_mask].copy()
        if side_df.empty:
            return None

        if option_type == "call":
            side_df = side_df.loc[side_df["strike_price"].astype(float) > spot]
        else:
            side_df = side_df.loc[side_df["strike_price"].astype(float) < spot]
        if side_df.empty:
            return None

        deltas = pd.to_numeric(side_df.get("delta"), errors="coerce")
        bad_delta = ~np.isfinite(deltas.to_numpy(dtype=float, na_value=np.nan))
        if bad_delta.any():
            side_df.loc[bad_delta, "delta"] = side_df.loc[bad_delta].apply(
                lambda row: self._compute_fallback_delta(row, spot, now, option_type),
                axis=1,
            )
            deltas = pd.to_numeric(side_df.get("delta"), errors="coerce")

        side_df = side_df.loc[np.isfinite(deltas)]
        if side_df.empty:
            return None

        delta_abs = pd.to_numeric(side_df["delta"], errors="coerce").abs()
        lower_bound = max(0.0, float(target_abs_delta) - max(0.0, self.max_delta_diff_lower))
        upper_bound = float(target_abs_delta) + max(0.0, self.max_delta_diff_upper)
        eligible = side_df.loc[(delta_abs >= lower_bound) & (delta_abs <= upper_bound)].copy()
        if eligible.empty:
            return None
        eligible_delta_abs = pd.to_numeric(eligible["delta"], errors="coerce").abs()
        diff = (eligible_delta_abs - target_abs_delta).abs()
        best_idx = diff.idxmin()
        if not np.isfinite(diff.loc[best_idx]):
            return None
        return eligible.loc[best_idx]

    def _compute_fallback_delta(self, row: pd.Series, spot: float, now: pd.Timestamp, option_type: str) -> float:
        exp = pd.Timestamp(row["expiration_date"])
        if exp.tzinfo is None:
            exp = exp.tz_localize("UTC")
        else:
            exp = exp.tz_convert("UTC")
        t_years = max((exp.value - now.value) / (365.25 * 86_400_000_000_000.0), 1e-6)
        strike = float(row["strike_price"])
        iv = row.get("mark_iv", self.default_iv)
        try:
            iv = float(iv)
        except Exception:
            iv = self.default_iv
        if not np.isfinite(iv) or iv <= 0:
            iv = self.default_iv
        if iv > 3.0:
            iv /= 100.0
        try:
            return float(bs_delta(spot, strike, t_years, sigma=iv, option_type=option_type, r=0.0))
        except Exception:
            return np.nan

    def _compute_quantity(self, context, spot: float) -> float:
        base_qty = max(self.quantity, 0.0)
        if not self.compound or spot <= 0:
            return base_qty
        equity = float(context.account.balance)
        raw_qty = equity * self.leverage / spot
        step = self.quantity_step if self.quantity_step > 0 else self.quantity
        if step > 0:
            raw_qty = math.floor(raw_qty / step) * step
        return max(base_qty, raw_qty)

    def _initial_entry_fraction(self) -> float:
        frac = float(self.initial_entry_fraction)
        if self.secondary_entry_fraction > 0 and frac >= 0.999:
            frac = max(0.0, 1.0 - self.secondary_entry_fraction)
        return max(0.0, frac)

    def _secondary_entry_fraction_for_now(self, context, now: pd.Timestamp) -> float:
        if self.secondary_entry_fraction <= 0:
            return 0.0
        if self._entry_tranches_done != 1:
            return 0.0
        if now.weekday() != self.DAY_MAP.get(self.entry_day, 4):
            return 0.0
        if now.hour != self.secondary_entry_hour or now.minute != self.secondary_entry_minute:
            return 0.0
        if self.secondary_entry_max_abs_move_pct > 0 and self._active_entry_spot and self._active_entry_spot > 0:
            move_pct = abs(float(context.underlying_price) / self._active_entry_spot - 1.0)
            if move_pct > self.secondary_entry_max_abs_move_pct:
                self.log(
                    f"Skip secondary entry: move from first tranche {move_pct:.2%} > {self.secondary_entry_max_abs_move_pct:.2%}"
                )
                self._entry_tranches_done = 2
                return 0.0
        return max(0.0, float(self.secondary_entry_fraction))

    def _apply_entry_size_adjustments(self, context, now: pd.Timestamp, qty: float) -> float:
        if qty <= 0:
            return qty
        step = self.quantity_step if self.quantity_step > 0 else self.quantity

        if self.entry_realized_vol_size_threshold > 0 and self.entry_realized_vol_size_factor > 0:
            lookback = self.entry_realized_vol_lookback_hours
            if lookback > 1:
                realized_vol = self._recent_realized_vol(context, now, lookback)
                if realized_vol is not None and realized_vol > self.entry_realized_vol_size_threshold:
                    scaled_qty = qty * self.entry_realized_vol_size_factor
                    if step > 0:
                        scaled_qty = math.floor(scaled_qty / step) * step
                    scaled_qty = max(0.0, scaled_qty)
                    if scaled_qty < qty:
                        self.log(
                            f"Reduce entry size on elevated RV: {realized_vol:.2%} > "
                            f"{self.entry_realized_vol_size_threshold:.2%}, qty {qty:.3f} -> {scaled_qty:.3f}"
                        )
                    qty = scaled_qty

        if self.drawdown_reduce_threshold > 0 and self.drawdown_reduce_factor > 0 and self._equity_peak > 0:
            current_equity = float(context.account.balance)
            drawdown = (self._equity_peak - current_equity) / self._equity_peak
            if drawdown >= self.drawdown_reduce_threshold:
                dd_qty = qty * self.drawdown_reduce_factor
                if step > 0:
                    dd_qty = math.floor(dd_qty / step) * step
                dd_qty = max(0.0, dd_qty)
                if dd_qty < qty:
                    self.log(
                        f"Reduce entry size on drawdown: {drawdown:.2%} >= {self.drawdown_reduce_threshold:.2%}, "
                        f"qty {qty:.3f} -> {dd_qty:.3f}"
                    )
                qty = dd_qty

        return qty

    def _open_entry_tranche(self, context, now: pd.Timestamp, fraction: float) -> None:
        chain = context.option_chain
        if chain.empty:
            return

        chain_df = self._filter_target_expiry(chain, now)
        if chain_df.empty:
            return

        spot = float(context.underlying_price)
        short_call = self._select_leg(chain_df, spot, now, "call", self.target_call_delta)
        short_put = self._select_leg(chain_df, spot, now, "put", self.target_put_delta)
        if short_call is None or short_put is None:
            return

        active_wing_delta = self._effective_entry_wing_delta(context, now)
        wing_call = None
        wing_put = None
        if active_wing_delta > 0:
            wing_call = self._select_leg(
                chain_df[chain_df["strike_price"] > short_call["strike_price"]],
                spot,
                now,
                "call",
                active_wing_delta,
            )
            wing_put = self._select_leg(
                chain_df[chain_df["strike_price"] < short_put["strike_price"]],
                spot,
                now,
                "put",
                active_wing_delta,
            )
            if wing_call is None or wing_put is None:
                return

        qty = self._compute_quantity(context, spot)
        qty = self._apply_entry_size_adjustments(context, now, qty)
        qty = qty * max(0.0, float(fraction))
        step = self.quantity_step if self.quantity_step > 0 else self.quantity
        if step > 0:
            qty = math.floor(qty / step) * step
        if qty <= 0:
            return

        short_call_name = str(short_call["instrument_name"])
        short_put_name = str(short_put["instrument_name"])
        tranche_idx = self._entry_tranches_done + 1
        realized_vol = None
        if self.entry_realized_vol_lookback_hours > 1:
            realized_vol = self._recent_realized_vol(context, now, self.entry_realized_vol_lookback_hours)

        short_call_iv = self._normalize_iv(short_call.get("mark_iv", self.default_iv))
        short_put_iv = self._normalize_iv(short_put.get("mark_iv", self.default_iv))
        short_call_premium = self._quote_reference_price(short_call)
        short_put_premium = self._quote_reference_price(short_put)

        self._entry_diagnostics.append(
            {
                "entry_time": pd.Timestamp(now),
                "tranche_index": tranche_idx,
                "fraction": float(fraction),
                "spot": spot,
                "rv_24h": None if realized_vol is None else float(realized_vol),
                "short_call_symbol": short_call_name,
                "short_call_delta": float(short_call.get("delta", np.nan)),
                "short_call_iv": short_call_iv,
                "short_call_premium": short_call_premium,
                "short_put_symbol": short_put_name,
                "short_put_delta": float(short_put.get("delta", np.nan)),
                "short_put_iv": short_put_iv,
                "short_put_premium": short_put_premium,
                "combined_short_premium_per_btc": short_call_premium + short_put_premium,
                "avg_short_iv": (short_call_iv + short_put_iv) / 2.0,
            }
        )

        if active_wing_delta > 0 and wing_call is not None and wing_put is not None:
            wing_call_name = str(wing_call["instrument_name"])
            wing_put_name = str(wing_put["instrument_name"])
            legs = {short_call_name, short_put_name, wing_call_name, wing_put_name}
            if len(legs) < 4:
                return
            self.log(
                "Open weekend IC tranche: "
                f"#{tranche_idx} SC={short_call_name} SP={short_put_name} "
                f"LC={wing_call_name} LP={wing_put_name} qty={qty:.3f} fraction={fraction:.0%}"
            )
            context.sell(short_call_name, qty)
            context.sell(short_put_name, qty)
            context.buy(wing_call_name, qty)
            context.buy(wing_put_name, qty)
        else:
            if short_call_name == short_put_name:
                return
            self.log(
                "Open weekend strangle tranche: "
                f"#{tranche_idx} SC={short_call_name} SP={short_put_name} qty={qty:.3f} fraction={fraction:.0%}"
            )
            context.sell(short_call_name, qty)
            context.sell(short_put_name, qty)

        if self._entry_tranches_done == 0:
            self._active_entry_time = pd.Timestamp(now)
            self._active_entry_spot = spot
            self._active_trade_entry_equity = float(context.account.balance)
            self._active_basket_pnl_peak_pct = 0.0
            self._last_trade_week = now.strftime("%G-W%V")
        self._entry_tranches_done += 1
        self._had_positions_prev = True

    def _effective_entry_wing_delta(self, context, now: pd.Timestamp) -> float:
        wing_delta = self.wing_delta
        if self.conditional_wing_rv_threshold > 0 and self.conditional_wing_delta > 0:
            lookback = self.entry_realized_vol_lookback_hours
            if lookback > 1:
                realized_vol = self._recent_realized_vol(context, now, lookback)
                if realized_vol is not None and realized_vol >= self.conditional_wing_rv_threshold:
                    wing_delta = max(wing_delta, self.conditional_wing_delta)
                    self.log(
                        f"Enable conditional wings: RV {realized_vol:.2%} >= {self.conditional_wing_rv_threshold:.2%}, wing_delta={wing_delta:.2f}"
                    )
        return wing_delta

    def _manage_open_position(self, context, now: pd.Timestamp) -> bool:
        total_unrealized = float(sum(p.unrealized_pnl for p in context.positions.values()))
        equity = float(context.account.equity(total_unrealized))

        basket_pnl_pct = self._basket_pnl_pct(context)
        week_id = now.strftime("%G-W%V")

        if basket_pnl_pct is not None:
            self._active_basket_pnl_peak_pct = max(self._active_basket_pnl_peak_pct, basket_pnl_pct)

        if (
            basket_pnl_pct is not None
            and self.trail_profit_activation_pct > 0
            and self.trail_profit_giveback_pct > 0
            and self._last_trail_profit_week != week_id
            and self._active_basket_pnl_peak_pct >= self.trail_profit_activation_pct
            and basket_pnl_pct <= self._active_basket_pnl_peak_pct - self.trail_profit_giveback_pct
        ):
            self.log(
                f"Trail profit reduce: peak {self._active_basket_pnl_peak_pct:.1f}% -> now {basket_pnl_pct:.1f}% "
                f"(giveback {self._active_basket_pnl_peak_pct - basket_pnl_pct:.1f}%)"
            )
            self._close_positions_by_fraction(context, self.trail_profit_close_fraction)
            self._last_trail_profit_week = week_id
            return True

        if (
            self.scheduled_reduce_fraction > 0
            and self._last_scheduled_reduce_week != week_id
            and self._is_scheduled_reduce_time(now)
        ):
            self.log(
                f"Scheduled reduce: close {self.scheduled_reduce_fraction:.0%} at {now.isoformat()}"
            )
            self._close_positions_by_fraction(context, self.scheduled_reduce_fraction)
            self._last_scheduled_reduce_week = week_id
            return True

        if (
            basket_pnl_pct is not None
            and self.early_profit_take_profit_pct > 0
            and self._is_early_profit_check_time(now)
            and self._last_early_profit_action_week != week_id
            and basket_pnl_pct >= self.early_profit_take_profit_pct
        ):
            self.log(
                f"Early weekend profit close: {basket_pnl_pct:.1f}% >= {self.early_profit_take_profit_pct:.1f}%"
            )
            self._close_positions_by_fraction(context, self.early_profit_close_fraction)
            self._last_early_profit_action_week = week_id
            return True

        if (
            basket_pnl_pct is not None
            and self.secondary_profit_take_profit_pct > 0
            and self._is_secondary_profit_check_time(now)
            and self._last_secondary_profit_action_week != week_id
            and basket_pnl_pct >= self.secondary_profit_take_profit_pct
        ):
            self.log(
                f"Secondary weekend profit close: {basket_pnl_pct:.1f}% >= {self.secondary_profit_take_profit_pct:.1f}%"
            )
            self._close_positions_by_fraction(context, self.secondary_profit_close_fraction)
            self._last_secondary_profit_action_week = week_id
            return True

        if (
            self.reactive_hedge_move_pct > 0
            and self.reactive_hedge_delta > 0
            and self.reactive_hedge_fraction > 0
            and self._active_entry_spot
            and self._active_entry_spot > 0
            and self._last_reactive_hedge_week != week_id
        ):
            spot_move_signed = float(context.underlying_price) / self._active_entry_spot - 1.0
            if abs(spot_move_signed) >= self.reactive_hedge_move_pct:
                hedge_side = "call" if spot_move_signed > 0 else "put"
                if self._open_reactive_hedge(context, now, hedge_side):
                    self.log(
                        f"Reactive hedge: spot move {spot_move_signed:.2%}, buy {hedge_side} hedge at |delta|≈{self.reactive_hedge_delta:.2f}"
                    )
                    self._last_reactive_hedge_week = week_id
                    return False

        if (
            self.directional_adjust_move_pct > 0
            and self._active_entry_spot
            and self._active_entry_spot > 0
            and self._is_directional_adjust_time(now)
            and self._last_directional_adjust_week != week_id
        ):
            spot_move_signed = float(context.underlying_price) / self._active_entry_spot - 1.0
            if abs(spot_move_signed) >= self.directional_adjust_move_pct:
                threatened_side = "call" if spot_move_signed > 0 else "put"
                reduced = self._reduce_option_side(context, threatened_side, self.directional_adjust_close_fraction)
                if reduced:
                    self.log(
                        f"Directional adjust: spot move {spot_move_signed:.2%}, reduce {threatened_side} side by {self.directional_adjust_close_fraction:.0%}"
                    )
                    self._last_directional_adjust_week = week_id
                    return True

        if self.underlying_move_stop_pct > 0 and self._active_entry_spot and self._active_entry_spot > 0:
            spot_move = abs(float(context.underlying_price) / self._active_entry_spot - 1.0)
            if spot_move >= self.underlying_move_stop_pct:
                self.log(
                    f"Close weekend basket: spot move {spot_move:.2%} "
                    f">= {self.underlying_move_stop_pct:.2%} from entry"
                )
                self._queue_exit_diagnostic("underlying_move_stop", now, basket_pnl_pct, context, total_unrealized)
                context.close_all()
                return True

        if self.max_loss_equity_pct > 0 and equity > 0:
            loss_ratio = max(0.0, -total_unrealized) / equity
            if loss_ratio >= self.max_loss_equity_pct:
                self.log(
                    f"Close weekend basket: unrealized loss/equity {loss_ratio:.2%} "
                    f">= {self.max_loss_equity_pct:.2%}"
                )
                self._queue_exit_diagnostic("max_loss_equity", now, basket_pnl_pct, context, total_unrealized)
                context.close_all()
                return True

        if basket_pnl_pct is None:
            return False

        if self.take_profit_pct > 0 and basket_pnl_pct >= self.take_profit_pct:
            self.log(
                f"Take profit on weekend basket: {basket_pnl_pct:.1f}% >= {self.take_profit_pct:.1f}%"
            )
            self._queue_exit_diagnostic("take_profit", now, basket_pnl_pct, context, total_unrealized)
            context.close_all()
            return True

        if self.stop_loss_pct > 0 and basket_pnl_pct <= -self.stop_loss_pct:
            if self.stop_loss_underlying_move_pct > 0 and self._active_entry_spot and self._active_entry_spot > 0:
                spot_move = abs(float(context.underlying_price) / self._active_entry_spot - 1.0)
                if spot_move < self.stop_loss_underlying_move_pct:
                    return False
            self.log(
                f"Stop loss on weekend basket: {basket_pnl_pct:.1f}% <= -{self.stop_loss_pct:.1f}%"
            )
            self._queue_exit_diagnostic("stop_loss", now, basket_pnl_pct, context, total_unrealized)
            context.close_all()
            return True

        return False

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

    def _close_positions_by_fraction(self, context, fraction: float) -> None:
        fraction = float(fraction)
        if fraction >= 0.999:
            context.close_all()
            return
        if fraction <= 0:
            return
        for name in list(context.positions.keys()):
            context.close_partial(name, fraction=fraction)

    def _reduce_option_side(self, context, option_type: str, fraction: float) -> bool:
        reduced = False
        for name in list(context.positions.keys()):
            if self._position_option_type(context, name) != option_type:
                continue
            context.close_partial(name, fraction=fraction)
            reduced = True
        return reduced

    def _position_option_type(self, context, instrument_name: str) -> str | None:
        engine = getattr(context, "_engine", None)
        inst = getattr(engine, "_instrument_dict", {}).get(instrument_name) if engine is not None else None
        option_type = None
        if inst is not None:
            option_type = inst.get("option_type")
        if option_type is None:
            parts = str(instrument_name).split("-")
            if parts:
                suffix = parts[-1].lower()
                if suffix.startswith("c"):
                    option_type = "call"
                elif suffix.startswith("p"):
                    option_type = "put"
        if option_type is None:
            return None
        option_type = str(option_type).lower()
        return "call" if option_type.startswith("c") else "put"

    def _open_reactive_hedge(self, context, now: pd.Timestamp, option_type: str) -> bool:
        chain = context.option_chain
        if chain.empty:
            return False
        chain_df = self._filter_target_expiry(chain, now)
        if chain_df.empty:
            return False

        boundary_strike = self._current_short_side_boundary(context, option_type)
        short_qty = self._current_short_side_qty(context, option_type)
        if boundary_strike is None or short_qty <= 0:
            return False

        side_df = chain_df.loc[
            chain_df["option_type"].astype(str).str.lower().str.startswith(option_type[0])
        ].copy()
        if side_df.empty:
            return False

        if option_type == "call":
            side_df = side_df.loc[pd.to_numeric(side_df["strike_price"], errors="coerce") > boundary_strike]
        else:
            side_df = side_df.loc[pd.to_numeric(side_df["strike_price"], errors="coerce") < boundary_strike]
        if side_df.empty:
            return False

        hedge_leg = self._select_leg(
            side_df,
            float(context.underlying_price),
            now,
            option_type,
            self.reactive_hedge_delta,
        )
        if hedge_leg is None:
            return False

        hedge_name = str(hedge_leg["instrument_name"])
        if hedge_name in context.positions:
            return False

        hedge_qty = short_qty * self.reactive_hedge_fraction
        step = self.quantity_step if self.quantity_step > 0 else self.quantity
        if step > 0:
            hedge_qty = math.floor(hedge_qty / step) * step
        hedge_qty = max(0.0, hedge_qty)
        if hedge_qty <= 0:
            return False

        context.buy(hedge_name, hedge_qty)
        return True

    def _current_short_side_qty(self, context, option_type: str) -> float:
        total = 0.0
        for name, pos in context.positions.items():
            if pos.direction.value != "short":
                continue
            if self._position_option_type(context, name) != option_type:
                continue
            total += float(pos.quantity)
        return total

    def _current_short_side_boundary(self, context, option_type: str) -> float | None:
        strikes: list[float] = []
        engine = getattr(context, "_engine", None)
        instrument_dict = getattr(engine, "_instrument_dict", {}) if engine is not None else {}
        for name, pos in context.positions.items():
            if pos.direction.value != "short":
                continue
            if self._position_option_type(context, name) != option_type:
                continue
            inst = instrument_dict.get(name, {})
            strike = inst.get("strike_price")
            try:
                strike = float(strike)
            except Exception:
                strike = np.nan
            if np.isfinite(strike):
                strikes.append(float(strike))
        if not strikes:
            return None
        return max(strikes) if option_type == "call" else min(strikes)

    def _is_early_profit_check_time(self, now: pd.Timestamp) -> bool:
        if not self.early_profit_close_day or self.early_profit_close_hour < 0:
            return False
        target_day = self.DAY_MAP.get(self.early_profit_close_day, -1)
        if target_day < 0 or now.weekday() != target_day:
            return False
        return now.hour == self.early_profit_close_hour and now.minute == self.early_profit_close_minute

    def _is_scheduled_reduce_time(self, now: pd.Timestamp) -> bool:
        if not self.scheduled_reduce_day or self.scheduled_reduce_hour < 0:
            return False
        target_day = self.DAY_MAP.get(self.scheduled_reduce_day, -1)
        if target_day < 0 or now.weekday() != target_day:
            return False
        return now.hour == self.scheduled_reduce_hour and now.minute == self.scheduled_reduce_minute

    def _is_directional_adjust_time(self, now: pd.Timestamp) -> bool:
        if not self.directional_adjust_day or self.directional_adjust_hour < 0:
            return False
        target_day = self.DAY_MAP.get(self.directional_adjust_day, -1)
        if target_day < 0 or now.weekday() != target_day:
            return False
        return now.hour == self.directional_adjust_hour and now.minute == self.directional_adjust_minute

    def _is_secondary_profit_check_time(self, now: pd.Timestamp) -> bool:
        if not self.secondary_profit_close_day or self.secondary_profit_close_hour < 0:
            return False
        target_day = self.DAY_MAP.get(self.secondary_profit_close_day, -1)
        if target_day < 0 or now.weekday() != target_day:
            return False
        return now.hour == self.secondary_profit_close_hour and now.minute == self.secondary_profit_close_minute

    def _passes_entry_filters(self, context, now: pd.Timestamp) -> bool:
        if self.entry_abs_move_lookback_hours > 0 and self.entry_abs_move_max_pct > 0:
            abs_move = self._recent_abs_move_pct(context, now, self.entry_abs_move_lookback_hours)
            if abs_move is None:
                return False
            if abs_move > self.entry_abs_move_max_pct:
                self.log(
                    f"Skip entry: {self.entry_abs_move_lookback_hours}h abs move {abs_move:.2%} "
                    f"> {self.entry_abs_move_max_pct:.2%}"
                )
                return False

        if self.entry_realized_vol_lookback_hours > 1 and self.entry_realized_vol_max > 0:
            realized_vol = self._recent_realized_vol(context, now, self.entry_realized_vol_lookback_hours)
            if realized_vol is None:
                return False
            if realized_vol > self.entry_realized_vol_max:
                self.log(
                    f"Skip entry: {self.entry_realized_vol_lookback_hours}h RV {realized_vol:.2%} "
                    f"> {self.entry_realized_vol_max:.2%}"
                )
                return False

        return True

    def _recent_abs_move_pct(self, context, now: pd.Timestamp, lookback_hours: int) -> float | None:
        window = self._get_underlying_window(context, now, lookback_hours)
        if window is None or len(window) < 2:
            return None
        start_price = float(window[0])
        end_price = float(window[-1])
        if start_price <= 0 or not np.isfinite(start_price) or not np.isfinite(end_price):
            return None
        return abs(end_price / start_price - 1.0)

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