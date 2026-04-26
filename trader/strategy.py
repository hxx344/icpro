"""Weekend vol live trading strategy – 周末波动率卖方策略.

当前实盘交易端仅保留 `WeekendVolStrategy`：
- 每周指定时间入场，持有至周日 08:00 UTC 结算
- Delta 选短腿，可按 `wing_delta` 决定是否带保护翼
- 支持 Bybit UTA 保证金估算、篮子止损、执行风控锁
- 适用于 Bybit BTC/ETH
"""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
from typing import Any

from loguru import logger

from trader.bybit_client import BybitOptionsClient, OptionTicker, _parse_symbol
from trader.config import ExchangeConfig, StrategyConfig
from trader.position_manager import PositionManager
from trader.storage import Storage


BYBIT_OPTION_DEFAULT_CONTRACT_UNIT = 1.0
BYBIT_OPTION_DEFAULT_TAKER_FEE_RATE = 0.0003
BYBIT_OPTION_DEFAULT_FEE_CAP_RATIO = 0.07
BYBIT_OPTION_DEFAULT_SHORT_MARGIN_RATIO = 0.10
BYBIT_OPTION_DEFAULT_SHORT_OTM_DEDUCTION_RATIO = 0.08
BYBIT_OPTION_DEFAULT_SHORT_MIN_MARGIN_RATIO = 0.05
WEEKEND_VOL_ENTRY_RETRY_WINDOW_MINUTES = 10


def _resolve_exchange_param(exchange_cfg: ExchangeConfig | None, field_name: str, default: float) -> float:
    if exchange_cfg is None:
        return default
    try:
        return float(getattr(exchange_cfg, field_name, default) or default)
    except Exception:
        return default


def _bybit_option_otm_amount(index_price: float, strike: float, option_type: str) -> float:
    option_type_norm = str(option_type or "").lower()
    if option_type_norm == "call":
        return max(float(strike or 0.0) - float(index_price or 0.0), 0.0)
    return max(float(index_price or 0.0) - float(strike or 0.0), 0.0)


def estimate_bybit_option_fee(
    index_price: float,
    order_price: float,
    contract_unit: float = BYBIT_OPTION_DEFAULT_CONTRACT_UNIT,
    transaction_fee_rate: float = BYBIT_OPTION_DEFAULT_TAKER_FEE_RATE,
    fee_cap_ratio: float = BYBIT_OPTION_DEFAULT_FEE_CAP_RATIO,
) -> float:
    index_component = max(float(index_price or 0.0), 0.0) * max(float(contract_unit or 0.0), 0.0) * max(float(transaction_fee_rate or 0.0), 0.0)
    order_component = max(float(fee_cap_ratio or 0.0), 0.0) * max(float(order_price or 0.0), 0.0)
    if index_component <= 0:
        return order_component
    if order_component <= 0:
        return index_component
    return min(index_component, order_component)


def estimate_bybit_short_open_margin_per_unit(
    index_price: float,
    strike: float,
    option_type: str,
    mark_price: float,
    order_price: float,
    contract_unit: float = BYBIT_OPTION_DEFAULT_CONTRACT_UNIT,
    transaction_fee_rate: float = BYBIT_OPTION_DEFAULT_TAKER_FEE_RATE,
    fee_cap_ratio: float = BYBIT_OPTION_DEFAULT_FEE_CAP_RATIO,
    short_margin_ratio: float = BYBIT_OPTION_DEFAULT_SHORT_MARGIN_RATIO,
    short_otm_deduction_ratio: float = BYBIT_OPTION_DEFAULT_SHORT_OTM_DEDUCTION_RATIO,
    short_min_margin_ratio: float = BYBIT_OPTION_DEFAULT_SHORT_MIN_MARGIN_RATIO,
) -> float:
    index_price = max(float(index_price or 0.0), 0.0)
    contract_unit = max(float(contract_unit or 0.0), 0.0)
    mark_price = max(float(mark_price or 0.0), 0.0)
    order_price = max(float(order_price or 0.0), 0.0)
    otm_amount = _bybit_option_otm_amount(index_price, strike, option_type)
    base_margin = max(
        short_min_margin_ratio * index_price,
        short_margin_ratio * index_price - short_otm_deduction_ratio * otm_amount,
    ) * contract_unit
    fee = estimate_bybit_option_fee(index_price, order_price, contract_unit, transaction_fee_rate, fee_cap_ratio)
    return max(0.0, base_margin + mark_price - order_price) + fee


def estimate_bybit_long_open_margin_per_unit(
    index_price: float,
    order_price: float,
    contract_unit: float = BYBIT_OPTION_DEFAULT_CONTRACT_UNIT,
    transaction_fee_rate: float = BYBIT_OPTION_DEFAULT_TAKER_FEE_RATE,
    fee_cap_ratio: float = BYBIT_OPTION_DEFAULT_FEE_CAP_RATIO,
) -> float:
    order_price = max(float(order_price or 0.0), 0.0)
    fee = estimate_bybit_option_fee(index_price, order_price, contract_unit, transaction_fee_rate, fee_cap_ratio)
    return order_price + fee


def estimate_bybit_combo_open_margin_per_unit(
    index_price: float,
    sell_call: OptionTicker | None = None,
    sell_put: OptionTicker | None = None,
    buy_call: OptionTicker | None = None,
    buy_put: OptionTicker | None = None,
    exchange_cfg: ExchangeConfig | None = None,
    contract_unit: float = BYBIT_OPTION_DEFAULT_CONTRACT_UNIT,
) -> float:
    transaction_fee_rate = _resolve_exchange_param(exchange_cfg, "option_taker_fee_rate", BYBIT_OPTION_DEFAULT_TAKER_FEE_RATE)
    fee_cap_ratio = _resolve_exchange_param(exchange_cfg, "option_fee_cap_ratio", BYBIT_OPTION_DEFAULT_FEE_CAP_RATIO)
    short_margin_ratio = _resolve_exchange_param(exchange_cfg, "short_option_margin_ratio", BYBIT_OPTION_DEFAULT_SHORT_MARGIN_RATIO)
    short_otm_deduction_ratio = _resolve_exchange_param(exchange_cfg, "short_option_otm_deduction_ratio", BYBIT_OPTION_DEFAULT_SHORT_OTM_DEDUCTION_RATIO)
    short_min_margin_ratio = _resolve_exchange_param(exchange_cfg, "short_option_min_margin_ratio", BYBIT_OPTION_DEFAULT_SHORT_MIN_MARGIN_RATIO)
    total = 0.0
    for short_leg in (sell_call, sell_put):
        if short_leg is None:
            continue
        mark_price = float(short_leg.mark_price or short_leg.bid_price or short_leg.ask_price or 0.0)
        order_price = float(short_leg.bid_price or short_leg.mark_price or 0.0)
        total += estimate_bybit_short_open_margin_per_unit(
            index_price=index_price,
            strike=float(short_leg.strike or 0.0),
            option_type=str(short_leg.option_type or ""),
            mark_price=mark_price,
            order_price=order_price,
            contract_unit=contract_unit,
            transaction_fee_rate=transaction_fee_rate,
            fee_cap_ratio=fee_cap_ratio,
            short_margin_ratio=short_margin_ratio,
            short_otm_deduction_ratio=short_otm_deduction_ratio,
            short_min_margin_ratio=short_min_margin_ratio,
        )

    for long_leg in (buy_call, buy_put):
        if long_leg is None:
            continue
        order_price = float(long_leg.ask_price or long_leg.mark_price or 0.0)
        total += estimate_bybit_long_open_margin_per_unit(
            index_price=index_price,
            order_price=order_price,
            contract_unit=contract_unit,
            transaction_fee_rate=transaction_fee_rate,
            fee_cap_ratio=fee_cap_ratio,
        )
    return total


# ======================================================================
# Weekend Volatility Selling Strategy
# ======================================================================

class WeekendVolStrategy:
    """Weekend volatility selling strategy on Bybit.

    Strategy logic (mirrors backtest_weekend_vol.py):
    ─────────────────────────────────────────────────
    1. Every **Friday at entry_time_utc** (default 16:00 UTC):
       a. Fetch BTC option chain for the coming **Sunday 08:00 UTC** expiry
       b. Compute Black-76 delta for each strike using mark_iv or default_iv
       c. Select short call: closest strike to |delta| = target_delta
       d. Select short put:  closest strike to |delta| = target_delta
         e. Select wing call: closest strike to |delta| = wing_delta (optional protection)
         f. Select wing put: closest strike to |delta| = wing_delta (optional protection)
    g. Use fixed quantity from config
    h. Execute weekend-vol position via PositionManager

     2. During holding period:
         a. Monitor basket PnL using option mark prices
         b. If basket PnL% <= -stop_loss_pct, close all legs early
         c. Otherwise hold to settlement

    Parameters (from StrategyConfig):
    ──────────────────────────────────
    - target_delta : 0.40 (|δ| for short legs)
    - wing_delta   : 0.05 (|δ| for protection legs, 0 → strangle)
    - max_delta_diff : 0.20 (actual |δ| vs target |δ| max deviation)
    - leverage     : 1.0 (仅用于预览展示)
    - entry_day    : "friday"
    - entry_time_utc : "18:00"
    - underlying   : "BTC"
    - quantity     : fixed option size
    """

    DAY_MAP = {
        "monday": 0, "tuesday": 1, "wednesday": 2,
        "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
    }

    def __init__(
        self,
        client: BybitOptionsClient,
        position_mgr: PositionManager,
        storage: Storage,
        config: StrategyConfig,
    ):
        self.client = client
        self.pos_mgr = position_mgr
        self.storage = storage
        self.cfg = config

        # Recovery: last trade week identifier to avoid duplicate entry
        self._last_trade_week: str = self.storage.load_state(
            "wv_last_trade_week", ""
        )
        self._last_order_attempt_at: str = ""
        self._last_order_status: str = "idle"
        self._last_order_message: str = ""
        self._last_order_group_id: str = ""
        self._last_order_exchange_positions_checked: bool = False
        self._last_order_exchange_positions_underlying: str = ""
        self._last_order_exchange_positions: list[dict[str, Any]] = []
        self._last_order_exchange_positions_error: str = ""
        self._last_guard_exchange_positions: list[dict[str, Any]] = []
        self._last_guard_exchange_positions_error: str = ""
        self._execution_risk_lock_active: bool = bool(self.storage.load_state("wv_execution_risk_lock_active", False))
        self._execution_risk_lock_since: str = str(self.storage.load_state("wv_execution_risk_lock_since", "") or "")
        self._execution_risk_lock_reason: str = str(self.storage.load_state("wv_execution_risk_lock_reason", "") or "")
        self._execution_risk_lock_group_id: str = str(self.storage.load_state("wv_execution_risk_lock_group_id", "") or "")
        self._execution_risk_lock_event: str = str(self.storage.load_state("wv_execution_risk_lock_event", "") or "")
        recovery = self.pos_mgr.sync_exchange_positions_to_local(self.cfg.underlying, replace_on_conflict=True)
        if recovery.get("restored"):
            week_id = datetime.now(timezone.utc).strftime("%G-W%V")
            self._last_trade_week = week_id
            self.storage.save_state("wv_last_trade_week", week_id)
            self._last_order_status = "recovered"
            self._last_order_message = f"启动时已从交易所恢复持仓: {', '.join(recovery.get('symbols') or [])}"
            logger.warning(
                f"[WeekendVol] Recovered exchange positions into local storage on startup: {recovery.get('symbols') or []}"
            )
        if self._execution_risk_lock_active:
            logger.warning(
                f"[WeekendVol] Execution risk lock active since {self._execution_risk_lock_since or '-'}: "
                f"{self._execution_risk_lock_reason or 'unknown'}"
            )

    def _set_execution_risk_lock(
        self,
        reason: str,
        event_type: str,
        group_id: str = "",
    ) -> None:
        reason_text = str(reason or "执行风控锁已触发")
        since = datetime.now(timezone.utc).isoformat()
        self._execution_risk_lock_active = True
        self._execution_risk_lock_since = since
        self._execution_risk_lock_reason = reason_text
        self._execution_risk_lock_group_id = str(group_id or "")
        self._execution_risk_lock_event = str(event_type or "")
        self.storage.save_state("wv_execution_risk_lock_active", True)
        self.storage.save_state("wv_execution_risk_lock_since", since)
        self.storage.save_state("wv_execution_risk_lock_reason", reason_text)
        self.storage.save_state("wv_execution_risk_lock_group_id", self._execution_risk_lock_group_id)
        self.storage.save_state("wv_execution_risk_lock_event", self._execution_risk_lock_event)
        self.storage.record_execution_event(
            event_type="execution_risk_lock_set",
            execution_state="risk_locked",
            severity="error",
            group_id=self._execution_risk_lock_group_id,
            underlying=self.cfg.underlying.upper(),
            execution_mode="market",
            message=reason_text,
            meta={"source_event": self._execution_risk_lock_event},
        )
        logger.error(f"[WeekendVol] Execution risk lock set: {reason_text}")

    def clear_execution_risk_lock(self) -> None:
        previous_reason = self._execution_risk_lock_reason or "manual_clear"
        self._execution_risk_lock_active = False
        self._execution_risk_lock_since = ""
        self._execution_risk_lock_reason = ""
        self._execution_risk_lock_group_id = ""
        self._execution_risk_lock_event = ""
        self.storage.save_state("wv_execution_risk_lock_active", False)
        self.storage.save_state("wv_execution_risk_lock_since", "")
        self.storage.save_state("wv_execution_risk_lock_reason", "")
        self.storage.save_state("wv_execution_risk_lock_group_id", "")
        self.storage.save_state("wv_execution_risk_lock_event", "")
        self.storage.record_execution_event(
            event_type="execution_risk_lock_cleared",
            execution_state="idle",
            severity="info",
            underlying=self.cfg.underlying.upper(),
            execution_mode="market",
            message="人工清除执行风控锁",
            meta={"previous_reason": previous_reason},
        )
        logger.warning("[WeekendVol] Execution risk lock cleared manually")

    def _get_exchange_position_guard_snapshot(self) -> tuple[list[dict[str, Any]], str]:
        try:
            positions = self.client.get_positions(self.cfg.underlying.upper())
            if not isinstance(positions, list):
                positions = []
            self._last_guard_exchange_positions = positions
            self._last_guard_exchange_positions_error = ""
            return positions, ""
        except Exception as e:
            self._last_guard_exchange_positions = []
            self._last_guard_exchange_positions_error = str(e)
            return [], str(e)

    # ------------------------------------------------------------------
    # Main entry: called periodically by engine / main loop
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Single strategy tick."""
        now = datetime.now(timezone.utc)

        # 1. Check settlement of open positions
        self._check_settlement(now)

        # 2. Check live stop management
        self._manage_open_positions(now)

        # 3. Check if it's time to open a new position
        if self._should_enter(now):
            self._try_open_weekend_vol_position(now)

    # ------------------------------------------------------------------
    # Entry logic
    # ------------------------------------------------------------------

    def _should_enter(self, now: datetime) -> bool:
        """Determine if we should open a new position now."""
        # Correct day of week?
        target_day = self.DAY_MAP.get(self.cfg.entry_day.lower(), 4)
        if now.weekday() != target_day:
            return False

        # Parse entry time
        parts = self.cfg.entry_time_utc.split(":")
        entry_min = int(parts[0]) * 60 + int(parts[1])
        now_min = now.hour * 60 + now.minute

        # Before entry time?
        if now_min < entry_min:
            return False

        # Too late? (fixed 10-minute retry window after entry)
        if now_min > entry_min + WEEKEND_VOL_ENTRY_RETRY_WINDOW_MINUTES:
            return False

        # Already traded this week?
        week_id = now.strftime("%G-W%V")  # ISO week
        if self._last_trade_week == week_id:
            return False

        # Max positions?
        if self.pos_mgr.open_position_count >= self.cfg.max_positions:
            return False

        # Exchange position guard
        exchange_positions, exchange_error = self._get_exchange_position_guard_snapshot()
        if exchange_positions:
            symbols = [p["symbol"] for p in exchange_positions]
            logger.warning(
                f"[WeekendVol] Exchange has {len(exchange_positions)} "
                f"existing position(s): {symbols} – skipping"
            )
            return False
        if exchange_error:
            logger.error(f"[WeekendVol] Position check failed: {exchange_error} – skipping")
            return False

        if self._execution_risk_lock_active:
            logger.error(
                f"[WeekendVol] Execution risk lock active since {self._execution_risk_lock_since or '-'}: "
                f"{self._execution_risk_lock_reason or 'unknown'} – skipping"
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Core: open weekend volatility position via delta-based strike selection
    # ------------------------------------------------------------------

    def _try_open_weekend_vol_position(self, now: datetime) -> None:
        """Fetch option chain, select strikes by delta, execute combo position."""
        ul = self.cfg.underlying.upper()
        self._mark_order_attempt("opening", f"开始扫描 {ul} 周末合约")
        logger.info(f"[WeekendVol] Scanning {ul} weekend options for target combo...")

        rv24 = self._get_entry_realized_vol(ul)
        if not self._passes_entry_rv_filter(rv24):
            return

        initial_candidate = self._resolve_entry_candidate(now, ul)
        if initial_candidate is None:
            return
        sell_call, sell_put, buy_call, buy_put, spot, T_years = initial_candidate

        refreshed_candidate = self._resolve_entry_candidate(now, ul)
        if refreshed_candidate is None:
            logger.warning("[WeekendVol] Final pre-trade refresh failed – skipping current attempt")
            return

        refreshed_sell_call, refreshed_sell_put, refreshed_buy_call, refreshed_buy_put, refreshed_spot, refreshed_T_years = refreshed_candidate
        original_symbols = (
            sell_put.symbol,
            sell_call.symbol,
            buy_put.symbol if buy_put else "",
            buy_call.symbol if buy_call else "",
        )
        refreshed_symbols = (
            refreshed_sell_put.symbol,
            refreshed_sell_call.symbol,
            refreshed_buy_put.symbol if refreshed_buy_put else "",
            refreshed_buy_call.symbol if refreshed_buy_call else "",
        )
        if refreshed_symbols != original_symbols:
            logger.warning(
                "[WeekendVol] Entry legs changed on final refresh; switching to latest snapshot: "
                f"old={original_symbols} new={refreshed_symbols}"
            )

        sell_call = refreshed_sell_call
        sell_put = refreshed_sell_put
        buy_call = refreshed_buy_call
        buy_put = refreshed_buy_put
        spot = refreshed_spot
        T_years = refreshed_T_years

        # --- Compute quantity ---
        quantity = self._compute_quantity(
            spot,
            sell_call=sell_call,
            sell_put=sell_put,
            buy_call=buy_call,
            buy_put=buy_put,
        )
        if quantity <= 0:
            logger.warning("[WeekendVol] Computed quantity <= 0, skipping")
            return

        # --- Execute ---
        week_id = now.strftime("%G-W%V")

        if self.cfg.wing_delta > 0 and buy_call and buy_put:
            logger.info(
                f"[WeekendVol] Opening winged weekend-vol structure:\n"
                f"  Sell Call: {sell_call.symbol} K={sell_call.strike}\n"
                f"  Buy Call:  {buy_call.symbol} K={buy_call.strike}\n"
                f"  Sell Put:  {sell_put.symbol} K={sell_put.strike}\n"
                f"  Buy Put:   {buy_put.symbol} K={buy_put.strike}\n"
                f"  Qty: {quantity:.3f}  Leverage: {self.cfg.leverage}x  "
                f"T={T_years*365.25:.1f}d"
            )
            try:
                condor = self.pos_mgr.open_winged_position(
                    sell_call_symbol=sell_call.symbol,
                    buy_call_symbol=buy_call.symbol,
                    sell_put_symbol=sell_put.symbol,
                    buy_put_symbol=buy_put.symbol,
                    sell_call_strike=sell_call.strike,
                    buy_call_strike=buy_call.strike,
                    sell_put_strike=sell_put.strike,
                    buy_put_strike=buy_put.strike,
                    quantity=quantity,
                    underlying_price=spot,
                )
            except Exception as e:
                self._record_live_order_failure(
                    underlying=ul,
                    message=f"实盘开仓异常: {e}",
                    status="error",
                )
                raise
        else:
            # No wings → wingless weekend-vol structure
            logger.info(
                f"[WeekendVol] Opening wingless weekend-vol structure:\n"
                f"  Sell Call: {sell_call.symbol} K={sell_call.strike}\n"
                f"  Sell Put:  {sell_put.symbol} K={sell_put.strike}\n"
                f"  Qty: {quantity:.3f}  Leverage: {self.cfg.leverage}x  "
                f"T={T_years*365.25:.1f}d"
            )
            try:
                condor = self.pos_mgr.open_short_strangle(
                    sell_call_symbol=sell_call.symbol,
                    sell_put_symbol=sell_put.symbol,
                    sell_call_strike=sell_call.strike,
                    sell_put_strike=sell_put.strike,
                    quantity=quantity,
                    underlying_price=spot,
                )
            except Exception as e:
                self._record_live_order_failure(
                    underlying=ul,
                    message=f"实盘开仓异常: {e}",
                    status="error",
                )
                raise

        if condor:
            self._last_trade_week = week_id
            self.storage.save_state("wv_last_trade_week", week_id)
            expected_legs = 4 if self.cfg.wing_delta > 0 and buy_call and buy_put else 2
            if len(condor.legs) < expected_legs:
                self._mark_order_partial(
                    group_id=condor.group_id,
                    message=f"实盘部分开仓: {condor.group_id}，已成交 {len(condor.legs)}/{expected_legs} 条腿",
                )
                logger.warning(
                    f"[WeekendVol] Position partially opened: {condor.group_id} "
                    f"legs={len(condor.legs)}/{expected_legs} premium={condor.total_premium:.6f}"
                )
            else:
                self._mark_order_success(
                    group_id=condor.group_id,
                    message=f"实盘开仓成功: {condor.group_id}",
                )
                logger.info(
                    f"[WeekendVol] Position opened: {condor.group_id} "
                    f"premium={condor.total_premium:.6f}"
                )
        else:
            self._record_live_order_failure(
                underlying=ul,
                message="实盘开仓失败：至少有一条腿未成交，或成交后本地记账失败",
                status="failed",
            )
            logger.error("[WeekendVol] Failed to open position")

    def _resolve_entry_candidate(
        self,
        now: datetime,
        underlying: str,
    ) -> tuple[OptionTicker, OptionTicker, OptionTicker | None, OptionTicker | None, float, float] | None:
        snapshot = self._load_weekend_market_snapshot(now, underlying)
        if snapshot is None:
            return None

        calls, puts, spot, T_years = snapshot
        selected = self._select_entry_legs(calls, puts, spot, T_years)
        if selected is None:
            return None

        sell_call, sell_put, buy_call, buy_put = selected
        return sell_call, sell_put, buy_call, buy_put, spot, T_years

    def _load_weekend_market_snapshot(
        self,
        now: datetime,
        underlying: str,
    ) -> tuple[list[OptionTicker], list[OptionTicker], float, float] | None:
        try:
            tickers = self.client.get_tickers(underlying)
        except Exception as e:
            logger.error(f"[WeekendVol] Failed to fetch tickers: {e}")
            return None

        if not tickers:
            logger.warning("[WeekendVol] No tickers available")
            return None

        spot = self.client.get_spot_price(underlying)
        if spot <= 0:
            prices = [t.underlying_price for t in tickers if t.underlying_price > 0]
            spot = prices[0] if prices else 0
        if spot <= 0:
            logger.error("[WeekendVol] Cannot determine spot price")
            return None

        logger.info(f"[WeekendVol] {underlying} spot = {spot:.2f}")

        sunday_expiry = self._next_sunday_0800(now)
        tolerance_hours = 2.0
        weekend_tickers = [
            t for t in tickers
            if abs((t.expiry - sunday_expiry).total_seconds()) < tolerance_hours * 3600
        ]
        if not weekend_tickers:
            logger.warning(
                f"[WeekendVol] No options expiring around {sunday_expiry.isoformat()} "
                f"(found {len(tickers)} total tickers)"
            )
            return None

        logger.info(
            f"[WeekendVol] Found {len(weekend_tickers)} tickers for Sunday expiry "
            f"{sunday_expiry.strftime('%Y-%m-%d %H:%M')} UTC"
        )

        try:
            self.client.enrich_greeks(weekend_tickers, underlying)
        except Exception as e:
            logger.warning(f"[WeekendVol] Failed to fetch greeks, will use Black-76 fallback: {e}")

        calls = [t for t in weekend_tickers if t.option_type == "call"]
        puts = [t for t in weekend_tickers if t.option_type == "put"]
        T_years = max((sunday_expiry - now).total_seconds() / (365.25 * 86400), 1e-6)
        return calls, puts, spot, T_years

    def _select_entry_legs(
        self,
        calls: list[OptionTicker],
        puts: list[OptionTicker],
        spot: float,
        T_years: float,
    ) -> tuple[OptionTicker, OptionTicker, OptionTicker | None, OptionTicker | None] | None:
        sell_call = self._find_by_delta(calls, spot, T_years, self.cfg.target_delta, "call")
        sell_put = self._find_by_delta(puts, spot, T_years, self.cfg.target_delta, "put")

        if not sell_call or not sell_put:
            logger.warning("[WeekendVol] Could not find both short legs by delta")
            return None

        if sell_call.strike <= spot or sell_put.strike >= spot:
            logger.warning(
                f"[WeekendVol] Reject non-OTM short legs: "
                f"call K={sell_call.strike} spot={spot}, put K={sell_put.strike} spot={spot}"
            )
            return None

        if sell_call.bid_price <= 0 or sell_put.bid_price <= 0:
            logger.warning("[WeekendVol] Short legs have no bid – skipping")
            return None

        buy_call: OptionTicker | None = None
        buy_put: OptionTicker | None = None
        if self.cfg.wing_delta > 0:
            wing_calls = [t for t in calls if float(t.strike) > float(sell_call.strike)]
            wing_puts = [t for t in puts if float(t.strike) < float(sell_put.strike)]
            buy_call = self._find_by_delta(wing_calls, spot, T_years, self.cfg.wing_delta, "call")
            buy_put = self._find_by_delta(wing_puts, spot, T_years, self.cfg.wing_delta, "put")

            if not buy_call or not buy_put:
                logger.warning("[WeekendVol] Could not find wing legs by delta")
                return None

            if sell_call.strike >= buy_call.strike:
                logger.warning(
                    f"[WeekendVol] Invalid call spread: sell K={sell_call.strike} "
                    f">= buy K={buy_call.strike}"
                )
                return None
            if sell_put.strike <= buy_put.strike:
                logger.warning(
                    f"[WeekendVol] Invalid put spread: sell K={sell_put.strike} "
                    f"<= buy K={buy_put.strike}"
                )
                return None

        return sell_call, sell_put, buy_call, buy_put

    def _mark_order_attempt(self, status: str, message: str, group_id: str = "") -> None:
        self._last_order_attempt_at = datetime.now(timezone.utc).isoformat()
        self._last_order_status = status
        self._last_order_message = message
        self._last_order_group_id = group_id
        self._last_order_exchange_positions_checked = False
        self._last_order_exchange_positions_underlying = ""
        self._last_order_exchange_positions = []
        self._last_order_exchange_positions_error = ""

    def _mark_order_success(self, group_id: str, message: str) -> None:
        self._last_order_attempt_at = datetime.now(timezone.utc).isoformat()
        self._last_order_status = "success"
        self._last_order_message = message
        self._last_order_group_id = group_id
        self._last_order_exchange_positions_checked = False
        self._last_order_exchange_positions_underlying = ""
        self._last_order_exchange_positions = []
        self._last_order_exchange_positions_error = ""

    def _mark_order_partial(self, group_id: str, message: str) -> None:
        self._last_order_attempt_at = datetime.now(timezone.utc).isoformat()
        self._last_order_status = "partial"
        self._last_order_message = message
        self._last_order_group_id = group_id
        self._last_order_exchange_positions_checked = False
        self._last_order_exchange_positions_underlying = ""
        self._last_order_exchange_positions = []
        self._last_order_exchange_positions_error = ""
        self._set_execution_risk_lock(message, event_type="position_open_partial", group_id=group_id)

    def _record_live_order_failure(self, underlying: str, message: str, status: str) -> None:
        self._last_order_attempt_at = datetime.now(timezone.utc).isoformat()
        self._last_order_status = status
        self._last_order_message = message
        self._last_order_group_id = ""
        self._last_order_exchange_positions_checked = True
        self._last_order_exchange_positions_underlying = underlying.upper()
        try:
            positions = self.client.get_positions(underlying.upper())
            self._last_order_exchange_positions = positions
            self._last_order_exchange_positions_error = ""
            if positions:
                logger.error(
                    f"[WeekendVol] Post-failure exchange position check found {len(positions)} live position(s): "
                    f"{[p.get('symbol') for p in positions]}"
                )
                self._set_execution_risk_lock(
                    f"{message}；失败后交易所仍有真实持仓残留",
                    event_type="position_open_failed",
                )
            else:
                logger.info(f"[WeekendVol] Post-failure exchange position check: no live {underlying.upper()} positions")
        except Exception as e:
            self._last_order_exchange_positions = []
            self._last_order_exchange_positions_error = str(e)
            logger.error(f"[WeekendVol] Post-failure exchange position check failed: {e}")
            self._set_execution_risk_lock(
                f"{message}；失败后交易所持仓复核异常: {e}",
                event_type="position_open_error",
            )

    def _manage_open_positions(self, now: datetime) -> None:
        if self.pos_mgr.open_position_count <= 0 or self.cfg.stop_loss_pct <= 0:
            return

        try:
            mark_prices = self.client.get_mark_prices(self.cfg.underlying.upper())
            basket_pnl_pct = self._basket_pnl_pct(mark_prices)
        except Exception as e:
            logger.warning(f"[WeekendVol] Failed to evaluate live stop: {e}")
            return

        if basket_pnl_pct is None:
            return

        if basket_pnl_pct <= -float(self.cfg.stop_loss_pct):
            move_filter_state = self._get_underlying_move_filter_state()
            move_filter_pct = float(getattr(self.cfg, "stop_loss_underlying_move_pct", 0.0) or 0.0)
            if move_filter_pct > 0:
                move_filter_error = str(move_filter_state.get("error") or "")
                if move_filter_error:
                    logger.warning(
                        f"[WeekendVol] Stop loss blocked: basket pnl {basket_pnl_pct:.1f}% hit threshold, "
                        f"but underlying move filter could not be evaluated ({move_filter_error})"
                    )
                    return
                if not bool(move_filter_state.get("passes_filter")):
                    logger.warning(
                        f"[WeekendVol] Stop loss blocked by underlying move filter: "
                        f"basket pnl {basket_pnl_pct:.1f}% <= -{self.cfg.stop_loss_pct:.1f}% but "
                        f"spot move={float(move_filter_state.get('max_abs_move_pct') or 0.0):.2f}% "
                        f"< required {move_filter_pct:.2f}%"
                    )
                    return
            logger.warning(
                f"[WeekendVol] Stop loss triggered: basket pnl {basket_pnl_pct:.1f}% "
                f"<= -{self.cfg.stop_loss_pct:.1f}%"
                + (
                    f", spot move={float(move_filter_state.get('max_abs_move_pct') or 0.0):.2f}%"
                    if move_filter_pct > 0
                    else ""
                )
            )
            self.pos_mgr.close_all(
                reason=f"weekend_vol_stop_loss_{self.cfg.stop_loss_pct:.0f}",
            )

    def _get_entry_realized_vol(self, underlying: str) -> float | None:
        if self.cfg.entry_realized_vol_lookback_hours <= 1 or self.cfg.entry_realized_vol_max <= 0:
            return None
        return self._compute_realized_vol(underlying, log_result=True)

    def _compute_realized_vol(self, underlying: str, log_result: bool = False) -> float | None:
        try:
            rv = self.client.get_realized_vol(underlying, self.cfg.entry_realized_vol_lookback_hours)
        except Exception as e:
            logger.warning(f"[WeekendVol] Failed to compute RV filter input: {e}")
            return None
        if rv is not None and log_result:
            logger.info(
                f"[WeekendVol] RV{self.cfg.entry_realized_vol_lookback_hours} = {rv:.2%}"
            )
        return rv

    def _passes_entry_rv_filter(self, realized_vol: float | None) -> bool:
        if self.cfg.entry_realized_vol_lookback_hours <= 1 or self.cfg.entry_realized_vol_max <= 0:
            return True
        if realized_vol is None:
            logger.warning("[WeekendVol] RV filter enabled but realized vol unavailable – skipping entry")
            return False
        if realized_vol > self.cfg.entry_realized_vol_max:
            logger.info(
                f"[WeekendVol] Skip entry: RV{self.cfg.entry_realized_vol_lookback_hours} "
                f"{realized_vol:.2%} > {self.cfg.entry_realized_vol_max:.2%}"
            )
            return False
        return True

    def _basket_pnl_pct(self, mark_prices: dict[str, float]) -> float | None:
        entry_credit = 0.0
        close_cost = 0.0

        for position in self.pos_mgr.open_positions.values():
            if not position.is_open:
                continue
            for leg in position.legs:
                sign = 1.0 if leg.side == "SELL" else -1.0
                mark = float(mark_prices.get(leg.symbol, leg.entry_price) or leg.entry_price)
                entry_credit += float(leg.entry_price) * float(leg.quantity) * sign
                close_cost += mark * float(leg.quantity) * sign

        if entry_credit <= 0:
            return None

        pnl = entry_credit - close_cost
        return pnl / entry_credit * 100.0

    def _current_basket_pnl_pct(self) -> float | None:
        if self.pos_mgr.open_position_count <= 0:
            return None
        try:
            mark_prices = self.client.get_mark_prices(self.cfg.underlying.upper())
        except Exception as e:
            logger.warning(f"[WeekendVol] Failed to fetch mark prices for status: {e}")
            return None
        return self._basket_pnl_pct(mark_prices)

    def _get_underlying_move_filter_state(self, current_spot: float | None = None) -> dict[str, Any]:
        threshold_pct = float(getattr(self.cfg, "stop_loss_underlying_move_pct", 0.0) or 0.0)
        state: dict[str, Any] = {
            "enabled": threshold_pct > 0,
            "threshold_pct": threshold_pct,
            "current_spot": None,
            "max_abs_move_pct": 0.0,
            "max_up_move_pct": 0.0,
            "max_down_move_pct": 0.0,
            "passes_filter": threshold_pct <= 0,
            "positions_checked": 0,
            "positions_with_entry_spot": 0,
            "reference_group_id": "",
            "reference_entry_spot": None,
            "error": "",
        }

        if self.pos_mgr.open_position_count <= 0:
            return state

        if current_spot is None:
            try:
                current_spot = float(self.client.get_spot_price(self.cfg.underlying.upper()) or 0.0)
            except Exception as e:
                state["error"] = str(e)
                return state

        current_spot = float(current_spot or 0.0)
        if current_spot <= 0:
            state["error"] = "invalid_current_spot"
            return state

        state["current_spot"] = current_spot

        for group_id, position in self.pos_mgr.open_positions.items():
            if not position.is_open:
                continue
            state["positions_checked"] += 1
            entry_spot = float(position.underlying_price or 0.0)
            if entry_spot <= 0:
                continue
            state["positions_with_entry_spot"] += 1
            move_pct = (current_spot / entry_spot - 1.0) * 100.0
            up_move_pct = max(move_pct, 0.0)
            down_move_pct = max(-move_pct, 0.0)
            abs_move_pct = abs(move_pct)
            if abs_move_pct >= float(state["max_abs_move_pct"]):
                state["max_abs_move_pct"] = abs_move_pct
                state["max_up_move_pct"] = up_move_pct
                state["max_down_move_pct"] = down_move_pct
                state["reference_group_id"] = group_id
                state["reference_entry_spot"] = entry_spot

        if state["positions_with_entry_spot"] <= 0:
            state["error"] = "missing_entry_spot"
            state["passes_filter"] = False if threshold_pct > 0 else True
            return state

        state["passes_filter"] = threshold_pct <= 0 or float(state["max_abs_move_pct"]) >= threshold_pct
        return state

    # ------------------------------------------------------------------
    # Delta-based strike selection
    # ------------------------------------------------------------------

    def _find_by_delta(
        self,
        tickers: list,
        spot: float,
        T_years: float,
        target_abs_delta: float,
        option_type: str,
    ) -> Any:
        """Find the ticker whose |delta| is closest to target.

        Prefers exchange-provided delta from /eapi/v1/mark.
        Falls back to Black-76 calculation when delta is not available.
        """
        best = None
        best_diff = float("inf")
        best_delta = 0.0
        used_exchange = False

        for t in tickers:
            if option_type == "call":
                if float(t.strike) <= float(spot):
                    continue
            else:
                if float(t.strike) >= float(spot):
                    continue

            # Prefer exchange delta
            if t.delta != 0.0:
                abs_d = abs(t.delta)
            else:
                # Fallback: compute via Black-76
                from options_backtest.pricing.black76 import delta as bs_delta
                iv = t.mark_iv or self.cfg.default_iv
                if iv <= 0:
                    iv = self.cfg.default_iv
                try:
                    d = bs_delta(spot, t.strike, T_years, sigma=iv,
                                 option_type=option_type, r=0.0)
                except (ValueError, ZeroDivisionError):
                    continue
                abs_d = abs(d)

            diff = abs(abs_d - target_abs_delta)
            if diff < best_diff:
                best_diff = diff
                best = t
                best_delta = abs_d
                used_exchange = t.delta != 0.0

        if best:
            source = "exchange" if used_exchange else "Black-76"
            if best_diff > self.cfg.max_delta_diff:
                if used_exchange:
                    logger.warning(
                        f"[WeekendVol] Reject {option_type} strike {best.symbol}: "
                        f"|δ|={best_delta:.4f} target={target_abs_delta:.4f} "
                        f"diff={best_diff:.4f} > max_delta_diff={self.cfg.max_delta_diff:.4f}"
                    )
                    return None
                logger.warning(
                    f"[WeekendVol] Accepting closest {option_type} strike via Black-76 fallback: "
                    f"{best.symbol} |δ|={best_delta:.4f} target={target_abs_delta:.4f} "
                    f"diff={best_diff:.4f} > max_delta_diff={self.cfg.max_delta_diff:.4f}"
                )
            logger.info(
                f"[WeekendVol] {option_type} strike selection: "
                f"K={best.strike} |δ|={best_delta:.4f} "
                f"(target={target_abs_delta}, diff={best_diff:.4f}) source={source} "
                f"IV={best.mark_iv:.1%} {best.symbol}"
            )

        return best

    # ------------------------------------------------------------------
    # Quantity computation (fixed size)
    # ------------------------------------------------------------------

    def _compute_quantity(
        self,
        spot: float,
        sell_call: OptionTicker | None = None,
        sell_put: OptionTicker | None = None,
        buy_call: OptionTicker | None = None,
        buy_put: OptionTicker | None = None,
    ) -> float:
        """Compute position size in fixed-size mode."""
        del spot, sell_call, sell_put, buy_call, buy_put
        base_qty = max(float(self.cfg.quantity or 0.0), 0.0)
        logger.info(f"[WeekendVol] fixed quantity mode: final={base_qty:.4f} {self.cfg.underlying.upper()}")
        return base_qty

    # ------------------------------------------------------------------
    # Settlement check
    # ------------------------------------------------------------------

    def _check_settlement(self, now: datetime) -> None:
        """Check if any open position has expired and record settlement.

        Bybit auto-settles expired options. We detect expiry from the
        symbol and mark the position as closed.
        """
        for gid in list(self.pos_mgr.open_positions.keys()):
            position = self.pos_mgr.open_positions[gid]
            if not position.is_open:
                continue

            # Check expiry from any leg
            for leg in position.legs:
                parsed = _parse_symbol(leg.symbol)
                if parsed and parsed["expiry"] <= now:
                    underlying = str(parsed.get("underlying") or self.cfg.underlying).upper()
                    settlement_spot = self.client.get_spot_price(underlying)
                    if settlement_spot <= 0:
                        settlement_spot = position.underlying_price
                    logger.info(
                        f"[WeekendVol] Position {gid} expired at "
                        f"{parsed['expiry'].isoformat()} – marking settled "
                        f"with {underlying} spot={settlement_spot:.4f}"
                    )

                    total_settlement_pnl = 0.0
                    for leg2 in position.legs:
                        settlement_price = self._settlement_price(leg2, settlement_spot)
                        pnl = self._settlement_pnl(leg2, settlement_price)
                        self.storage.close_trade(
                            trade_id=leg2.trade_id,
                            close_price=settlement_price,
                            pnl=pnl,
                        )
                        total_settlement_pnl += pnl
                        logger.info(
                            f"[WeekendVol] Settled {leg2.side} {leg2.option_type} {leg2.symbol}: "
                            f"entry={leg2.entry_price:.4f} settle={settlement_price:.4f} pnl={pnl:.4f}"
                        )
                    logger.info(f"[WeekendVol] Position {gid} settlement pnl={total_settlement_pnl:.4f}")
                    position.is_open = False
                    del self.pos_mgr.open_positions[gid]
                    break

    @staticmethod
    def _settlement_price(leg, settlement_spot: float) -> float:
        if settlement_spot <= 0:
            return 0.0
        if leg.option_type == "call":
            return max(settlement_spot - leg.strike, 0.0)
        return max(leg.strike - settlement_spot, 0.0)

    @staticmethod
    def _settlement_pnl(leg, settlement_price: float) -> float:
        if leg.side == "SELL":
            return (leg.entry_price - settlement_price) * leg.quantity
        return (settlement_price - leg.entry_price) * leg.quantity

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _next_sunday_0800(now: datetime) -> datetime:
        """Find the next Sunday 08:00 UTC from now (or this Sunday if today is Fri/Sat)."""
        days_until_sunday = (6 - now.weekday()) % 7
        if days_until_sunday == 0 and now.hour >= 8:
            days_until_sunday = 7  # already past this Sunday 08:00
        sunday = now.replace(hour=8, minute=0, second=0, microsecond=0) + timedelta(days=days_until_sunday)
        return sunday

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Return current strategy status."""
        rv24 = self._compute_realized_vol(self.cfg.underlying.upper(), log_result=False)
        basket_pnl_pct = self._current_basket_pnl_pct()
        local_open_positions = self.pos_mgr.open_position_count
        move_filter_state = self._get_underlying_move_filter_state() if local_open_positions > 0 else {
            "enabled": float(getattr(self.cfg, "stop_loss_underlying_move_pct", 0.0) or 0.0) > 0,
            "threshold_pct": float(getattr(self.cfg, "stop_loss_underlying_move_pct", 0.0) or 0.0),
            "current_spot": None,
            "max_abs_move_pct": 0.0,
            "max_up_move_pct": 0.0,
            "max_down_move_pct": 0.0,
            "passes_filter": False,
            "positions_checked": 0,
            "positions_with_entry_spot": 0,
            "reference_group_id": "",
            "reference_entry_spot": None,
            "error": "",
        }
        exchange_positions = list(self._last_guard_exchange_positions or [])
        exchange_position_count = len(exchange_positions)
        effective_open_positions = local_open_positions
        position_source = "local"
        if exchange_position_count > 0 and local_open_positions <= 0:
            effective_open_positions = 1
            position_source = "exchange"
        return {
            "strategy": "WeekendVol",
            "mode": "weekend_vol",
            "underlying": self.cfg.underlying,
            "target_delta": self.cfg.target_delta,
            "wing_delta": self.cfg.wing_delta,
            "max_delta_diff": self.cfg.max_delta_diff,
            "leverage": self.cfg.leverage,
            "entry_day": self.cfg.entry_day,
            "entry_time_utc": self.cfg.entry_time_utc,
            "entry_realized_vol_lookback_hours": self.cfg.entry_realized_vol_lookback_hours,
            "entry_realized_vol_max": self.cfg.entry_realized_vol_max,
            "entry_realized_vol_current": rv24,
            "stop_loss_pct": self.cfg.stop_loss_pct,
            "stop_loss_underlying_move_pct": self.cfg.stop_loss_underlying_move_pct,
            "basket_pnl_pct": basket_pnl_pct,
            "current_spot": move_filter_state.get("current_spot"),
            "stop_loss_underlying_move_abs_pct": move_filter_state.get("max_abs_move_pct"),
            "stop_loss_underlying_move_up_pct": move_filter_state.get("max_up_move_pct"),
            "stop_loss_underlying_move_down_pct": move_filter_state.get("max_down_move_pct"),
            "stop_loss_underlying_move_filter_passed": move_filter_state.get("passes_filter"),
            "stop_loss_underlying_move_reference_group_id": move_filter_state.get("reference_group_id"),
            "stop_loss_underlying_move_reference_entry_spot": move_filter_state.get("reference_entry_spot"),
            "stop_loss_underlying_move_error": move_filter_state.get("error"),
            "last_order_attempt_at": self._last_order_attempt_at,
            "last_order_status": self._last_order_status,
            "last_order_message": self._last_order_message,
            "last_order_group_id": self._last_order_group_id,
            "last_order_exchange_positions_checked": self._last_order_exchange_positions_checked,
            "last_order_exchange_positions_underlying": self._last_order_exchange_positions_underlying,
            "last_order_exchange_positions": self._last_order_exchange_positions,
            "last_order_exchange_positions_error": self._last_order_exchange_positions_error,
            "execution_risk_lock_active": self._execution_risk_lock_active,
            "execution_risk_lock_since": self._execution_risk_lock_since,
            "execution_risk_lock_reason": self._execution_risk_lock_reason,
            "execution_risk_lock_group_id": self._execution_risk_lock_group_id,
            "execution_risk_lock_event": self._execution_risk_lock_event,
            "last_trade_week": self._last_trade_week,
            "open_positions": effective_open_positions,
            "local_open_positions": local_open_positions,
            "exchange_position_count": exchange_position_count,
            "exchange_position_symbols": [str(p.get("symbol") or "") for p in exchange_positions],
            "exchange_position_check_error": self._last_guard_exchange_positions_error,
            "position_source": position_source,
            "max_positions": self.cfg.max_positions,
            "positions": self.pos_mgr.summary(),
        }
