"""Core backtest engine – fixed time‑step loop.

Orchestrates: data loading → time iteration → strategy calls →
order execution → position / account updates → equity recording.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from options_backtest.config import Config
from options_backtest.data.loader import DataLoader, ArrayChain
from options_backtest.engine.account import Account
from options_backtest.engine.matcher import Matcher
from options_backtest.engine.position import PositionManager
from options_backtest.engine.settlement import check_and_settle
from options_backtest.pricing.iv_solver import implied_volatility_btc
from options_backtest.pricing.black76 import call_price, put_price
from options_backtest.utils import to_utc_timestamp

if TYPE_CHECKING:
    from options_backtest.strategy.base import BaseStrategy

# ---------------------------------------------------------------------------
# Cached instrument name parsing
# ---------------------------------------------------------------------------
_parsed_instrument_cache: dict[str, dict | None] = {}


def _cached_parse_instrument(name: str) -> dict | None:
    """Parse instrument name with caching."""
    result = _parsed_instrument_cache.get(name)
    if result is not None or name in _parsed_instrument_cache:
        return result
    from options_backtest.data.fetcher import parse_instrument_name
    result = parse_instrument_name(name)
    _parsed_instrument_cache[name] = result
    return result


# ---------------------------------------------------------------------------
# Strategy context – the object visible to strategies
# ---------------------------------------------------------------------------

class StrategyContext:
    """Read‑only context passed to strategy callbacks each time step."""

    __slots__ = (
        "current_time", "underlying_price", "_option_chain",
        "_chain_builder", "positions", "account", "_engine",
        "_ts_ns",
    )

    def __init__(
        self,
        current_time: datetime,
        underlying_price: float,
        positions: dict,
        account: Account,
        _engine: "BacktestEngine",
        *,
        _chain_builder=None,
        _option_chain: pd.DataFrame | None = None,
        _ts_ns: int = 0,
    ):
        self.current_time = current_time
        self.underlying_price = underlying_price
        self._option_chain = _option_chain
        self._chain_builder = _chain_builder
        self.positions = positions
        self.account = account
        self._engine = _engine
        self._ts_ns = _ts_ns

    @property
    def option_chain(self):
        """Lazy-build option chain on first access."""
        if self._option_chain is None:
            if self._chain_builder is not None:
                self._option_chain = self._chain_builder()
            else:
                self._option_chain = ArrayChain({}, 0)
        return self._option_chain

    def get_instrument_dte(self, instrument_name: str) -> float:
        """Get days-to-expiry for an instrument without building the chain.

        Returns 999.0 if the instrument is not found.
        """
        inst = self._engine._instrument_dict.get(instrument_name)
        if inst is None:
            return 999.0
        exp_ns = inst.get("_expiry_ns", 0)
        if exp_ns == 0:
            return 999.0
        dte_ns = exp_ns - self._ts_ns
        return dte_ns / 86_400_000_000_000  # ns → days

    # -- action helpers (delegate to engine) --

    def buy(self, instrument_name: str, quantity: float = 1.0) -> None:
        """Submit a buy order."""
        from options_backtest.data.models import Direction, OrderRequest
        order = OrderRequest(
            instrument_name=instrument_name,
            direction=Direction.LONG,
            quantity=quantity,
        )
        self._engine._pending_orders.append(order)

    def sell(self, instrument_name: str, quantity: float = 1.0) -> None:
        """Submit a sell (short) order."""
        from options_backtest.data.models import Direction, OrderRequest
        order = OrderRequest(
            instrument_name=instrument_name,
            direction=Direction.SHORT,
            quantity=quantity,
        )
        self._engine._pending_orders.append(order)

    def close(self, instrument_name: str) -> None:
        """Close an existing position completely."""
        pos = self.positions.get(instrument_name)
        if pos is None:
            return
        from options_backtest.data.models import Direction, OrderRequest
        close_dir = Direction.SHORT if pos.direction.value == "long" else Direction.LONG
        order = OrderRequest(
            instrument_name=instrument_name,
            direction=close_dir,
            quantity=pos.quantity,
        )
        self._engine._pending_orders.append(order)

    def close_all(self) -> None:
        """Close every open position."""
        for name in list(self.positions.keys()):
            self.close(name)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Fixed time‑step backtest engine."""

    def __init__(self, config: Config, strategy: "BaseStrategy"):
        self.config = config
        self.strategy = strategy

        # Margin mode (must be set before Matcher)
        self._margin_usd: bool = config.backtest.margin_mode.upper() == "USD"

        self.loader = DataLoader(data_dir="data")
        self.account = Account(initial_balance=config.account.initial_balance)
        self.position_mgr = PositionManager()
        self.matcher = Matcher(config.execution, margin_usd=self._margin_usd)

        self._pending_orders: list = []

        # Data (loaded in run())
        self._instruments_df: pd.DataFrame = pd.DataFrame()
        self._underlying_df: pd.DataFrame = pd.DataFrame()
        self._option_data: dict[str, pd.DataFrame] = {}
        self._settlements_df: pd.DataFrame = pd.DataFrame()

        # Pre-indexed data for fast lookups (built in _load_data)
        self._ohlcv_index: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        self._instrument_dict: dict[str, dict] = {}
        self._settlement_index: dict[str, float] = {}  # instrument_name → index_price
        # Arithmetic OHLCV index: name → (start_ns, step_ns, length, open, close, low, high, vol)
        self._ohlcv_arith: dict[str, tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        self._iv_observations: dict[str, tuple[int, float]] = {}
        self._realized_vol_ts: np.ndarray = np.array([], dtype="datetime64[ns]")
        self._realized_vol_values: np.ndarray = np.array([], dtype=float)
        self._dvol_ts: np.ndarray = np.array([], dtype="datetime64[ns]")
        self._dvol_values: np.ndarray = np.array([], dtype=float)

        # Data source tracking
        self._mark_source_market: int = 0   # mark prices from real OHLCV
        self._mark_source_synth: int = 0    # mark prices from Black-76
        self._quote_source_market: int = 0  # order quotes from real OHLCV
        self._quote_source_synth: int = 0   # order quotes from Black-76
        self._chain_source_market: int = 0  # chain entries from real OHLCV
        self._chain_source_synth: int = 0   # chain entries from Black-76

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Execute the backtest and return a results dict."""
        cfg = self.config.backtest
        underlying = cfg.underlying

        logger.info(f"=== Backtest: {cfg.name} ===")
        logger.info(f"Period: {cfg.start_date} → {cfg.end_date}, step={cfg.time_step}")

        # 1. Load data
        self._load_data(underlying, cfg.start_date, cfg.end_date, cfg.time_step)

        # 2. Pre-extract numpy arrays for fast iteration
        ts_values = self._underlying_df["timestamp"].values          # datetime64[ns, UTC]
        open_values = (self._underlying_df["open"].values.astype(np.float64)
                   if "open" in self._underlying_df.columns
                   else self._underlying_df["close"].values.astype(np.float64))
        close_values = self._underlying_df["close"].values.astype(np.float64)
        daily_open_close_mode = cfg.time_step == "1d"
        n_steps = len(ts_values)
        if n_steps == 0:
            raise RuntimeError("No underlying data for the given period")

        logger.info(f"Time steps: {n_steps}")

        # 2b. USD margin: convert initial balance from coin to USD
        if self._margin_usd:
            first_price = float(open_values[0] if daily_open_close_mode else close_values[0])
            if first_price > 0:
                usd_balance = self.account.initial_balance * first_price
                self.account.initial_balance = usd_balance
                self.account.balance = usd_balance
                logger.info(f"USD margin mode: initial balance = {usd_balance:,.2f} USD "
                            f"({self.config.account.initial_balance} coin × {first_price:,.2f})")

        # 3. Initialise strategy
        ts0 = pd.Timestamp(ts_values[0])
        initial_ctx = self._build_context(ts0, float(open_values[0] if daily_open_close_mode else close_values[0]))
        self.strategy.initialize(initial_ctx)

        # Local references for hot-path (avoid repeated attribute lookups)
        position_mgr = self.position_mgr
        account = self.account
        matcher = self.matcher
        instruments_df = self._instruments_df
        settlements_df = self._settlements_df
        instrument_dict = self._instrument_dict
        strategy = self.strategy
        pending_orders = self._pending_orders

        # Pre-convert all timestamps to pd.Timestamp (avoids per-step conversion)
        ts_pd_all = pd.DatetimeIndex(ts_values).tz_localize("UTC")

        # 4. Iterate
        for i in tqdm(range(n_steps), desc="Backtesting"):
            ts_np = ts_values[i]               # numpy datetime64
            open_price = float(open_values[i])
            close_price = float(close_values[i])
            ts_pd = ts_pd_all[i]               # pre-converted pd.Timestamp

            has_positions = bool(position_mgr.positions)

            # 4a. Update marks / settle for non-daily modes before strategy acts
            if has_positions and not daily_open_close_mode:
                mark_prices = self._get_mark_prices_fast(ts_np, close_price)
                position_mgr.update_marks(mark_prices)

                check_and_settle(
                    ts_np, position_mgr, account,
                    matcher, instrument_dict, settlements_df,
                    margin_usd=self._margin_usd,
                    settlement_index=self._settlement_index,
                )

            # 4c. Build context (lazy chain) and call strategy
            ctx = self._build_context(ts_pd, open_price if daily_open_close_mode else close_price)
            pending_orders.clear()
            strategy.on_step(ctx)

            # 4d. Process pending orders (reuse ctx)
            if pending_orders:
                self._process_orders(
                    ts_np,
                    open_price if daily_open_close_mode else close_price,
                    ctx,
                    price_field="open" if daily_open_close_mode else "close",
                )

            if daily_open_close_mode and position_mgr.positions:
                mark_prices = self._get_mark_prices_fast(ts_np, close_price)
                position_mgr.update_marks(mark_prices)
                self._close_expiring_positions_at_bar_close(ts_np, close_price, ctx)

            # 4e. Record equity
            account.record_equity(ts_pd, position_mgr.total_unrealized_pnl, close_price)

        # 5. Force‑close remaining positions at last mark
        if position_mgr.positions:
            logger.info(f"Force‑closing {len(position_mgr.positions)} remaining positions")
            last_ts = ts_pd_all[-1]
            last_price = float(close_values[-1])
            ctx = self._build_context(last_ts, last_price)
            ctx.close_all()
            self._process_orders(ts_values[-1], last_price, ctx, price_field="close")
            account.record_equity(last_ts, position_mgr.total_unrealized_pnl, last_price)

        # 6. Call strategy on_end (no‑op in base, but strategies can use it)
        # 7. Package results
        results = self._build_results()
        logger.info(f"Backtest complete. Total return: {results['total_return']:.2%}")
        return results

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self, underlying: str, start: str, end: str, step: str) -> None:
        res_map = {"1m": "1", "5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "1D"}
        resolution = res_map.get(step, "1D")
        use_bs_only = self.config.backtest.use_bs_only

        self._instruments_df = self.loader.load_instruments(underlying)
        self._underlying_df = self.loader.load_underlying(underlying, resolution, start, end)
        if use_bs_only:
            self._option_data = {}
            logger.info("BS-only mode enabled: skip option OHLCV loading")
        else:
            self._option_data = self.loader.load_all_option_data(
                self._instruments_df, start, end, resolution=resolution,
            )
        if self.config.backtest.iv_mode == "surface":
            if use_bs_only:
                logger.warning("IV surface mode requested but BS-only is enabled; fallback to fixed IV")
            else:
                logger.info("IV surface mode enabled: use dynamic IV(K,T,t) from recent market observations")
        elif self.config.backtest.iv_mode == "proxy":
            self._prepare_proxy_iv_inputs()
            logger.info("IV proxy mode enabled: DVOL/realized-vol with K-T-t adjustments")
        self._settlements_df = self.loader.load_settlements(underlying)

        # --- Pre-filter instruments to only those relevant to the period ---
        # This dramatically reduces the DataFrame size for build_option_chain
        exp_col = ("expiration_timestamp"
                   if "expiration_timestamp" in self._instruments_df.columns
                   else "expiration_date")
        exps = pd.to_datetime(self._instruments_df[exp_col], utc=True)
        ts_start = to_utc_timestamp(start)
        ts_end = to_utc_timestamp(end)
        # Keep instruments expiring during or up to 90 days after the period
        relevant = (exps >= ts_start) & (exps <= ts_end + pd.Timedelta(days=90))
        full_count = len(self._instruments_df)
        self._instruments_df = self._instruments_df.loc[relevant].reset_index(drop=True)
        logger.info(f"Pre-filtered instruments: {len(self._instruments_df)} / {full_count}")

        # --- Pre-compute columns for build_option_chain (avoids repeat work) ---
        idf = self._instruments_df
        exp_col = ("expiration_timestamp"
                   if "expiration_timestamp" in idf.columns
                   else "expiration_date")
        idf["_expiry_dt"] = pd.to_datetime(idf[exp_col], utc=True)
        idf["_expiry_ns"] = idf["_expiry_dt"].values.astype("int64")  # nanoseconds
        strike_col = "strike_price" if "strike_price" in idf.columns else "strike"
        if strike_col != "strike_price":
            idf["strike_price"] = idf[strike_col]
        if "option_type" not in idf.columns:
            idf["option_type"] = "call"
        idf["option_type"] = idf["option_type"].str.lower()
        # Pre-sort so build_option_chain can skip sorting
        idf = idf.sort_values(["_expiry_dt", "strike_price"]).reset_index(drop=True)
        self._instruments_df = idf

        # --- Pre-build OHLCV index: numpy arrays for O(log n) searchsorted ---
        self._ohlcv_index = {}
        self._ohlcv_arith = {}       # O(1) arithmetic index for regular grids
        _HOUR_NS = 3600 * 1_000_000_000
        for name, df in self._option_data.items():
            ts_arr = np.asarray(df["timestamp"].values)  # sorted datetime64
            open_arr = (np.asarray(df["open"].values, dtype=np.float64)
                        if "open" in df.columns else np.asarray(df["close"].values, dtype=np.float64))
            close_arr = np.asarray(df["close"].values, dtype=np.float64)
            low_arr = (np.asarray(df["low"].values, dtype=np.float64)
                       if "low" in df.columns else close_arr.copy())
            high_arr = (np.asarray(df["high"].values, dtype=np.float64)
                        if "high" in df.columns else close_arr.copy())
            vol_arr = (np.asarray(df["volume"].values, dtype=np.float64)
                       if "volume" in df.columns else np.zeros(len(df), dtype=np.float64))
            self._ohlcv_index[name] = (ts_arr, open_arr, close_arr, low_arr, high_arr, vol_arr)

            # Build O(1) arithmetic index for regularly-spaced (hourly) grids
            n_rows = len(ts_arr)
            if n_rows >= 2:
                ts_ns = ts_arr.view("int64")  # zero-copy view
                start_ns = int(ts_ns[0])
                step_ns = int(ts_ns[1] - ts_ns[0])
                if step_ns > 0:
                    # Check regularity: all steps should equal step_ns
                    expected_end_ns = start_ns + step_ns * (n_rows - 1)
                    if int(ts_ns[-1]) == expected_end_ns:
                        # Regular grid → O(1) index
                        self._ohlcv_arith[name] = (
                            start_ns, step_ns, n_rows,
                            open_arr, close_arr, low_arr, high_arr, vol_arr,
                        )
            elif n_rows == 1:
                ts_ns = ts_arr.view("int64")  # zero-copy view
                self._ohlcv_arith[name] = (
                    int(ts_ns[0]), _HOUR_NS, 1,
                    open_arr, close_arr, low_arr, high_arr, vol_arr,
                )

        # --- Pre-build instrument dict for O(1) lookup (fast vectorised) ---
        records = self._instruments_df.to_dict(orient="records")
        names_list = self._instruments_df["instrument_name"].tolist()
        self._instrument_dict = {names_list[i]: records[i] for i in range(len(names_list))}

        # --- Pre-extract instrument columns as numpy arrays for build_option_chain ---
        idf = self._instruments_df
        self._inst_arrays = {
            "instrument_name": idf["instrument_name"].values,
            "_expiry_ns": idf["_expiry_ns"].values if "_expiry_ns" in idf.columns else None,
            "strike_price": idf["strike_price"].values.astype(np.float64),
            "option_type": idf["option_type"].values,
            "_expiry_dt": idf["_expiry_dt"].values if "_expiry_dt" in idf.columns else None,
        }

        # --- Pre-build settlement index: instrument_name → index_price (O(1) lookup) ---
        self._settlement_index = {}
        sdf = self._settlements_df
        if not sdf.empty and "instrument_name" in sdf.columns and "index_price" in sdf.columns:
            for inst_name, idx_price in zip(sdf["instrument_name"].values, sdf["index_price"].values):
                self._settlement_index[str(inst_name)] = float(idx_price)

        logger.info(f"Pre-indexed {len(self._ohlcv_index)} OHLCV series, "
                    f"{len(self._instrument_dict)} instruments, "
                    f"{len(self._settlement_index)} settlement prices")

    # ------------------------------------------------------------------
    # Context helpers
    # ------------------------------------------------------------------

    def _build_context(self, timestamp, underlying_price: float) -> StrategyContext:
        # Lazy chain builder – avoids building the full option chain
        # unless the strategy actually accesses context.option_chain
        def _lazy_chain():
            sc = {"market": 0, "synth": 0}
            result = self.loader.build_option_chain(
                self._instruments_df, self._option_data, timestamp, underlying_price,
                ohlcv_index=self._ohlcv_index,
                source_counter=sc,
                prefer_market_data=not self.config.backtest.use_bs_only,
                ohlcv_arith=self._ohlcv_arith,
                inst_arrays=self._inst_arrays,
                market_price_field="open" if self.config.backtest.time_step == "1d" else "close",
            )
            self._chain_source_market += sc["market"]
            self._chain_source_synth += sc["synth"]
            return result

        ts_ns = timestamp.value if hasattr(timestamp, 'value') else int(pd.Timestamp(timestamp).value)

        return StrategyContext(
            current_time=timestamp,
            underlying_price=underlying_price,
            positions=self.position_mgr.positions,
            account=self.account,
            _engine=self,
            _chain_builder=_lazy_chain,
            _ts_ns=ts_ns,
        )

    def _get_mark_prices_fast(self, ts_np, underlying_price: float) -> dict[str, float]:
        """Get latest mark (close) price using pre-indexed numpy arrays.

        Uses searchsorted for O(log n) lookups instead of O(n) boolean masks.
        Falls back to Black-76 synthetic pricing when OHLCV is missing.

        Returns prices in coin or USD depending on margin_mode.
        """
        marks: dict[str, float] = {}
        ohlcv_index = self._ohlcv_index
        ohlcv_arith = self._ohlcv_arith
        use_market_data = not self.config.backtest.use_bs_only
        margin_usd = self._margin_usd
        # Pre-compute values needed for synthetic fallback (avoid per-position overhead)
        _ts_ns_val = int(np.datetime64(ts_np, 'ns').view('int64')) if not isinstance(ts_np, (int, np.integer)) else int(ts_np)

        for name in self.position_mgr.positions:
            # Try O(1) arithmetic index first (regular hourly grids)
            if use_market_data:
                arith = ohlcv_arith.get(name)
                if arith is not None:
                    start_ns, step_ns, a_len, a_close = arith[0], arith[1], arith[2], arith[4]
                    j = (_ts_ns_val - start_ns) // step_ns
                    if j < 0:
                        j = -1
                    elif j >= a_len:
                        j = a_len - 1
                    if j >= 0:
                        market_mark = float(a_close[j])
                        self._mark_source_market += 1
                        self._update_iv_observation(name, ts_np, market_mark, underlying_price)
                        marks[name] = market_mark * underlying_price if margin_usd else market_mark
                        continue

                # Fallback: O(log n) searchsorted
                idx_data = ohlcv_index.get(name)
                if idx_data is not None:
                    ts_arr, close_arr = idx_data[0], idx_data[2]
                    j = np.searchsorted(ts_arr, ts_np, side="right") - 1
                    if j >= 0:
                        market_mark = float(close_arr[j])
                        self._mark_source_market += 1
                        self._update_iv_observation(name, ts_np, market_mark, underlying_price)
                        marks[name] = market_mark * underlying_price if margin_usd else market_mark
                        continue

            # Synthetic fallback
            if underlying_price <= 0:
                continue
            parsed = _cached_parse_instrument(name)
            if parsed is None:
                continue
            exp_ts = to_utc_timestamp(parsed["expiration_date"])
            # Use nanosecond arithmetic to avoid pd.Timestamp creation
            exp_ns = exp_ts.value
            T = max((exp_ns - _ts_ns_val) / (365.25 * 86400 * 1e9), 0.0001)
            strike = parsed["strike_price"]
            iv = self._resolve_dynamic_iv(name, ts_np, underlying_price, strike, T)
            if parsed["option_type"] == "call":
                usd_price = call_price(underlying_price, strike, T, iv, r=0.0)
            else:
                usd_price = put_price(underlying_price, strike, T, iv, r=0.0)
            marks[name] = usd_price if margin_usd else usd_price / underlying_price
            self._mark_source_synth += 1

        return marks

    # ------------------------------------------------------------------
    # Order processing
    # ------------------------------------------------------------------

    def _process_orders(self, ts_np, underlying_price: float, ctx=None, price_field: str = "close") -> None:
        from options_backtest.data.models import Direction

        for order in self._pending_orders:
            # Look up current bid / ask / mark for the instrument
            bid, ask, mark = self._get_quotes_fast(order.instrument_name, ts_np, underlying_price, price_field=price_field)
            if mark is None or mark <= 0:
                logger.warning(f"No quote for {order.instrument_name}, skipping order")
                continue

            fill = self.matcher.execute(
                order, pd.Timestamp(ts_np) if not isinstance(ts_np, pd.Timestamp) else ts_np,
                bid, ask, mark, underlying_price,
            )
            if fill is None:
                continue

            self.position_mgr.apply_fill(fill)
            self.account.pay_fee(fill.fee)

            if fill.direction == Direction.LONG:
                self.account.withdraw(fill.fill_price * fill.quantity)
            else:
                self.account.deposit(fill.fill_price * fill.quantity)

            # Reuse existing context instead of rebuilding
            if ctx is not None:
                self.strategy.on_fill(ctx, fill)

        self._pending_orders.clear()

    def _get_quotes_fast(self, instrument_name: str, ts_np, underlying_price: float = 0.0, price_field: str = "close"):
        """Return (bid, ask, mark) using pre-indexed OHLCV + searchsorted.

        Falls back to Black-76 when no OHLCV data is available.
        Returns prices in coin or USD depending on margin_mode.
        """
        margin_usd = self._margin_usd
        if not self.config.backtest.use_bs_only:
            # O(1) arithmetic index for regular grids
            arith = self._ohlcv_arith.get(instrument_name)
            if arith is not None:
                start_ns, step_ns, a_len, a_open, a_close, a_low, a_high, _ = arith
                _ts_ns = int(np.datetime64(ts_np, 'ns').view('int64')) if not isinstance(ts_np, (int, np.integer)) else int(ts_np)
                j = (_ts_ns - start_ns) // step_ns
                if j < 0:
                    j = -1
                elif j >= a_len:
                    j = a_len - 1
                if j >= 0:
                    market_mark = float(a_open[j] if price_field == "open" else a_close[j])
                    self._quote_source_market += 1
                    self._update_iv_observation(instrument_name, ts_np, market_mark, underlying_price)
                    if self.config.backtest.time_step == "1d" and price_field in {"open", "close"}:
                        px = market_mark * underlying_price if margin_usd else market_mark
                        return px, px, px
                    mark = market_mark * underlying_price if margin_usd else market_mark
                    bid, ask = self._derive_market_bid_ask(mark)
                    return bid, ask, mark

            # Fallback: O(log n) searchsorted
            idx_data = self._ohlcv_index.get(instrument_name)
            if idx_data is not None:
                ts_arr, open_arr, close_arr, low_arr, high_arr, _ = idx_data
                j = np.searchsorted(ts_arr, ts_np, side="right") - 1
                if j >= 0:
                    market_mark = float(open_arr[j] if price_field == "open" else close_arr[j])
                    self._quote_source_market += 1
                    self._update_iv_observation(instrument_name, ts_np, market_mark, underlying_price)
                    if self.config.backtest.time_step == "1d" and price_field in {"open", "close"}:
                        px = market_mark * underlying_price if margin_usd else market_mark
                        return px, px, px
                    mark = market_mark * underlying_price if margin_usd else market_mark
                    bid, ask = self._derive_market_bid_ask(mark)
                    return bid, ask, mark

        # Synthetic pricing
        if underlying_price <= 0:
            return None, None, None

        parsed = _cached_parse_instrument(instrument_name)
        if parsed is None:
            return None, None, None

        exp_ts = to_utc_timestamp(parsed["expiration_date"])
        # Use nanosecond arithmetic to avoid pd.Timestamp creation
        _ts_ns_val = int(np.datetime64(ts_np, 'ns').view('int64')) if not isinstance(ts_np, (int, np.integer)) else int(ts_np)
        T = max((exp_ts.value - _ts_ns_val) / (365.25 * 86400 * 1e9), 0.0001)
        strike = parsed["strike_price"]
        iv = self._resolve_dynamic_iv(instrument_name, ts_np, underlying_price, strike, T)
        if parsed["option_type"] == "call":
            usd = call_price(underlying_price, strike, T, iv, r=0.0)
        else:
            usd = put_price(underlying_price, strike, T, iv, r=0.0)
        mark = usd if margin_usd else usd / underlying_price
        spread = mark * 0.02
        self._quote_source_synth += 1
        return max(mark - spread, 0.0), mark + spread, mark

    def _close_expiring_positions_at_bar_close(self, ts_np, underlying_price: float, ctx=None) -> None:
        """For 1D bars, close positions expiring within the current bar at the bar close price."""
        from options_backtest.data.models import Direction, Fill

        bar_end_ns = int(np.datetime64(ts_np, 'ns').view('int64')) + 86400 * 1_000_000_000
        for name in list(self.position_mgr.positions.keys()):
            inst_data = self._instrument_dict.get(name)
            if not inst_data:
                continue

            exp_ns = inst_data.get("_expiry_ns")
            if exp_ns is None or int(exp_ns) > bar_end_ns:
                continue

            _, _, close_mark = self._get_quotes_fast(name, ts_np, underlying_price, price_field="close")
            if close_mark is None or close_mark <= 0:
                continue

            pos = self.position_mgr.positions.get(name)
            if pos is None:
                continue

            exit_dir = Direction.LONG if pos.direction == Direction.SHORT else Direction.SHORT
            fee = self.matcher._compute_fee(close_mark, pos.quantity, underlying_price)
            fill = Fill(
                timestamp=pd.Timestamp(ts_np) if not isinstance(ts_np, pd.Timestamp) else ts_np,
                instrument_name=name,
                direction=exit_dir,
                quantity=pos.quantity,
                fill_price=close_mark,
                fee=fee,
                underlying_price=underlying_price,
            )
            self.position_mgr.apply_fill(fill)
            self.account.pay_fee(fill.fee)

            if exit_dir == Direction.LONG:
                self.account.withdraw(fill.fill_price * fill.quantity)
            else:
                self.account.deposit(fill.fill_price * fill.quantity)

            if ctx is not None:
                self.strategy.on_fill(ctx, fill)

    def _derive_market_bid_ask(self, mark: float) -> tuple[float, float]:
        """Build a conservative quote proxy from OHLC-only market data.

        Daily CDD files do not contain real order-book bid/ask. Using OHLC low/high
        as instantaneous bid/ask introduces look-ahead bias, so we instead apply a
        fixed spread around the observed close/mark.
        """
        spread_pct = max(float(self.config.execution.market_quote_spread_pct), 0.0)
        spread = max(mark * spread_pct, self.config.execution.slippage * 2)
        half_spread = spread / 2.0
        return max(mark - half_spread, 0.0), mark + half_spread

    def _resolve_dynamic_iv(
        self,
        instrument_name: str,
        ts_np,
        underlying_price: float,
        strike: float,
        maturity_years: float,
    ) -> float:
        """Resolve IV for synthetic pricing using IV(K,T,t) surface fallback."""
        base_iv = float(self.config.backtest.fixed_iv)
        if self.config.backtest.iv_mode == "proxy":
            return self._resolve_proxy_iv(ts_np, underlying_price, strike, maturity_years)
        if self.config.backtest.iv_mode != "surface":
            return base_iv
        if self.config.backtest.use_bs_only:
            return base_iv
        if underlying_price <= 0 or strike <= 0 or maturity_years <= 0:
            return base_iv

        if not self._iv_observations:
            return base_iv

        now_ns = pd.Timestamp(ts_np, tz="UTC").value
        max_age_ns = int(pd.Timedelta(days=7).value)

        target_log_moneyness = float(np.log(strike / underlying_price))
        target_sqrt_t = float(np.sqrt(max(maturity_years, 1e-6)))

        score_rows: list[tuple[float, float]] = []

        for observed_name, (obs_ns, observed_iv) in self._iv_observations.items():
            age_ns = now_ns - obs_ns
            if age_ns < 0 or age_ns > max_age_ns:
                continue
            if not np.isfinite(observed_iv):
                continue
            if observed_iv <= 0.01 or observed_iv > 5.0:
                continue

            meta = self._instrument_dict.get(observed_name)
            if meta is None:
                continue

            observed_strike = float(meta.get("strike_price", 0.0))
            exp_ns = int(meta.get("_expiry_ns", 0))
            if observed_strike <= 0 or exp_ns <= now_ns:
                continue

            observed_t = (exp_ns - now_ns) / (365.25 * 86400 * 1e9)
            if observed_t <= 0:
                continue

            observed_log_m = float(np.log(observed_strike / underlying_price))
            observed_sqrt_t = float(np.sqrt(max(observed_t, 1e-6)))

            d_log_m = (observed_log_m - target_log_moneyness) / 0.15
            d_sqrt_t = (observed_sqrt_t - target_sqrt_t) / 0.20
            d_time = age_ns / max_age_ns
            distance2 = d_log_m * d_log_m + d_sqrt_t * d_sqrt_t + 0.25 * d_time * d_time
            weight = float(np.exp(-0.5 * distance2))
            if weight <= 1e-6:
                continue

            score_rows.append((weight, float(observed_iv)))

        if not score_rows:
            return base_iv

        score_rows.sort(key=lambda item: item[0], reverse=True)
        top_rows = score_rows[:24]

        total_weight = sum(weight for weight, _ in top_rows)
        if total_weight <= 0:
            return base_iv

        iv_value = sum(weight * iv for weight, iv in top_rows) / total_weight
        if not np.isfinite(iv_value):
            return base_iv
        return float(np.clip(iv_value, 0.05, 3.0))

    def _update_iv_observation(
        self,
        instrument_name: str,
        ts_np,
        market_mark_btc: float,
        underlying_price: float,
    ) -> None:
        """Update surface observation cache with implied vol from a market mark."""
        if self.config.backtest.iv_mode != "surface":
            return
        if self.config.backtest.use_bs_only:
            return
        if market_mark_btc <= 0 or underlying_price <= 0:
            return

        meta = self._instrument_dict.get(instrument_name)
        if meta is None:
            return

        strike = float(meta.get("strike_price", 0.0))
        option_type = str(meta.get("option_type", "call"))
        exp_ns = int(meta.get("_expiry_ns", 0))
        now_ns = pd.Timestamp(ts_np, tz="UTC").value
        if strike <= 0 or exp_ns <= now_ns:
            return

        maturity = (exp_ns - now_ns) / (365.25 * 86400 * 1e9)
        if maturity <= 0:
            return

        iv = implied_volatility_btc(
            market_mark_btc,
            underlying_price,
            strike,
            maturity,
            option_type=option_type,
            r=0.0,
        )
        if not np.isfinite(iv):
            return
        if iv <= 0.01 or iv > 5.0:
            return

        self._iv_observations[instrument_name] = (now_ns, float(iv))

    def _prepare_proxy_iv_inputs(self) -> None:
        """Prepare DVOL / realized-vol time series for proxy IV mode."""
        self._realized_vol_ts = np.array([], dtype="datetime64[ns]")
        self._realized_vol_values = np.array([], dtype=float)
        self._dvol_ts = np.array([], dtype="datetime64[ns]")
        self._dvol_values = np.array([], dtype=float)

        if self._underlying_df.empty:
            return

        lookback_days = max(int(self.config.backtest.iv_proxy_lookback_days), 5)
        steps_per_day = {
            "1m": 1440,
            "5m": 288,
            "15m": 96,
            "1h": 24,
            "4h": 6,
            "1d": 1,
        }.get(self.config.backtest.time_step, 24)
        window = max(lookback_days * steps_per_day, 24)

        df = self._underlying_df[["timestamp", "close"]].copy()
        df = df.reset_index(drop=True)
        close = df["close"].astype(float)
        log_ret = np.log(close / close.shift(1))
        annual_factor = np.sqrt(max(365 * steps_per_day, 1))
        rv = log_ret.rolling(window=window, min_periods=max(window // 3, 12)).std() * annual_factor
        rv = rv.clip(lower=self.config.backtest.iv_min, upper=self.config.backtest.iv_max)

        valid_mask = rv.notna()
        if valid_mask.any():
            self._realized_vol_ts = df.loc[valid_mask, "timestamp"].values
            self._realized_vol_values = rv.loc[valid_mask].values.astype(float)

        dvol_path_cfg = str(self.config.backtest.dvol_path or "").strip()
        if not dvol_path_cfg:
            return

        dvol_path = Path(dvol_path_cfg)
        if not dvol_path.is_absolute():
            dvol_path = Path("data") / dvol_path
        if not dvol_path.exists():
            logger.warning(f"DVOL file not found: {dvol_path}, fallback to realized-vol only")
            return

        try:
            if dvol_path.suffix.lower() == ".parquet":
                ddf = pd.read_parquet(dvol_path)
            else:
                ddf = pd.read_csv(dvol_path)
        except Exception as exc:
            logger.warning(f"Failed to load DVOL file {dvol_path}: {exc}")
            return

        if ddf.empty:
            return

        ts_col = "timestamp" if "timestamp" in ddf.columns else ddf.columns[0]
        val_col = "dvol" if "dvol" in ddf.columns else (
            "value" if "value" in ddf.columns else ddf.columns[1] if len(ddf.columns) > 1 else None
        )
        if val_col is None:
            logger.warning(f"DVOL file {dvol_path} has no value column")
            return

        ddf[ts_col] = pd.to_datetime(ddf[ts_col], utc=True, errors="coerce")
        ddf[val_col] = pd.to_numeric(ddf[val_col], errors="coerce")
        ddf = ddf.dropna(subset=[ts_col, val_col]).sort_values(by=[ts_col])
        if ddf.empty:
            return

        vals = ddf[val_col].astype(float).values
        if np.nanmedian(vals) > 3.0:
            vals = vals / 100.0
        vals = np.clip(vals, self.config.backtest.iv_min, self.config.backtest.iv_max)

        self._dvol_ts = ddf[ts_col].values
        self._dvol_values = vals

    def _resolve_proxy_iv(self, ts_np, underlying_price: float, strike: float, maturity_years: float) -> float:
        """Resolve proxy IV(K,T,t) without requiring historical option OHLCV."""
        base_iv = float(self.config.backtest.fixed_iv)
        if underlying_price <= 0 or strike <= 0 or maturity_years <= 0:
            return base_iv

        ts64 = pd.Timestamp(ts_np, tz="UTC").to_datetime64()

        anchor_iv = np.nan
        if self._dvol_ts.size > 0:
            j = np.searchsorted(self._dvol_ts, ts64, side="right") - 1
            if j >= 0:
                anchor_iv = float(self._dvol_values[j])

        if not np.isfinite(anchor_iv) and self._realized_vol_ts.size > 0:
            j = np.searchsorted(self._realized_vol_ts, ts64, side="right") - 1
            if j >= 0:
                anchor_iv = float(self._realized_vol_values[j])

        if not np.isfinite(anchor_iv):
            anchor_iv = base_iv

        log_m = abs(float(np.log(strike / underlying_price)))
        target_t = 30.0 / 365.25
        term_diff = float(np.sqrt(max(maturity_years, 1e-6)) - np.sqrt(target_t))

        smile_mult = 1.0 + float(self.config.backtest.iv_smile_slope) * log_m
        term_mult = 1.0 + float(self.config.backtest.iv_term_slope) * term_diff

        iv = anchor_iv * smile_mult * term_mult
        return float(np.clip(iv, self.config.backtest.iv_min, self.config.backtest.iv_max))

    # ------------------------------------------------------------------
    # Results packaging
    # ------------------------------------------------------------------

    def _build_results(self) -> dict:
        equity = self.account.equity_history
        if not equity:
            return {"total_return": 0.0}

        eq_values = [e[1] for e in equity]
        initial = self.account.initial_balance
        final = eq_values[-1]

        # USD equity depends on margin mode
        if self._margin_usd:
            # Equity IS already in USD
            initial_usd = initial
            final_usd = final
        else:
            # Coin margin: convert to USD
            initial_underlying = equity[0][4] if len(equity[0]) > 4 and equity[0][4] > 0 else 1.0
            final_underlying = equity[-1][4] if len(equity[-1]) > 4 and equity[-1][4] > 0 else 1.0
            initial_usd = initial * initial_underlying
            final_usd = final * final_underlying

        # Data source statistics
        mark_total = self._mark_source_market + self._mark_source_synth
        quote_total = self._quote_source_market + self._quote_source_synth
        chain_total = self._chain_source_market + self._chain_source_synth

        return {
            "underlying": self.config.backtest.underlying,
            "margin_mode": "USD" if self._margin_usd else "coin",
            "initial_balance": initial,
            "final_equity": final,
            "total_return": (final - initial) / initial if initial > 0 else 0.0,
            "initial_usd": initial_usd,
            "final_usd": final_usd,
            "total_return_usd": (final_usd - initial_usd) / initial_usd if initial_usd > 0 else 0.0,
            "total_fee_paid": self.account.total_fee_paid,
            "total_trades": len(self.position_mgr.closed_trades),
            "equity_history": equity,
            "closed_trades": self.position_mgr.closed_trades,
            "data_source": {
                "mark_market": self._mark_source_market,
                "mark_synth": self._mark_source_synth,
                "mark_total": mark_total,
                "mark_market_pct": self._mark_source_market / mark_total if mark_total > 0 else 0.0,
                "quote_market": self._quote_source_market,
                "quote_synth": self._quote_source_synth,
                "quote_total": quote_total,
                "quote_market_pct": self._quote_source_market / quote_total if quote_total > 0 else 0.0,
                "chain_market": self._chain_source_market,
                "chain_synth": self._chain_source_synth,
                "chain_total": chain_total,
                "chain_market_pct": self._chain_source_market / chain_total if chain_total > 0 else 0.0,
            },
        }
