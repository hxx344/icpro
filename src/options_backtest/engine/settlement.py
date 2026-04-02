"""Expiry settlement logic.

Handles automatic exercise / expiry of Deribit options at UTC 08:00
on the expiration date.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from loguru import logger

from options_backtest.engine.account import Account
from options_backtest.engine.matcher import Matcher
from options_backtest.engine.position import PositionManager
from options_backtest.utils import to_utc_timestamp


def check_and_settle(
    timestamp,
    position_mgr: PositionManager,
    account: Account,
    matcher: Matcher,
    instruments: pd.DataFrame | dict,
    settlements_df: pd.DataFrame,
    *,
    margin_usd: bool = False,
    settlement_index: dict[str, float] | None = None,
) -> list[str]:
    """Check for expired positions and settle them.

    Parameters
    ----------
    timestamp : current backtest time (UTC) – datetime, pd.Timestamp, or numpy datetime64
    position_mgr : the position manager
    account : the account to credit / debit
    matcher : used for delivery fee calculation
    instruments : instrument catalogue (DataFrame or pre-built dict[name→row])
    settlements_df : settlement / delivery prices
    settlement_index : optional pre-built dict[instrument_name → index_price] for O(1) lookup

    Returns
    -------
    List of instrument names that were settled.
    """
    ts = to_utc_timestamp(timestamp)
    settled: list[str] = []

    # Support both DataFrame and dict formats
    if isinstance(instruments, dict):
        inst_dict = instruments
    else:
        inst_dict = None

    # Check each open position
    for name in list(position_mgr.positions.keys()):
        # Find instrument info (dict lookup = O(1))
        if inst_dict is not None:
            inst_data = inst_dict.get(name)
            if inst_data is None:
                continue
            # inst_data is a dict
            # Fast path: use pre-computed _expiry_ns (nanoseconds)
            exp_ns = inst_data.get("_expiry_ns")
            if exp_ns is not None:
                # Deribit settles at UTC 08:00 on expiry date
                # Convert ns to pd.Timestamp for settlement_time calculation
                expiry = pd.Timestamp(exp_ns, unit="ns", tz="UTC")
            else:
                exp_col = "expiration_timestamp" if "expiration_timestamp" in inst_data else "expiration_date"
                expiry = to_utc_timestamp(inst_data[exp_col])
            strike = inst_data.get("strike_price", inst_data.get("strike", 0))
            opt_type_raw = inst_data.get("option_type", "call")
        else:
            inst_rows = instruments[instruments["instrument_name"] == name]
            if inst_rows.empty:
                continue
            inst = inst_rows.iloc[0]
            exp_col = "expiration_timestamp" if "expiration_timestamp" in inst.index else "expiration_date"
            expiry = to_utc_timestamp(inst[exp_col])
            strike = inst.get("strike", inst.get("strike_price", 0))
            opt_type_raw = inst.get("option_type", "call")

        # Deribit settles at UTC 08:00 on expiry date
        settlement_time = expiry.replace(hour=8, minute=0, second=0, microsecond=0)

        if ts < settlement_time:
            continue

        # Fast path: use pre-built settlement index (O(1) dict lookup)
        settlement_price = None
        if settlement_index is not None:
            settlement_price = settlement_index.get(name)
        # Fallback to DataFrame scan if not found
        if settlement_price is None:
            settlement_price = _find_settlement_price(settlements_df, expiry, name)
        if settlement_price is None:
            logger.warning(f"No settlement price found for {name}, skipping")
            continue

        # Deribit instrument_name encodes type as last char:
        # e.g. BTC-26MAR26-80000-C → "C" = call
        if isinstance(opt_type_raw, str):
            opt_type = opt_type_raw
        else:
            opt_type = "call" if name.endswith("-C") else "put"

        delivery_fee = matcher.cfg.delivery_fee
        delivery_fee_max_pct = matcher.cfg.delivery_fee_max_pct
        # USD margin: convert delivery fee from coin to USD
        if margin_usd:
            delivery_fee = delivery_fee * settlement_price

        pnl = position_mgr.settle_expired(
            instrument_name=name,
            settlement_price_usd=settlement_price,
            strike_price=strike,
            option_type=opt_type,
            timestamp=timestamp,
            delivery_fee_per_qty=delivery_fee,
            delivery_fee_max_pct=delivery_fee_max_pct,
            margin_usd=margin_usd,
        )

        account.balance += pnl
        settled.append(name)

    if settled:
        logger.debug(f"Settled {len(settled)} position(s) at {timestamp}")

    return settled


def _find_settlement_price(
    settlements_df: pd.DataFrame,
    expiry: pd.Timestamp,
    instrument_name: str,
) -> float | None:
    """Look up the delivery / index price for a given instrument at expiry."""
    if settlements_df.empty:
        return None

    # Try matching by instrument name directly (our settlement data has this)
    if "instrument_name" in settlements_df.columns:
        matches = settlements_df[settlements_df["instrument_name"] == instrument_name]
        if not matches.empty:
            if "index_price" in matches.columns:
                return float(matches.iloc[0]["index_price"])

    # Fallback: match by date
    for date_col in ("settlement_timestamp", "date", "expiration_date", "delivery_date"):
        if date_col in settlements_df.columns:
            target = expiry.normalize()
            try:
                col_ts = pd.to_datetime(settlements_df[date_col])
                matches = settlements_df[col_ts.dt.normalize() == target]
            except Exception:
                continue
            if not matches.empty:
                for price_col in ("index_price", "delivery_price", "price"):
                    if price_col in matches.columns:
                        return float(matches.iloc[0][price_col])

    return None
