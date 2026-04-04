"""Helpers for computing dashboard preview metrics for option orders."""

from __future__ import annotations

import math
from typing import Any

from trader.binance_client import OptionTicker
from trader.strategy import estimate_binance_combo_open_margin_per_unit


def compute_option_order_preview(
    *,
    mode: str,
    spot: float,
    base_quantity: float,
    compound: bool,
    equity: float = 0.0,
    available_balance: float = 0.0,
    leverage: float = 1.0,
    max_capital_pct: float = 0.0,
    otm_pct: float = 0.0,
    wing_width_pct: float = 0.0,
    sell_call: OptionTicker,
    sell_put: OptionTicker,
    buy_call: OptionTicker | None = None,
    buy_put: OptionTicker | None = None,
) -> dict[str, Any]:
    """Compute preview quantity, premium and margin metrics for dashboard use."""
    is_iron_condor = buy_call is not None and buy_put is not None
    quantity = float(base_quantity or 0.0)
    equity_source = "配置基础数量"
    equity = max(float(equity or 0.0), 0.0)
    available_balance = max(float(available_balance or 0.0), 0.0)
    spot = max(float(spot or 0.0), 0.0)

    margin_per_unit = estimate_binance_combo_open_margin_per_unit(
        index_price=spot,
        sell_call=sell_call,
        sell_put=sell_put,
        buy_call=buy_call,
        buy_put=buy_put,
    )

    if compound and equity > 0 and spot > 0:
        mode_norm = str(mode or "").lower()
        if mode_norm == "weekend_vol":
            raw_qty = (equity * float(leverage or 0.0)) / spot
            quantity = math.floor(raw_qty * 10) / 10
            qty_by_margin = 0.0
            if margin_per_unit > 0:
                budget = available_balance if available_balance > 0 else equity
                qty_by_margin = math.floor(budget / margin_per_unit * 10) / 10
                quantity = min(quantity, qty_by_margin)
            equity_source = f"实时权益(复利 {float(leverage or 0.0):.0f}x)"
        else:
            margin_budget = equity * float(max_capital_pct or 0.0)
            if available_balance > 0:
                margin_budget = min(margin_budget, available_balance)
            if margin_per_unit <= 0:
                if mode_norm == "strangle":
                    margin_per_unit = float(otm_pct or 0.0) * spot * 2
                else:
                    wing_width = float(wing_width_pct or 0.0) * spot
                    margin_per_unit = wing_width * 2 if wing_width > 0 else float(otm_pct or 0.0) * spot * 2
            if margin_per_unit > 0:
                scaled = math.floor(margin_budget / margin_per_unit * 100) / 100
                quantity = scaled if 0 < scaled < base_quantity else max(base_quantity, scaled)
            equity_source = "实时权益(复利)"

    sc_bid = float(sell_call.bid_price or 0.0)
    sp_bid = float(sell_put.bid_price or 0.0)
    premium_per_unit = sc_bid + sp_bid
    total_max_loss = None

    if not is_iron_condor:
        margin_used = margin_per_unit * quantity if margin_per_unit > 0 else float(otm_pct or 0.0) * spot * 2 * quantity
    else:
        assert buy_call is not None
        assert buy_put is not None
        lc_ask = float(buy_call.ask_price or 0.0)
        lp_ask = float(buy_put.ask_price or 0.0)
        premium_per_unit = (sc_bid + sp_bid) - (lc_ask + lp_ask)
        call_width = float(buy_call.strike or 0.0) - float(sell_call.strike or 0.0)
        put_width = float(sell_put.strike or 0.0) - float(buy_put.strike or 0.0)
        max_wing = max(call_width, put_width)
        total_max_loss = (max_wing - premium_per_unit) * quantity
        margin_used = margin_per_unit * quantity if margin_per_unit > 0 else max_wing * 2 * quantity

    return {
        "quantity": quantity,
        "equity": equity,
        "available_balance": available_balance,
        "equity_source": equity_source,
        "margin_per_unit": margin_per_unit,
        "margin_used": margin_used,
        "premium_per_unit": premium_per_unit,
        "total_premium": premium_per_unit * quantity,
        "break_even_upper": float(sell_call.strike or 0.0) + premium_per_unit,
        "break_even_lower": float(sell_put.strike or 0.0) - premium_per_unit,
        "total_max_loss": total_max_loss,
        "is_iron_condor": is_iron_condor,
    }
