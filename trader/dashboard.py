"""Trader Dashboard – 交易管理面板.

Streamlit 前端，用于:
- 实时查看策略状态和持仓
- 资产曲线图表
- 交易统计与历史
- 成交历史查询
- 手动操作 (平仓、暂停)

Usage:
    streamlit run trader/dashboard.py -- --config configs/trader/weekend_vol_btc.yaml
"""

from __future__ import annotations

import base64
import json
import math
import os
import platform
import sys
import threading
import time as _time
import traceback
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, cast

# ---------------------------------------------------------------------------
# OS-aware Streamlit server config (injected before Streamlit reads config)
# ---------------------------------------------------------------------------
# Streamlit reads STREAMLIT_SERVER_* env vars as config overrides.
# We detect the OS here and set sensible defaults so .streamlit/config.toml
# stays platform-agnostic (only theme / browser settings).
# ---------------------------------------------------------------------------
_is_linux = platform.system() == "Linux"
_dashboard_public = os.environ.get("DASHBOARD_PUBLIC", "false").strip().lower() in {"1", "true", "yes", "on"}
_allow_no_auth = os.environ.get("DASHBOARD_ALLOW_NO_AUTH", "false").strip().lower() in {"1", "true", "yes", "on"}

def _set_default(env_key: str, value: str) -> None:
    """Set env var only if not already set (allow manual override)."""
    if env_key not in os.environ:
        os.environ[env_key] = value

if _is_linux:
    # Linux default: safe local bind. Public exposure must be explicit.
    _set_default("STREAMLIT_SERVER_HEADLESS", "true")
    _set_default("STREAMLIT_SERVER_ADDRESS", "0.0.0.0" if _dashboard_public else "127.0.0.1")
    _set_default("STREAMLIT_SERVER_PORT", "8501")
    _set_default("STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION", "true")
else:
    # Windows / macOS: open browser, localhost only
    _set_default("STREAMLIT_SERVER_HEADLESS", "false")
    _set_default("STREAMLIT_SERVER_ADDRESS", "localhost")
    _set_default("STREAMLIT_SERVER_PORT", "8501")

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trader.config import load_config
from trader.storage import Storage
from trader.engine import get_engine, reset_engine, TradingEngine
from trader.bybit_client import BybitOptionsClient, OptionTicker
from trader.dashboard_expiry import resolve_test_order_expiry_target, summarize_available_expiries
from trader.position_manager import PositionManager, OptionPosition, PositionLeg
from trader.order_preview import compute_option_order_preview

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="期权交易面板",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Config & Storage initialization
# ---------------------------------------------------------------------------

@st.cache_resource
def init_storage(db_path: str) -> Storage:
    """Initialize SQLite storage (cached across reruns)."""
    return Storage(db_path)


_TEST_ORDER_TASKS: dict[str, dict] = {}
_TEST_ORDER_TASKS_LOCK = threading.Lock()


def _get_test_order_task(task_id: str | None) -> dict | None:
    if not task_id:
        return None
    with _TEST_ORDER_TASKS_LOCK:
        task = _TEST_ORDER_TASKS.get(task_id)
        if task is None:
            return None
        snapshot = dict(task)
        snapshot["legs"] = [dict(leg) for leg in task.get("legs", [])]
        return snapshot


def _get_latest_test_order_task_id() -> str | None:
    with _TEST_ORDER_TASKS_LOCK:
        if not _TEST_ORDER_TASKS:
            return None
        return max(
            _TEST_ORDER_TASKS.items(),
            key=lambda item: str(item[1].get("updated_at") or item[1].get("created_at") or ""),
        )[0]


def _update_test_order_task(task_id: str, **updates) -> None:
    with _TEST_ORDER_TASKS_LOCK:
        task = _TEST_ORDER_TASKS.setdefault(task_id, {})
        task.update(updates)
        task["updated_at"] = datetime.now(timezone.utc).isoformat()


def _format_test_order_message(progress: dict) -> str:
    event = progress.get("event", "")
    if progress.get("message"):
        return str(progress["message"])
    mapping = {
        "started": "开始追单",
        "initial_order": "已提交初始委托",
        "poll": "正在检查成交状态",
        "amend_cycle": "正在更新挂单价格",
        "market_fallback_zone": "进入兜底成交阶段",
        "deadline_forced_market": "已强制进入兜底成交",
        "finished": "追单结束",
        "market_order_submit": "正在发送市价单",
        "market_order_filled": "市价单已成交",
        "market_order_result": "市价单返回结果",
        "position_open_start": "开始创建测试仓位",
        "position_open_success": "测试仓位已成交",
        "position_open_failed": "测试仓位创建失败，已回滚",
        "position_open_error": "测试下单异常",
    }
    return mapping.get(event, "测试下单处理中")


def _infer_underlying_from_symbols(symbols: list[str]) -> str:
    for symbol in symbols:
        _symbol = str(symbol or "").strip().upper()
        if _symbol and "-" in _symbol:
            return _symbol.split("-", 1)[0]
    return ""


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return default


def _exchange_position_snapshot_rows(positions: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "合约": p.get("symbol", ""),
            "方向": p.get("side", ""),
            "数量": _to_float(p.get("quantity"), 0.0),
            "开仓价": f"${_to_float(p.get('entryPrice'), 0.0):.4f}",
            "未实现PnL": f"${_to_float(p.get('unrealizedPnl'), 0.0):.4f}",
        }
        for p in positions
    ]


def _get_test_stop_loss_runtime(task: dict, engine_ref: TradingEngine) -> dict[str, object]:
    order_summary = task.get("order_summary") or {}
    group_id = str(task.get("result_group_id") or "")
    attached = bool(task.get("attached_to_engine_for_stop_loss"))
    stop_loss_pct = float(order_summary.get("stop_loss_pct") or 0.0)
    stop_loss_underlying_move_pct = float(order_summary.get("stop_loss_underlying_move_pct") or 0.0)

    runtime: dict[str, object] = {
        "attached": attached,
        "group_id": group_id,
        "stop_loss_pct": stop_loss_pct,
        "stop_loss_underlying_move_pct": stop_loss_underlying_move_pct,
        "quantity": float(order_summary.get("quantity") or 0.0),
        "state": "not_attached" if not attached else "unknown",
    }
    if not attached:
        return runtime
    if not group_id:
        runtime["state"] = "missing_group_id"
        return runtime
    if not engine_ref or not engine_ref.is_running or engine_ref.pos_mgr is None or engine_ref.client is None:
        runtime["state"] = "engine_unavailable"
        return runtime

    position = engine_ref.pos_mgr.open_positions.get(group_id)
    if position is None or not position.is_open:
        runtime["state"] = "not_open"
        return runtime

    runtime["state"] = "active"
    runtime["legs"] = len(position.legs)
    runtime["entry_credit"] = float(position.total_premium or 0.0)
    runtime["entry_underlying_price"] = float(position.underlying_price or 0.0)
    runtime["sell_put_strike"] = float(position.sell_put.strike) if position.sell_put else None
    runtime["sell_call_strike"] = float(position.sell_call.strike) if position.sell_call else None

    underlying = _infer_underlying_from_symbols([leg.symbol for leg in position.legs])
    runtime["underlying"] = underlying
    if not underlying:
        runtime["state"] = "active_no_underlying"
        return runtime

    try:
        current_spot = float(engine_ref.client.get_spot_price(underlying) or 0.0)
    except Exception as exc:
        runtime["spot_price_error"] = str(exc)
        current_spot = 0.0
    if current_spot > 0:
        runtime["current_spot"] = current_spot
        entry_underlying_price = float(position.underlying_price or 0.0)
        if entry_underlying_price > 0:
            underlying_move_pct = abs((current_spot / entry_underlying_price - 1.0) * 100.0)
            runtime["underlying_move_pct"] = abs(
                (current_spot / entry_underlying_price - 1.0) * 100.0
            )
            runtime["underlying_move_filter_passed"] = (
                stop_loss_underlying_move_pct <= 0
                or underlying_move_pct >= stop_loss_underlying_move_pct
            )

    try:
        mark_prices = engine_ref.client.get_mark_prices(underlying)
    except Exception as exc:
        runtime["mark_prices_error"] = str(exc)
        return runtime

    entry_credit = 0.0
    close_cost = 0.0
    for leg in position.legs:
        sign = 1.0 if leg.side == "SELL" else -1.0
        mark = float(mark_prices.get(leg.symbol, leg.entry_price) or leg.entry_price)
        entry_credit += float(leg.entry_price) * float(leg.quantity) * sign
        close_cost += mark * float(leg.quantity) * sign

    basket_pnl = entry_credit - close_cost
    runtime["entry_credit"] = entry_credit
    runtime["close_cost"] = close_cost
    runtime["basket_pnl"] = basket_pnl
    runtime["basket_pnl_pct"] = (basket_pnl / entry_credit * 100.0) if entry_credit > 0 else None
    if stop_loss_pct > 0 and entry_credit > 0:
        quantity = float(order_summary.get("quantity") or 0.0)
        stop_loss_trigger_close_cost = entry_credit * (1 + stop_loss_pct / 100.0)
        stop_loss_trigger_price_boundary = (
            stop_loss_trigger_close_cost / quantity
            if quantity > 0
            else None
        )
        runtime["stop_loss_trigger_close_cost"] = stop_loss_trigger_close_cost
        runtime["stop_loss_trigger_price_boundary"] = stop_loss_trigger_price_boundary
        runtime["stop_loss_trigger_pnl"] = -(entry_credit * stop_loss_pct / 100.0)
        sell_put_strike = float(position.sell_put.strike) if position.sell_put else None
        sell_call_strike = float(position.sell_call.strike) if position.sell_call else None
        if sell_put_strike is not None:
            runtime["stop_loss_trigger_spot_low"] = sell_put_strike - float(stop_loss_trigger_price_boundary or 0.0)
        if sell_call_strike is not None:
            runtime["stop_loss_trigger_spot_high"] = sell_call_strike + float(stop_loss_trigger_price_boundary or 0.0)
    return runtime


def _build_expiry_payoff_figure(
    legs: list[dict],
    underlying: str,
    chart_title: str,
    spot_price: float = 0.0,
) -> go.Figure | None:
    """Build an expiry payoff chart from current position legs."""
    import numpy as _np

    valid_legs = [
        leg for leg in legs
        if float(leg.get("quantity") or 0.0) > 0 and float(leg.get("strike") or 0.0) > 0
    ]
    if not valid_legs:
        return None

    strikes = [float(leg["strike"]) for leg in valid_legs]
    anchors = list(strikes)
    if spot_price > 0:
        anchors.append(float(spot_price))

    price_lo_anchor = min(anchors)
    price_hi_anchor = max(anchors)
    span = max(price_hi_anchor - price_lo_anchor, max(price_hi_anchor * 0.12, 100.0))
    price_lo = max(0.0, price_lo_anchor - span * 0.25)
    price_hi = price_hi_anchor + span * 0.25
    prices = _np.linspace(price_lo, price_hi, 600)

    pnl = _np.zeros_like(prices, dtype=float)
    for leg in valid_legs:
        strike = float(leg.get("strike") or 0.0)
        quantity = float(leg.get("quantity") or 0.0)
        entry_price = float(leg.get("entry_price") or 0.0)
        option_type = str(leg.get("option_type") or "").lower()
        side = str(leg.get("side") or "").upper()

        if option_type == "call":
            intrinsic = _np.maximum(prices - strike, 0.0)
        else:
            intrinsic = _np.maximum(strike - prices, 0.0)

        leg_pnl = (entry_price - intrinsic) if side == "SELL" else (intrinsic - entry_price)
        pnl += leg_pnl * quantity

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prices,
        y=pnl,
        mode="lines",
        line=dict(color="royalblue", width=2.5),
        name="P&L",
        hovertemplate=f"{underlying}: $%{{x:,.0f}}<br>P&L: $%{{y:,.2f}}<extra></extra>",
    ))

    pnl_pos = _np.where(pnl > 0, pnl, 0)
    fig.add_trace(go.Scatter(
        x=prices, y=pnl_pos,
        fill="tozeroy",
        fillcolor="rgba(0,200,83,0.15)",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    pnl_neg = _np.where(pnl < 0, pnl, 0)
    fig.add_trace(go.Scatter(
        x=prices, y=pnl_neg,
        fill="tozeroy",
        fillcolor="rgba(255,82,82,0.15)",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    show_strike_labels = len(valid_legs) <= 8
    seen_markers: set[tuple[float, str, str]] = set()
    for leg in sorted(valid_legs, key=lambda item: float(item.get("strike") or 0.0)):
        strike = float(leg.get("strike") or 0.0)
        side = str(leg.get("side") or "").upper()
        option_type = str(leg.get("option_type") or "").lower()
        marker_key = (strike, side, option_type)
        if marker_key in seen_markers:
            continue
        seen_markers.add(marker_key)
        role = str(leg.get("role") or f"{side} {option_type}")
        fig.add_vline(
            x=strike,
            line_dash="dot",
            line_color="green" if side == "BUY" else "red",
            opacity=0.5,
            annotation_text=f"{role} ${strike:,.0f}" if show_strike_labels else None,
            annotation_position="top",
        )

    if spot_price > 0:
        fig.add_vline(
            x=spot_price,
            line_dash="dash",
            line_color="orange",
            opacity=0.7,
            annotation_text=f"Spot ${spot_price:,.0f}",
            annotation_position="bottom right",
        )

    crossings = []
    for idx in range(len(prices) - 1):
        y0 = float(pnl[idx])
        y1 = float(pnl[idx + 1])
        if abs(y0) < 1e-9:
            crossings.append(float(prices[idx]))
            continue
        if y0 * y1 < 0:
            x0 = float(prices[idx])
            x1 = float(prices[idx + 1])
            root = x0 - y0 * (x1 - x0) / (y1 - y0)
            crossings.append(root)

    dedup_crossings: list[float] = []
    for root in crossings:
        if all(abs(root - existing) > 1.0 for existing in dedup_crossings):
            dedup_crossings.append(root)

    for root in dedup_crossings[:6]:
        fig.add_vline(
            x=root,
            line_dash="dashdot",
            line_color="purple",
            opacity=0.35,
            annotation_text=f"BE ${root:,.0f}",
        )

    fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.3)
    fig.update_layout(
        title=chart_title,
        xaxis_title=f"{underlying} 到期价格 ($)",
        yaxis_title="盈亏 ($)",
        height=420,
        hovermode="x unified",
        template="plotly_white",
        showlegend=False,
        margin=dict(l=50, r=20, t=56, b=20),
    )
    return fig


def _start_test_order_task(
    task_id: str,
    exchange_cfg,
    storage: Storage,
    order_params: dict,
    engine_ref: TradingEngine,
    attach_to_engine_for_stop_loss: bool = False,
) -> None:
    def _runner() -> None:
        last_progress: dict[str, object] = {}

        def _infer_underlying_from_order() -> str:
            for _key in ("sell_call_symbol", "sell_put_symbol", "buy_call_symbol", "buy_put_symbol"):
                _symbol = str(order_params.get(_key) or "").strip().upper()
                if _symbol and "-" in _symbol:
                    return _symbol.split("-", 1)[0]
            return ""

        def _capture_exchange_positions_snapshot() -> dict[str, object]:
            _underlying = _infer_underlying_from_order()
            if not _underlying:
                return {
                    "exchange_positions_checked": False,
                    "exchange_positions_error": "无法从测试订单参数推断标的，未执行持仓复核",
                }
            try:
                _positions = task_client.get_positions(_underlying)
                return {
                    "exchange_positions_checked": True,
                    "exchange_positions_underlying": _underlying,
                    "exchange_positions": _positions,
                    "exchange_positions_count": len(_positions),
                }
            except Exception as _snapshot_exc:
                return {
                    "exchange_positions_checked": True,
                    "exchange_positions_underlying": _underlying,
                    "exchange_positions_error": str(_snapshot_exc),
                }

        def _recover_test_position_from_snapshot(snapshot: dict[str, object]) -> OptionPosition | None:
            _positions = snapshot.get("exchange_positions") or []
            if not isinstance(_positions, list) or not _positions:
                return None

            _expected_roles = [
                ("sell_put", order_params.get("sell_put_symbol"), order_params.get("sell_put_strike"), "put", "SELL"),
                ("sell_call", order_params.get("sell_call_symbol"), order_params.get("sell_call_strike"), "call", "SELL"),
            ]
            if order_params.get("buy_put_symbol"):
                _expected_roles.append(("buy_put", order_params.get("buy_put_symbol"), order_params.get("buy_put_strike"), "put", "BUY"))
            if order_params.get("buy_call_symbol"):
                _expected_roles.append(("buy_call", order_params.get("buy_call_symbol"), order_params.get("buy_call_strike"), "call", "BUY"))

            _positions_by_symbol = {
                str(p.get("symbol") or "").upper(): p
                for p in _positions
                if abs(float(p.get("quantity") or 0.0)) > 0
            }

            _matched: list[tuple[str, str, float | None, str, str, dict]] = []
            for _role, _symbol, _strike, _option_type, _side in _expected_roles:
                _sym = str(_symbol or "").upper()
                _pos = _positions_by_symbol.get(_sym)
                if _pos is None:
                    return None
                _matched.append((_role, _sym, _strike, _option_type, _side, _pos))

            _group_id = str(last_progress.get("group_id") or f"REC_{task_id[:8]}")
            _position = OptionPosition(
                group_id=_group_id,
                entry_time=datetime.now(timezone.utc),
                underlying_price=float(order_params.get("underlying_price") or 0.0),
                is_open=True,
            )
            _total_premium = 0.0
            for _role, _sym, _strike, _option_type, _side, _pos in _matched:
                _entry_price = float(_pos.get("entryPrice") or 0.0)
                _qty = abs(float(_pos.get("quantity") or order_params.get("quantity") or 0.0))
                _leg = PositionLeg(
                    symbol=_sym,
                    side=_side,
                    option_type=_option_type,
                    strike=float(_strike or 0.0),
                    quantity=_qty,
                    entry_price=_entry_price,
                    trade_id=0,
                    order_id="",
                )
                setattr(_position, _role, _leg)
                _total_premium += _entry_price * _qty if _side == "SELL" else -_entry_price * _qty
            _position.total_premium = _total_premium
            return _position

        def _on_progress(progress: dict) -> None:
            last_progress.clear()
            last_progress.update(progress)
            legs = progress.get("legs") or []
            fill_ratio = float(progress.get("fill_ratio") or 0.0)
            _update_test_order_task(
                task_id,
                state="running",
                status=progress.get("event", "running"),
                message=_format_test_order_message(progress),
                percent=min(max(int(fill_ratio * 100), 0), 100),
                legs=[dict(leg) for leg in legs],
                progress=progress,
            )

        try:
            _update_test_order_task(
                task_id,
                state="running",
                status="starting",
                message="正在初始化测试下单任务",
                percent=0,
            )
            task_client = BybitOptionsClient(exchange_cfg)
            task_pos_mgr = PositionManager(
                task_client,
                storage,
            )

            _is_strangle = not order_params.get("buy_call_symbol") and not order_params.get("buy_put_symbol")
            if _is_strangle:
                position = task_pos_mgr.open_short_strangle(
                    status_callback=_on_progress,
                    sell_call_symbol=order_params["sell_call_symbol"],
                    sell_put_symbol=order_params["sell_put_symbol"],
                    sell_call_strike=order_params["sell_call_strike"],
                    sell_put_strike=order_params["sell_put_strike"],
                    quantity=order_params["quantity"],
                    underlying_price=order_params["underlying_price"],
                )
            else:
                position = task_pos_mgr.open_winged_position(
                    status_callback=_on_progress,
                    **order_params,
                )

            if position is not None:
                _group_id = str(getattr(position, "group_id", ""))
                if attach_to_engine_for_stop_loss and engine_ref.pos_mgr is not None:
                    engine_ref.pos_mgr.open_positions[_group_id] = position
                _snapshot = _capture_exchange_positions_snapshot()
                _update_test_order_task(
                    task_id,
                    state="success",
                    status="success",
                    message=(
                        f"测试下单成功：{_group_id}；已接入策略止损管理"
                        if attach_to_engine_for_stop_loss
                        else f"测试下单成功：{_group_id}；未接入自动止损"
                    ),
                    percent=100,
                    result_group_id=_group_id,
                    attached_to_engine_for_stop_loss=attach_to_engine_for_stop_loss,
                    recovered_via_exchange_snapshot=False,
                    **_snapshot,
                )
            else:
                _snapshot = _capture_exchange_positions_snapshot()
                _recovered_position = _recover_test_position_from_snapshot(_snapshot)
                if _recovered_position is not None:
                    _group_id = str(_recovered_position.group_id)
                    if attach_to_engine_for_stop_loss and engine_ref.pos_mgr is not None:
                        engine_ref.pos_mgr.open_positions[_group_id] = _recovered_position
                    _update_test_order_task(
                        task_id,
                        state="success",
                        status="success",
                        message=(
                            f"测试下单已通过交易所持仓复核确认为成功：{_group_id}；已接入策略止损管理"
                            if attach_to_engine_for_stop_loss
                            else f"测试下单已通过交易所持仓复核确认为成功：{_group_id}；未接入自动止损"
                        ),
                        percent=100,
                        result_group_id=_group_id,
                        attached_to_engine_for_stop_loss=attach_to_engine_for_stop_loss,
                        recovered_via_exchange_snapshot=True,
                        **_snapshot,
                    )
                else:
                    _update_test_order_task(
                        task_id,
                        state="failed",
                        status="failed",
                        message=str(last_progress.get("message") or "测试下单失败：至少有一条腿未成交或被回滚"),
                        recovered_via_exchange_snapshot=False,
                        **_snapshot,
                    )
        except Exception as exc:
            _snapshot = _capture_exchange_positions_snapshot()
            _recovered_position = _recover_test_position_from_snapshot(_snapshot)
            if _recovered_position is not None:
                _group_id = str(_recovered_position.group_id)
                if attach_to_engine_for_stop_loss and engine_ref.pos_mgr is not None:
                    engine_ref.pos_mgr.open_positions[_group_id] = _recovered_position
                _update_test_order_task(
                    task_id,
                    state="success",
                    status="success",
                    message=(
                        f"测试下单虽有异常，但交易所持仓复核确认已成功：{_group_id}；已接入策略止损管理"
                        if attach_to_engine_for_stop_loss
                        else f"测试下单虽有异常，但交易所持仓复核确认已成功：{_group_id}；未接入自动止损"
                    ),
                    percent=100,
                    result_group_id=_group_id,
                    attached_to_engine_for_stop_loss=attach_to_engine_for_stop_loss,
                    recovered_via_exchange_snapshot=True,
                    error=str(exc),
                    traceback=traceback.format_exc(),
                    **_snapshot,
                )
            else:
                _update_test_order_task(
                    task_id,
                    state="error",
                    status="error",
                    message=f"测试下单异常: {exc}",
                    error=str(exc),
                    traceback=traceback.format_exc(),
                    recovered_via_exchange_snapshot=False,
                    **_snapshot,
                )

    thread = threading.Thread(target=_runner, name=f"test-order-{task_id[:8]}", daemon=True)
    _update_test_order_task(task_id, thread_name=thread.name)
    thread.start()


def get_config_path() -> str:
    """Get config path from CLI args or default."""
    # streamlit run trader/dashboard.py -- --config path/to/config.yaml
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--config" and i + 1 < len(args):
            return args[i + 1]
    return "configs/trader/weekend_vol_btc.yaml"


config_path = get_config_path()
cfg = load_config(config_path)          # ← This also loads .env into os.environ
storage = init_storage(cfg.storage.db_path)

# ---------------------------------------------------------------------------
# Login authentication (credentials from env: DASHBOARD_USER / DASHBOARD_PASS)
# ---------------------------------------------------------------------------

def _check_login() -> bool:
    """Show login form and validate credentials. Returns True if authenticated."""
    # Load expected credentials from environment
    expected_user = os.environ.get("DASHBOARD_USER", "").strip()
    expected_pass = os.environ.get("DASHBOARD_PASS", "").strip()

    # If no credentials configured, skip auth
    if not expected_user or not expected_pass:
        if _dashboard_public and not _allow_no_auth:
            st.error("公网访问已启用，但未配置 DASHBOARD_USER / DASHBOARD_PASS。已拒绝访问。")
            st.stop()
        if not _allow_no_auth:
            st.warning("当前未配置 Dashboard 登录凭据，仅建议在本机或受保护网络使用。")
        return True

    # Already authenticated this session
    if st.session_state.get("authenticated"):
        return True

    # --- Login form ---
    st.markdown(
        "<h2 style='text-align:center; margin-top:80px;'>🦅 期权交易面板</h2>"
        "<p style='text-align:center; color:#888;'>请输入用户名和密码登录</p>",
        unsafe_allow_html=True,
    )
    _col_l, col_form, _col_r = st.columns([1, 1.5, 1])
    with col_form:
        with st.form("login_form"):
            username = st.text_input("用户名")
            password = st.text_input("密码", type="password")
            submitted = st.form_submit_button("登录", use_container_width=True, type="primary")

        if submitted:
            if username == expected_user and password == expected_pass:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("用户名或密码错误")

    return False


if not _check_login():
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("📊 期权交易面板")

# --- Mode selector ---
_mode_options = ["🔒 只读模式", "🟢 交易模式"]
_mode_param_to_label = {
    "readonly": "🔒 只读模式",
    "trade": "🟢 交易模式",
}
_mode_label_to_param = {v: k for k, v in _mode_param_to_label.items()}
_mode_from_query = _mode_param_to_label.get(str(st.query_params.get("mode", "")).strip().lower())
if "trading_mode" not in st.session_state:
    st.session_state.trading_mode = _mode_from_query or "🔒 只读模式"
elif _mode_from_query and st.session_state.trading_mode != _mode_from_query:
    st.session_state.trading_mode = _mode_from_query

trading_mode = st.sidebar.radio(
    "运行模式",
    _mode_options,
    index=_mode_options.index(st.session_state.trading_mode),
    horizontal=True,
    help="只读模式: 查看行情/持仓/统计，不可启动引擎或下单\n交易模式: 解锁全部功能",
)
st.session_state.trading_mode = trading_mode
st.query_params["mode"] = _mode_label_to_param.get(trading_mode, "readonly")
is_trade_mode = "交易" in trading_mode
st.sidebar.markdown(f"**策略:** {cfg.name}")
st.sidebar.markdown(f"**标的:** {cfg.strategy.underlying}")
_wing_label = "带翼结构" if getattr(cfg.strategy, 'wing_delta', 0.0) > 0 else "无翼结构"
st.sidebar.markdown(
    f"**模式:** 周末波动率 | **结构:** {_wing_label} | "
    f"**Δ:** {getattr(cfg.strategy, 'target_delta', 0.40):.0%} / {getattr(cfg.strategy, 'wing_delta', 0.05):.0%} | "
    f"**杠杆:** {getattr(cfg.strategy, 'leverage', 1.0):.0f}x"
)
st.sidebar.markdown(f"**数据库:** `{cfg.storage.db_path}`")

st.sidebar.divider()

# ---------------------------------------------------------------------------
# Engine Control (交易引擎控制)
# ---------------------------------------------------------------------------

engine = get_engine(cfg)
engine_status = engine.status()


def _rerun_preserve_nav_page(current_page: str | None = None) -> None:
    st.session_state["nav_page"] = current_page or st.session_state.get("nav_page", "📊 总览")
    st.rerun()

if not is_trade_mode:
    # Readonly mode: show status only, no engine controls
    if engine.is_running:
        st.sidebar.success(f"🟢 引擎运行中  ({engine_status['uptime_str']})")
    else:
        st.sidebar.info("🔒 只读模式 — 切换到交易模式以启动引擎")
elif engine.is_running:
    st.sidebar.success(f"🟢 引擎运行中  ({engine_status['uptime_str']})")
    st.sidebar.caption(
        f"Tick: {engine_status['tick_count']}  |  "
        f"持仓: {engine_status['open_positions']}  |  "
        f"错误: {engine_status['error_count']}"
    )
    if engine_status["last_error"]:
        st.sidebar.warning(f"最近错误: {engine_status['last_error']}")

    col_stop, col_close = st.sidebar.columns(2)
    with col_stop:
        if st.button("⏹ 停止引擎", use_container_width=True, type="secondary"):
            engine.stop()
            _rerun_preserve_nav_page()
    with col_close:
        if st.button("🚨 全部平仓", use_container_width=True, type="primary"):
            pnl = engine.close_all_positions()
            st.sidebar.info(f"已平仓, PnL: ${pnl:,.4f}")
            _rerun_preserve_nav_page()
else:
    if is_trade_mode:
        st.sidebar.error("🔴 引擎未运行")
        if st.sidebar.button("🚀 启动引擎", use_container_width=True, type="primary"):
            ok = engine.start()
            if ok:
                st.sidebar.success("引擎已启动!")
            else:
                st.sidebar.error(f"启动失败: {engine.status().get('last_error', '未知错误')}")
            _rerun_preserve_nav_page()

st.sidebar.divider()

# Date range filter
st.sidebar.subheader("📅 日期筛选")
date_range = st.sidebar.date_input(
    "日期范围",
    value=(
        datetime.now(timezone.utc).date() - timedelta(days=30),
        datetime.now(timezone.utc).date(),
    ),
    max_value=datetime.now(timezone.utc).date(),
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date = str(date_range[0])
    end_date = str(date_range[1]) + "T23:59:59"
else:
    start_date = None
    end_date = None

st.sidebar.divider()

auto_refresh = st.sidebar.checkbox("⏱ 自动刷新", value=True)
refresh_sec = st.sidebar.selectbox("刷新间隔", [10, 30, 60, 120], index=0)
_current_nav_page = st.session_state.get("nav_page", "📊 总览")
_pause_auto_refresh = _current_nav_page in {"🔧 策略配置", "📋 成交历史"}

if auto_refresh and not _pause_auto_refresh:
    # Fragment-based auto-refresh: triggers full app rerun without hard browser reload,
    # preserving session_state (selected page, mode, etc.)
    st.session_state["_app_run_ts"] = _time.time()

    @st.fragment(run_every=timedelta(seconds=refresh_sec))
    def _auto_refresh():
        # Only trigger full rerun on timer-based fragment reruns, not the initial script run
        if _time.time() - st.session_state.get("_app_run_ts", 0) > 2:
            st.rerun(scope="app")

    _auto_refresh()
    st.sidebar.caption(f"每 {refresh_sec} 秒自动刷新")
elif auto_refresh and _pause_auto_refresh:
    _pause_reason = {
        "🔧 策略配置": "策略配置页已暂停整页自动刷新，避免表单/下单预览卡顿",
        "📋 成交历史": "成交历史页已暂停整页自动刷新，避免导出 CSV 时媒体文件失效",
    }
    st.sidebar.caption(_pause_reason.get(_current_nav_page, "当前页面已暂停整页自动刷新"))

# Manual refresh
if st.sidebar.button("🔄 立即刷新", use_container_width=True):
    st.rerun()

# --------------------------------------------------------------------------
# Navigation
# --------------------------------------------------------------------------

if st.session_state.get("nav_page") == "💰 损益记录":
    st.session_state["nav_page"] = "📊 总览"

page = st.sidebar.radio(
    "导航",
    ["📊 总览", "📈 资产曲线", "📋 成交历史", "🔧 策略配置", "🖥 引擎状态"],
    key="nav_page",
)

# ==========================================================================
# PAGE: 总览 (Overview)
# ==========================================================================

if page == "📊 总览":
    st.title("📊 策略总览")

    @st.cache_resource
    def _get_client(_exchange_cfg) -> BybitOptionsClient:
        return BybitOptionsClient(_exchange_cfg)

    client = _get_client(cfg.exchange)

    # --- Fetch LIVE exchange data for KPIs ---
    has_creds = bool(cfg.exchange.api_key and cfg.exchange.api_secret)
    live_account = None
    exchange_positions = []

    if has_creds and not cfg.exchange.simulate_private:
        # 1) Live account balance
        try:
            _acct = client.get_account()
            if not _acct.raw.get("simulated"):
                live_account = _acct
        except Exception:
            pass

        # 2) Live exchange positions
        try:
            if hasattr(client, "get_positions"):
                exchange_positions = client.get_positions(cfg.strategy.underlying.upper())
        except Exception as e:
            st.warning(f"获取交易所持仓失败: {e}")

    # Sum unrealized PnL from exchange positions
    live_upnl = sum(
        float(p.get("unrealizedPnl") or p.get("unrealizedPNL") or 0)
        for p in exchange_positions
    )

    # --- DB stored data (historical) ---
    stats = storage.get_trade_stats()
    equity_curve = storage.get_equity_curve(start_date, end_date)
    # Compute derived stats
    total_pnl = stats["total_pnl"]
    total_fees = stats["total_fees"]
    net_pnl = total_pnl - total_fees

    # Drawdown (from historical equity curve)
    peak = 0.0
    max_dd = 0.0
    for snap in equity_curve:
        eq = snap["total_equity"]
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # --- Determine display values: prefer LIVE, fall back to DB snapshot ---
    if live_account is not None:
        latest_equity = live_account.total_balance
        latest_upnl = live_upnl if exchange_positions else live_account.unrealized_pnl
        _data_source = "实时"
    elif equity_curve:
        latest_equity = equity_curve[-1]["total_equity"]
        latest_upnl = equity_curve[-1]["unrealized_pnl"]
        _data_source = "历史快照"
    else:
        latest_equity = 0
        latest_upnl = 0
        _data_source = "无数据"

    # --- Auto-record equity snapshot when live data is available ---
    _skip_auto_snapshot_once = bool(st.session_state.pop("_skip_overview_auto_snapshot_once", False))
    if live_account is not None and not _skip_auto_snapshot_once:
        try:
            _spot = client.get_spot_price(cfg.strategy.underlying)
            storage.record_equity_snapshot(
                total_equity=latest_equity,
                available_balance=live_account.available_balance,
                unrealized_pnl=latest_upnl,
                position_count=len(exchange_positions),
                underlying_price=_spot,
            )
        except Exception:
            pass  # non-critical

    # KPI row
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("账户权益", f"${latest_equity:,.2f}")
    with col2:
        st.metric("未实现盈亏", f"${latest_upnl:,.2f}",
                   delta=f"${latest_upnl:,.2f}" if latest_upnl != 0 else None)
    with col3:
        st.metric("累计已实现 PnL", f"${net_pnl:,.2f}",
                   delta=f"${net_pnl:,.2f}" if net_pnl != 0 else None)
    with col4:
        st.metric("胜率", f"{stats['win_rate']:.1f}%")
    with col5:
        st.metric("最大回撤", f"{max_dd*100:.2f}%")
    with col6:
        st.metric("累计手续费", f"${total_fees:,.2f}")

    st.caption(f"📡 数据来源: **{_data_source}**")

    st.session_state.setdefault("overview_clear_history_confirm", False)
    with st.expander("🗑 数据清理", expanded=bool(st.session_state.get("overview_clear_history_confirm"))):
        st.caption("仅清空本地数据库中的历史记录，不会影响交易所真实持仓。")
        if not st.session_state.get("overview_clear_history_confirm"):
            if st.button("清除历史", type="secondary", key="overview_clear_history_prepare"):
                st.session_state["overview_clear_history_confirm"] = True
                st.rerun()
        else:
            st.warning("确认后将删除本地数据库中的成交历史、资产快照、每日损益和策略状态，且不可恢复。")
            _clear_col1, _clear_col2 = st.columns(2)
            with _clear_col1:
                if st.button("确认清除数据库", type="primary", key="overview_clear_history_confirm_btn"):
                    with st.spinner("正在清除数据库历史..."):
                        storage.clear_all_data()
                    st.cache_data.clear()
                    st.session_state["overview_clear_history_confirm"] = False
                    st.session_state["_skip_overview_auto_snapshot_once"] = True
                    st.session_state.pop("preview_test_order_task_id", None)
                    st.session_state.pop("test_order_apply_strategy_stop_loss", None)
                    st.success("数据库历史已清除。")
                    st.rerun()
            with _clear_col2:
                if st.button("取消", key="overview_clear_history_cancel_btn"):
                    st.session_state["overview_clear_history_confirm"] = False
                    st.rerun()

    st.divider()

    # --- Open Positions (local strategy + exchange) ---
    st.subheader("🔓 当前持仓")
    open_trades = storage.get_open_trades()
    _overview_spot_cache: dict[str, float] = {}

    if not open_trades and exchange_positions:
        st.warning("本地数据库当前持仓为空，但交易所存在实时持仓。当前以交易所持仓为准；本地账本与交易所已不一致。")

    if open_trades:
        # Group by trade_group
        groups: dict[str, list[dict]] = {}
        for t in open_trades:
            gid = t["trade_group"]
            groups.setdefault(gid, []).append(t)

        for gid, legs in groups.items():
            with st.expander(f"🦅 {gid} ({len(legs)} 腿)", expanded=True):
                leg_rows = []
                total_premium = 0.0
                payoff_legs: list[dict] = []
                group_underlying = ""
                group_spot = 0.0
                for t in legs:
                    meta = json.loads(t.get("meta", "{}"))
                    role = meta.get("leg_role", "?")
                    opt_type = meta.get("option_type", "?")
                    strike = meta.get("strike", 0)
                    symbol = str(t.get("symbol", ""))
                    group_underlying = group_underlying or (symbol.split("-", 1)[0].upper() if symbol else "")
                    group_spot = group_spot or float(meta.get("underlying_price") or 0.0)
                    premium_contrib = t["price"] * t["quantity"]
                    if t["side"] == "SELL":
                        total_premium += premium_contrib
                    else:
                        total_premium -= premium_contrib

                    leg_rows.append({
                        "角色": role,
                        "方向": t["side"],
                        "类型": opt_type.upper(),
                        "行权价": f"${strike:,.0f}",
                        "数量": t["quantity"],
                        "开仓价": f"${t['price']:.4f}",
                        "手续费": f"${t['fee']:.4f}",
                        "合约": symbol,
                        "时间": t["timestamp"][:19],
                    })
                    payoff_legs.append({
                        "role": role,
                        "symbol": symbol,
                        "side": str(t.get("side") or "").upper(),
                        "option_type": str(opt_type).lower(),
                        "strike": float(strike or 0.0),
                        "quantity": float(t.get("quantity") or 0.0),
                        "entry_price": float(t.get("price") or 0.0),
                    })

                df_legs = pd.DataFrame(leg_rows)
                st.dataframe(df_legs, width='stretch', hide_index=True)
                st.caption(f"净权利金: **${total_premium:.4f}**")

                if group_underlying:
                    if group_underlying not in _overview_spot_cache:
                        try:
                            _overview_spot_cache[group_underlying] = float(client.get_spot_price(group_underlying) or 0.0)
                        except Exception:
                            _overview_spot_cache[group_underlying] = 0.0
                    chart_spot = _overview_spot_cache.get(group_underlying, 0.0) or group_spot
                else:
                    chart_spot = group_spot

                payoff_fig = _build_expiry_payoff_figure(
                    payoff_legs,
                    underlying=group_underlying or cfg.strategy.underlying.upper(),
                    chart_title=f"当前持仓到期盈亏 | {gid}",
                    spot_price=chart_spot,
                )
                if payoff_fig is not None:
                    st.plotly_chart(payoff_fig, width='stretch')
    elif exchange_positions:
        _total_ex_upnl = sum(float(p.get("unrealizedPnl") or 0) for p in exchange_positions)
        ex_rows = []
        for p in exchange_positions:
            ex_rows.append(
                {
                    "合约": p.get("symbol", ""),
                    "方向": p.get("side", ""),
                    "数量": p.get("quantity", 0.0),
                    "开仓价": f"${float(p.get('entryPrice', 0.0)):.4f}",
                    "未实现PnL": f"${float(p.get('unrealizedPnl', 0.0)):.4f}",
                }
            )
        st.dataframe(pd.DataFrame(ex_rows), width='stretch', hide_index=True)
        st.caption(f"合计未实现盈亏: **${_total_ex_upnl:,.4f}**  |  持仓数: **{len(exchange_positions)}**")
    elif not has_creds:
        st.caption("未配置 API Key/Secret，无法查询交易所私有持仓。")
    else:
        st.info("当前无持仓")

    st.divider()

    # --- Recent Trades ---
    st.subheader("⏳ 最近成交")
    recent = storage.get_all_trades(limit=10)
    if recent:
        rows = []
        for t in recent:
            meta = json.loads(t.get("meta", "{}"))
            rows.append({
                "时间": t["timestamp"][:19],
                "状态": "🟢 持仓" if t["is_open"] else "🔴 已平",
                "方向": t["side"],
                "合约": t["symbol"],
                "数量": t["quantity"],
                "价格": f"${t['price']:.4f}",
                "PnL": f"${t['pnl']:.4f}" if not t["is_open"] else "-",
                "组": t["trade_group"][:20],
            })
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    elif exchange_positions:
        st.info("暂无本地成交记录；当前持仓来自交易所实时持仓。")
    else:
        st.info("暂无成交记录")

    # --- Mini equity chart ---
    if equity_curve and len(equity_curve) > 1:
        st.subheader("📈 资产走势")
        df_eq = pd.DataFrame(equity_curve)
        df_eq["timestamp"] = pd.to_datetime(df_eq["timestamp"])
        fig_mini = go.Figure()
        fig_mini.add_trace(go.Scatter(
            x=df_eq["timestamp"], y=df_eq["total_equity"],
            fill="tozeroy", line=dict(color="#00e396", width=1.8),
            fillcolor="rgba(0,227,150,0.08)",
            hovertemplate="%{y:,.2f}<extra></extra>",
        ))
        fig_mini.update_layout(
            height=220, margin=dict(l=50, r=12, t=8, b=16),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="权益 (USD)",
            font=dict(color="#b0b0b0", size=11),
            hovermode="x unified",
            hoverlabel=dict(bgcolor="#1e1e2f", font_size=12),
        )
        fig_mini.update_xaxes(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#b0b0b0"))
        fig_mini.update_yaxes(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#b0b0b0"))
        st.plotly_chart(fig_mini, width='stretch')


# ==========================================================================
# PAGE: 资产曲线 (Equity Curve)
# ==========================================================================

elif page == "📈 资产曲线":
    st.title("📈 资产曲线")

    _EQ_PLOT_MAX_POINTS = 1500

    @st.cache_data(ttl=10, show_spinner=False)
    def _load_equity_curve_view(
        db_path: str,
        db_mtime: float,
        start_date_value: str | None,
        end_date_value: str | None,
        max_points: int,
    ) -> tuple[dict, list[dict]]:
        _ = db_mtime
        cached_storage = Storage(db_path)
        try:
            stats = cached_storage.get_equity_curve_stats(start_date_value, end_date_value)
            rows = cached_storage.get_equity_curve_sampled(start_date_value, end_date_value, max_points=max_points)
            return stats, rows
        finally:
            cached_storage.close()

    # 手动刷新实时权益快照；避免每次切页都同步请求交易所导致页面卡顿
    @st.cache_resource
    def _get_client_eq(_exchange_cfg) -> BybitOptionsClient:
        return BybitOptionsClient(_exchange_cfg)

    _eq_client = _get_client_eq(cfg.exchange)
    _eq_has_creds = bool(cfg.exchange.api_key and cfg.exchange.api_secret)
    _eq_can_refresh_live = _eq_has_creds and not cfg.exchange.simulate_private

    _eq_col_info, _eq_col_action = st.columns([3, 1])
    with _eq_col_info:
        st.caption("本页默认直接读取本地权益快照，避免切页时实时请求交易所造成卡顿。")
    with _eq_col_action:
        _eq_refresh_clicked = st.button(
            "🔄 刷新实时快照",
            use_container_width=True,
            disabled=not _eq_can_refresh_live,
            key="equity_curve_refresh_snapshot",
        )

    if _eq_refresh_clicked:
        try:
            with st.spinner("正在拉取交易所账户并写入权益快照..."):
                _eq_acct = _eq_client.get_account()
                if not _eq_acct.raw.get("simulated") and _eq_acct.total_balance > 0:
                    _eq_positions = []
                    try:
                        _eq_positions = _eq_client.get_positions(cfg.strategy.underlying.upper())
                    except Exception:
                        pass
                    _eq_upnl = sum(
                        float(p.get("unrealizedPnl") or p.get("unrealizedPNL") or 0)
                        for p in _eq_positions
                    ) if _eq_positions else _eq_acct.unrealized_pnl
                    _eq_spot = _eq_client.get_spot_price(cfg.strategy.underlying)
                    storage.record_equity_snapshot(
                        total_equity=_eq_acct.total_balance,
                        available_balance=_eq_acct.available_balance,
                        unrealized_pnl=_eq_upnl,
                        position_count=len(_eq_positions),
                        underlying_price=_eq_spot,
                    )
                    st.success("实时权益快照已刷新。")
                else:
                    st.info("当前未获取到有效实时账户权益，已保留历史快照显示。")
        except Exception as e:
            st.warning(f"刷新实时权益快照失败：{e}")

    _eq_db_path = str(Path(cfg.storage.db_path).resolve())
    _eq_db_mtime = Path(_eq_db_path).stat().st_mtime if Path(_eq_db_path).exists() else 0.0
    equity_stats, equity_curve = _load_equity_curve_view(
        _eq_db_path,
        _eq_db_mtime,
        start_date,
        end_date,
        _EQ_PLOT_MAX_POINTS,
    )

    if not equity_curve or int(equity_stats.get("point_count") or 0) <= 0:
        st.info("暂无资产数据。交易程序运行后将自动记录。")
    else:
        df = pd.DataFrame(equity_curve)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # --- Dark theme constants ---
        _BG = "rgba(0,0,0,0)"
        _GRID = "rgba(255,255,255,0.06)"
        _TEXT = "#b0b0b0"
        _EQUITY_COLOR = "#00e396"   # vibrant green
        _BALANCE_COLOR = "#775dd0"  # muted purple
        _POS_COLOR = "#feb019"      # warm amber
        _UP_COLOR = "#00e396"
        _DN_COLOR = "#ff4560"
        _DD_COLOR = "#ff4560"

        _dark_axis: dict[str, Any] = dict(
            gridcolor=_GRID, zerolinecolor=_GRID,
            tickfont=dict(color=_TEXT, size=11),
            title_font=dict(color=_TEXT, size=12),
        )
        _dark_layout: dict[str, Any] = dict(
            paper_bgcolor=_BG, plot_bgcolor=_BG,
            font=dict(color=_TEXT, size=12),
            hovermode="x unified",
            hoverlabel=dict(bgcolor="#1e1e2f", font_size=12),
        )

        # --- Drawdown series (precompute) ---
        total_equity_series = df["total_equity"].astype(float)
        peak_series = total_equity_series.cummax()
        plot_dd_series = ((peak_series - total_equity_series) / peak_series).fillna(0.0).mul(100.0)
        plot_df = df.reset_index(drop=True)
        plot_dd_series = plot_dd_series.reset_index(drop=True)

        # --- KPI summary row ---
        total_points = int(equity_stats.get("point_count") or len(df))
        if total_points > 1:
            first_eq = float(equity_stats.get("first_equity") or 0.0)
            last_eq = float(equity_stats.get("last_equity") or 0.0)
            total_ret = (last_eq - first_eq) / first_eq * 100 if first_eq > 0 else 0
            max_dd_val = float(equity_stats.get("max_drawdown_pct") or 0.0)
        else:
            first_eq = last_eq = total_ret = max_dd_val = 0

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("最新权益", f"${last_eq:,.2f}")
        with k2:
            st.metric("期间收益率", f"{total_ret:+.2f}%")
        with k3:
            st.metric("最大回撤", f"{max_dd_val:.2f}%")
        with k4:
            st.metric("数据点数", f"{total_points:,}")

        if len(plot_df) < total_points:
            st.caption(f"为提升加载速度，图表已从 {total_points:,} 个数据点抽样显示为 {len(plot_df):,} 个点。")

        st.divider()

        # ======== Main equity chart ========
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.55, 0.22, 0.23],
            vertical_spacing=0.03,
        )

        # -- Row 1: Equity area + available balance --
        fig.add_trace(
            go.Scattergl(
                x=plot_df["timestamp"], y=plot_df["total_equity"],
                name="权益",
                line=dict(color=_EQUITY_COLOR, width=2),
                fill="tozeroy",
                fillcolor="rgba(0,227,150,0.08)",
                hovertemplate="%{y:,.2f}",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scattergl(
                x=plot_df["timestamp"], y=plot_df["available_balance"],
                name="可用余额",
                line=dict(color=_BALANCE_COLOR, width=1.2, dash="dot"),
                hovertemplate="%{y:,.2f}",
            ),
            row=1, col=1,
        )

        # -- Row 2: Unrealized PnL bars --
        upnl_colors = [_UP_COLOR if v >= 0 else _DN_COLOR for v in plot_df["unrealized_pnl"]]
        fig.add_trace(
            go.Bar(
                x=plot_df["timestamp"], y=plot_df["unrealized_pnl"],
                name="未实现 PnL",
                marker_color=upnl_colors,
                marker_line_width=0,
                opacity=0.75,
                hovertemplate="%{y:,.4f}",
            ),
            row=2, col=1,
        )

        # -- Row 3: Drawdown area (inverted) --
        fig.add_trace(
            go.Scattergl(
                x=plot_df["timestamp"], y=plot_dd_series,
                name="回撤 %",
                line=dict(color=_DD_COLOR, width=1.2),
                fill="tozeroy",
                fillcolor="rgba(255,69,96,0.12)",
                hovertemplate="%{y:.2f}%",
            ),
            row=3, col=1,
        )

        # -- Layout --
        fig.update_layout(
            height=620,
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                bgcolor="rgba(0,0,0,0)", font=dict(color=_TEXT, size=11),
            ),
            margin=dict(l=60, r=16, t=24, b=16),
            **_dark_layout,
        )

        # Axes styling
        for row_i in range(1, 4):
            fig.update_xaxes(row=row_i, col=1, **_dark_axis, showticklabels=(row_i == 3))
            fig.update_yaxes(row=row_i, col=1, **_dark_axis)

        fig.update_yaxes(title_text="账户权益", row=1, col=1)
        fig.update_yaxes(title_text="未实现PnL", row=2, col=1)
        fig.update_yaxes(title_text="回撤 %", autorange="reversed", row=3, col=1)
        fig.update_xaxes(rangeslider_visible=False)

        st.plotly_chart(fig, width='stretch')

        # ======== Position count + underlying overlay ========
        show_extra_charts = st.toggle(
            "显示附加图表（标的价格 / 持仓数）",
            value=False,
            key="equity_show_extra_charts",
            help="默认关闭以加快加载速度，需要时再展开。",
        )

        has_ul = "underlying_price" in plot_df.columns and plot_df["underlying_price"].sum() > 0
        if show_extra_charts and has_ul:
            fig2 = make_subplots(
                rows=1, cols=1,
                specs=[[{"secondary_y": True}]],
            )
            fig2.add_trace(
                go.Scattergl(
                    x=plot_df["timestamp"], y=plot_df["underlying_price"],
                    name="标的价格 (USD)",
                    line=dict(color="#00b4d8", width=1.5),
                    hovertemplate="$%{y:,.2f}",
                ),
                secondary_y=False,
            )
            fig2.add_trace(
                go.Scattergl(
                    x=plot_df["timestamp"], y=plot_df["position_count"],
                    name="持仓数", mode="lines+markers",
                    line=dict(color=_POS_COLOR, width=1.5),
                    marker=dict(size=4, color=_POS_COLOR),
                    hovertemplate="%{y}",
                ),
                secondary_y=True,
            )
            fig2.update_layout(
                height=260,
                margin=dict(l=60, r=60, t=8, b=16),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=_TEXT, size=11),
                ),
                **_dark_layout,
            )
            fig2.update_xaxes(**_dark_axis)
            fig2.update_yaxes(title_text="标的价格 (USD)", secondary_y=False, **_dark_axis)
            fig2.update_yaxes(title_text="持仓数", secondary_y=True, **_dark_axis)
            st.plotly_chart(fig2, width='stretch')
        elif show_extra_charts:
            # Position count only
            fig_pos = go.Figure()
            fig_pos.add_trace(go.Scattergl(
                x=plot_df["timestamp"], y=plot_df["position_count"],
                name="持仓数", mode="lines+markers",
                line=dict(color=_POS_COLOR, width=2),
                marker=dict(size=4),
                fill="tozeroy", fillcolor="rgba(254,176,25,0.08)",
            ))
            fig_pos.update_layout(
                height=200, margin=dict(l=60, r=16, t=8, b=16),
                yaxis_title="持仓数", **_dark_layout,
            )
            fig_pos.update_xaxes(**_dark_axis)
            fig_pos.update_yaxes(**_dark_axis)
            st.plotly_chart(fig_pos, width='stretch')


# ==========================================================================
# PAGE: 成交历史 (Trade History)
# ==========================================================================

elif page == "📋 成交历史":
    st.title("📋 成交历史")

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filter_status = st.selectbox("状态", ["全部", "持仓中", "已平仓"])
    with col_f2:
        filter_side = st.selectbox("方向", ["全部", "BUY", "SELL"])
    with col_f3:
        limit = st.number_input("显示条数", min_value=10, max_value=1000,
                                value=100, step=50)

    # Fetch trades
    all_trades = storage.get_all_trades(limit=limit, start_date=start_date,
                                        end_date=end_date)

    # Apply filters
    if filter_status == "持仓中":
        all_trades = [t for t in all_trades if t["is_open"]]
    elif filter_status == "已平仓":
        all_trades = [t for t in all_trades if not t["is_open"]]

    if filter_side != "全部":
        all_trades = [t for t in all_trades if t["side"] == filter_side]

    # Stats
    stats = storage.get_trade_stats()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总成交", stats["total_trades"])
    with col2:
        st.metric("持仓中", stats["open_trades"])
    with col3:
        st.metric("已平仓", stats["closed_trades"])
    with col4:
        st.metric("胜率", f"{stats['win_rate']:.1f}%")

    st.divider()

    if not all_trades:
        st.info("暂无成交记录")
    else:
        # Build dataframe
        rows = []
        for t in all_trades:
            meta = json.loads(t.get("meta", "{}"))
            rows.append({
                "ID": t["id"],
                "时间": t["timestamp"][:19],
                "状态": "🟢 持仓" if t["is_open"] else "🔴 已平",
                "方向": t["side"],
                "合约": t["symbol"],
                "角色": meta.get("leg_role", "-"),
                "类型": meta.get("option_type", "-").upper(),
                "行权价": meta.get("strike", "-"),
                "数量": t["quantity"],
                "开仓价": t["price"],
                "平仓价": t["close_price"] if not t["is_open"] else None,
                "PnL": t["pnl"] if not t["is_open"] else None,
                "手续费": t["fee"],
                "OrderID": t.get("order_id", "-"),
                "组": t["trade_group"],
                "平仓时间": t.get("close_timestamp", "")[:19] if t.get("close_timestamp") else "",
            })

        df = pd.DataFrame(rows)

        # Highlight PnL
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            column_config={
                "PnL": st.column_config.NumberColumn(
                    "PnL", format="$%.4f",
                ),
                "开仓价": st.column_config.NumberColumn(
                    "开仓价", format="$%.4f",
                ),
                "平仓价": st.column_config.NumberColumn(
                    "平仓价", format="$%.4f",
                ),
                "手续费": st.column_config.NumberColumn(
                    "手续费", format="$%.4f",
                ),
            },
        )

        # --- Group-level PnL breakdown ---
        st.subheader("🦅 组合损益")
        closed_trades = [t for t in all_trades if not t["is_open"]]
        if closed_trades:
            group_pnl: dict[str, float] = {}
            group_fees: dict[str, float] = {}
            group_time: dict[str, str] = {}
            for t in closed_trades:
                gid = t["trade_group"]
                group_pnl[gid] = group_pnl.get(gid, 0) + t["pnl"]
                group_fees[gid] = group_fees.get(gid, 0) + t["fee"]
                if gid not in group_time:
                    group_time[gid] = t["timestamp"][:19]

            group_rows = []
            for gid in group_pnl:
                net = group_pnl[gid] - group_fees[gid]
                group_rows.append({
                    "组ID": gid,
                    "时间": group_time[gid],
                    "总PnL": group_pnl[gid],
                    "总手续费": group_fees[gid],
                    "净收益": net,
                    "结果": "✅ 盈利" if net > 0 else "❌ 亏损",
                })

            df_groups = pd.DataFrame(group_rows)
            st.dataframe(
                df_groups,
                width="stretch",
                hide_index=True,
                column_config={
                    "总PnL": st.column_config.NumberColumn(format="$%.4f"),
                    "总手续费": st.column_config.NumberColumn(format="$%.4f"),
                    "净收益": st.column_config.NumberColumn(format="$%.4f"),
                },
            )

        # --- Export ---
        st.divider()
        csv = df.to_csv(index=False).encode("utf-8")
        csv_filename = f"trades_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        csv_b64 = base64.b64encode(csv).decode("ascii")
        st.markdown(
            (
                f'<a href="data:text/csv;base64,{csv_b64}" download="{csv_filename}" '
                'style="display:inline-block;padding:0.45rem 0.9rem;border-radius:0.5rem;'
                'background:#1f77b4;color:white;text-decoration:none;font-weight:600;">'
                '📥 导出 CSV</a>'
            ),
            unsafe_allow_html=True,
        )


# ==========================================================================
# PAGE: 策略配置 (Strategy Config)
# ==========================================================================

elif page == "🔧 策略配置":
    st.title("🔧 策略配置")

    # --- Load current YAML ---
    try:
        import yaml as _yaml
    except ImportError:
        import importlib
        _yaml = importlib.import_module("yaml")

    try:
        with open(config_path, "r", encoding="utf-8") as _f:
            _raw_yaml = _yaml.safe_load(_f) or {}
    except FileNotFoundError:
        _raw_yaml = {}
        st.error(f"配置文件不存在: {config_path}")

    _strategy = _raw_yaml.get("strategy", {})
    _exchange = _raw_yaml.get("exchange", {})
    _storage = _raw_yaml.get("storage", {})
    _monitor = _raw_yaml.get("monitor", {})
    _pv_margin_per_unit = 0.0
    _pv_available = 0.0
    _pv_max_open_quantity = 0.0
    _pv_max_open_groups = 0.0
    _pv_margin_hint = "当前无法根据实时行情估算最大可开仓位"

    def _render_strategy_form() -> None:
        with st.form("config_editor", border=True):
            st.subheader("📄 策略参数")

            s_col1, s_col2, s_col3 = st.columns(3)

            with s_col1:
                ed_underlying = st.selectbox(
                    "标的", ["ETH", "BTC"],
                    index=0 if _strategy.get("underlying", "ETH") == "ETH" else 1,
                )
                st.markdown("**▸ Weekend Vol Delta 参数**")
                ed_target_delta = st.number_input("短腿目标 |Δ|", min_value=0.05, max_value=0.90, value=float(_strategy.get("target_delta", 0.40)), step=0.05, format="%.2f", help="Short legs 的 |delta| 目标 (weekend_vol)")
                ed_wing_delta = st.number_input("翼 |Δ| (保护腿)", min_value=0.0, max_value=0.50, value=float(_strategy.get("wing_delta", 0.05)), step=0.01, format="%.2f", help="Long legs 的 |delta| 目标 (0=无翼结构)")
                ed_leverage = st.number_input("杠杆倍数", min_value=0.5, max_value=10.0, value=float(_strategy.get("leverage", 1.0)), step=0.5, format="%.1f", help="仓位大小的杠杆倍数 (weekend_vol)")
                ed_rv_hours = st.number_input("RV 回看小时数", min_value=0, max_value=168, value=int(_strategy.get("entry_realized_vol_lookback_hours", 24)), step=1, help="入场 RV 过滤的历史小时数；0 表示关闭过滤")
                ed_rv_max = st.number_input("RV 上限", min_value=0.0, max_value=5.0, value=float(_strategy.get("entry_realized_vol_max", 1.20)), step=0.05, format="%.2f", help="仅当年化 RV 不超过该值时允许开仓")
                ed_stop_loss_pct = st.number_input("组合止损 %", min_value=0.0, max_value=1000.0, value=float(_strategy.get("stop_loss_pct", 200.0)), step=10.0, format="%.1f", help="组合 PnL% 低于 -该值时触发止损；0 表示关闭")
                ed_stop_loss_underlying_move_pct = st.number_input("标的单边止损过滤 %", min_value=0.0, max_value=50.0, value=float(_strategy.get("stop_loss_underlying_move_pct", 0.0)), step=0.5, format="%.1f", help="仅当标的价格相对开仓价单方向波动达到该百分比时，组合止损才允许触发；0 表示关闭")
                _day_options = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
                _cur_day = _strategy.get("entry_day", "friday").lower()
                _day_index = _day_options.index(_cur_day) if _cur_day in _day_options else 4
                ed_entry_day = st.selectbox("开仓日", _day_options, index=_day_index, help="每周开仓的日子 (weekend_vol 通常选 friday)")
                ed_default_iv = st.number_input("默认 IV", min_value=0.1, max_value=3.0, value=float(_strategy.get("default_iv", 0.60)), step=0.05, format="%.2f", help="mark_iv 不可用时的回退 IV (weekend_vol)")
                import datetime as _dt
                _entry_str = _strategy.get("entry_time_utc", "08:00")
                _entry_parts = _entry_str.split(":")
                _entry_time = _dt.time(int(_entry_parts[0]), int(_entry_parts[1]))
                ed_entry_time = st.time_input("开仓时间 (UTC)", value=_entry_time, step=_dt.timedelta(minutes=1), help="每周 weekend_vol 开仓的 UTC 时间（精确到分钟）")

            with s_col2:
                ed_quantity = st.number_input("固定数量 (张)", min_value=0.0, max_value=100.0, value=float(_strategy.get("quantity", 5.0)), step=0.1, format="%.3f", help="Bybit 实盘固定下单数量，可直接在前端修改")
                st.caption("当前策略已改为固定数量模式，前端修改后保存即可生效。")
                if _pv_ok and _pv_margin_per_unit > 0:
                    st.caption(_pv_margin_hint)
                else:
                    st.caption("当前无法根据实时行情估算最大可开仓位，请先确认下方下单预览可正常获取行情。")
                ed_max_pos = st.number_input("最大持仓组数", min_value=1, max_value=20, value=_strategy.get("max_positions", 1), step=1)

            with s_col3:
                st.info("固定数量模式")
                st.caption("复利已停用；仓位大小完全由“固定数量 (张)”控制。")
                st.markdown("**▸ Bybit 手续费 / 保证金参数**")
                ed_maker_fee_rate = st.number_input("Maker 费率", min_value=0.0, max_value=0.01, value=float(_exchange.get("option_maker_fee_rate", 0.0002)), step=0.0001, format="%.4f")
                ed_taker_fee_rate = st.number_input("Taker 费率", min_value=0.0, max_value=0.01, value=float(_exchange.get("option_taker_fee_rate", 0.0003)), step=0.0001, format="%.4f")
                ed_fee_cap_ratio = st.number_input("手续费上限系数", min_value=0.0, max_value=1.0, value=float(_exchange.get("option_fee_cap_ratio", 0.07)), step=0.01, format="%.2f")
                ed_short_margin_ratio = st.number_input("空头保证金比率", min_value=0.0, max_value=1.0, value=float(_exchange.get("short_option_margin_ratio", 0.10)), step=0.01, format="%.2f")
                ed_short_otm_deduction_ratio = st.number_input("OTM 抵扣比率", min_value=0.0, max_value=1.0, value=float(_exchange.get("short_option_otm_deduction_ratio", 0.08)), step=0.01, format="%.2f")
                ed_short_min_margin_ratio = st.number_input("最小保证金比率", min_value=0.0, max_value=1.0, value=float(_exchange.get("short_option_min_margin_ratio", 0.05)), step=0.01, format="%.2f")

            st.divider()
            st.subheader("🔌 API & 运行参数")
            r_col1, r_col2 = st.columns(2)

            with r_col1:
                ed_testnet = st.checkbox("测试网模式", value=_exchange.get("testnet", True), help="开启后使用模拟账户，不真实下单")
                ed_timeout = st.number_input("API 超时 (秒)", min_value=3, max_value=60, value=_exchange.get("timeout", 10), step=1)
                _check_interval_value = max(1, int(_monitor.get("check_interval_sec", 60) or 60))
                _heartbeat_value = max(60, int(_monitor.get("heartbeat_interval_sec", 300) or 300))
                ed_check_interval = st.number_input("策略循环间隔 (秒)", min_value=1, max_value=600, value=_check_interval_value, step=1)
                ed_heartbeat = st.number_input("心跳间隔 (秒)", min_value=60, max_value=3600, value=_heartbeat_value, step=60)

            with r_col2:
                _snapshot_value = max(300, int(_monitor.get("equity_snapshot_interval_sec", 3600) or 3600))
                ed_snapshot = st.number_input("资产快照间隔 (秒)", min_value=300, max_value=86400, value=_snapshot_value, step=300)
                ed_db_path = st.text_input("数据库路径", value=_storage.get("db_path", "./data/trader.db"))
                ed_log_dir = st.text_input("日志目录", value=_storage.get("log_dir", "./logs"))
                ed_log_level = st.selectbox("日志级别", ["DEBUG", "INFO", "WARNING", "ERROR"], index=["DEBUG", "INFO", "WARNING", "ERROR"].index(_storage.get("log_level", "INFO")))

            st.divider()
            _save_col1, _save_col2 = st.columns([1, 4])
            with _save_col1:
                _submitted = st.form_submit_button("💾 保存配置", use_container_width=True, type="primary")
            with _save_col2:
                st.caption("保存后需要重启引擎才能生效")

        if _submitted:
            _new_cfg = {
                "name": _raw_yaml.get("name", cfg.name),
                "exchange": {
                    "api_key": "",
                    "api_secret": "",
                    "testnet": ed_testnet,
                    "timeout": ed_timeout,
                    "account_currency": _exchange.get("account_currency", "USDT"),
                    "simulate_private": _exchange.get("simulate_private", False),
                    "option_maker_fee_rate": round(float(ed_maker_fee_rate), 6),
                    "option_taker_fee_rate": round(float(ed_taker_fee_rate), 6),
                    "option_fee_cap_ratio": round(float(ed_fee_cap_ratio), 4),
                    "short_option_margin_ratio": round(float(ed_short_margin_ratio), 4),
                    "short_option_otm_deduction_ratio": round(float(ed_short_otm_deduction_ratio), 4),
                    "short_option_min_margin_ratio": round(float(ed_short_min_margin_ratio), 4),
                },
                "strategy": {
                    "mode": "weekend_vol",
                    "underlying": ed_underlying,
                    "target_delta": round(ed_target_delta, 4),
                    "wing_delta": round(ed_wing_delta, 4),
                    "leverage": round(ed_leverage, 2),
                    "entry_realized_vol_lookback_hours": int(ed_rv_hours),
                    "entry_realized_vol_max": round(ed_rv_max, 4),
                    "stop_loss_pct": round(ed_stop_loss_pct, 4),
                    "stop_loss_underlying_move_pct": round(ed_stop_loss_underlying_move_pct, 4),
                    "entry_day": ed_entry_day,
                    "default_iv": round(ed_default_iv, 4),
                    "entry_time_utc": ed_entry_time.strftime("%H:%M"),
                    "quantity": round(float(ed_quantity), 4),
                    "max_positions": ed_max_pos,
                    "compound": False,
                },
                "storage": {
                    "db_path": ed_db_path,
                    "log_dir": ed_log_dir,
                    "log_level": ed_log_level,
                    "log_rotation": _storage.get("log_rotation", "1 day"),
                    "log_retention": _storage.get("log_retention", "30 days"),
                },
                "monitor": {
                    "check_interval_sec": ed_check_interval,
                    "heartbeat_interval_sec": ed_heartbeat,
                    "equity_snapshot_interval_sec": ed_snapshot,
                },
            }
            try:
                _header = (
                    "# ============================================================\n"
                    f"# {_new_cfg['name']}\n"
                    "# ============================================================\n"
                    "# 由交易面板自动保存\n"
                    "# ============================================================\n\n"
                )
                with open(config_path, "w", encoding="utf-8") as _wf:
                    _wf.write(_header)
                    _yaml.dump(_new_cfg, _wf, default_flow_style=False, allow_unicode=True, sort_keys=False)
                st.success(f"✅ 配置已保存到 {config_path}")
                st.caption("出于安全考虑，API Key / Secret 不会由面板写回 YAML，请继续使用环境变量或 .env。")
                st.info("⚠️ 请重启引擎使新配置生效 (先停止再启动)")
            except Exception as _ex:
                st.error(f"保存失败: {_ex}")

    st.divider()

    # ------------------------------------------------------------------
    # 📋  下单预览 — 根据实时行情模拟下一次开仓
    # ------------------------------------------------------------------
    _pv_is_ic = getattr(cfg.strategy, "wing_delta", 0) > 0
    _pv_mode_label = "Weekend Vol 带翼结构" if _pv_is_ic else "Weekend Vol 无翼结构"
    st.subheader(f"📋 下单预览 ({_pv_mode_label})（实时行情）")

    @st.cache_resource
    def _get_client_preview(_exchange_cfg) -> BybitOptionsClient:
        return BybitOptionsClient(_exchange_cfg)

    _pv_client = _get_client_preview(cfg.exchange)

    _pv_ok = False
    try:
        _pv_ul = cfg.strategy.underlying.upper()
        _pv_spot = _pv_client.get_spot_price(_pv_ul)
        _pv_tickers = _pv_client.get_tickers(_pv_ul)

        if _pv_spot <= 0:
            _prices = [t.underlying_price for t in _pv_tickers if t.underlying_price > 0]
            _pv_spot = _prices[0] if _prices else 0

        _now_utc = datetime.now(timezone.utc)
        _tolerance_h = 2.0
        _active_expiry_target, _sunday_expiry_target, _friday_expiry_target = resolve_test_order_expiry_target(
            _pv_tickers,
            _now_utc,
            tolerance_hours=_tolerance_h,
        )
        _available_expiry_summaries = summarize_available_expiries(_pv_tickers)
        _sunday_exp = _sunday_expiry_target.expiry
        _pv_today = list(_active_expiry_target.tickers)
        _target_dte_days = round((_sunday_exp - _now_utc).total_seconds() / 86400, 1)
        _dte_label_str = _sunday_expiry_target.label
        _active_dte_label_str = _active_expiry_target.label
        _test_order_used_friday_fallback = _active_expiry_target.is_fallback and bool(_active_expiry_target.tickers)

        def _render_available_expiry_panel() -> None:
            with st.expander("查看 Bybit 当前可用到期日", expanded=True):
                st.caption(f"Bybit 返回期权 ticker 数量: {len(_pv_tickers)}")
                if _available_expiry_summaries:
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "到期日": _row.label,
                                    "合约数量": _row.contract_count,
                                }
                                for _row in _available_expiry_summaries
                            ]
                        ),
                        width="stretch",
                        hide_index=True,
                    )
                else:
                    st.warning("当前未从 Bybit 获取到任何可用期权到期日。")
                    st.caption("这通常表示：接口没返回期权 ticker、网络受限、API 网关异常，或当前标的没有拉到期权行情。")

        if _pv_spot > 0 and _pv_today:
            # --- Strike selection ---
            _sell_call: OptionTicker | None = None
            _sell_put: OptionTicker | None = None
            _buy_call: OptionTicker | None = None
            _buy_put: OptionTicker | None = None
            def _best_by_delta(tks, otype, target_delta, spot_val, T_yr, fallback_iv):
                """Delta-based: find closest |delta| to target."""
                cs = [t for t in tks if t.option_type == otype]
                if not cs:
                    return None
                best_t = None
                best_diff = float("inf")
                for _t in cs:
                    if _t.delta != 0.0:
                        abs_d = abs(_t.delta)
                    else:
                        try:
                            from options_backtest.pricing.black76 import delta as _bs_delta
                            _iv = _t.mark_iv or fallback_iv
                            if _iv <= 0:
                                _iv = fallback_iv
                            _d = _bs_delta(spot_val, _t.strike, T_yr, sigma=_iv,
                                           option_type=otype, r=0.0)
                            abs_d = abs(_d)
                        except Exception:
                            continue
                    diff = abs(abs_d - target_delta)
                    if diff < best_diff:
                        best_diff = diff
                        best_t = _t
                return best_t

            def _format_preview_delta(ticker, spot_val, fallback_iv):
                """Format contract delta for preview table.

                Prefer exchange delta; fall back to Black-76 estimate when missing.
                Uses `~` prefix for estimated values.
                """
                if ticker is None:
                    return "-"

                if ticker.delta != 0.0:
                    return f"{ticker.delta:+.3f}"

                try:
                    from options_backtest.pricing.black76 import delta as _bs_delta

                    _iv = ticker.mark_iv or fallback_iv
                    if _iv <= 0:
                        _iv = fallback_iv
                    if _iv <= 0 or spot_val <= 0:
                        return "-"

                    _T_yr = max(ticker.dte_hours / (24 * 365.25), 1e-6)
                    _d = _bs_delta(
                        spot_val,
                        ticker.strike,
                        _T_yr,
                        sigma=_iv,
                        option_type=ticker.option_type,
                        r=0.0,
                    )
                    return f"~{_d:+.3f}"
                except Exception:
                    return "-"

            # Enrich tickers with greeks from exchange
            try:
                _pv_client.enrich_greeks(_pv_today, _pv_ul)
            except Exception:
                pass

            _active_expiry = _active_expiry_target.expiry
            _T_years = max((_active_expiry - _now_utc).total_seconds() / (365.25 * 86400), 1e-6)
            _tgt_delta = getattr(cfg.strategy, "target_delta", 0.40)
            _wing_d = getattr(cfg.strategy, "wing_delta", 0.05)
            _def_iv = getattr(cfg.strategy, "default_iv", 0.60)

            _sell_call = _best_by_delta(_pv_today, "call", _tgt_delta, _pv_spot, _T_years, _def_iv)
            _sell_put = _best_by_delta(_pv_today, "put", _tgt_delta, _pv_spot, _T_years, _def_iv)

            if _wing_d > 0:
                _buy_call = _best_by_delta(_pv_today, "call", _wing_d, _pv_spot, _T_years, _def_iv)
                _buy_put = _best_by_delta(_pv_today, "put", _wing_d, _pv_spot, _T_years, _def_iv)

            _is_ic = _pv_is_ic

            _has_all_legs = (
                all([_sell_call, _buy_call, _sell_put, _buy_put])
                if _is_ic
                else all([_sell_call, _sell_put])
            )

            if _has_all_legs:
                assert _sell_call is not None and _sell_put is not None
                _sell_call_t: OptionTicker = _sell_call
                _sell_put_t: OptionTicker = _sell_put
                _buy_call_t: OptionTicker | None = None
                _buy_put_t: OptionTicker | None = None
                _lc_bid = 0.0
                _lc_ask = 0.0
                _lp_bid = 0.0
                _lp_ask = 0.0
                _call_width = 0.0
                _put_width = 0.0
                _total_max_loss = 0.0
                if _is_ic:
                    assert _buy_call is not None and _buy_put is not None
                    _buy_call_t = _buy_call
                    _buy_put_t = _buy_put

                _base_qty = float(getattr(cfg.strategy, "quantity", 0.0) or 0.0)
                _pv_equity = 0.0
                _pv_available = 0.0
                _pv_equity_src = "固定数量配置"
                try:
                    _pv_acct = _pv_client.get_account()
                    _pv_equity = _pv_acct.total_balance
                    _pv_available = max(float(getattr(_pv_acct, "available_balance", 0.0) or 0.0), 0.0)
                except Exception:
                    pass

                _preview = compute_option_order_preview(
                    spot=_pv_spot,
                    base_quantity=_base_qty,
                    compound=False,
                    equity=_pv_equity,
                    available_balance=_pv_available,
                    leverage=getattr(cfg.strategy, "leverage", 1.0),
                    exchange_cfg=cfg.exchange,
                    sell_call=_sell_call_t,
                    sell_put=_sell_put_t,
                    buy_call=_buy_call_t,
                    buy_put=_buy_put_t,
                )
                _pv_qty = float(_preview["quantity"])
                _pv_equity_src = _preview["equity_source"]
                _margin_per = float(_preview["margin_per_unit"])
                _pv_margin_per_unit = _margin_per
                if _margin_per > 0 and _pv_available > 0:
                    _pv_max_open_quantity = math.floor((_pv_available / _margin_per) * 1000.0) / 1000.0
                    if _pv_qty > 0:
                        _pv_max_open_groups = math.floor((_pv_available / (_margin_per * _pv_qty)) * 1000.0) / 1000.0
                    _pv_margin_hint = (
                        f"按当前实时盘口估算：可用保证金约 USD {_pv_available:,.0f}，"
                        f"每组保证金约 USD {_margin_per:,.0f}（每腿 1 张），最多可开约 {_pv_max_open_quantity:.3f} 组。"
                    )
                elif _margin_per > 0:
                    _pv_margin_hint = (
                        f"按当前实时盘口估算：每组保证金约 USD {_margin_per:,.0f}（每腿 1 张），"
                        "但当前未获取到可用保证金余额。"
                    )

                # Prices in USD (Bybit option native quote unit)
                _preview_fallback_iv = getattr(cfg.strategy, "default_iv", 0.60)
                _sc_bid = _sell_call_t.bid_price
                _sc_ask = _sell_call_t.ask_price
                _sp_bid = _sell_put_t.bid_price
                _sp_ask = _sell_put_t.ask_price

                def _preview_leg_row(
                    title: str,
                    ticker: OptionTicker,
                    bid: float,
                    ask: float,
                    direction: str,
                ) -> dict[str, str]:
                    return {
                        "腿": title,
                        "合约": ticker.symbol,
                        "行权价": f"${ticker.strike:,.0f}",
                        "距Spot": f"{(ticker.strike/_pv_spot - 1)*100:+.1f}%",
                        "Delta": _format_preview_delta(ticker, _pv_spot, _preview_fallback_iv),
                        "方向": direction,
                        "Bid (USDT)": f"${bid:.2f}",
                        "Ask (USDT)": f"${ask:.2f}",
                        "DTE": f"{ticker.dte_hours:.0f}h ({ticker.dte_hours/24:.1f}d)",
                    }

                if not _is_ic:
                    # ---- Weekend vol without wings P&L ----
                    _premium_per = float(_preview["premium_per_unit"])
                    _total_premium = float(_preview["total_premium"])
                    _margin_used = float(_preview["margin_used"])
                    _be_upper = float(_preview["break_even_upper"])
                    _be_lower = float(_preview["break_even_lower"])
                else:
                    # ---- Weekend vol with wings P&L ----
                    assert _buy_call_t is not None and _buy_put_t is not None
                    _lc_bid = _buy_call_t.bid_price
                    _lc_ask = _buy_call_t.ask_price
                    _lp_bid = _buy_put_t.bid_price
                    _lp_ask = _buy_put_t.ask_price
                    _premium_per = float(_preview["premium_per_unit"])
                    _call_width = _buy_call_t.strike - _sell_call_t.strike
                    _put_width = _sell_put_t.strike - _buy_put_t.strike
                    _max_wing = max(_call_width, _put_width)
                    _max_loss_per = _max_wing - _premium_per
                    _total_premium = float(_preview["total_premium"])
                    _total_max_loss = float(_preview["total_max_loss"] or (_max_loss_per * _pv_qty))
                    _margin_used = float(_preview["margin_used"])
                    _be_upper = float(_preview["break_even_upper"])
                    _be_lower = float(_preview["break_even_lower"])

                _pv_ok = True

                # ---- Display ----
                if _test_order_used_friday_fallback:
                    st.warning(
                        f"⚠️ 周日到期合约当前不可用，以下合约已仅为测试下单自动切换到 {_active_dte_label_str}。"
                        "此回退不会影响正式策略实际选腿。"
                    )
                st.caption(
                    f"当前测试下单目标到期日: {_active_dte_label_str} | "
                    f"原始周日目标: {_dte_label_str}"
                )
                st.caption(f"数据来源: Bybit 实时行情 | {_pv_equity_src} | {_active_dte_label_str}")
                _render_available_expiry_panel()

                # KPI row
                kc1, kc2, kc3, kc4 = st.columns(4)
                with kc1:
                    st.metric(f"{_pv_ul} 现货", f"${_pv_spot:,.2f}")
                with kc2:
                    st.metric("下单数量", f"{_pv_qty:.2f} 张")
                with kc3:
                    if _pv_equity > 0:
                        st.metric("账户权益", f"${_pv_equity:,.2f}")
                    else:
                        st.metric("账户权益", "-")
                with kc4:
                    if _pv_equity > 0:
                        st.metric("保证金占比", f"{_margin_used / _pv_equity * 100:.1f}%")
                    else:
                        st.metric("保证金占比", "-")

                # Legs table
                if not _is_ic:
                    st.markdown("##### 🦵 两条腿明细")
                    _legs_data = [
                        _preview_leg_row("① Short Put (卖出收权利金)", _sell_put_t, _sp_bid, _sp_ask, "🔴 卖出"),
                        _preview_leg_row("② Short Call (卖出收权利金)", _sell_call_t, _sc_bid, _sc_ask, "🔴 卖出"),
                    ]
                else:
                    st.markdown("##### 🦵 四条腿明细")
                    assert _buy_call_t is not None and _buy_put_t is not None
                    _legs_data = [
                        _preview_leg_row("① Long Put (买入保护)", _buy_put_t, _lp_bid, _lp_ask, "🟢 买入"),
                        _preview_leg_row("② Short Put (卖出收权利金)", _sell_put_t, _sp_bid, _sp_ask, "🔴 卖出"),
                        _preview_leg_row("③ Short Call (卖出收权利金)", _sell_call_t, _sc_bid, _sc_ask, "🔴 卖出"),
                        _preview_leg_row("④ Long Call (买入保护)", _buy_call_t, _lc_bid, _lc_ask, "🟢 买入"),
                    ]
                st.dataframe(pd.DataFrame(_legs_data), width="stretch", hide_index=True)

                # Manual test order
                st.markdown("##### 🧪 测试下单")
                _test_qty = 0.01
                _active_test_task_id = st.session_state.get("preview_test_order_task_id") or _get_latest_test_order_task_id()
                if _active_test_task_id and not st.session_state.get("preview_test_order_task_id"):
                    st.session_state["preview_test_order_task_id"] = _active_test_task_id
                _active_test_task = _get_test_order_task(_active_test_task_id)
                _test_task_running = bool(_active_test_task and _active_test_task.get("state") in {"queued", "running"})
                _test_disabled_reason = None
                _sl_cfg_pct = float(getattr(cfg.strategy, "stop_loss_pct", 0.0) or 0.0)
                _sl_supported = _sl_cfg_pct > 0 and engine.is_running and is_trade_mode
                if not is_trade_mode:
                    _test_disabled_reason = "当前为只读模式，切换到交易模式后才可发送测试单。"
                elif _test_task_running:
                    _test_disabled_reason = "已有测试下单任务在执行，请等待完成。"

                _test_leg_count = 4 if _is_ic else 2
                _test_button_label = f"🧪 测试{_test_leg_count}腿市价单 0.01"
                st.caption(
                    f"按当前预览目标固定发送 {_test_leg_count} 条腿市价单，每条腿数量 0.01，用于验证真实下单链路。"
                )
                _auto_sl_pref_key = f"preview_auto_sl_pref_weekend_vol_{_test_leg_count}"
                _auto_sl_widget_key = f"preview_auto_sl_widget_weekend_vol_{_test_leg_count}"
                if _auto_sl_pref_key not in st.session_state:
                    st.session_state[_auto_sl_pref_key] = False
                _tc1, _tc2 = st.columns([1, 2])
                with _tc1:
                    _auto_stop_loss = st.checkbox(
                        "应用策略自动止损",
                        value=bool(st.session_state.get(_auto_sl_pref_key, False)),
                        disabled=not _sl_supported,
                        help="勾选后，测试单成交后会接入当前引擎持仓管理，并按策略里的止损线自动监控平仓。",
                        key=_auto_sl_widget_key,
                    )
                    st.session_state[_auto_sl_pref_key] = bool(_auto_stop_loss)
                    _preview_test_clicked = st.button(
                        _test_button_label,
                        key=(
                            f"preview_test_combo_weekend_vol_"
                            f"{_sell_put.symbol}_{_sell_call.symbol}_"
                            f"{_buy_put.symbol if _buy_put else 'none'}_"
                            f"{_buy_call.symbol if _buy_call else 'none'}"
                        ),
                        use_container_width=True,
                        type="secondary",
                        disabled=_test_disabled_reason is not None,
                    )
                with _tc2:
                    if _is_ic:
                        st.write(
                            f"固定数量: {_test_qty:.2f} | "
                            f"Long Put `{_buy_put.symbol if _buy_put else '-'}` / "
                            f"Short Put `{_sell_put.symbol}` / "
                            f"Short Call `{_sell_call.symbol}` / "
                            f"Long Call `{_buy_call.symbol if _buy_call else '-'}`"
                        )
                    else:
                        st.write(
                            f"固定数量: {_test_qty:.2f} | "
                            f"Short Put `{_sell_put.symbol}` / "
                            f"Short Call `{_sell_call.symbol}`"
                        )

                if not _sl_supported:
                    if _sl_cfg_pct <= 0:
                        st.caption("当前策略未配置止损线，无法对测试单启用自动止损。")
                    elif not engine.is_running:
                        st.caption("引擎未运行，无法接管测试单并执行自动止损。")

                if _sl_cfg_pct > 0:
                    _test_total_premium = _premium_per * _test_qty
                    if not _is_ic:
                        _current_close_cost_per = _sp_ask + _sc_ask
                    else:
                        _current_close_cost_per = (_sp_ask + _sc_ask) - (_lp_bid + _lc_bid)
                    _current_close_cost_total = _current_close_cost_per * _test_qty
                    _current_test_pnl_total = _test_total_premium - _current_close_cost_total
                    _current_test_pnl_pct = (
                        (_current_test_pnl_total / _test_total_premium) * 100.0
                        if _test_total_premium > 0
                        else None
                    )
                    _stop_loss_trigger_cost_per = _premium_per * (1 + _sl_cfg_pct / 100.0)
                    _stop_loss_trigger_cost_total = _stop_loss_trigger_cost_per * _test_qty
                    _stop_loss_trigger_loss_total = _test_total_premium * _sl_cfg_pct / 100.0
                    _stop_loss_trigger_spot_low = _sell_put.strike - _stop_loss_trigger_cost_per
                    _stop_loss_trigger_spot_high = _sell_call.strike + _stop_loss_trigger_cost_per
                    _sl1, _sl2, _sl3, _sl4 = st.columns(4)
                    with _sl1:
                        st.metric("止损线", f"-{_sl_cfg_pct:.0f}%")
                    with _sl2:
                        st.metric("测试篮子净收权利金(0.01)", f"${_test_total_premium:.2f}")
                    with _sl3:
                        st.metric(
                            "当前测试篮子PnL(0.01)",
                            f"${_current_test_pnl_total:.2f}",
                            delta=(
                                f"回补 ${_current_close_cost_total:.2f} / {_current_test_pnl_pct:+.0f}%"
                                if _current_test_pnl_pct is not None
                                else f"回补 ${_current_close_cost_total:.2f}"
                            ),
                            delta_color="normal" if _current_test_pnl_total >= 0 else "inverse",
                        )
                    with _sl4:
                        st.metric(
                            "止损触发价格(近似)",
                            f"${_stop_loss_trigger_spot_low:,.0f} / ${_stop_loss_trigger_spot_high:,.0f}",
                            delta=f"测试0.01回补成本 ${_stop_loss_trigger_cost_total:.2f}",
                            delta_color="inverse",
                        )

                    st.caption(
                        "止损按实时篮子 PnL% 触发。上面所有数字都按测试数量 "
                        f"{_test_qty:.2f} 计算：当前测试篮子 PnL = 净收权利金 - 当前理论回补成本；"
                        f"当标的价格大致到达 ${_stop_loss_trigger_spot_low:,.0f} / ${_stop_loss_trigger_spot_high:,.0f} 一带时，"
                        f"通常对应接近止损触发区；"
                        f"按测试数量 {_test_qty:.2f} 折算，等价于回补成本 ${_stop_loss_trigger_cost_total:.2f}。"
                    )
                    if _auto_stop_loss and _sl_supported:
                        st.info(
                            f"本次测试单成交后将接入引擎持仓管理，并按当前策略止损线 -{_sl_cfg_pct:.0f}% 自动监控。"
                        )

                if _test_disabled_reason:
                    st.info(_test_disabled_reason)
                elif _preview_test_clicked:
                    _new_task_id = uuid.uuid4().hex
                    st.session_state["preview_test_order_task_id"] = _new_task_id
                    _update_test_order_task(
                        _new_task_id,
                        state="queued",
                        status="queued",
                        message="测试下单任务已创建，等待后台执行",
                        percent=0,
                        legs=[],
                        created_at=datetime.now(timezone.utc).isoformat(),
                        order_summary={
                            "sell_call": _sell_call.symbol,
                            "buy_call": _buy_call.symbol if _buy_call else None,
                            "sell_put": _sell_put.symbol,
                            "buy_put": _buy_put.symbol if _buy_put else None,
                            "quantity": _test_qty,
                            "execution_mode": "market",
                            "apply_strategy_stop_loss": bool(_auto_stop_loss and _sl_supported),
                            "stop_loss_pct": _sl_cfg_pct,
                            "stop_loss_underlying_move_pct": float(getattr(cfg.strategy, "stop_loss_underlying_move_pct", 0.0) or 0.0),
                        },
                    )
                    _start_test_order_task(
                        _new_task_id,
                        cfg.exchange,
                        storage,
                        {
                            "sell_call_symbol": _sell_call.symbol,
                            "buy_call_symbol": _buy_call.symbol if _buy_call else None,
                            "sell_put_symbol": _sell_put.symbol,
                            "buy_put_symbol": _buy_put.symbol if _buy_put else None,
                            "sell_call_strike": _sell_call.strike,
                            "buy_call_strike": _buy_call.strike if _buy_call else None,
                            "sell_put_strike": _sell_put.strike,
                            "buy_put_strike": _buy_put.strike if _buy_put else None,
                            "quantity": _test_qty,
                            "underlying_price": _pv_spot,
                        },
                        engine,
                        attach_to_engine_for_stop_loss=bool(_auto_stop_loss and _sl_supported),
                    )
                    st.rerun()

                _active_test_task_id = st.session_state.get("preview_test_order_task_id")
                _active_test_task = _get_test_order_task(_active_test_task_id)
                if _active_test_task:
                    assert isinstance(_active_test_task_id, str)
                    @st.fragment(run_every=1)
                    def _render_test_order_progress(task_id: str):
                        _task = _get_test_order_task(task_id)
                        if not _task:
                            return

                        _state = _task.get("state", "unknown")
                        _message = _task.get("message", "-")
                        _percent = int(_task.get("percent") or 0)
                        _progress = _task.get("progress") or {}
                        _legs = _task.get("legs") or []
                        _runtime = _get_test_stop_loss_runtime(_task, engine)

                        st.progress(max(0, min(_percent, 100)), text=f"测试下单进度：{_percent}% | {_message}")
                        if _state in {"queued", "running"}:
                            st.info(_message)
                        elif _state == "success":
                            st.success(_message)
                        elif _state in {"failed", "error"}:
                            st.error(_message)
                        if _task.get("recovered_via_exchange_snapshot"):
                            st.warning("本次测试单是通过交易所持仓复核确认为成功的；前序下单返回或本地记账曾出现异常。")

                        _elapsed = _progress.get("elapsed_sec")
                        _remaining = _progress.get("remaining_sec")
                        _filled_legs = _progress.get("filled_legs", 0)
                        _total_legs = _progress.get("total_legs", len(_legs))
                        _meta_cols = st.columns(4)
                        with _meta_cols[0]:
                            st.metric("任务状态", _state)
                        with _meta_cols[1]:
                            st.metric("已成交腿数", f"{_filled_legs}/{_total_legs}")
                        with _meta_cols[2]:
                            st.metric("已耗时", f"{_elapsed:.0f}s" if isinstance(_elapsed, (int, float)) else "-")
                        with _meta_cols[3]:
                            st.metric("剩余追单时长", f"{_remaining:.0f}s" if isinstance(_remaining, (int, float)) else "-")

                        if _task.get("result_group_id") or _task.get("attached_to_engine_for_stop_loss"):
                            st.markdown("##### 🛡️ 测试单自动止损状态")
                            _sl_cols = st.columns(4)
                            with _sl_cols[0]:
                                st.metric("测试组ID", str(_task.get("result_group_id") or "-"))
                            with _sl_cols[1]:
                                st.metric("自动止损", "已启用" if _runtime.get("attached") else "未启用")
                            with _sl_cols[2]:
                                _runtime_state = str(_runtime.get("state") or "unknown")
                                _runtime_state_label = {
                                    "active": "监控中",
                                    "active_no_underlying": "监控中",
                                    "not_open": "已不在引擎持仓",
                                    "engine_unavailable": "引擎不可用",
                                    "missing_group_id": "缺少组ID",
                                    "not_attached": "未接入",
                                }.get(_runtime_state, _runtime_state)
                                st.metric("引擎接管状态", _runtime_state_label)
                            with _sl_cols[3]:
                                _rt_sl_pct = _to_float(_runtime.get("stop_loss_pct"), 0.0)
                                st.metric("策略止损线", f"-{_rt_sl_pct:.0f}%" if _rt_sl_pct else "-")

                            if _runtime.get("attached") and _runtime.get("state") in {"active", "active_no_underlying"}:
                                _basket_pnl = _to_float(_runtime.get("basket_pnl"), 0.0)
                                _basket_pnl_pct_raw = _runtime.get("basket_pnl_pct")
                                _basket_pnl_pct = _to_float(_basket_pnl_pct_raw, 0.0) if isinstance(_basket_pnl_pct_raw, (int, float, str)) else None
                                _close_cost = _to_float(_runtime.get("close_cost"), 0.0)
                                _trigger_spot_low = _runtime.get("stop_loss_trigger_spot_low")
                                _trigger_spot_high = _runtime.get("stop_loss_trigger_spot_high")
                                _trigger_close_cost = _to_float(_runtime.get("stop_loss_trigger_close_cost"), 0.0)
                                _trigger_pnl_raw = _runtime.get("stop_loss_trigger_pnl")
                                _trigger_pnl = _to_float(_trigger_pnl_raw, 0.0) if _trigger_pnl_raw is not None else None
                                _rt_cols = st.columns(3)
                                with _rt_cols[0]:
                                    st.metric(
                                        "当前篮子PnL",
                                        f"${_basket_pnl:.2f}",
                                        delta=(
                                            f"{_basket_pnl_pct:+.0f}%"
                                            if _basket_pnl_pct is not None
                                            else None
                                        ),
                                        delta_color="normal" if _basket_pnl >= 0 else "inverse",
                                    )
                                with _rt_cols[1]:
                                    st.metric("当前理论回补成本", f"${_close_cost:.2f}")
                                with _rt_cols[2]:
                                    st.metric(
                                        "止损触发价格(近似)",
                                        (
                                            f"${_to_float(_trigger_spot_low):,.0f} / ${_to_float(_trigger_spot_high):,.0f}"
                                            if _trigger_spot_low is not None and _trigger_spot_high is not None
                                            else "-"
                                        ),
                                        delta=(
                                            f"当前仓回补成本 ${_trigger_close_cost:.2f} | 触发PnL ${_trigger_pnl:.2f}"
                                            if _trigger_pnl is not None
                                            else None
                                        ),
                                        delta_color="inverse",
                                    )
                                if _runtime.get("mark_prices_error"):
                                    st.warning(f"测试单已接入自动止损，但当前获取标记价格失败：{_runtime['mark_prices_error']}")
                                else:
                                    st.success("该测试单当前已接入引擎持仓管理，自动止损正在按策略规则监控。")
                            elif _runtime.get("attached") and _runtime.get("state") == "not_open":
                                st.info("该测试单之前已接入自动止损，但当前已不在引擎持仓中，可能已被手动平仓、止损平仓，或引擎已重建持仓状态。")
                            elif _runtime.get("attached") and _runtime.get("state") == "engine_unavailable":
                                st.warning("该测试单已标记为接入自动止损，但当前引擎未运行或不可用，无法确认实时监控状态。")
                            elif not _runtime.get("attached"):
                                st.caption("本次测试单未接入自动止损，仅用于验证下单链路。")

                        if _legs:
                            _legs_df = pd.DataFrame([
                                {
                                    "腿": leg.get("leg_role", "-"),
                                    "合约": leg.get("symbol", "-"),
                                    "方向": leg.get("side", "-"),
                                    "状态": leg.get("status", "-"),
                                    "已成交": f"{float(leg.get('filled_qty') or 0.0):.4f}/{float(leg.get('quantity') or 0.0):.4f}",
                                    "当前委托价": f"${float(leg.get('current_price') or 0.0):.2f}" if leg.get("current_price") else "-",
                                    "成交均价": f"${float(leg.get('avg_price') or 0.0):.2f}" if leg.get("avg_price") else "-",
                                    "尝试次数": int(leg.get("attempts") or 0),
                                }
                                for leg in _legs
                            ])
                            st.dataframe(_legs_df, width="stretch", hide_index=True)

                        _snapshot_checked = bool(_task.get("exchange_positions_checked"))
                        _snapshot_underlying = str(_task.get("exchange_positions_underlying") or "")
                        _snapshot_error = _task.get("exchange_positions_error")
                        _snapshot_positions = _task.get("exchange_positions") or []
                        _snapshot_rows = _exchange_position_snapshot_rows(_snapshot_positions)
                        if _snapshot_checked and (_snapshot_error or _snapshot_positions):
                            st.markdown("##### 📦 当前测试持仓复核")
                            if _snapshot_error:
                                st.warning(f"持仓复核失败（{_snapshot_underlying or '-'}）：{_snapshot_error}")
                            elif _snapshot_positions:
                                st.dataframe(pd.DataFrame(_snapshot_rows), width="stretch", hide_index=True)

                        if _task.get("traceback"):
                            with st.expander("查看异常堆栈"):
                                st.code(_task["traceback"])

                        if _state in {"failed", "error"}:
                            st.markdown("##### 🔎 失败后交易所持仓复核")
                            if _snapshot_checked:
                                if _snapshot_error:
                                    st.warning(f"持仓复核失败（{_snapshot_underlying or '-'}）：{_snapshot_error}")
                                elif _snapshot_positions:
                                    st.error(
                                        f"复核发现 {_snapshot_underlying} 仍有 {len(_snapshot_positions)} 条真实持仓，请立即核对是否存在残留仓位。"
                                    )
                                    st.dataframe(pd.DataFrame(_snapshot_rows), width="stretch", hide_index=True)
                                else:
                                    st.success(f"复核完成：{_snapshot_underlying} 当前无真实持仓残留。")
                            else:
                                st.caption("本次失败未完成交易所持仓复核。")

                        if _state in {"success", "failed", "error"}:
                            if st.button("清除测试进度", key=f"clear_test_task_{task_id}"):
                                with _TEST_ORDER_TASKS_LOCK:
                                    _TEST_ORDER_TASKS.pop(task_id, None)
                                st.session_state.pop("preview_test_order_task_id", None)
                                st.rerun()

                    _render_test_order_progress(_active_test_task_id)

                # P&L summary
                st.markdown("##### 💰 盈亏概要")
                if not _is_ic:
                    pl1, pl2, pl3, pl4 = st.columns(4)
                    with pl1:
                        _clr = "normal" if _premium_per > 0 else "inverse"
                        st.metric("净权利金/组", f"${_premium_per:.2f}", delta=f"总计 ${_total_premium:.2f}", delta_color=_clr)
                    with pl2:
                        st.metric("最大收益", f"${_total_premium:.2f}",
                                  delta=f"{_total_premium/_pv_equity*100:.2f}% 权益" if _pv_equity > 0 else None,
                                  delta_color="normal")
                    with pl3:
                        st.metric("最大亏损", "∞ (无保护翼)")
                    with pl4:
                        st.metric("估算保证金", f"${_margin_used:,.0f}")
                else:
                    pl1, pl2, pl3, pl4 = st.columns(4)
                    with pl1:
                        _clr = "normal" if _premium_per > 0 else "inverse"
                        st.metric("净权利金/组", f"${_premium_per:.2f}", delta=f"总计 ${_total_premium:.2f}", delta_color=_clr)
                    with pl2:
                        st.metric("最大收益", f"${_total_premium:.2f}",
                                  delta=f"{_total_premium/_pv_equity*100:.2f}% 权益" if _pv_equity > 0 else None,
                                  delta_color="normal")
                    with pl3:
                        st.metric("最大亏损 (单侧)", f"${_total_max_loss:.2f}",
                                  delta=f"{_total_max_loss/_pv_equity*100:.1f}% 权益" if _pv_equity > 0 else None,
                                  delta_color="inverse")
                    with pl4:
                        _rr = _total_premium / _total_max_loss if _total_max_loss > 0 else 0
                        st.metric("盈亏比", f"1 : {1/_rr:.1f}" if _rr > 0 else "-")

                # Breakeven
                be1, be2, be3, be4 = st.columns(4)
                with be1:
                    st.metric("下方盈亏平衡", f"${_be_lower:,.0f}")
                with be2:
                    st.metric("上方盈亏平衡", f"${_be_upper:,.0f}")
                with be3:
                    st.metric("保证金占用", f"${_margin_used:,.0f}")
                with be4:
                    _lev = _pv_spot * _pv_qty / _pv_equity if _pv_equity > 0 else 0
                    st.metric("名义杠杆 (单侧)", f"{_lev:.2f}x")

                # ---------- Plotly payoff chart ----------
                st.markdown("##### 📈 到期盈亏曲线")
                import numpy as _np

                if not _is_ic:
                    _price_lo = _sell_put_t.strike * 0.85
                    _price_hi = _sell_call_t.strike * 1.15
                    _prices = _np.linspace(_price_lo, _price_hi, 500)

                    def _payoff(S):
                        sc = -_np.maximum(S - _sell_call_t.strike, 0)
                        sp = -_np.maximum(_sell_put_t.strike - S, 0)
                        return (sc + sp + _premium_per) * _pv_qty

                    _pnl = _payoff(_prices)
                    _chart_title = f"{_pv_mode_label} 到期盈亏  |  {_pv_ul} Spot=${_pv_spot:,.0f}  |  Qty={_pv_qty:.0f}"
                    _strike_lines = [
                        (_sell_put_t.strike,  f"Short Put ${_sell_put_t.strike:,.0f}",  "red"),
                        (_sell_call_t.strike, f"Short Call ${_sell_call_t.strike:,.0f}", "red"),
                    ]
                    # No max loss line for wingless structure (unlimited)
                    _max_loss_line = None
                else:
                    assert _buy_call_t is not None and _buy_put_t is not None
                    _buy_call_payoff: OptionTicker = _buy_call_t
                    _buy_put_payoff: OptionTicker = _buy_put_t
                    _price_lo = _buy_put_payoff.strike * 0.92
                    _price_hi = _buy_call_t.strike * 1.08
                    _prices = _np.linspace(_price_lo, _price_hi, 500)

                    def _payoff(S):
                        sc = -_np.maximum(S - _sell_call_t.strike, 0)
                        lc = _np.maximum(S - _buy_call_payoff.strike, 0)
                        sp = -_np.maximum(_sell_put_t.strike - S, 0)
                        lp = _np.maximum(_buy_put_payoff.strike - S, 0)
                        return (sc + lc + sp + lp + _premium_per) * _pv_qty

                    _pnl = _payoff(_prices)
                    _chart_title = f"{_pv_mode_label} 到期盈亏  |  {_pv_ul} Spot=${_pv_spot:,.0f}  |  Qty={_pv_qty:.0f}"
                    _strike_lines = [
                        (_buy_put_t.strike,   f"Long Put ${_buy_put_t.strike:,.0f}",   "green"),
                        (_sell_put_t.strike,  f"Short Put ${_sell_put_t.strike:,.0f}",  "red"),
                        (_sell_call_t.strike, f"Short Call ${_sell_call_t.strike:,.0f}", "red"),
                        (_buy_call_t.strike,  f"Long Call ${_buy_call_t.strike:,.0f}",  "green"),
                    ]
                    _max_loss_line = -_total_max_loss

                _fig = go.Figure()
                _fig.add_trace(go.Scatter(
                    x=_prices, y=_pnl,
                    mode="lines", line=dict(color="royalblue", width=2.5),
                    name="P&L", hovertemplate=f"{_pv_ul}: $%{{x:,.0f}}<br>P&L: $%{{y:,.2f}}<extra></extra>",
                ))
                _pnl_pos = _np.where(_pnl > 0, _pnl, 0)
                _fig.add_trace(go.Scatter(
                    x=_prices, y=_pnl_pos, fill="tozeroy",
                    fillcolor="rgba(0,200,83,0.15)", line=dict(width=0),
                    showlegend=False, hoverinfo="skip",
                ))
                _pnl_neg = _np.where(_pnl < 0, _pnl, 0)
                _fig.add_trace(go.Scatter(
                    x=_prices, y=_pnl_neg, fill="tozeroy",
                    fillcolor="rgba(255,82,82,0.15)", line=dict(width=0),
                    showlegend=False, hoverinfo="skip",
                ))

                for _k, _lbl, _clr2 in _strike_lines:
                    _fig.add_vline(x=_k, line_dash="dot", line_color=_clr2,
                                   opacity=0.5, annotation_text=_lbl,
                                   annotation_position="top")

                _fig.add_vline(x=_pv_spot, line_dash="dash", line_color="orange",
                               opacity=0.7, annotation_text=f"Spot ${_pv_spot:,.0f}",
                               annotation_position="bottom right")
                _fig.add_vline(x=_be_lower, line_dash="dashdot", line_color="purple",
                               opacity=0.4, annotation_text=f"BE ${_be_lower:,.0f}")
                _fig.add_vline(x=_be_upper, line_dash="dashdot", line_color="purple",
                               opacity=0.4, annotation_text=f"BE ${_be_upper:,.0f}")
                _fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.3)

                # Max profit / loss annotations
                _fig.add_hline(y=_total_premium, line_dash="dot",
                               line_color="green", opacity=0.3,
                               annotation_text=f"Max Profit ${_total_premium:.0f}")
                if _max_loss_line is not None:
                    _fig.add_hline(y=_max_loss_line, line_dash="dot",
                                   line_color="red", opacity=0.3,
                                   annotation_text=f"Max Loss ${_max_loss_line:.0f}")

                _fig.update_layout(
                    title=_chart_title,
                    xaxis_title=f"{_pv_ul} 到期价格 ($)",
                    yaxis_title="盈亏 ($)",
                    height=450,
                    hovermode="x unified",
                    template="plotly_white",
                    showlegend=False,
                )

                st.plotly_chart(_fig, width='stretch')

                # Text summary
                with st.expander("📊 完整计算明细"):
                    if not _is_ic:
                        _mode_detail = (
                            f"模式: {_pv_mode_label}  |Δ|={getattr(cfg.strategy, 'target_delta', 0.40):.0%}  "
                            f"到期: {_target_dte_days:.1f}天"
                        )
                        st.code(
                            f"标的: {_pv_ul}  现货: ${_pv_spot:,.2f}\n"
                            f"{_mode_detail}\n"
                            f"───────────────────────────────\n"
                            f"Short Put:  {_sell_put.symbol}  K=${_sell_put.strike:,.0f}\n"
                            f"Short Call: {_sell_call.symbol}  K=${_sell_call.strike:,.0f}\n"
                            f"───────────────────────────────\n"
                            f"数量: {_pv_qty:.2f} 张  (fixed={_base_qty})\n"
                            f"权益: ${_pv_equity:,.2f}  leverage: {getattr(cfg.strategy, 'leverage', 1.0):.1f}x\n"
                            f"估算保证金: ${_margin_used:,.0f}\n"
                            f"───────────────────────────────\n"
                            f"净权利金/组: ${_premium_per:.2f}  (sell_call_bid={_sc_bid:.2f} + sell_put_bid={_sp_bid:.2f})\n"
                            f"总权利金: ${_total_premium:.2f}\n"
                            f"最大亏损: 无限 (无保护翼)\n"
                            f"盈亏平衡: ${_be_lower:,.0f} ~ ${_be_upper:,.0f}\n"
                            f"名义杠杆(单侧): {_lev:.2f}x\n",
                            language=None,
                        )
                    else:
                        _mode_detail = (
                            f"模式: {_pv_mode_label}  |Δ|={getattr(cfg.strategy, 'target_delta', 0.40):.0%}  "
                            f"翼Δ={getattr(cfg.strategy, 'wing_delta', 0.05):.0%}  到期: {_target_dte_days:.1f}天"
                        )
                        _margin_pct_str = f" ({_margin_used/_pv_equity*100:.1f}% 权益)" if _pv_equity > 0 else ""
                        assert _buy_call_t is not None and _buy_put_t is not None
                        st.code(
                            f"标的: {_pv_ul}  现货: ${_pv_spot:,.2f}\n"
                            f"{_mode_detail}\n"
                            f"───────────────────────────────\n"
                            f"Long Put:   {_buy_put_t.symbol}  K=${_buy_put_t.strike:,.0f}\n"
                            f"Short Put:  {_sell_put_t.symbol}  K=${_sell_put_t.strike:,.0f}\n"
                            f"Short Call: {_sell_call_t.symbol}  K=${_sell_call_t.strike:,.0f}\n"
                            f"Long Call:  {_buy_call_t.symbol}  K=${_buy_call_t.strike:,.0f}\n"
                            f"───────────────────────────────\n"
                            f"Put侧翼宽:  ${_put_width:,.0f}\n"
                            f"Call侧翼宽: ${_call_width:,.0f}\n"
                            f"───────────────────────────────\n"
                            f"数量: {_pv_qty:.2f} 张  (fixed={_base_qty})\n"
                            f"权益: ${_pv_equity:,.2f}  leverage: {getattr(cfg.strategy, 'leverage', 1.0):.1f}x\n"
                            f"保证金: ${_margin_used:,.0f}{_margin_pct_str}\n"
                            f"───────────────────────────────\n"
                            f"净权利金/组: ${_premium_per:.2f}  (sell_call_bid={_sc_bid:.2f} + sell_put_bid={_sp_bid:.2f}"
                            f" - buy_call_ask={_lc_ask:.2f} - buy_put_ask={_lp_ask:.2f})\n"
                            f"总权利金: ${_total_premium:.2f}\n"
                            f"最大亏损(单侧): ${_total_max_loss:.2f}\n"
                            f"盈亏平衡: ${_be_lower:,.0f} ~ ${_be_upper:,.0f}\n"
                            f"名义杠杆(单侧): {_lev:.2f}x\n",
                            language=None,
                        )
            else:
                if _test_order_used_friday_fallback:
                    st.warning(
                        f"⚠️ 周日到期合约不可用，已尝试回退到 {_active_dte_label_str}，"
                        f"但仍无法找到完整的合约腿（共 {len(_pv_today)} 个 ticker）。"
                    )
                else:
                    st.warning(f"⚠️ 无法找到完整的合约腿 ({_active_dte_label_str}，共 {len(_pv_today)} 个 ticker)")
                _render_available_expiry_panel()
        else:
            if _pv_spot <= 0:
                st.warning("⚠️ 无法获取现货价格")
            else:
                if _friday_expiry_target.tickers:
                    st.warning(
                        f"⚠️ 周日到期合约不可用 ({_dte_label_str})；"
                        f"可在测试下单时自动切换到 {_friday_expiry_target.label}。"
                    )
                else:
                    st.warning(
                        f"⚠️ 目标合约不可用 ({_dte_label_str})，且最近周五到期合约也不可用"
                    )
                _render_available_expiry_panel()
    except Exception as _pv_ex:
        st.error(f"下单预览失败: {_pv_ex}")
        import traceback
        st.code(traceback.format_exc())

    if not _pv_ok:
        st.info(f"💡 下单预览需要实时行情数据，请确保网络连通且有可匹配合约")

    st.divider()

    _render_strategy_form()

    st.divider()

    # --- Strategy State ---
    st.subheader("💾 策略状态 (持久化)")
    state_keys = ["last_trade_date", "day_start_equity", "day_realized_pnl",
                   "day_fees", "day_trade_count"]
    state_data = []
    for k in state_keys:
        v = storage.load_state(k)
        state_data.append({"键": k, "值": str(v) if v is not None else "-"})

    st.dataframe(pd.DataFrame(state_data), width="stretch",
                  hide_index=True)

    st.divider()

    # --- Performance Stats ---
    st.subheader("📊 综合绩效")
    stats = storage.get_trade_stats()
    daily_pnl = storage.get_daily_pnl()
    equity_curve = storage.get_equity_curve()

    # Drawdown
    peak = 0.0
    max_dd = 0.0
    for snap in equity_curve:
        eq = snap["total_equity"]
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # Daily returns for Sharpe
    daily_returns = []
    for d in daily_pnl:
        if d["starting_equity"] > 0:
            ret = (d["ending_equity"] - d["starting_equity"]) / d["starting_equity"]
            daily_returns.append(ret)

    avg_daily = sum(daily_returns) / len(daily_returns) if daily_returns else 0
    profitable_days = sum(1 for r in daily_returns if r > 0)
    loss_days = sum(1 for r in daily_returns if r < 0)

    if len(daily_returns) > 1:
        import math
        mean_ret = avg_daily
        std_ret = math.sqrt(
            sum((r - mean_ret) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
        )
        sharpe = (mean_ret / std_ret * math.sqrt(365)) if std_ret > 0 else 0
    else:
        sharpe = 0

    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    with perf_col1:
        st.metric("总成交笔数", stats["total_trades"])
        st.metric("已平仓", stats["closed_trades"])
        st.metric("持仓中", stats["open_trades"])
    with perf_col2:
        st.metric("胜率", f"{stats['win_rate']:.1f}%")
        st.metric("盈利笔数", stats["win_count"])
        st.metric("亏损笔数", stats["loss_count"])
    with perf_col3:
        st.metric("累计 PnL", f"${stats['total_pnl']:,.2f}")
        st.metric("累计手续费", f"${stats['total_fees']:,.2f}")
        st.metric("最大回撤", f"{max_dd*100:.2f}%")
    with perf_col4:
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        st.metric("日均收益率", f"{avg_daily*100:.4f}%")
        st.metric("运行天数", len(daily_pnl))


# ==========================================================================
# PAGE: 引擎状态 (Engine Status)
# ==========================================================================

elif page == "🖥 引擎状态":
    st.title("🖥 交易引擎状态")

    es = engine.status()

    # --- Status banner ---
    if es["running"]:
        st.success(f"🟢 引擎正在运行  —  已运行 {es['uptime_str']}")
    else:
        st.error("🔴 引擎未运行")
        if is_trade_mode:
            st.info("💡 点击左侧 **🚀 启动引擎** 按钮开始交易")
        else:
            st.info("💡 切换到侧边栏 **🟢 交易模式** 后方可启动引擎")

    st.divider()

    # --- Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("运行状态", "运行中" if es["running"] else "已停止")
    with col2:
        st.metric("运行时长", es["uptime_str"])
    with col3:
        st.metric("策略 Tick 次数", es["tick_count"])
    with col4:
        st.metric("当前持仓组数", es["open_positions"])

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("检查间隔", f"{es['check_interval']}s")
    with col6:
        last_ago = es.get("last_tick_ago_sec")
        st.metric("上次 Tick", f"{last_ago:.0f}s 前" if last_ago else "-")
    with col7:
        st.metric("错误次数", es["error_count"])
    with col8:
        env = "🧪 Testnet" if cfg.exchange.testnet else "🔴 Production"
        st.metric("API 环境", env)

    strategy_status = es.get("strategy_status") or {}
    if strategy_status.get("position_source") == "exchange":
        _symbols = strategy_status.get("exchange_position_symbols") or []
        st.caption(f"当前持仓组数按交易所实时持仓兜底显示；本地数据库为空，交易所合约: {', '.join(_symbols) if _symbols else '-'}")
    col9, col10, col11, col12 = st.columns(4)
    with col9:
        rv_now = strategy_status.get("entry_realized_vol_current")
        st.metric(
            "当前 RV24",
            f"{rv_now:.2%}" if isinstance(rv_now, (int, float)) else "-",
        )
    with col10:
        rv_cap = strategy_status.get("entry_realized_vol_max")
        st.metric(
            "RV 上限",
            f"{rv_cap:.2f}" if isinstance(rv_cap, (int, float)) and rv_cap > 0 else "关闭",
        )
    with col11:
        basket_pnl = strategy_status.get("basket_pnl_pct")
        st.metric(
            "篮子 PnL%",
            f"{basket_pnl:.1f}%" if isinstance(basket_pnl, (int, float)) else "-",
        )
    with col12:
        sl_pct = strategy_status.get("stop_loss_pct")
        sl_move_pct = strategy_status.get("stop_loss_underlying_move_pct")
        st.metric(
            "止损线",
            (
                f"-{sl_pct:.0f}% / {sl_move_pct:.1f}%"
                if isinstance(sl_pct, (int, float)) and sl_pct > 0 and isinstance(sl_move_pct, (int, float)) and sl_move_pct > 0
                else (f"-{sl_pct:.0f}%" if isinstance(sl_pct, (int, float)) and sl_pct > 0 else "关闭")
            ),
        )

    _last_order_status = str(strategy_status.get("last_order_status") or "")
    _last_order_message = str(strategy_status.get("last_order_message") or "")
    _last_order_attempt_at = str(strategy_status.get("last_order_attempt_at") or "")
    _last_order_group_id = str(strategy_status.get("last_order_group_id") or "")

    if _last_order_status and _last_order_status != "idle":
        st.subheader("🧾 最近一次实盘开仓结果")
        _lc1, _lc2, _lc3, _lc4 = st.columns(4)
        with _lc1:
            st.metric("状态", _last_order_status)
        with _lc2:
            st.metric("尝试时间", _last_order_attempt_at or "-")
        with _lc3:
            st.metric("组合ID", _last_order_group_id or "-")
        with _lc4:
            st.metric("持仓复核", "已执行" if strategy_status.get("last_order_exchange_positions_checked") else "未执行")

            if _last_order_status == "success":
                st.success(_last_order_message or "最近一次实盘开仓成功")
            elif _last_order_status in {"failed", "error"}:
                st.error(_last_order_message or "最近一次实盘开仓失败")
            else:
                st.info(_last_order_message or "最近一次实盘开仓处理中")

            if _last_order_status in {"failed", "error"}:
                st.markdown("##### 🔎 实盘失败后交易所持仓复核")
                _snapshot_underlying = str(strategy_status.get("last_order_exchange_positions_underlying") or "")
                _snapshot_error = strategy_status.get("last_order_exchange_positions_error")
                _snapshot_positions = strategy_status.get("last_order_exchange_positions") or []
                if _snapshot_error:
                    st.warning(f"持仓复核失败（{_snapshot_underlying or '-'}）：{_snapshot_error}")
                elif _snapshot_positions:
                    _snapshot_rows = [
                        {
                            "合约": p.get("symbol", ""),
                            "方向": p.get("side", ""),
                            "数量": float(p.get("quantity") or 0.0),
                            "开仓价": f"${float(p.get('entryPrice') or 0.0):.4f}",
                            "未实现PnL": f"${float(p.get('unrealizedPnl') or 0.0):.4f}",
                        }
                        for p in _snapshot_positions
                    ]
                    st.error(
                        f"复核发现 {_snapshot_underlying} 仍有 {len(_snapshot_positions)} 条真实持仓，请立即核对是否存在残留仓位。"
                    )
                    st.dataframe(pd.DataFrame(_snapshot_rows), width="stretch", hide_index=True)
                elif strategy_status.get("last_order_exchange_positions_checked"):
                    st.success(f"复核完成：{_snapshot_underlying or cfg.strategy.underlying} 当前无真实持仓残留。")

    execution_metrics = es.get("execution_metrics") or {}
    recent_execution_events = es.get("recent_execution_events") or []
    if execution_metrics:
        st.subheader("🩺 执行健康")
        _ec1, _ec2, _ec3, _ec4, _ec5, _ec6 = st.columns(6)
        with _ec1:
            st.metric("开仓尝试", int(execution_metrics.get("open_attempts") or 0))
        with _ec2:
            st.metric("成功率", f"{float(execution_metrics.get('success_rate_pct') or 0.0):.1f}%")
        with _ec3:
            st.metric("部分成交", int(execution_metrics.get("open_partials") or 0))
        with _ec4:
            st.metric("失败次数", int(execution_metrics.get("open_failures") or 0))
        with _ec5:
            st.metric("风险告警", int(execution_metrics.get("risk_alerts") or 0))
        with _ec6:
            st.metric("平均完成时长", f"{float(execution_metrics.get('avg_open_duration_sec') or 0.0):.1f}s")

    if recent_execution_events:
        st.markdown("##### 最近执行事件")
        event_rows = [
            {
                "时间": str(item.get("timestamp") or ""),
                "事件": str(item.get("event_type") or ""),
                "状态": str(item.get("execution_state") or ""),
                "级别": str(item.get("severity") or ""),
                "组合ID": str(item.get("group_id") or ""),
                "标的": str(item.get("underlying") or ""),
                "模式": str(item.get("execution_mode") or ""),
                "消息": str(item.get("message") or ""),
            }
            for item in recent_execution_events
        ]
        st.dataframe(pd.DataFrame(event_rows), width="stretch", hide_index=True)

    risk_lock_active = bool(
        strategy_status.get("execution_risk_lock_active")
        or storage.load_state("wv_execution_risk_lock_active", False)
    )
    if risk_lock_active:
        risk_lock_since = str(
            strategy_status.get("execution_risk_lock_since")
            or storage.load_state("wv_execution_risk_lock_since", "")
            or ""
        )
        risk_lock_reason = str(
            strategy_status.get("execution_risk_lock_reason")
            or storage.load_state("wv_execution_risk_lock_reason", "")
            or "执行风控锁已触发"
        )
        risk_lock_event = str(
            strategy_status.get("execution_risk_lock_event")
            or storage.load_state("wv_execution_risk_lock_event", "")
            or ""
        )
        st.warning(
            f"执行风控锁已生效：{risk_lock_reason}"
            + (f" | 触发时间: {risk_lock_since}" if risk_lock_since else "")
            + (f" | 来源事件: {risk_lock_event}" if risk_lock_event else "")
        )
        if st.button(
            "🧯 清除执行风控锁",
            use_container_width=True,
            disabled=not is_trade_mode,
            help="仅在确认当前无残留仓位、无未处理执行风险后再清除。",
        ):
            _clear_lock = getattr(getattr(engine, "strategy", None), "clear_execution_risk_lock", None)
            if callable(_clear_lock):
                typed_clear_lock = cast(Callable[[], None], _clear_lock)
                typed_clear_lock()
            else:
                storage.save_state("wv_execution_risk_lock_active", False)
                storage.save_state("wv_execution_risk_lock_since", "")
                storage.save_state("wv_execution_risk_lock_reason", "")
                storage.save_state("wv_execution_risk_lock_group_id", "")
                storage.save_state("wv_execution_risk_lock_event", "")
                storage.record_execution_event(
                    event_type="execution_risk_lock_cleared",
                    execution_state="idle",
                    severity="info",
                    underlying=cfg.strategy.underlying.upper(),
                    execution_mode="market",
                    message="人工清除执行风控锁（引擎未运行）",
                )
            st.success("执行风控锁已清除")
            _rerun_preserve_nav_page("🖥 引擎状态")

    st.divider()

    # --- Error log ---
    if es["last_error"]:
        st.subheader("⚠️ 最近错误")
        st.error(es["last_error"])

    # --- Engine actions ---
    st.subheader("⚡ 引擎操作")

    if not is_trade_mode:
        st.info("🔒 当前为只读模式，引擎操作已禁用。切换到 **交易模式** 以解锁。")

    col_a1, col_a2, col_a3 = st.columns(3)

    with col_a1:
        if es["running"]:
            if st.button("⏹ 停止引擎", use_container_width=True, type="secondary",
                          disabled=not is_trade_mode):
                engine.stop()
                _rerun_preserve_nav_page("🔧 策略配置")
        else:
            if st.button("🚀 启动引擎", use_container_width=True, type="primary",
                          disabled=not is_trade_mode):
                ok = engine.start()
                if not ok:
                    st.error(f"启动失败: {engine.status().get('last_error')}")
                _rerun_preserve_nav_page("🔧 策略配置")

    with col_a2:
        if st.button("🚨 紧急全部平仓", use_container_width=True, type="primary",
                      disabled=(not es["running"] or not is_trade_mode)):
            pnl = engine.close_all_positions()
            st.success(f"已平仓, PnL: ${pnl:,.4f}")
            _rerun_preserve_nav_page("🔧 策略配置")

    with col_a3:
        if st.button("🔄 重置引擎", use_container_width=True,
                      disabled=(es["running"] or not is_trade_mode),
                      help="停止并释放引擎实例，下次启动将重新初始化"):
            reset_engine()
            _rerun_preserve_nav_page("🔧 策略配置")

    st.divider()

    # --- Config summary ---
    st.subheader("📋 运行配置摘要")
    run_info = {
        "配置项": [
            "策略名", "模式", "结构", "标的", "短腿 |Δ|", "翼 |Δ|",
            "杠杆", "开仓日/时间", "RV过滤", "止损", "固定数量", "模式", "API环境", "检查间隔",
        ],
        "值": [
            cfg.name,
            "Weekend Vol (delta选行权价)",
            "带翼结构" if getattr(cfg.strategy, 'wing_delta', 0.0) > 0 else "无翼结构",
            cfg.strategy.underlying,
            f"{getattr(cfg.strategy, 'target_delta', 0.40):.0%}",
            f"{getattr(cfg.strategy, 'wing_delta', 0.05):.0%}",
            f"{getattr(cfg.strategy, 'leverage', 1.0):.1f}x",
            f"{getattr(cfg.strategy, 'entry_day', 'friday')} {cfg.strategy.entry_time_utc} UTC",
            (
                f"RV{getattr(cfg.strategy, 'entry_realized_vol_lookback_hours', 0)} <= "
                f"{getattr(cfg.strategy, 'entry_realized_vol_max', 0.0):.2f}"
                if getattr(cfg.strategy, 'entry_realized_vol_lookback_hours', 0) > 1
                and getattr(cfg.strategy, 'entry_realized_vol_max', 0.0) > 0
                else "关闭"
            ),
            (
                (
                    f"-{getattr(cfg.strategy, 'stop_loss_pct', 0.0):.0f}% + 标的单边{getattr(cfg.strategy, 'stop_loss_underlying_move_pct', 0.0):.1f}%"
                    if getattr(cfg.strategy, 'stop_loss_pct', 0.0) > 0 and getattr(cfg.strategy, 'stop_loss_underlying_move_pct', 0.0) > 0
                    else f"-{getattr(cfg.strategy, 'stop_loss_pct', 0.0):.0f}%"
                )
                if getattr(cfg.strategy, 'stop_loss_pct', 0.0) > 0
                else "关闭"
            ),
            str(cfg.strategy.quantity),
            "固定数量",
            "测试网" if cfg.exchange.testnet else "正式环境",
            f"{cfg.monitor.check_interval_sec}s",
        ],
    }
    st.dataframe(pd.DataFrame(run_info), width="stretch", hide_index=True)

    st.caption(
        "💡 **一体化启动方式:** 只需运行 "
        "`streamlit run trader/dashboard.py -- --config configs/trader/weekend_vol_btc.yaml` "
        "然后切换到 **🟢 交易模式** 并点击 **🚀 启动引擎** 即可同时管理前端与后端。"
    )


# ==========================================================================
# Auto-refresh is handled by inline JS timer above (no external dependency)
# ==========================================================================
