"""Trader Dashboard – 交易管理面板.

Streamlit 前端，用于:
- 实时查看策略状态和持仓
- 资产曲线图表
- 损益记录与统计
- 成交历史查询
- 手动操作 (平仓、暂停)

Usage:
    streamlit run dashboard_app.py -- --config configs/trader_iron_condor_0dte.yaml
"""

from __future__ import annotations

import hmac
import json
import os
import platform
import requests
import sys
import threading
import time as _time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# OS-aware Streamlit server config (injected before Streamlit reads config)
# ---------------------------------------------------------------------------
# Streamlit reads STREAMLIT_SERVER_* env vars as config overrides.
# We detect the OS here and set sensible defaults so .streamlit/config.toml
# stays platform-agnostic (only theme / browser settings).
# ---------------------------------------------------------------------------
_is_linux = platform.system() == "Linux"
_dashboard_public = os.environ.get("DASHBOARD_PUBLIC", "false").strip().lower() in {"1", "true", "yes", "on"}

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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from loguru import logger

from trader.config import load_config, TraderConfig
from trader.storage import Storage
from trader.engine import get_engine, reset_engine, TradingEngine
from trader.binance_client import BinanceOptionsClient
from trader.position_manager import PositionManager
from trader.limit_chaser import ChaserConfig as DashboardChaserConfig

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="铁鹰交易面板",
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


def _new_error_id(prefix: str = "dash") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def _log_exception_with_id(message: str, exc: Exception, *, prefix: str = "dash") -> str:
    error_id = _new_error_id(prefix)
    logger.exception(f"{message} | error_id={error_id} | error={exc}")
    return error_id


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
    chaser_cfg,
    order_params: dict,
    engine_ref: TradingEngine,
) -> None:
    def _runner() -> None:
        last_progress: dict[str, object] = {}

        def _infer_underlying_from_order() -> str:
            for _key in ("sell_call_symbol", "sell_put_symbol", "buy_call_symbol", "buy_put_symbol"):
                _symbol = str(order_params.get(_key) or "").strip().upper()
                if _symbol and "-" in _symbol:
                    return _symbol.split("-", 1)[0]
            return ""

        def _capture_exchange_positions_on_failure() -> dict[str, object]:
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
            task_client = BinanceOptionsClient(exchange_cfg)
            task_pos_mgr = PositionManager(
                task_client,
                storage,
                chaser_config=DashboardChaserConfig(
                    window_seconds=chaser_cfg.window_seconds,
                    poll_interval_sec=chaser_cfg.poll_interval_sec,
                    tick_size_usdt=chaser_cfg.tick_size_usdt,
                    market_fallback_sec=chaser_cfg.market_fallback_sec,
                    max_amend_attempts=chaser_cfg.max_amend_attempts,
                ),
            )

            _is_strangle = not order_params.get("buy_call_symbol") and not order_params.get("buy_put_symbol")
            if _is_strangle:
                condor = task_pos_mgr.open_short_strangle(
                    execution_mode="market",
                    status_callback=_on_progress,
                    sell_call_symbol=order_params["sell_call_symbol"],
                    sell_put_symbol=order_params["sell_put_symbol"],
                    sell_call_strike=order_params["sell_call_strike"],
                    sell_put_strike=order_params["sell_put_strike"],
                    quantity=order_params["quantity"],
                    underlying_price=order_params["underlying_price"],
                )
            else:
                condor = task_pos_mgr.open_iron_condor(
                    execution_mode="market",
                    status_callback=_on_progress,
                    **order_params,
                )

            if condor is not None:
                _group_id = str(getattr(condor, "group_id", ""))
                if engine_ref.pos_mgr is not None:
                    engine_ref.pos_mgr.open_condors[_group_id] = condor
                _update_test_order_task(
                    task_id,
                    state="success",
                    status="success",
                    message=f"测试下单成功：{_group_id}",
                    percent=100,
                    result_group_id=_group_id,
                )
            else:
                _snapshot = _capture_exchange_positions_on_failure()
                _update_test_order_task(
                    task_id,
                    state="failed",
                    status="failed",
                    message=str(last_progress.get("message") or "测试下单失败：至少有一条腿未成交或被回滚"),
                    **_snapshot,
                )
        except Exception as exc:
            _snapshot = _capture_exchange_positions_on_failure()
            error_id = _log_exception_with_id("Test order task failed", exc, prefix="test")
            _update_test_order_task(
                task_id,
                state="error",
                status="error",
                message=f"测试下单异常，请查看服务端日志。错误编号: {error_id}",
                error=str(exc),
                error_id=error_id,
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
    return "configs/trader/short_strangle_7dte.yaml"


config_path = get_config_path()
cfg = load_config(config_path)          # ← This also loads .env into os.environ
storage = init_storage(cfg.storage.db_path)

# ---------------------------------------------------------------------------
# Login authentication (env: DASHBOARD_READONLY_* / DASHBOARD_TRADER_* / legacy DASHBOARD_*)
# ---------------------------------------------------------------------------

def _get_dashboard_credentials() -> list[dict[str, str]]:
    credentials: list[dict[str, str]] = []

    def _append_role(role: str, user_key: str, pass_key: str) -> None:
        user = os.environ.get(user_key, "").strip()
        password = os.environ.get(pass_key, "").strip()
        if bool(user) != bool(password):
            raise RuntimeError(f"{user_key} 和 {pass_key} 必须同时配置")
        if user and password:
            credentials.append({"role": role, "user": user, "password": password})

    _append_role("readonly", "DASHBOARD_READONLY_USER", "DASHBOARD_READONLY_PASS")
    _append_role("trader", "DASHBOARD_TRADER_USER", "DASHBOARD_TRADER_PASS")
    return credentials

def _check_login() -> bool:
    """Show login form and validate credentials. Returns True if authenticated."""
    try:
        credentials = _get_dashboard_credentials()
    except RuntimeError as exc:
        error_id = _log_exception_with_id("Dashboard credential validation failed", exc, prefix="auth")
        st.error(f"Dashboard 当前不可用，请联系管理员。错误编号: {error_id}")
        st.stop()

    if not credentials:
        error_id = _new_error_id("auth")
        logger.error(f"Dashboard credentials missing | error_id={error_id}")
        st.error(f"Dashboard 当前不可用，请联系管理员。错误编号: {error_id}")
        st.stop()

    # Already authenticated this session
    if st.session_state.get("authenticated") and st.session_state.get("auth_role") in {"readonly", "trader"}:
        return True
    if st.session_state.get("authenticated"):
        st.session_state["authenticated"] = False
        st.session_state.pop("auth_role", None)

    # --- Login form ---
    st.markdown("## 🦅 铁鹰交易面板")
    st.caption("请输入用户名和密码登录")
    _col_l, col_form, _col_r = st.columns([1, 1.5, 1])
    with col_form:
        with st.form("login_form"):
            username = st.text_input("用户名")
            password = st.text_input("密码", type="password")
            submitted = st.form_submit_button("登录", width='stretch', type="primary")

        if submitted:
            matched = next(
                (
                    item for item in credentials
                    if hmac.compare_digest(username, item["user"])
                    and hmac.compare_digest(password, item["password"])
                ),
                None,
            )
            if matched is None:
                st.error("用户名或密码错误")
            else:
                st.session_state["authenticated"] = True
                st.session_state["auth_role"] = matched["role"]
                st.rerun()

    return False


if not _check_login():
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("📊 期权交易面板")

# --- Mode selector ---
auth_role = str(st.session_state.get("auth_role", "readonly") or "readonly")
can_trade = auth_role == "trader"
_mode_options = ["🔒 只读模式"] + (["🟢 交易模式"] if can_trade else [])
_mode_param_to_label = {
    "readonly": "🔒 只读模式",
    "trade": "🟢 交易模式",
}
_mode_label_to_param = {v: k for k, v in _mode_param_to_label.items()}
_mode_from_query = _mode_param_to_label.get(str(st.query_params.get("mode", "")).strip().lower())
if not can_trade:
    _mode_from_query = "🔒 只读模式"
if "trading_mode" not in st.session_state:
    st.session_state.trading_mode = _mode_from_query or "🔒 只读模式"
elif _mode_from_query and st.session_state.trading_mode != _mode_from_query:
    st.session_state.trading_mode = _mode_from_query

if st.session_state.trading_mode not in _mode_options:
    st.session_state.trading_mode = "🔒 只读模式"

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
st.sidebar.caption("当前权限: 交易管理员" if can_trade else "当前权限: 只读用户")
st.sidebar.markdown(f"**策略:** {cfg.name}")
st.sidebar.markdown(f"**标的:** {cfg.strategy.underlying}")
_sidebar_mode = getattr(cfg.strategy, 'mode', 'iron_condor')
if _sidebar_mode == "weekend_vol":
    _mode_label = "周末波动率"
    _delta_label = getattr(cfg.strategy, 'target_delta', 0.40)
    _wing_d_label = getattr(cfg.strategy, 'wing_delta', 0.05)
    st.sidebar.markdown(
        f"**模式:** {_mode_label} | **Δ:** {_delta_label:.0%} / {_wing_d_label:.0%} | "
        f"**杠杆:** {getattr(cfg.strategy, 'leverage', 1.0):.0f}x"
    )
elif _sidebar_mode == "strangle":
    _mode_label = "裸卖双卖"
    _dte_label = getattr(cfg.strategy, 'target_dte_days', 0)
    st.sidebar.markdown(f"**模式:** {_mode_label} | **DTE:** {_dte_label}d | **OTM:** ±{cfg.strategy.otm_pct*100:.0f}%")
else:
    _mode_label = "铁鹰"
    _dte_label = getattr(cfg.strategy, 'target_dte_days', 0)
    st.sidebar.markdown(f"**模式:** {_mode_label} | **DTE:** {_dte_label}d | **OTM:** ±{cfg.strategy.otm_pct*100:.0f}%")
st.sidebar.markdown(f"**数据库:** `{cfg.storage.db_path}`")

st.sidebar.divider()

# ---------------------------------------------------------------------------
# Engine Control (交易引擎控制)
# ---------------------------------------------------------------------------

engine = get_engine(cfg)
engine_status = engine.status()

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
        if st.button("⏹ 停止引擎", width='stretch', type="secondary"):
            engine.stop()
            st.rerun()
    with col_close:
        if st.button("🚨 全部平仓", width='stretch', type="primary"):
            pnl = engine.close_all_positions()
            st.sidebar.info(f"已平仓, PnL: ${pnl:,.4f}")
            st.rerun()
else:
    if is_trade_mode:
        st.sidebar.error("🔴 引擎未运行")
        if st.sidebar.button("🚀 启动引擎", width='stretch', type="primary"):
            ok = engine.start()
            if ok:
                st.sidebar.success("引擎已启动!")
            else:
                st.sidebar.error(f"启动失败: {engine.status().get('last_error', '未知错误')}")
            st.rerun()

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
_pause_auto_refresh = _current_nav_page == "🔧 策略配置"

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
    st.sidebar.caption("策略配置页已暂停整页自动刷新，避免表单/下单预览卡顿")

# Manual refresh
if st.sidebar.button("🔄 立即刷新", width='stretch'):
    st.rerun()

# --------------------------------------------------------------------------
# Navigation
# --------------------------------------------------------------------------

page = st.sidebar.radio(
    "导航",
    ["📊 总览", "📈 资产曲线", "💰 损益记录", "📋 成交历史", "📡 期权行情", "🔧 策略配置", "🖥 引擎状态"],
    key="nav_page",
)

# ==========================================================================
# PAGE: 总览 (Overview)
# ==========================================================================

if page == "📊 总览":
    st.title("📊 策略总览")

    @st.cache_resource
    def _get_client(_exchange_cfg) -> BinanceOptionsClient:
        return BinanceOptionsClient(_exchange_cfg)

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
    daily_pnl = storage.get_daily_pnl(start_date, end_date)

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
    if live_account is not None:
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

    st.divider()

    # --- Open Positions (local strategy + exchange) ---
    st.subheader("🔓 当前持仓")
    open_trades = storage.get_open_trades()
    _overview_spot_cache: dict[str, float] = {}

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
    elif not exchange_positions:
        st.info("当前无持仓")
    # If no local trades but exchange has positions, skip the "无持仓" message
    # since exchange positions section below will show them

    st.subheader("🏦 交易所实时持仓")
    if not has_creds:
        st.caption("未配置 API Key/Secret，跳过交易所私有持仓查询")
    elif exchange_positions:
        # Compute total entry value and total unrealized PnL
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
    else:
        st.caption("交易所当前无持仓")

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

    # Try to get live account data and auto-record a snapshot
    @st.cache_resource
    def _get_client_eq(_exchange_cfg) -> BinanceOptionsClient:
        return BinanceOptionsClient(_exchange_cfg)

    _eq_client = _get_client_eq(cfg.exchange)
    _eq_has_creds = bool(cfg.exchange.api_key and cfg.exchange.api_secret)

    if _eq_has_creds and not cfg.exchange.simulate_private:
        try:
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
        except Exception:
            pass

    equity_curve = storage.get_equity_curve(start_date, end_date)

    if not equity_curve:
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

        _dark_axis: dict[str, object] = dict(
            gridcolor=_GRID, zerolinecolor=_GRID,
            tickfont=dict(color=_TEXT, size=11),
            title_font=dict(color=_TEXT, size=12),
        )
        _dark_layout: dict[str, object] = dict(
            paper_bgcolor=_BG, plot_bgcolor=_BG,
            font=dict(color=_TEXT, size=12),
            hovermode="x unified",
            hoverlabel=dict(bgcolor="#1e1e2f", font_size=12),
        )

        # --- Drawdown series (precompute) ---
        peak_series = df["total_equity"].cummax()
        dd_series = (peak_series - df["total_equity"]) / peak_series * 100
        dd_series = dd_series.fillna(0)

        # --- KPI summary row ---
        if len(df) > 1:
            first_eq = df["total_equity"].iloc[0]
            last_eq = df["total_equity"].iloc[-1]
            total_ret = (last_eq - first_eq) / first_eq * 100 if first_eq > 0 else 0
            max_dd_val = dd_series.max()
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
            st.metric("数据点数", f"{len(df):,}")

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
            go.Scatter(
                x=df["timestamp"], y=df["total_equity"],
                name="权益",
                line=dict(color=_EQUITY_COLOR, width=2),
                fill="tozeroy",
                fillcolor="rgba(0,227,150,0.08)",
                hovertemplate="%{y:,.2f}",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"], y=df["available_balance"],
                name="可用余额",
                line=dict(color=_BALANCE_COLOR, width=1.2, dash="dot"),
                hovertemplate="%{y:,.2f}",
            ),
            row=1, col=1,
        )

        # -- Row 2: Unrealized PnL bars --
        upnl_colors = [_UP_COLOR if v >= 0 else _DN_COLOR for v in df["unrealized_pnl"]]
        fig.add_trace(
            go.Bar(
                x=df["timestamp"], y=df["unrealized_pnl"],
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
            go.Scatter(
                x=df["timestamp"], y=dd_series,
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
            paper_bgcolor=_BG,
            plot_bgcolor=_BG,
            font=dict(color=_TEXT, size=12),
            hovermode="x unified",
            hoverlabel=dict(bgcolor="#1e1e2f", font_size=12),
        )

        # Axes styling
        for row_i in range(1, 4):
            fig.update_xaxes(
                row=row_i,
                col=1,
                gridcolor=_GRID,
                zerolinecolor=_GRID,
                tickfont=dict(color=_TEXT, size=11),
                title_font=dict(color=_TEXT, size=12),
                showticklabels=(row_i == 3),
            )
            fig.update_yaxes(
                row=row_i,
                col=1,
                gridcolor=_GRID,
                zerolinecolor=_GRID,
                tickfont=dict(color=_TEXT, size=11),
                title_font=dict(color=_TEXT, size=12),
            )

        fig.update_yaxes(title_text="账户权益", row=1, col=1)
        fig.update_yaxes(title_text="未实现PnL", row=2, col=1)
        fig.update_yaxes(title_text="回撤 %", autorange="reversed", row=3, col=1)
        fig.update_xaxes(rangeslider_visible=False)

        st.plotly_chart(fig, width='stretch')

        # ======== Position count + underlying overlay ========
        has_ul = "underlying_price" in df.columns and df["underlying_price"].sum() > 0
        if has_ul:
            fig2 = make_subplots(
                rows=1, cols=1,
                specs=[[{"secondary_y": True}]],
            )
            fig2.add_trace(
                go.Scatter(
                    x=df["timestamp"], y=df["underlying_price"],
                    name="标的价格 (USD)",
                    line=dict(color="#00b4d8", width=1.5),
                    hovertemplate="$%{y:,.2f}",
                ),
                secondary_y=False,
            )
            fig2.add_trace(
                go.Scatter(
                    x=df["timestamp"], y=df["position_count"],
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
                paper_bgcolor=_BG,
                plot_bgcolor=_BG,
                font=dict(color=_TEXT, size=12),
                hovermode="x unified",
                hoverlabel=dict(bgcolor="#1e1e2f", font_size=12),
            )
            fig2.update_xaxes(
                gridcolor=_GRID,
                zerolinecolor=_GRID,
                tickfont=dict(color=_TEXT, size=11),
                title_font=dict(color=_TEXT, size=12),
            )
            fig2.update_yaxes(
                title_text="标的价格 (USD)",
                secondary_y=False,
                gridcolor=_GRID,
                zerolinecolor=_GRID,
                tickfont=dict(color=_TEXT, size=11),
                title_font=dict(color=_TEXT, size=12),
            )
            fig2.update_yaxes(
                title_text="持仓数",
                secondary_y=True,
                gridcolor=_GRID,
                zerolinecolor=_GRID,
                tickfont=dict(color=_TEXT, size=11),
                title_font=dict(color=_TEXT, size=12),
            )
            st.plotly_chart(fig2, width='stretch')
        else:
            # Position count only
            fig_pos = go.Figure()
            fig_pos.add_trace(go.Scatter(
                x=df["timestamp"], y=df["position_count"],
                name="持仓数", mode="lines+markers",
                line=dict(color=_POS_COLOR, width=2),
                marker=dict(size=4),
                fill="tozeroy", fillcolor="rgba(254,176,25,0.08)",
            ))
            fig_pos.update_layout(
                height=200, margin=dict(l=60, r=16, t=8, b=16),
                yaxis_title="持仓数",
                paper_bgcolor=_BG,
                plot_bgcolor=_BG,
                font=dict(color=_TEXT, size=12),
                hovermode="x unified",
                hoverlabel=dict(bgcolor="#1e1e2f", font_size=12),
            )
            fig_pos.update_xaxes(
                gridcolor=_GRID,
                zerolinecolor=_GRID,
                tickfont=dict(color=_TEXT, size=11),
                title_font=dict(color=_TEXT, size=12),
            )
            fig_pos.update_yaxes(
                gridcolor=_GRID,
                zerolinecolor=_GRID,
                tickfont=dict(color=_TEXT, size=11),
                title_font=dict(color=_TEXT, size=12),
            )
            st.plotly_chart(fig_pos, width='stretch')


# ==========================================================================
# PAGE: 损益记录 (PnL Records)
# ==========================================================================

elif page == "💰 损益记录":
    st.title("💰 每日损益记录")

    daily_pnl = storage.get_daily_pnl(start_date, end_date)

    if not daily_pnl:
        st.info("暂无每日损益数据。")
    else:
        df = pd.DataFrame(daily_pnl)
        df["date"] = pd.to_datetime(df["date"])
        df["daily_return"] = (df["ending_equity"] - df["starting_equity"])
        df["daily_return_pct"] = df.apply(
            lambda r: (r["ending_equity"] - r["starting_equity"]) / r["starting_equity"] * 100
            if r["starting_equity"] > 0 else 0,
            axis=1,
        )
        df["cumulative_pnl"] = df["realized_pnl"].cumsum()

        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_rpnl = df["realized_pnl"].sum()
            st.metric("累计已实现 PnL", f"${total_rpnl:,.2f}")
        with col2:
            total_fees = df["total_fees"].sum()
            st.metric("累计手续费", f"${total_fees:,.2f}")
        with col3:
            profitable = (df["daily_return"] > 0).sum()
            total_days = len(df)
            st.metric("盈利天数", f"{profitable} / {total_days}")
        with col4:
            avg_daily = df["daily_return_pct"].mean()
            st.metric("日均收益率", f"{avg_daily:.3f}%")

        st.divider()

        # --- Daily PnL bar chart ---
        st.subheader("📊 每日收益")
        fig_pnl = go.Figure()

        colors = ["#00C49A" if v >= 0 else "#FF6B6B" for v in df["daily_return"]]
        fig_pnl.add_trace(go.Bar(
            x=df["date"], y=df["daily_return"],
            name="日收益 (USD)",
            marker_color=colors,
        ))

        fig_pnl.update_layout(
            height=350, margin=dict(l=50, r=20, t=10, b=20),
            yaxis_title="USD",
        )
        st.plotly_chart(fig_pnl, width='stretch')

        # --- Cumulative PnL chart ---
        st.subheader("📈 累计已实现 PnL")
        fig_cum = px.area(df, x="date", y="cumulative_pnl",
                          color_discrete_sequence=["#8884d8"])
        fig_cum.update_layout(height=250, margin=dict(l=50, r=20, t=10, b=20),
                              yaxis_title="USD")
        st.plotly_chart(fig_cum, width='stretch')

        # --- Daily return distribution ---
        st.subheader("📊 日收益率分布")
        fig_hist = px.histogram(df, x="daily_return_pct", nbins=30,
                                color_discrete_sequence=["#00C49A"],
                                labels={"daily_return_pct": "日收益率 (%)"})
        fig_hist.update_layout(height=250, margin=dict(l=50, r=20, t=10, b=20))
        st.plotly_chart(fig_hist, width='stretch')

        # --- Daily PnL table ---
        st.subheader("📋 每日明细")
        df_display = pd.DataFrame(df[["date", "starting_equity", "ending_equity",
                  "realized_pnl", "unrealized_pnl", "total_fees",
                  "trade_count", "daily_return_pct"]].copy())
        df_display.columns = ["日期", "日初权益", "日末权益", "已实现PnL",
                               "未实现PnL", "手续费", "交易笔数", "日收益率%"]
        df_display["日期"] = df_display["日期"].dt.strftime("%Y-%m-%d")

        # Format numbers
        for col in ["日初权益", "日末权益", "已实现PnL", "未实现PnL", "手续费"]:
            df_display[col] = df_display[col].map(lambda x: f"${x:,.2f}")
        df_display["日收益率%"] = df_display["日收益率%"].map(lambda x: f"{x:.3f}%")

        st.dataframe(df_display, width="stretch", hide_index=True)


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
        st.subheader("🦅 铁鹰组合损益")
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
        st.download_button(
            "📥 导出 CSV",
            data=csv,
            file_name=f"trades_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
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

    # ---- Editable form ----
    with st.form("config_editor", border=True):
        st.subheader("📄 策略参数")
        _cur_mode = _strategy.get("mode", "strangle")
        _mode_options = ["strangle", "iron_condor", "weekend_vol"]
        _mode_index = _mode_options.index(_cur_mode) if _cur_mode in _mode_options else 0

        s_col1, s_col2, s_col3 = st.columns(3)

        with s_col1:
            ed_mode = st.selectbox(
                "策略模式", _mode_options,
                index=_mode_index,
                help="strangle: 裸卖双卖(2腿)  iron_condor: 铁鹰(4腿)  weekend_vol: 周末波动率(delta选行权价)",
            )
            ed_underlying = st.selectbox(
                "标的", ["ETH", "BTC"],
                index=0 if _strategy.get("underlying", "ETH") == "ETH" else 1,
            )

            # --- OTM%-based params (strangle / iron_condor) ---
            st.markdown("**▸ OTM% 选行权价** *(strangle / iron_condor)*")
            ed_otm_pct = st.number_input(
                "短腿 OTM %", min_value=1.0, max_value=50.0,
                value=_strategy.get("otm_pct", 0.10) * 100,
                step=1.0, format="%.1f", help="短腿离现货的距离百分比 (strangle/iron_condor)",
            )
            ed_wing_width = st.number_input(
                "翼宽 % (仅铁鹰)", min_value=0.0, max_value=20.0,
                value=_strategy.get("wing_width_pct", 0.02) * 100,
                step=0.5, format="%.1f", help="保护翼额外偏移百分比 (strangle模式忽略)",
            )
            ed_target_dte = st.number_input(
                "目标 DTE (天)", min_value=0, max_value=30,
                value=_strategy.get("target_dte_days", 7),
                step=1, help="目标到期天数 (0=0DTE当日到期)",
            )
            ed_dte_window = st.number_input(
                "DTE 容差 (小时)", min_value=6, max_value=168,
                value=_strategy.get("dte_window_hours", 48),
                step=6, help="目标DTE ± 容差范围内的合约都可选",
            )

            # --- Delta-based params (weekend_vol) ---
            st.markdown("**▸ Delta 选行权价** *(weekend_vol)*")
            ed_target_delta = st.number_input(
                "短腿目标 |Δ|", min_value=0.05, max_value=0.90,
                value=float(_strategy.get("target_delta", 0.40)),
                step=0.05, format="%.2f", help="Short legs 的 |delta| 目标 (weekend_vol)",
            )
            ed_wing_delta = st.number_input(
                "翼 |Δ| (保护腿)", min_value=0.0, max_value=0.50,
                value=float(_strategy.get("wing_delta", 0.05)),
                step=0.01, format="%.2f", help="Long legs 的 |delta| 目标 (0=无翼/strangle)",
            )
            ed_leverage = st.number_input(
                "杠杆倍数", min_value=0.5, max_value=10.0,
                value=float(_strategy.get("leverage", 1.0)),
                step=0.5, format="%.1f", help="仓位大小的杠杆倍数 (weekend_vol)",
            )
            ed_rv_hours = st.number_input(
                "RV 回看小时数", min_value=0, max_value=168,
                value=int(_strategy.get("entry_realized_vol_lookback_hours", 24)),
                step=1, help="入场 RV 过滤的历史小时数；0 表示关闭过滤",
            )
            ed_rv_max = st.number_input(
                "RV 上限", min_value=0.0, max_value=5.0,
                value=float(_strategy.get("entry_realized_vol_max", 1.20)),
                step=0.05, format="%.2f", help="仅当年化 RV 不超过该值时允许开仓",
            )
            ed_stop_loss_pct = st.number_input(
                "组合止损 %", min_value=0.0, max_value=1000.0,
                value=float(_strategy.get("stop_loss_pct", 200.0)),
                step=10.0, format="%.1f", help="组合 PnL% 低于 -该值时触发止损；0 表示关闭",
            )
            _day_options = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            _cur_day = _strategy.get("entry_day", "friday").lower()
            _day_index = _day_options.index(_cur_day) if _cur_day in _day_options else 4
            ed_entry_day = st.selectbox(
                "开仓日", _day_options, index=_day_index,
                help="每周开仓的日子 (weekend_vol 通常选 friday)",
            )
            ed_default_iv = st.number_input(
                "默认 IV", min_value=0.1, max_value=3.0,
                value=float(_strategy.get("default_iv", 0.60)),
                step=0.05, format="%.2f", help="mark_iv 不可用时的回退 IV (weekend_vol)",
            )

            import datetime as _dt
            _entry_str = _strategy.get("entry_time_utc", "08:00")
            _entry_parts = _entry_str.split(":")
            _entry_time = _dt.time(int(_entry_parts[0]), int(_entry_parts[1]))
            ed_entry_time = st.time_input(
                "开仓时间 (UTC)",
                value=_entry_time,
                step=_dt.timedelta(minutes=1),
                help="每日开仓的 UTC 时间（精确到分钟）",
            )

        with s_col2:
            ed_quantity = st.number_input(
                "基础数量", min_value=0.001, max_value=100.0,
                value=float(_strategy.get("quantity", 0.01)),
                step=0.01, format="%.3f", help="每组策略的合约数量",
            )
            ed_max_pos = st.number_input(
                "最大持仓组数", min_value=1, max_value=20,
                value=_strategy.get("max_positions", 1),
                step=1,
            )

        with s_col3:
            ed_max_cap = st.number_input(
                "最大资金比例 %", min_value=5.0, max_value=100.0,
                value=_strategy.get("max_capital_pct", 0.30) * 100,
                step=5.0, format="%.0f",
            )
            ed_compound = st.checkbox(
                "复利模式", value=_strategy.get("compound", True),
                help="根据账户权益自动调整下单数量",
            )
            ed_wait_midpoint = st.checkbox(
                "等待中点开仓", value=_strategy.get("wait_for_midpoint", False),
                help="等现货价格到达两个最近行权价的中点再开仓",
            )

        st.divider()
        st.subheader("🔌 API & 运行参数")
        r_col1, r_col2 = st.columns(2)

        with r_col1:
            ed_testnet = st.checkbox(
                "测试网模式", value=_exchange.get("testnet", True),
                help="开启后使用模拟账户，不真实下单",
            )
            ed_timeout = st.number_input(
                "API 超时 (秒)", min_value=3, max_value=60,
                value=_exchange.get("timeout", 10), step=1,
            )
            ed_check_interval = st.number_input(
                "策略循环间隔 (秒)", min_value=10, max_value=600,
                value=_monitor.get("check_interval_sec", 60), step=10,
            )
            ed_heartbeat = st.number_input(
                "心跳间隔 (秒)", min_value=60, max_value=3600,
                value=_monitor.get("heartbeat_interval_sec", 300), step=60,
            )

        with r_col2:
            ed_snapshot = st.number_input(
                "资产快照间隔 (秒)", min_value=300, max_value=86400,
                value=_monitor.get("equity_snapshot_interval_sec", 3600), step=300,
            )
            ed_db_path = st.text_input(
                "数据库路径", value=_storage.get("db_path", "./data/trader.db"),
            )
            ed_log_dir = st.text_input(
                "日志目录", value=_storage.get("log_dir", "./logs"),
            )
            ed_log_level = st.selectbox(
                "日志级别", ["DEBUG", "INFO", "WARNING", "ERROR"],
                index=["DEBUG", "INFO", "WARNING", "ERROR"].index(
                    _storage.get("log_level", "INFO")
                ),
            )

        st.divider()
        _save_col1, _save_col2 = st.columns([1, 4])
        with _save_col1:
            _submitted = st.form_submit_button("💾 保存配置", width='stretch', type="primary")
        with _save_col2:
            st.caption("保存后需要重启引擎才能生效")

    # ---- Handle save ----
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
            },
            "strategy": {
                "mode": ed_mode,
                "underlying": ed_underlying,
                "otm_pct": round(ed_otm_pct / 100, 4),
                "wing_width_pct": round(ed_wing_width / 100, 4),
                "target_dte_days": ed_target_dte,
                "dte_window_hours": ed_dte_window,
                "target_delta": round(ed_target_delta, 4),
                "wing_delta": round(ed_wing_delta, 4),
                "leverage": round(ed_leverage, 2),
                "entry_realized_vol_lookback_hours": int(ed_rv_hours),
                "entry_realized_vol_max": round(ed_rv_max, 4),
                "stop_loss_pct": round(ed_stop_loss_pct, 4),
                "entry_day": ed_entry_day,
                "default_iv": round(ed_default_iv, 4),
                "entry_time_utc": ed_entry_time.strftime("%H:%M"),
                "quantity": ed_quantity,
                "max_positions": ed_max_pos,
                "max_capital_pct": round(ed_max_cap / 100, 4),
                "compound": ed_compound,
                "wait_for_midpoint": ed_wait_midpoint,
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
            # Write with comments header
            _header = (
                "# ============================================================\n"
                f"# {_new_cfg['name']}\n"
                "# ============================================================\n"
                "# 由交易面板自动保存\n"
                "# ============================================================\n\n"
            )
            with open(config_path, "w", encoding="utf-8") as _wf:
                _wf.write(_header)
                _yaml.dump(
                    _new_cfg, _wf,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )
            st.success(f"✅ 配置已保存到 {config_path}")
            st.caption("出于安全考虑，API Key / Secret 不会由面板写回 YAML，请继续使用环境变量或 .env。")
            st.info("⚠️ 请重启引擎使新配置生效 (先停止再启动)")
        except Exception as _ex:
            st.error(f"保存失败: {_ex}")

    st.divider()

    # ------------------------------------------------------------------
    # 📋  下单预览 — 根据实时行情模拟下一次开仓
    # ------------------------------------------------------------------
    _pv_mode = getattr(cfg.strategy, "mode", "iron_condor")
    _pv_weekend_is_ic = getattr(cfg.strategy, "wing_delta", 0) > 0
    if _pv_mode == "weekend_vol":
        _pv_mode_label = "Weekend Vol Iron Condor" if _pv_weekend_is_ic else "Weekend Vol Short Strangle"
    else:
        _pv_mode_labels = {
            "strangle": "Short Strangle",
            "iron_condor": "Iron Condor",
        }
        _pv_mode_label = _pv_mode_labels.get(_pv_mode, _pv_mode)
    st.subheader(f"📋 下单预览 ({_pv_mode_label})（实时行情）")

    @st.cache_resource
    def _get_client_preview(_exchange_cfg) -> BinanceOptionsClient:
        return BinanceOptionsClient(_exchange_cfg)

    _pv_client = _get_client_preview(cfg.exchange)

    _pv_ok = False
    try:
        _pv_ul = cfg.strategy.underlying.upper()
        _pv_spot = _pv_client.get_spot_price(_pv_ul)
        _pv_tickers = _pv_client.get_tickers(_pv_ul)
        _now_utc = datetime.now(timezone.utc)
        _sunday_exp = _now_utc

        if _pv_spot <= 0:
            _prices = [t.underlying_price for t in _pv_tickers if t.underlying_price > 0]
            _pv_spot = _prices[0] if _prices else 0

        # --- Filter tickers based on mode ---
        if _pv_mode == "weekend_vol":
            # Weekend vol: filter by nearest Sunday 08:00 UTC expiry
            from datetime import timedelta as _td
            _now_utc = datetime.now(timezone.utc)
            _days_to_sunday = (6 - _now_utc.weekday()) % 7
            if _days_to_sunday == 0 and _now_utc.hour >= 8:
                _days_to_sunday = 7
            _sunday_exp = _now_utc.replace(hour=8, minute=0, second=0, microsecond=0) + _td(days=_days_to_sunday)
            _tolerance_h = 2.0
            _pv_today = [
                t for t in _pv_tickers
                if abs((t.expiry - _sunday_exp).total_seconds()) < _tolerance_h * 3600
            ]
            _target_dte_days = round((_sunday_exp - _now_utc).total_seconds() / 86400, 1)
            _dte_label_str = f"周日到期 ({_sunday_exp.strftime('%Y-%m-%d %H:%M')} UTC)"
        else:
            # Strangle / iron_condor: filter by DTE window
            _target_dte_days = getattr(cfg.strategy, "target_dte_days", 0)
            _dte_window = getattr(cfg.strategy, "dte_window_hours", 48)
            _target_dte_h = _target_dte_days * 24
            _dte_min = max(0.1, _target_dte_h - _dte_window)
            _dte_max = _target_dte_h + _dte_window
            _pv_today = [t for t in _pv_tickers if _dte_min < t.dte_hours < _dte_max]
            _dte_label_str = f"DTE≈{_target_dte_days}天"

        if _pv_spot > 0 and _pv_today:
            # --- Strike selection ---
            _sell_call = None
            _sell_put = None
            _buy_call = None
            _buy_put = None

            def _best_by_strike(tks, otype, target):
                """OTM%-based: find closest strike to target price."""
                cs = [t for t in tks if t.option_type == otype]
                if not cs:
                    return None
                cs.sort(key=lambda _t: abs(_t.strike - target))
                return cs[0]

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

            if _pv_mode == "weekend_vol":
                # Enrich tickers with greeks from exchange
                try:
                    _pv_client.enrich_greeks(_pv_today, _pv_ul)
                except Exception:
                    pass

                _T_years = max((_sunday_exp - _now_utc).total_seconds() / (365.25 * 86400), 1e-6)
                _tgt_delta = getattr(cfg.strategy, "target_delta", 0.40)
                _wing_d = getattr(cfg.strategy, "wing_delta", 0.05)
                _def_iv = getattr(cfg.strategy, "default_iv", 0.60)

                _sell_call = _best_by_delta(_pv_today, "call", _tgt_delta, _pv_spot, _T_years, _def_iv)
                _sell_put = _best_by_delta(_pv_today, "put", _tgt_delta, _pv_spot, _T_years, _def_iv)

                if _wing_d > 0:
                    _buy_call = _best_by_delta(_pv_today, "call", _wing_d, _pv_spot, _T_years, _def_iv)
                    _buy_put = _best_by_delta(_pv_today, "put", _wing_d, _pv_spot, _T_years, _def_iv)
            else:
                _otm = cfg.strategy.otm_pct
                _wing = cfg.strategy.wing_width_pct
                _sc_target = _pv_spot * (1 + _otm)
                _sp_target = _pv_spot * (1 - _otm)

                _sell_call = _best_by_strike(_pv_today, "call", _sc_target)
                _sell_put = _best_by_strike(_pv_today, "put", _sp_target)

                if _pv_mode == "iron_condor":
                    _lc_target = _pv_spot * (1 + _otm + _wing)
                    _lp_target = _pv_spot * (1 - _otm - _wing)
                    _buy_call = _best_by_strike(_pv_today, "call", _lc_target)
                    _buy_put = _best_by_strike(_pv_today, "put", _lp_target)

            # Determine if we're doing an IC (4-leg) or strangle (2-leg)
            _is_ic = _pv_mode == "iron_condor" or (_pv_mode == "weekend_vol" and getattr(cfg.strategy, "wing_delta", 0) > 0)

            _has_all_legs = (
                all([_sell_call, _buy_call, _sell_put, _buy_put])
                if _is_ic
                else all([_sell_call, _sell_put])
            )

            if _has_all_legs:
                if _sell_call is None or _sell_put is None:
                    raise RuntimeError("Preview leg selection returned incomplete short legs")
                if _is_ic and (_buy_call is None or _buy_put is None):
                    raise RuntimeError("Preview leg selection returned incomplete wing legs")

                _sell_call_t: Any = _sell_call
                _sell_put_t: Any = _sell_put
                _buy_call_t: Any = _buy_call
                _buy_put_t: Any = _buy_put

                # Compute quantity
                import math as _math
                _base_qty = cfg.strategy.quantity
                _pv_qty = _base_qty
                _pv_equity = 0.0
                _pv_equity_src = "配置基础数量"
                if cfg.strategy.compound:
                    try:
                        _pv_acct = _pv_client.get_account()
                        _pv_equity = _pv_acct.total_balance
                        if _pv_equity > 0 and _pv_spot > 0:
                            if _pv_mode == "weekend_vol":
                                # Weekend vol: leverage-based sizing
                                _lev_cfg = getattr(cfg.strategy, "leverage", 1.0)
                                _pv_qty_raw = (_pv_equity * _lev_cfg) / _pv_spot
                                _pv_qty = max(_base_qty, _math.floor(_pv_qty_raw * 10) / 10)
                                _pv_equity_src = f"实时权益(复利 {_lev_cfg:.0f}x)"
                            else:
                                _max_notional = _pv_equity * cfg.strategy.max_capital_pct
                                if _pv_mode == "strangle":
                                    _margin_per = _otm * _pv_spot * 2
                                else:
                                    _ww = _wing * _pv_spot
                                    _margin_per = _ww * 2 if _ww > 0 else _otm * _pv_spot * 2
                                if _margin_per > 0:
                                    _scaled = _math.floor(_max_notional / _margin_per * 100) / 100
                                    _pv_qty = max(_base_qty, _scaled)
                                _pv_equity_src = "实时权益(复利)"
                    except Exception:
                        _pv_equity_src = "权益获取失败, 用基础数量"

                # Prices in USD (Binance option native quote unit)
                _preview_fallback_iv = getattr(cfg.strategy, "default_iv", 0.60)
                _total_max_loss = 0.0
                _sc_bid = _sell_call_t.bid_price
                _sc_ask = _sell_call_t.ask_price
                _sp_bid = _sell_put_t.bid_price
                _sp_ask = _sell_put_t.ask_price

                if not _is_ic:
                    # ---- Strangle / weekend_vol without wings P&L ----
                    _premium_per = _sc_bid + _sp_bid
                    _total_premium = _premium_per * _pv_qty
                    if _pv_mode == "weekend_vol":
                        _margin_used = _pv_spot * _pv_qty  # notional
                    else:
                        _margin_used = _otm * _pv_spot * 2 * _pv_qty
                    _be_upper = _sell_call_t.strike + _premium_per
                    _be_lower = _sell_put_t.strike - _premium_per
                else:
                    # ---- Iron Condor P&L ----
                    _lc_bid = _buy_call_t.bid_price
                    _lc_ask = _buy_call_t.ask_price
                    _lp_bid = _buy_put_t.bid_price
                    _lp_ask = _buy_put_t.ask_price
                    _premium_per = (_sc_bid + _sp_bid) - (_lc_ask + _lp_ask)
                    _call_width = _buy_call_t.strike - _sell_call_t.strike
                    _put_width = _sell_put_t.strike - _buy_put_t.strike
                    _max_wing = max(_call_width, _put_width)
                    _max_loss_per = _max_wing - _premium_per
                    _total_premium = _premium_per * _pv_qty
                    _total_max_loss = _max_loss_per * _pv_qty
                    _margin_used = _max_wing * 2 * _pv_qty
                    _be_upper = _sell_call_t.strike + _premium_per
                    _be_lower = _sell_put_t.strike - _premium_per

                _pv_ok = True

                # ---- Display ----
                st.caption(f"数据来源: Binance 实时行情 | {_pv_equity_src} | {_dte_label_str}")

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
                        {
                            "腿": "① Short Put (卖出收权利金)",
                            "合约": _sell_put_t.symbol,
                            "行权价": f"${_sell_put_t.strike:,.0f}",
                            "距Spot": f"{(_sell_put_t.strike/_pv_spot - 1)*100:+.1f}%",
                            "Delta": _format_preview_delta(_sell_put_t, _pv_spot, _preview_fallback_iv),
                            "方向": "🔴 卖出",
                            "Bid (USDT)": f"${_sp_bid:.2f}",
                            "Ask (USDT)": f"${_sp_ask:.2f}",
                            "DTE": f"{_sell_put_t.dte_hours:.0f}h ({_sell_put_t.dte_hours/24:.1f}d)",
                        },
                        {
                            "腿": "② Short Call (卖出收权利金)",
                            "合约": _sell_call_t.symbol,
                            "行权价": f"${_sell_call_t.strike:,.0f}",
                            "距Spot": f"{(_sell_call_t.strike/_pv_spot - 1)*100:+.1f}%",
                            "Delta": _format_preview_delta(_sell_call_t, _pv_spot, _preview_fallback_iv),
                            "方向": "🔴 卖出",
                            "Bid (USDT)": f"${_sc_bid:.2f}",
                            "Ask (USDT)": f"${_sc_ask:.2f}",
                            "DTE": f"{_sell_call_t.dte_hours:.0f}h ({_sell_call_t.dte_hours/24:.1f}d)",
                        },
                    ]
                else:
                    st.markdown("##### 🦵 四条腿明细")
                    _legs_data = [
                        {
                            "腿": "① Long Put (买入保护)",
                            "合约": _buy_put_t.symbol,
                            "行权价": f"${_buy_put_t.strike:,.0f}",
                            "距Spot": f"{(_buy_put_t.strike/_pv_spot - 1)*100:+.1f}%",
                            "Delta": _format_preview_delta(_buy_put_t, _pv_spot, _preview_fallback_iv),
                            "方向": "🟢 买入",
                            "Bid (USDT)": f"${_lp_bid:.2f}",
                            "Ask (USDT)": f"${_lp_ask:.2f}",
                            "DTE": f"{_buy_put_t.dte_hours:.0f}h ({_buy_put_t.dte_hours/24:.1f}d)",
                        },
                        {
                            "腿": "② Short Put (卖出收权利金)",
                            "合约": _sell_put_t.symbol,
                            "行权价": f"${_sell_put_t.strike:,.0f}",
                            "距Spot": f"{(_sell_put_t.strike/_pv_spot - 1)*100:+.1f}%",
                            "Delta": _format_preview_delta(_sell_put_t, _pv_spot, _preview_fallback_iv),
                            "方向": "🔴 卖出",
                            "Bid (USDT)": f"${_sp_bid:.2f}",
                            "Ask (USDT)": f"${_sp_ask:.2f}",
                            "DTE": f"{_sell_put_t.dte_hours:.0f}h ({_sell_put_t.dte_hours/24:.1f}d)",
                        },
                        {
                            "腿": "③ Short Call (卖出收权利金)",
                            "合约": _sell_call_t.symbol,
                            "行权价": f"${_sell_call_t.strike:,.0f}",
                            "距Spot": f"{(_sell_call_t.strike/_pv_spot - 1)*100:+.1f}%",
                            "Delta": _format_preview_delta(_sell_call_t, _pv_spot, _preview_fallback_iv),
                            "方向": "🔴 卖出",
                            "Bid (USDT)": f"${_sc_bid:.2f}",
                            "Ask (USDT)": f"${_sc_ask:.2f}",
                            "DTE": f"{_sell_call_t.dte_hours:.0f}h ({_sell_call_t.dte_hours/24:.1f}d)",
                        },
                        {
                            "腿": "④ Long Call (买入保护)",
                            "合约": _buy_call_t.symbol,
                            "行权价": f"${_buy_call_t.strike:,.0f}",
                            "距Spot": f"{(_buy_call_t.strike/_pv_spot - 1)*100:+.1f}%",
                            "Delta": _format_preview_delta(_buy_call_t, _pv_spot, _preview_fallback_iv),
                            "方向": "🟢 买入",
                            "Bid (USDT)": f"${_lc_bid:.2f}",
                            "Ask (USDT)": f"${_lc_ask:.2f}",
                            "DTE": f"{_buy_call_t.dte_hours:.0f}h ({_buy_call_t.dte_hours/24:.1f}d)",
                        },
                    ]
                st.dataframe(pd.DataFrame(_legs_data), width="stretch", hide_index=True)

                # Manual test order
                st.markdown("##### 🧪 测试下单")
                _test_qty = 0.01
                _active_test_task_id = st.session_state.get("preview_test_order_task_id")
                _active_test_task = _get_test_order_task(_active_test_task_id)
                _test_task_running = bool(_active_test_task and _active_test_task.get("state") in {"queued", "running"})
                _test_disabled_reason = None
                if not is_trade_mode:
                    _test_disabled_reason = "当前为只读模式，切换到交易模式后才可发送测试单。"
                elif _test_task_running:
                    _test_disabled_reason = "已有测试下单任务在执行，请等待完成。"

                _test_leg_count = 4 if _is_ic else 2
                _test_button_label = f"🧪 测试{_test_leg_count}腿市价单 0.01"
                st.caption(
                    f"按当前预览目标固定发送 {_test_leg_count} 条腿市价单，每条腿数量 0.01，用于验证真实下单链路。"
                )
                _tc1, _tc2 = st.columns([1, 2])
                with _tc1:
                    _preview_test_clicked = st.button(
                        _test_button_label,
                        key=(
                            f"preview_test_combo_{_pv_mode}_"
                            f"{_sell_put.symbol}_{_sell_call.symbol}_"
                            f"{_buy_put.symbol if _buy_put else 'none'}_"
                            f"{_buy_call.symbol if _buy_call else 'none'}"
                        ),
                        width='stretch',
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
                        },
                    )
                    _start_test_order_task(
                        _new_task_id,
                        cfg.exchange,
                        storage,
                        cfg.chaser,
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
                    )
                    st.rerun()

                _active_test_task_id = st.session_state.get("preview_test_order_task_id")
                _active_test_task = _get_test_order_task(_active_test_task_id)
                if _active_test_task:
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

                        st.progress(max(0, min(_percent, 100)), text=f"测试下单进度：{_percent}% | {_message}")
                        if _state in {"queued", "running"}:
                            st.info(_message)
                        elif _state == "success":
                            st.success(_message)
                        elif _state in {"failed", "error"}:
                            st.error(_message)

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

                        if _task.get("error_id"):
                            st.caption(f"错误编号: {_task['error_id']}（详细堆栈仅写入服务端日志）")

                        if _state in {"failed", "error"}:
                            _snapshot_checked = bool(_task.get("exchange_positions_checked"))
                            _snapshot_underlying = str(_task.get("exchange_positions_underlying") or "")
                            _snapshot_error = _task.get("exchange_positions_error")
                            _snapshot_positions = _task.get("exchange_positions") or []

                            st.markdown("##### 🔎 失败后交易所持仓复核")
                            if _snapshot_checked:
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

                    if _active_test_task_id is not None:
                        _render_test_order_progress(str(_active_test_task_id))

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

                _max_loss_line: float | None = None

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
                else:
                    _price_lo = _buy_put_t.strike * 0.92
                    _price_hi = _buy_call_t.strike * 1.08
                    _prices = _np.linspace(_price_lo, _price_hi, 500)

                    def _payoff(S):
                        sc = -_np.maximum(S - _sell_call_t.strike, 0)
                        lc = _np.maximum(S - _buy_call_t.strike, 0)
                        sp = -_np.maximum(_sell_put_t.strike - S, 0)
                        lp = _np.maximum(_buy_put_t.strike - S, 0)
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
                        _mode_detail = f"模式: {_pv_mode_label}"
                        if _pv_mode == "weekend_vol":
                            _mode_detail += f"  |Δ|={getattr(cfg.strategy, 'target_delta', 0.40):.0%}"
                        else:
                            _mode_detail += f"  OTM: {_otm*100:.1f}%  DTE: {_target_dte_days}天"
                        st.code(
                            f"标的: {_pv_ul}  现货: ${_pv_spot:,.2f}\n"
                            f"{_mode_detail}\n"
                            f"───────────────────────────────\n"
                            f"Short Put:  {_sell_put_t.symbol}  K=${_sell_put_t.strike:,.0f}\n"
                            f"Short Call: {_sell_call_t.symbol}  K=${_sell_call_t.strike:,.0f}\n"
                            f"───────────────────────────────\n"
                            f"数量: {_pv_qty:.2f} 张  (base={_base_qty}, compound={cfg.strategy.compound})\n"
                            f"权益: ${_pv_equity:,.2f}  max_capital: {cfg.strategy.max_capital_pct*100:.0f}%\n"
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
                        _mode_detail = f"模式: {_pv_mode_label}"
                        if _pv_mode == "weekend_vol":
                            _mode_detail += f"  |Δ|={getattr(cfg.strategy, 'target_delta', 0.40):.0%}  翼Δ={getattr(cfg.strategy, 'wing_delta', 0.05):.0%}"
                        else:
                            _mode_detail += f"  OTM: {_otm*100:.1f}%  翼宽: {_wing*100:.1f}%"
                        _margin_pct_str = f" ({_margin_used/_pv_equity*100:.1f}% 权益)" if _pv_equity > 0 else ""
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
                            f"数量: {_pv_qty:.2f} 张  (base={_base_qty}, compound={cfg.strategy.compound})\n"
                            f"权益: ${_pv_equity:,.2f}  max_capital: {cfg.strategy.max_capital_pct*100:.0f}%\n"
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
                st.warning(f"⚠️ 无法找到完整的合约腿 ({_dte_label_str}，共 {len(_pv_today)} 个 ticker)")
        else:
            if _pv_spot <= 0:
                st.warning("⚠️ 无法获取现货价格")
            else:
                st.warning(f"⚠️ 目标合约不可用 ({_dte_label_str})")
    except Exception as _pv_ex:
        _error_id = _log_exception_with_id("Order preview failed", _pv_ex, prefix="preview")
        st.error(f"下单预览失败，请稍后重试。错误编号: {_error_id}")

    if not _pv_ok:
        st.info(f"💡 下单预览需要实时行情数据，请确保网络连通且有可匹配合约")

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
    if getattr(cfg.strategy, "mode", "") == "weekend_vol":
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
            st.metric(
                "止损线",
                f"-{sl_pct:.0f}%" if isinstance(sl_pct, (int, float)) and sl_pct > 0 else "关闭",
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
            if st.button("⏹ 停止引擎", width='stretch', type="secondary",
                          disabled=not is_trade_mode):
                engine.stop()
                st.rerun()
        else:
            if st.button("🚀 启动引擎", width='stretch', type="primary",
                          disabled=not is_trade_mode):
                ok = engine.start()
                if not ok:
                    st.error(f"启动失败: {engine.status().get('last_error')}")
                st.rerun()

    with col_a2:
        if st.button("🚨 紧急全部平仓", width='stretch', type="primary",
                      disabled=(not es["running"] or not is_trade_mode)):
            pnl = engine.close_all_positions()
            st.success(f"已平仓, PnL: ${pnl:,.4f}")
            st.rerun()

    with col_a3:
        if st.button("🔄 重置引擎", width='stretch',
                      disabled=(es["running"] or not is_trade_mode),
                      help="停止并释放引擎实例，下次启动将重新初始化"):
            reset_engine()
            st.rerun()

    st.divider()

    # --- Config summary ---
    st.subheader("📋 运行配置摘要")
    _cs_mode = getattr(cfg.strategy, "mode", "iron_condor")
    if _cs_mode == "weekend_vol":
        run_info = {
            "配置项": [
                "策略名", "模式", "标的", "短腿 |Δ|", "翼 |Δ|",
                "杠杆", "开仓日/时间", "RV过滤", "止损", "基础数量", "复利", "API环境",
            ],
            "值": [
                cfg.name,
                "Weekend Vol (delta选行权价)",
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
                    f"-{getattr(cfg.strategy, 'stop_loss_pct', 0.0):.0f}%"
                    if getattr(cfg.strategy, 'stop_loss_pct', 0.0) > 0
                    else "关闭"
                ),
                str(cfg.strategy.quantity),
                "✅" if cfg.strategy.compound else "❌",
                "测试网" if cfg.exchange.testnet else "正式环境",
            ],
        }
    else:
        run_info = {
            "配置项": [
                "策略名", "模式", "标的", "OTM%", "翼宽%", "开仓时间",
                "基础数量", "复利", "API环境", "检查间隔",
            ],
            "值": [
                cfg.name,
                "裸卖双卖" if _cs_mode == "strangle" else "铁鹰",
                cfg.strategy.underlying,
                f"±{cfg.strategy.otm_pct*100:.0f}%",
                f"{cfg.strategy.wing_width_pct*100:.0f}%",
                f"{cfg.strategy.entry_time_utc} UTC",
                str(cfg.strategy.quantity),
                "✅" if cfg.strategy.compound else "❌",
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
# PAGE: 期权行情 (Option Market Quotes)
# ==========================================================================

elif page == "📡 期权行情":
    st.title(f"📡 {cfg.strategy.underlying} 期权实时行情")

    # --- Initialize client (cached) ---
    @st.cache_resource
    def _get_client(_exchange_cfg) -> BinanceOptionsClient:
        return BinanceOptionsClient(_exchange_cfg)

    client = _get_client(cfg.exchange)
    ul = cfg.strategy.underlying.upper()

    # --- Spot price ---
    try:
        spot = client.get_spot_price(ul)
    except Exception:
        spot = 0.0

    # --- Fetch tickers ---
    try:
        tickers = client.get_tickers(ul)
        _market_source = "client"
    except Exception as e:
        tickers = []
        _market_source = "error"
        st.error(f"获取行情失败: {e}")

    # --- Header row: spot price + pricing unit toggle ---
    _hdr_col1, _hdr_col2 = st.columns([3, 1])
    with _hdr_col1:
        if spot > 0:
            st.metric(f"{ul}/USD 指数价格", f"${spot:,.2f}")
    with _hdr_col2:
        price_unit = st.radio(
            "计价单位",
            ["USD (原始)", f"{ul} (折算)"],
            key="mkt_price_unit",
            horizontal=True,
            help=f"Binance 期权原始报价为 USD。选择 {ul} (折算) 会按当前标的价格换算。",
        )
    _is_usd = price_unit.startswith("USD")

    rows = []
    if tickers:
        for t in tickers:
            dte_h = t.dte_hours
            moneyness = t.moneyness_pct
            spread_pct = (t.spread / t.mid_price * 100) if t.mid_price > 0 else 0
            ul_px = t.underlying_price if t.underlying_price > 0 else spot
            rows.append({
                "合约": t.symbol,
                "类型": "CALL" if t.option_type == "call" else "PUT",
                "行权价": t.strike,
                "到期": t.expiry.strftime("%m-%d %H:%M"),
                "DTE(h)": round(dte_h, 1),
                "Bid": t.bid_price,
                "Ask": t.ask_price,
                "Mid": round(t.mid_price, 6),
                "Mark": t.mark_price,
                "Last": t.last_price,
                "BidCoin": round(t.bid_price / ul_px, 6) if ul_px > 0 else 0.0,
                "AskCoin": round(t.ask_price / ul_px, 6) if ul_px > 0 else 0.0,
                "MidCoin": round(t.mid_price / ul_px, 6) if ul_px > 0 else 0.0,
                "MarkCoin": round(t.mark_price / ul_px, 6) if ul_px > 0 else 0.0,
                "LastCoin": round(t.last_price / ul_px, 6) if ul_px > 0 else 0.0,
                "价差%": round(spread_pct, 2),
                "OTM%": round(moneyness, 2),
                "24h量": t.volume_24h,
                "OI": t.open_interest,
                "标的价": ul_px,
            })
    else:
        # Dashboard-level fallback: production public feed (read-only)
        try:
            resp = requests.get("https://eapi.binance.com/eapi/v1/ticker", timeout=10)
            raw = resp.json()
            if isinstance(raw, list):
                _market_source = "production-fallback"
                now_utc = datetime.now(timezone.utc)
                for item in raw:
                    symbol = str(item.get("symbol", ""))
                    if not symbol.startswith(f"{ul}-"):
                        continue
                    # Fallback parse inline (ETH-260327-2400-C)
                    try:
                        ul0, yymmdd, strike_s, cp = symbol.split("-")
                        yy, mm, dd = int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
                        expiry = datetime(2000 + yy, mm, dd, 8, 0, tzinfo=timezone.utc)
                        strike = float(strike_s)
                        opt_type = "CALL" if cp == "C" else "PUT"
                    except Exception:
                        continue

                    ul_px = float(item.get("underlyingPrice") or item.get("exercisePrice") or spot or 0)
                    bid_u = float(item.get("bidPrice") or 0)
                    ask_u = float(item.get("askPrice") or 0)
                    last_u = float(item.get("lastPrice") or 0)
                    mark_u = (bid_u + ask_u) / 2 if bid_u > 0 and ask_u > 0 else last_u

                    bid = bid_u
                    ask = ask_u
                    mid = (bid + ask) / 2 if bid > 0 and ask > 0 else mark_u
                    mark = mark_u
                    last = last_u
                    bid_coin = (bid_u / ul_px) if ul_px > 0 else 0.0
                    ask_coin = (ask_u / ul_px) if ul_px > 0 else 0.0
                    mid_coin = (mid / ul_px) if ul_px > 0 else 0.0
                    mark_coin = (mark_u / ul_px) if ul_px > 0 else 0.0
                    last_coin = (last_u / ul_px) if ul_px > 0 else 0.0

                    spread_pct = ((ask - bid) / mid * 100) if mid > 0 else 0
                    dte_h = max((expiry - now_utc).total_seconds() / 3600.0, 0.0)
                    moneyness = ((strike / ul_px - 1.0) * 100) if ul_px > 0 else 0.0

                    rows.append({
                        "合约": symbol,
                        "类型": opt_type,
                        "行权价": strike,
                        "到期": expiry.strftime("%m-%d %H:%M"),
                        "DTE(h)": round(dte_h, 1),
                        "Bid": bid,
                        "Ask": ask,
                        "Mid": round(mid, 6),
                        "Mark": mark,
                        "Last": last,
                        "BidCoin": round(bid_coin, 6),
                        "AskCoin": round(ask_coin, 6),
                        "MidCoin": round(mid_coin, 6),
                        "MarkCoin": round(mark_coin, 6),
                        "LastCoin": round(last_coin, 6),
                        "价差%": round(spread_pct, 2),
                        "OTM%": round(moneyness, 2),
                        "24h量": float(item.get("volume") or 0),
                        "OI": float(item.get("amount") or 0),
                        "标的价": ul_px,
                    })
        except Exception:
            pass

    st.caption(
        f"共 {len(rows)} 个合约  |  数据源: {_market_source}  |  更新时间: "
        f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC"
    )

    if not rows:
        st.warning(
            "未获取到期权行情。若当前为 testnet，系统会尝试使用生产公共行情；"
            "请检查网络连通性与交易所接口可用性。"
        )

    if rows:
        df_all = pd.DataFrame(rows)

        st.divider()

        # --- Filters ---
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            type_filter = st.selectbox("类型", ["全部", "CALL", "PUT"], key="mkt_type")
        with col_f2:
            expiries = sorted(df_all["到期"].unique())
            # Auto-select nearest 08:00 UTC expiry (0DTE)
            _default_idx = 0
            if expiries:
                _08_expiries = [e for e in expiries if "08:00" in e]
                if _08_expiries:
                    _now_str = datetime.now(timezone.utc).strftime("%m-%d %H:%M")
                    # pick the first 08:00 expiry >= now, else the last one
                    _future_08 = [e for e in _08_expiries if e >= _now_str]
                    _pick = _future_08[0] if _future_08 else _08_expiries[-1]
                    _default_idx = 1 + expiries.index(_pick)  # +1 for "全部"
                else:
                    _default_idx = 1  # first expiry
            exp_filter = st.selectbox("到期日", ["全部"] + expiries, index=_default_idx, key="mkt_exp")
        with col_f3:
            _sort_choices = ["行权价", "DTE(h)", "Mark", "OTM%", "24h量", "OI"]
            sort_col = st.selectbox("排序", _sort_choices, key="mkt_sort")
        with col_f4:
            only_liquid = st.checkbox("仅显示有报价", value=True, key="mkt_liquid")

        df_show = df_all.copy()
        if type_filter != "全部":
            df_show = df_show[df_show["类型"] == type_filter]
        if exp_filter != "全部":
            df_show = df_show[df_show["到期"] == exp_filter]
        if only_liquid:
            df_show = df_show[(df_show["Bid"] > 0) | (df_show["Ask"] > 0)]
        df_show = df_show.sort_values(sort_col, ascending=(sort_col not in ["24h量", "OI", "Mark"]))

        st.caption(f"筛选后 {len(df_show)} / {len(df_all)} 个合约")

        # --- Select display columns based on pricing unit ---
        if _is_usd:
            _price_cols = ["Bid", "Ask", "Mid", "Mark", "Last"]
            _display = df_show[["合约", "类型", "行权价", "到期", "DTE(h)"]
                               + _price_cols
                               + ["价差%", "OTM%", "24h量", "OI", "标的价"]].copy()
            _col_cfg = {
                "行权价": st.column_config.NumberColumn(format="$%.0f"),
                "Bid": st.column_config.NumberColumn("Bid (USD)", format="$%.2f"),
                "Ask": st.column_config.NumberColumn("Ask (USD)", format="$%.2f"),
                "Mid": st.column_config.NumberColumn("Mid (USD)", format="$%.2f"),
                "Mark": st.column_config.NumberColumn("Mark (USD)", format="$%.2f"),
                "Last": st.column_config.NumberColumn("Last (USD)", format="$%.2f"),
                "标的价": st.column_config.NumberColumn(format="$%.2f"),
                "价差%": st.column_config.NumberColumn(format="%.2f%%"),
                "OTM%": st.column_config.NumberColumn(format="%.2f%%"),
            }
        else:
            _price_cols = ["BidCoin", "AskCoin", "MidCoin", "MarkCoin", "LastCoin"]
            _display = df_show[["合约", "类型", "行权价", "到期", "DTE(h)"]
                               + _price_cols
                               + ["价差%", "OTM%", "24h量", "OI", "标的价"]].copy()
            _display = _display.rename(columns={
                "BidCoin": "Bid", "AskCoin": "Ask", "MidCoin": "Mid", "MarkCoin": "Mark", "LastCoin": "Last",
            })
            _col_cfg = {
                "行权价": st.column_config.NumberColumn(format="$%.0f"),
                "Bid": st.column_config.NumberColumn(f"Bid ({ul})", format="%.6f"),
                "Ask": st.column_config.NumberColumn(f"Ask ({ul})", format="%.6f"),
                "Mid": st.column_config.NumberColumn(f"Mid ({ul})", format="%.6f"),
                "Mark": st.column_config.NumberColumn(f"Mark ({ul})", format="%.6f"),
                "Last": st.column_config.NumberColumn(f"Last ({ul})", format="%.6f"),
                "标的价": st.column_config.NumberColumn(format="$%.2f"),
                "价差%": st.column_config.NumberColumn(format="%.2f%%"),
                "OTM%": st.column_config.NumberColumn(format="%.2f%%"),
            }

        st.dataframe(
            _display,
            width="stretch",
            hide_index=True,
            column_config=_col_cfg,
        )

        st.divider()

        # --- T-shaped option chain (for selected expiry) ---
        if exp_filter != "全部":
            st.subheader(f"📋 T 型报价 — {exp_filter}")
            df_exp = df_all[df_all["到期"] == exp_filter].copy()
            calls = df_exp[df_exp["类型"] == "CALL"].set_index("行权价")
            puts = df_exp[df_exp["类型"] == "PUT"].set_index("行权价")
            all_strikes = sorted(set(calls.index) | set(puts.index))

            # Determine ATM strike
            atm_strike = min(all_strikes, key=lambda s: abs(s - spot)) if spot > 0 else 0

            # Filter strikes: show nearby range around ATM
            nearby_n = st.slider("显示行权价数量 (ATM 上下各)", 5, 30, 12, key="chain_range")
            if atm_strike > 0:
                atm_idx = all_strikes.index(atm_strike)
                lo = max(0, atm_idx - nearby_n)
                hi = min(len(all_strikes), atm_idx + nearby_n + 1)
                display_strikes = all_strikes[lo:hi]
            else:
                display_strikes = all_strikes

            # Choose price column keys based on unit
            _bid_key = "Bid" if _is_usd else "BidCoin"
            _ask_key = "Ask" if _is_usd else "AskCoin"
            _mark_key = "Mark" if _is_usd else "MarkCoin"

            # Build T-shaped rows: Call side | Strike | Put side
            chain_rows = []
            for k in display_strikes:
                row: dict = {}
                # --- Call side (left) ---
                if k in calls.index:
                    c = calls.loc[k]
                    row["C_Vol"] = c["24h量"]
                    row["C_OI"] = c["OI"]
                    row["C_Mark"] = c[_mark_key]
                    row["C_Ask"] = c[_ask_key]
                    row["C_Bid"] = c[_bid_key]
                else:
                    row["C_Vol"] = row["C_OI"] = row["C_Mark"] = row["C_Ask"] = row["C_Bid"] = 0
                # --- Strike (center) ---
                row["行权价"] = k
                # --- Put side (right) ---
                if k in puts.index:
                    p = puts.loc[k]
                    row["P_Bid"] = p[_bid_key]
                    row["P_Ask"] = p[_ask_key]
                    row["P_Mark"] = p[_mark_key]
                    row["P_OI"] = p["OI"]
                    row["P_Vol"] = p["24h量"]
                else:
                    row["P_Bid"] = row["P_Ask"] = row["P_Mark"] = row["P_OI"] = row["P_Vol"] = 0
                chain_rows.append(row)

            df_chain = pd.DataFrame(chain_rows, columns=[
                "C_Vol", "C_OI", "C_Mark", "C_Ask", "C_Bid",
                "行权价",
                "P_Bid", "P_Ask", "P_Mark", "P_OI", "P_Vol",
            ])

            if atm_strike > 0:
                _unit_label = "USD" if _is_usd else ul
                st.caption(f"▶ ATM ≈ ${atm_strike:,.0f}  |  现货 ${spot:,.2f}  |  计价: {_unit_label}  |  共 {len(display_strikes)} 档")

            # --- Build styled HTML table for proper T-shape with ATM highlight ---
            # Find the two strikes that bracket the spot price (for separator line)
            _spot_above = None
            _spot_below = None
            for _si, _sk in enumerate(display_strikes):
                if _sk >= spot and _spot_above is None:
                    _spot_above = _sk
                    _spot_below = display_strikes[_si - 1] if _si > 0 else None
                    break

            def _render_t_chain(df: pd.DataFrame, atm_k: float, spot_v: float,
                                spot_above_k: float | None, spot_below_k: float | None,
                                is_usd: bool = False, asset: str = "ETH") -> str:
                """Render T-shaped chain as styled HTML table."""
                _unit_str = "USD" if is_usd else asset
                html_parts = [
                    "<style>",
                    ".tchain { border-collapse:collapse; width:100%; font-size:14px; font-family:'Consolas','Courier New',monospace; }",
                    ".tchain th { background:#1a1a2e; color:#e0e0e0; padding:6px 10px; text-align:center; border-bottom:2px solid #555; font-weight:600; }",
                    ".tchain td { padding:5px 10px; text-align:right; border-bottom:1px solid #333; color:#e8e8e8; }",
                    ".tchain .call-bg { background:#0a2e1a; color:#b0f0d0; }",
                    ".tchain .put-bg  { background:#2e0a0a; color:#f0b0b0; }",
                    ".tchain .strike  { text-align:center; font-weight:bold; background:#1a1a2e; color:#ffffff; min-width:90px; font-size:14px; }",
                    ".tchain .atm     { background:#3d3d00 !important; color:#fff !important; }",
                    ".tchain .atm-strike { text-align:center; background:#4a4a00 !important; color:#ffd700; font-size:15px; font-weight:bold; min-width:90px; }",
                    ".tchain .zero    { color:#666; }",
                    ".tchain tr:hover td { filter:brightness(1.3); }",
                    ".tchain .spot-line td { border-bottom:2px solid #f0c040 !important; }",
                    ".tchain .spot-row td { background:#2a2a10; border-top:2px dashed #f0c040; border-bottom:2px dashed #f0c040; }",
                    ".tchain .spot-row .spot-label { text-align:center; color:#ffd700; font-weight:bold; font-size:13px; letter-spacing:1px; }",
                    "</style>",
                    '<table class="tchain"><thead><tr>',
                    f'<th colspan="5" style="color:#4aeaaa; font-size:15px;">── CALL ({_unit_str}) ──</th>',
                    '<th style="color:#aaa;">行权价</th>',
                    f'<th colspan="5" style="color:#ff8080; font-size:15px;">── PUT ({_unit_str}) ──</th>',
                    "</tr><tr>",
                    "<th>Vol</th><th>OI</th><th>Mark</th><th>Ask</th><th>Bid</th>",
                    '<th style="color:#888;">K</th>',
                    "<th>Bid</th><th>Ask</th><th>Mark</th><th>OI</th><th>Vol</th>",
                    "</tr></thead><tbody>",
                ]

                spot_row_inserted = False
                prev_k = None

                for _, r in df.iterrows():
                    k = r["行权价"]

                    # Insert spot price separator row between the two bracketing strikes
                    if (not spot_row_inserted and spot_v > 0
                            and prev_k is not None and prev_k < spot_v <= k):
                        html_parts.append(
                            '<tr class="spot-row">'
                            '<td colspan="5"></td>'
                            f'<td class="spot-label">▸ ${spot_v:,.2f} ◂</td>'
                            '<td colspan="5"></td>'
                            '</tr>'
                        )
                        spot_row_inserted = True

                    is_atm = abs(k - atm_k) < 0.01
                    atm_cls = " atm" if is_atm else ""
                    strike_cls = "atm-strike" if is_atm else "strike"

                    def _fmt_price(v: float) -> str:
                        if v == 0:
                            return '<span class="zero">-</span>'
                        if is_usd:
                            return f"${v:,.2f}"
                        return f"{v:.6f}"

                    def _fmt_int(v: float) -> str:
                        iv = int(v)
                        if iv == 0:
                            return '<span class="zero">-</span>'
                        return f"{iv:,}"

                    html_parts.append("<tr>")
                    # Call side
                    html_parts.append(f'<td class="call-bg{atm_cls}">{_fmt_int(r["C_Vol"])}</td>')
                    html_parts.append(f'<td class="call-bg{atm_cls}">{_fmt_int(r["C_OI"])}</td>')
                    html_parts.append(f'<td class="call-bg{atm_cls}">{_fmt_price(r["C_Mark"])}</td>')
                    html_parts.append(f'<td class="call-bg{atm_cls}">{_fmt_price(r["C_Ask"])}</td>')
                    html_parts.append(f'<td class="call-bg{atm_cls}">{_fmt_price(r["C_Bid"])}</td>')
                    # Strike (centered)
                    html_parts.append(f'<td class="{strike_cls}">${k:,.0f}</td>')
                    # Put side
                    html_parts.append(f'<td class="put-bg{atm_cls}">{_fmt_price(r["P_Bid"])}</td>')
                    html_parts.append(f'<td class="put-bg{atm_cls}">{_fmt_price(r["P_Ask"])}</td>')
                    html_parts.append(f'<td class="put-bg{atm_cls}">{_fmt_price(r["P_Mark"])}</td>')
                    html_parts.append(f'<td class="put-bg{atm_cls}">{_fmt_int(r["P_OI"])}</td>')
                    html_parts.append(f'<td class="put-bg{atm_cls}">{_fmt_int(r["P_Vol"])}</td>')
                    html_parts.append("</tr>")
                    prev_k = k

                html_parts.append("</tbody></table>")
                return "\n".join(html_parts)

            st.html(_render_t_chain(df_chain, atm_strike, spot, _spot_above, _spot_below,
                                    is_usd=_is_usd, asset=ul))

    else:
        st.info("暂无行情数据")


# ==========================================================================
# Auto-refresh is handled by inline JS timer above (no external dependency)
# ==========================================================================
