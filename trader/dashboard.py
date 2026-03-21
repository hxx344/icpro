"""Trader Dashboard – 交易管理面板.

Streamlit 前端，用于:
- 实时查看策略状态和持仓
- 资产曲线图表
- 损益记录与统计
- 成交历史查询
- 手动操作 (平仓、暂停)

Usage:
    streamlit run trader/dashboard.py -- --config configs/trader_iron_condor_0dte.yaml
"""

from __future__ import annotations

import json
import os
import platform
import requests
import sys
import time as _time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# OS-aware Streamlit server config (injected before Streamlit reads config)
# ---------------------------------------------------------------------------
# Streamlit reads STREAMLIT_SERVER_* env vars as config overrides.
# We detect the OS here and set sensible defaults so .streamlit/config.toml
# stays platform-agnostic (only theme / browser settings).
# ---------------------------------------------------------------------------
_is_linux = platform.system() == "Linux"

def _set_default(env_key: str, value: str) -> None:
    """Set env var only if not already set (allow manual override)."""
    if env_key not in os.environ:
        os.environ[env_key] = value

if _is_linux:
    # Linux VPS: headless, bind all interfaces, disable CORS+XSRF together
    _set_default("STREAMLIT_SERVER_HEADLESS", "true")
    _set_default("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")
    _set_default("STREAMLIT_SERVER_PORT", "8501")
    _set_default("STREAMLIT_SERVER_ENABLE_CORS", "false")
    _set_default("STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION", "false")
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

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trader.config import load_config, TraderConfig
from trader.storage import Storage
from trader.engine import get_engine, reset_engine, TradingEngine
from trader.binance_client import BinanceOptionsClient

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
# Login authentication (credentials from env: DASHBOARD_USER / DASHBOARD_PASS)
# ---------------------------------------------------------------------------

def _check_login() -> bool:
    """Show login form and validate credentials. Returns True if authenticated."""
    # Load expected credentials from environment
    expected_user = os.environ.get("DASHBOARD_USER", "").strip()
    expected_pass = os.environ.get("DASHBOARD_PASS", "").strip()

    # If no credentials configured, skip auth
    if not expected_user or not expected_pass:
        return True

    # Already authenticated this session
    if st.session_state.get("authenticated"):
        return True

    # --- Login form ---
    st.markdown(
        "<h2 style='text-align:center; margin-top:80px;'>🦅 铁鹰交易面板</h2>"
        "<p style='text-align:center; color:#888;'>请输入用户名和密码登录</p>",
        unsafe_allow_html=True,
    )
    _col_l, col_form, _col_r = st.columns([1, 1.5, 1])
    with col_form:
        with st.form("login_form"):
            username = st.text_input("用户名")
            password = st.text_input("密码", type="password")
            submitted = st.form_submit_button("登录", width='stretch', type="primary")

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
# Config & Storage initialization
# ---------------------------------------------------------------------------

@st.cache_resource
def init_storage(db_path: str) -> Storage:
    """Initialize SQLite storage (cached across reruns)."""
    return Storage(db_path)


def get_config_path() -> str:
    """Get config path from CLI args or default."""
    # streamlit run trader/dashboard.py -- --config path/to/config.yaml
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--config" and i + 1 < len(args):
            return args[i + 1]
    return "configs/trader_iron_condor_0dte.yaml"


config_path = get_config_path()
cfg = load_config(config_path)
storage = init_storage(cfg.storage.db_path)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🦅 铁鹰交易面板")

# --- Mode selector ---
if "trading_mode" not in st.session_state:
    st.session_state.trading_mode = "🔒 只读模式"

trading_mode = st.sidebar.radio(
    "运行模式",
    ["🔒 只读模式", "🟢 交易模式"],
    index=["🔒 只读模式", "🟢 交易模式"].index(st.session_state.trading_mode),
    horizontal=True,
    help="只读模式: 查看行情/持仓/统计，不可启动引擎或下单\n交易模式: 解锁全部功能",
)
st.session_state.trading_mode = trading_mode
is_trade_mode = "交易" in trading_mode
st.sidebar.markdown(f"**策略:** {cfg.name}")
st.sidebar.markdown(f"**标的:** {cfg.strategy.underlying}")
st.sidebar.markdown(f"**OTM:** ±{cfg.strategy.otm_pct*100:.0f}%")
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

if auto_refresh:
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
                for t in legs:
                    meta = json.loads(t.get("meta", "{}"))
                    role = meta.get("leg_role", "?")
                    opt_type = meta.get("option_type", "?")
                    strike = meta.get("strike", 0)
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
                        "合约": t["symbol"],
                        "时间": t["timestamp"][:19],
                    })

                df_legs = pd.DataFrame(leg_rows)
                st.dataframe(df_legs, width='stretch', hide_index=True)
                st.caption(f"净权利金: **${total_premium:.4f}**")
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

        _dark_axis = dict(
            gridcolor=_GRID, zerolinecolor=_GRID,
            tickfont=dict(color=_TEXT, size=11),
            title_font=dict(color=_TEXT, size=12),
        )
        _dark_layout = dict(
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
                **_dark_layout,
            )
            fig2.update_xaxes(**_dark_axis)
            fig2.update_yaxes(title_text="标的价格 (USD)", secondary_y=False, **_dark_axis)
            fig2.update_yaxes(title_text="持仓数", secondary_y=True, **_dark_axis)
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
                yaxis_title="持仓数", **_dark_layout,
            )
            fig_pos.update_xaxes(**_dark_axis)
            fig_pos.update_yaxes(**_dark_axis)
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
    _exchange = _raw_yaml.get("exchange") or _raw_yaml.get("deribit", {})
    _storage = _raw_yaml.get("storage", {})
    _monitor = _raw_yaml.get("monitor", {})

    # ---- Editable form ----
    with st.form("config_editor", border=True):
        st.subheader("📄 策略参数")
        s_col1, s_col2, s_col3 = st.columns(3)

        with s_col1:
            ed_underlying = st.selectbox(
                "标的", ["ETH", "BTC"],
                index=0 if _strategy.get("underlying", "ETH") == "ETH" else 1,
            )
            ed_otm_pct = st.number_input(
                "短腿 OTM %", min_value=1.0, max_value=50.0,
                value=_strategy.get("otm_pct", 0.08) * 100,
                step=1.0, format="%.1f", help="短腿离现货的距离百分比",
            )
            ed_wing_width = st.number_input(
                "翼宽 %", min_value=0.5, max_value=20.0,
                value=_strategy.get("wing_width_pct", 0.02) * 100,
                step=0.5, format="%.1f", help="保护翼额外偏移百分比",
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
                step=0.01, format="%.3f", help="每组铁鹰的合约数量",
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
                help="等现货价格到达两个最近行权价的中点再开仓，保证铁鹰两侧对称",
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
                "api_key": _exchange.get("api_key") or _exchange.get("client_id", ""),
                "api_secret": _exchange.get("api_secret") or _exchange.get("client_secret", ""),
                "testnet": ed_testnet,
                "timeout": ed_timeout,
                "account_currency": _exchange.get("account_currency", "USDT"),
                "simulate_private": _exchange.get("simulate_private", False),
            },
            "strategy": {
                "underlying": ed_underlying,
                "otm_pct": round(ed_otm_pct / 100, 4),
                "wing_width_pct": round(ed_wing_width / 100, 4),
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
            st.info("⚠️ 请重启引擎使新配置生效 (先停止再启动)")
        except Exception as _ex:
            st.error(f"保存失败: {_ex}")

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
    run_info = {
        "配置项": [
            "策略名", "标的", "OTM%", "翼宽%", "开仓时间",
            "基础数量", "复利", "API环境", "检查间隔",
        ],
        "值": [
            cfg.name,
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
        "`streamlit run trader/dashboard.py -- --config configs/trader_iron_condor_0dte.yaml` "
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
            [f"{ul} (原始)", "USD"],
            key="mkt_price_unit",
            horizontal=True,
            help=f"Binance 期权价格已转换为 {ul} 计价。选择 USD 会自动乘以标的价格换算",
        )
    _is_usd = price_unit == "USD"

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
                "Bid$": round(t.bid_price * ul_px, 2),
                "Ask$": round(t.ask_price * ul_px, 2),
                "Mid$": round(t.mid_price * ul_px, 2),
                "Mark$": round(t.mark_price * ul_px, 2),
                "Last$": round(t.last_price * ul_px, 2),
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

                    if ul_px > 0:
                        bid = bid_u / ul_px
                        ask = ask_u / ul_px
                        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else (mark_u / ul_px if ul_px > 0 else 0)
                        mark = mark_u / ul_px
                        last = last_u / ul_px
                    else:
                        bid = bid_u
                        ask = ask_u
                        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else mark_u
                        mark = mark_u
                        last = last_u

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
                        "Bid$": round(bid * ul_px, 2),
                        "Ask$": round(ask * ul_px, 2),
                        "Mid$": round(mid * ul_px, 2),
                        "Mark$": round(mark * ul_px, 2),
                        "Last$": round(last * ul_px, 2),
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
            _price_cols = ["Bid$", "Ask$", "Mid$", "Mark$", "Last$"]
            _display = df_show[["合约", "类型", "行权价", "到期", "DTE(h)"]
                               + _price_cols
                               + ["价差%", "OTM%", "24h量", "OI", "标的价"]].copy()
            _display = _display.rename(columns={
                "Bid$": "Bid", "Ask$": "Ask", "Mid$": "Mid", "Mark$": "Mark", "Last$": "Last",
            })
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
            _price_cols = ["Bid", "Ask", "Mid", "Mark", "Last"]
            _display = df_show[["合约", "类型", "行权价", "到期", "DTE(h)"]
                               + _price_cols
                               + ["价差%", "OTM%", "24h量", "OI", "标的价"]].copy()
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
            _bid_key = "Bid$" if _is_usd else "Bid"
            _ask_key = "Ask$" if _is_usd else "Ask"
            _mark_key = "Mark$" if _is_usd else "Mark"

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
