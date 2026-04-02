"""Options Monitor Dashboard – 期权行情监控面板.

实时监控 Deribit / OKX / Binance 的期权行情，
标记最高 APR 的 Short Put (SP) 和 Covered Call (CC) 机会，
并给出收益预估。

Usage:
    streamlit run monitor_app.py
"""

from __future__ import annotations

import hmac
import os
import time
import uuid
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from loguru import logger

from monitor.exchanges import fetch_all_quotes


def _new_error_id(prefix: str = "monitor") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def _log_exception_with_id(message: str, exc: Exception, *, prefix: str = "monitor") -> str:
    error_id = _new_error_id(prefix)
    logger.exception(f"{message} | error_id={error_id} | error={exc}")
    return error_id


def _get_monitor_credentials() -> tuple[str, str]:
    user = os.environ.get("MONITOR_USER", "").strip()
    password = os.environ.get("MONITOR_PASS", "").strip()
    if bool(user) != bool(password):
        raise RuntimeError("MONITOR_USER 和 MONITOR_PASS 必须同时配置")
    if not user or not password:
        raise RuntimeError("未配置 Monitor 登录凭据")
    return user, password


def _check_login() -> bool:
    try:
        expected_user, expected_pass = _get_monitor_credentials()
    except RuntimeError as exc:
        error_id = _log_exception_with_id("Monitor credential validation failed", exc, prefix="auth")
        st.error(f"Monitor 当前不可用，请联系管理员。错误编号: {error_id}")
        st.stop()

    if st.session_state.get("monitor_authenticated") is True:
        return True

    st.markdown("## 🔐 Monitor 登录")
    st.caption("监控面板默认需要登录后访问。")

    with st.form("monitor_login"):
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        submitted = st.form_submit_button("登录", width="stretch", type="primary")

    if submitted:
        if hmac.compare_digest(username, expected_user) and hmac.compare_digest(password, expected_pass):
            st.session_state["monitor_authenticated"] = True
            st.rerun()
        st.error("用户名或密码错误")

    return False


@st.cache_data(ttl=15, show_spinner="正在获取交易所数据...")
def load_quotes(
    underlyings: tuple[str, ...],
    exchanges: tuple[str, ...],
) -> list[dict]:
    """Fetch and serialise quotes (dicts for caching)."""
    quotes = fetch_all_quotes(
        underlyings=list(underlyings),
        exchanges=list(exchanges),
        timeout=10,
    )
    rows = []
    for q in quotes:
        rows.append({
            "exchange": q.exchange,
            "underlying": q.underlying,
            "instrument": q.instrument,
            "strategy": q.strategy,
            "option_type": q.option_type,
            "strike": q.strike,
            "expiry": q.expiry.strftime("%Y-%m-%d %H:%M"),
            "dte": round(q.dte, 2),
            "bid_usd": round(q.bid_usd, 2),
            "ask_usd": round(q.ask_usd, 2),
            "mark_usd": round(q.mark_usd, 2),
            "extrinsic": round(q.extrinsic_bid, 2),
            "mid_usd": round(q.mid_usd, 2),
            "spread_pct": round(q.spread_pct, 1),
            "underlying_price": round(q.underlying_price, 2),
            "moneyness_pct": round(q.moneyness_pct, 2),
            "apr": round(q.apr, 1),
            "net_apr": round(q.net_apr, 1),
            "iv": round(q.iv * 100, 1) if q.iv else 0,
            "volume_24h": round(q.volume_24h, 2),
            "open_interest": round(q.open_interest, 2),
            "daily_yield": round(q.daily_yield_per_unit, 2),
        })
    return rows


def _highlight_apr(val: object) -> str:
    if isinstance(val, (int, float)):
        if val >= 500:
            return "background-color: #ff4444; color: white; font-weight: bold"
        if val >= 200:
            return "background-color: #ff8c00; color: white; font-weight: bold"
        if val >= 100:
            return "background-color: #ffd700; font-weight: bold"
    return ""


def main() -> None:
    st.set_page_config(
        page_title="期权监控面板",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if not _check_login():
        st.stop()

    st.sidebar.title("⚙️ 筛选设置")

    sel_underlyings = st.sidebar.multiselect(
        "标的资产",
        ["BTC", "ETH"],
        default=["BTC", "ETH"],
    )
    sel_exchanges = st.sidebar.multiselect(
        "交易所",
        ["Deribit", "OKX", "Binance"],
        default=["Deribit", "OKX", "Binance"],
    )
    sel_strategy = st.sidebar.selectbox(
        "策略类型",
        ["全部", "CC (卖Call)", "SP (卖Put)"],
        index=0,
    )
    dte_range = st.sidebar.slider(
        "到期天数 (DTE)",
        min_value=0.0,
        max_value=90.0,
        value=(0.0, 30.0),
        step=0.5,
    )
    moneyness_range = st.sidebar.slider(
        "Moneyness 范围 (%)",
        min_value=-20.0,
        max_value=20.0,
        value=(-5.0, 5.0),
        step=0.5,
        help="0% = ATM，±5% = 近ATM",
    )
    min_bid_usd = st.sidebar.number_input(
        "最小 Bid (USD)",
        min_value=0.0,
        value=1.0,
        step=1.0,
        help="过滤掉无流动性的期权",
    )
    capital_usd = st.sidebar.number_input(
        "投入资金 (USD)",
        min_value=100.0,
        value=10000.0,
        step=1000.0,
    )
    auto_refresh = st.sidebar.checkbox("自动刷新", value=False)
    refresh_interval = st.sidebar.selectbox(
        "刷新间隔 (秒)",
        [10, 30, 60, 120],
        index=1,
    )

    st.title("📊 期权行情监控面板")
    st.caption("实时对比 Deribit / OKX / Binance 期权，寻找最佳 CC/SP 机会")

    col_fetch, col_time = st.columns([1, 3])
    with col_fetch:
        if st.button("🔄 刷新数据", type="primary"):
            load_quotes.clear()
    with col_time:
        st.caption(f"当前时间 (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")

    if not sel_underlyings or not sel_exchanges:
        st.warning("请至少选择一个标的资产和交易所")
        st.stop()

    try:
        raw_rows = load_quotes(
            underlyings=tuple(sorted(sel_underlyings)),
            exchanges=tuple(sorted(sel_exchanges)),
        )
    except Exception as exc:
        error_id = _log_exception_with_id("Monitor quote load failed", exc)
        st.error(f"获取行情失败，请稍后重试。错误编号: {error_id}")
        st.stop()

    if not raw_rows:
        st.error("未获取到任何数据，请检查网络连接或交易所可用性")
        st.stop()

    df = pd.DataFrame(raw_rows)
    mask = pd.Series(True, index=df.index)

    if sel_strategy == "CC (卖Call)":
        mask &= df["strategy"] == "CC"
    elif sel_strategy == "SP (卖Put)":
        mask &= df["strategy"] == "SP"

    mask &= (df["dte"] >= dte_range[0]) & (df["dte"] <= dte_range[1])
    mask &= (df["moneyness_pct"] >= moneyness_range[0]) & (df["moneyness_pct"] <= moneyness_range[1])
    mask &= df["bid_usd"] >= min_bid_usd

    filtered = df[mask].copy()
    if filtered.empty:
        st.warning("当前筛选条件下无数据，请调整筛选参数")
        st.stop()

    filtered = filtered.sort_values("apr", ascending=False).reset_index(drop=True)

    st.markdown("---")
    st.subheader("📈 市场概览")

    spot_data = filtered.groupby(["exchange", "underlying"])["underlying_price"].first().unstack()
    spot_cols = st.columns(len(sel_exchanges))
    for i, exch in enumerate(sel_exchanges):
        with spot_cols[i]:
            st.markdown(f"**{exch}**")
            if exch in spot_data.index:
                for ul in sel_underlyings:
                    if ul in spot_data.columns:
                        price = spot_data.loc[exch, ul]
                        if pd.notna(price) and price > 0:
                            st.metric(ul, f"${price:,.0f}")

    st.markdown("---")
    st.subheader("🏆 TOP 20 最高 APR 机会")

    top_n = filtered.head(20)
    display_cols = [
        "exchange", "underlying", "instrument", "strategy",
        "strike", "dte", "bid_usd", "extrinsic", "apr", "net_apr",
        "spread_pct", "moneyness_pct", "iv", "volume_24h",
    ]
    col_rename = {
        "exchange": "交易所",
        "underlying": "标的",
        "instrument": "合约",
        "strategy": "策略",
        "strike": "行权价",
        "dte": "DTE(天)",
        "bid_usd": "Bid($)",
        "extrinsic": "时间价值($)",
        "apr": "APR(%)",
        "net_apr": "净APR(%)",
        "spread_pct": "Spread(%)",
        "moneyness_pct": "偏离ATM(%)",
        "iv": "IV(%)",
        "volume_24h": "24h成交量",
    }

    styled = (
        top_n[display_cols]
        .rename(columns=col_rename)
        .style
        .map(_highlight_apr, subset=["APR(%)", "净APR(%)"])
        .format({
            "行权价": "${:,.0f}",
            "DTE(天)": "{:.1f}",
            "Bid($)": "${:,.2f}",
            "时间价值($)": "${:,.2f}",
            "APR(%)": "{:.0f}%",
            "净APR(%)": "{:.0f}%",
            "Spread(%)": "{:.1f}%",
            "偏离ATM(%)": "{:+.1f}%",
            "IV(%)": "{:.0f}%",
            "24h成交量": "{:,.0f}",
        })
    )
    st.dataframe(styled, width="stretch", height=500)

    st.markdown("---")
    st.subheader("💰 收益预估")

    if not top_n.empty:
        best = top_n.iloc[0]
        col1, col2, col3 = st.columns([1, 1, 1.5])

        with col1:
            st.markdown("**最佳机会**")
            st.markdown(f"""
| 项目 | 值 |
|------|-----|
| 交易所 | **{best['exchange']}** |
| 合约 | `{best['instrument']}` |
| 策略 | **{best['strategy']}** ({'卖Call' if best['strategy'] == 'CC' else '卖Put'}) |
| 行权价 | ${best['strike']:,.0f} |
| DTE | {best['dte']:.1f} 天 |
| Bid | ${best['bid_usd']:,.2f} |
| 时间价值 | ${best['extrinsic']:,.2f} |
| APR (时间价值) | **{best['apr']:.0f}%** |
| 净APR | {best['net_apr']:.0f}% |
| Spread | {best['spread_pct']:.1f}% |
""")

        with col2:
            st.markdown(f"**投入 ${capital_usd:,.0f} 的预估收益**")
            ul_price = best["underlying_price"]
            contracts = capital_usd / ul_price if ul_price > 0 else 0
            _strike_y = best["strike"]
            _is_call_y = best["strategy"] == "CC"
            _intrinsic = max(0, ul_price - _strike_y) if _is_call_y else max(0, _strike_y - ul_price)
            _extrinsic = max(0, best["bid_usd"] - _intrinsic)
            premium_total = _extrinsic * contracts
            dte = max(best["dte"], 0.01)
            daily_yield = premium_total / dte
            st.markdown(f"""
| 期限 | 预估收益 (USD) |
|------|---------------|
| 每次到期 ({dte:.1f}天) | **${premium_total:,.2f}** |
| 每天 | ${daily_yield:,.2f} |
| 每周 | ${daily_yield * 7:,.2f} |
| 每月 | ${daily_yield * 30:,.2f} |
| 每年 (APR) | **${daily_yield * 365:,.2f}** |

> 📌 以 ${capital_usd:,.0f} 买入 **{contracts:.4f}** {best['underlying']}，
> 卖出相同数量的 **{best['instrument']}**
""")

        with col3:
            import numpy as np
            import plotly.graph_objects as go

            st.markdown("**📉 到期收益曲线**")
            _strike = best["strike"]
            _spot = best["underlying_price"]
            _premium = best["bid_usd"]
            _is_call = best["strategy"] == "CC"
            _lo = _strike * 0.80
            _hi = _strike * 1.20
            _prices = np.linspace(_lo, _hi, 200)

            if _is_call:
                _pnl = (_prices - _spot) + _premium - np.maximum(0, _prices - _strike)
            else:
                _pnl = _premium - np.maximum(0, _strike - _prices)
            _pnl_total = _pnl * contracts

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=_prices, y=_pnl_total,
                mode="lines",
                line=dict(color="#2196F3", width=2.5),
                name="到期收益",
                fill="tozeroy",
                fillcolor="rgba(33,150,243,0.15)",
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
            fig.add_vline(x=_strike, line_dash="dot", line_color="red", line_width=1, annotation_text=f"K={_strike:,.0f}", annotation_position="top left", annotation_font_size=10)
            fig.add_vline(x=_spot, line_dash="dot", line_color="green", line_width=1, annotation_text=f"现价={_spot:,.0f}", annotation_position="top right", annotation_font_size=10)
            _be = _spot - _premium if _is_call else _strike - _premium
            fig.add_vline(x=_be, line_dash="dashdot", line_color="orange", line_width=1, annotation_text=f"盈亏平衡={_be:,.0f}", annotation_position="bottom right", annotation_font_size=9)
            _max_profit = _premium * contracts
            fig.add_annotation(x=_strike if not _is_call else _strike * 1.05, y=_max_profit, text=f"最大盈利: ${_max_profit:,.0f}", showarrow=True, arrowhead=2, font=dict(size=10, color="green"))
            fig.update_layout(
                height=280,
                margin=dict(l=10, r=10, t=30, b=40),
                xaxis_title=f"{best['underlying']} 到期价格 (USD)",
                yaxis_title="收益 (USD)",
                xaxis=dict(tickformat=",.0f"),
                yaxis=dict(tickformat=",.0f"),
                showlegend=False,
                title=dict(text=f"{'Covered Call' if _is_call else 'Short Put'} 到期损益 (USD)", font=dict(size=13)),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**📉 币本位到期收益曲线**")
            _premium_coin = _premium / _spot
            if _is_call:
                _pnl_coin = _premium_coin - np.maximum(0, _prices - _strike) / _prices
            else:
                _pnl_coin = _premium_coin - np.maximum(0, _strike - _prices) / _prices
            _pnl_coin_total = _pnl_coin * contracts

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=_prices, y=_pnl_coin_total,
                mode="lines",
                line=dict(color="#FF9800", width=2.5),
                name="币本位收益",
                fill="tozeroy",
                fillcolor="rgba(255,152,0,0.15)",
            ))
            fig2.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
            fig2.add_vline(x=_strike, line_dash="dot", line_color="red", line_width=1, annotation_text=f"K={_strike:,.0f}", annotation_position="top left", annotation_font_size=10)
            fig2.add_vline(x=_spot, line_dash="dot", line_color="green", line_width=1, annotation_text=f"现价={_spot:,.0f}", annotation_position="top right", annotation_font_size=10)
            _max_coin = _premium_coin * contracts
            fig2.add_annotation(x=_strike if not _is_call else _strike * 1.05, y=_max_coin, text=f"最大盈利: {_max_coin:.6f} {best['underlying']}", showarrow=True, arrowhead=2, font=dict(size=10, color="green"))
            fig2.update_layout(
                height=280,
                margin=dict(l=10, r=10, t=30, b=40),
                xaxis_title=f"{best['underlying']} 到期价格 (USD)",
                yaxis_title=f"收益 ({best['underlying']})",
                xaxis=dict(tickformat=",.0f"),
                yaxis=dict(tickformat=".6f"),
                showlegend=False,
                title=dict(text=f"{'Covered Call' if _is_call else 'Short Put'} 到期损益 (币本位)", font=dict(size=13)),
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**TOP 5 对比**")
        compare_rows = []
        for _, row in top_n.head(5).iterrows():
            ul_p = row["underlying_price"]
            c = capital_usd / ul_p if ul_p > 0 else 0
            p = row["bid_usd"] * c
            d = max(row["dte"], 0.01)
            dy = p / d
            compare_rows.append({
                "交易所": row["exchange"],
                "合约": row["instrument"],
                "策略": row["strategy"],
                "APR(%)": f"{row['apr']:.0f}%",
                "每次收益($)": f"${p:,.2f}",
                "日收益($)": f"${dy:,.2f}",
                "月收益($)": f"${dy * 30:,.2f}",
                "年收益($)": f"${dy * 365:,.2f}",
            })
        st.dataframe(pd.DataFrame(compare_rows), width="stretch")

    st.markdown("---")
    st.subheader("📋 各交易所对比 (同档位)")
    if len(sel_exchanges) > 1:
        compare_df = filtered.copy()
        compare_df["key"] = compare_df["underlying"] + " " + compare_df["strategy"] + " " + compare_df["strike"].astype(str) + " " + compare_df["expiry"]
        key_exchanges = compare_df.groupby("key")["exchange"].nunique()
        multi_keys = key_exchanges[key_exchanges > 1].index
        if len(multi_keys) > 0:
            multi = compare_df[compare_df["key"].isin(multi_keys)].copy().sort_values("apr", ascending=False)
            pivot = multi.pivot_table(
                index=["underlying", "strategy", "strike", "expiry", "dte"],
                columns="exchange",
                values=["bid_usd", "apr"],
                aggfunc="first",
            )
            if not pivot.empty:
                st.dataframe(pivot.head(30).style.format("{:.1f}"), width="stretch")
            else:
                st.info("未找到多个交易所的相同档位合约")
        else:
            st.info("未找到多个交易所的相同档位合约")

    st.markdown("---")
    with st.expander(f"📄 完整数据 ({len(filtered)} 条)", expanded=False):
        full_cols = [
            "exchange", "underlying", "instrument", "strategy",
            "strike", "expiry", "dte", "bid_usd", "ask_usd",
            "mid_usd", "spread_pct", "apr", "net_apr",
            "moneyness_pct", "iv", "volume_24h", "open_interest",
        ]
        st.dataframe(filtered[full_cols].rename(columns=col_rename), width="stretch", height=600)

    if auto_refresh:
        time.sleep(refresh_interval)
        load_quotes.clear()
        st.rerun()


if __name__ == "__main__":
    main()
