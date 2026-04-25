from __future__ import annotations

import hmac
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BYBIT_BASE_URL = "https://api.bybit.com"
MARKET_DATA_DIR = REPO_ROOT / "data" / "market_data"
DEFAULT_PANEL_USERNAME = "admin"
DEFAULT_PANEL_PASSWORD = "change-me-now"
MONTH_MAP = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}
BYBIT_SYMBOL_RE = re.compile(
    r"^(?P<ul>[A-Z]+)-(?P<day>\d{1,2})(?P<mon>[A-Z]{3})(?P<yy>\d{2})-(?P<strike>\d+(?:\.\d+)?)-(?P<cp>[CP])(?:-(?P<settle>[A-Z]+))?$"
)


@dataclass(frozen=True)
class OptionQuote:
    symbol: str
    underlying: str
    strike: float
    option_type: str
    expiry: datetime
    bid: float
    ask: float
    mark: float
    last: float
    underlying_price: float
    volume_24h: float
    open_interest: float
    delta: float
    mark_iv: float

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.mark

    @property
    def spread(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return self.ask - self.bid
        return 0.0

    @property
    def dte_days(self) -> float:
        return max((self.expiry - datetime.now(timezone.utc)).total_seconds() / 86400.0, 0.0)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value or default)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _normalize_iv(value: Any) -> float:
    iv = _float(value, 0.0)
    if iv > 3.0:
        iv /= 100.0
    return max(iv, 0.0)


def parse_bybit_symbol(symbol: str) -> dict[str, Any] | None:
    m = BYBIT_SYMBOL_RE.match(symbol)
    if not m:
        return None
    month = MONTH_MAP.get(m.group("mon"))
    if month is None:
        return None
    return {
        "underlying": m.group("ul"),
        "expiry": datetime(2000 + int(m.group("yy")), month, int(m.group("day")), 8, 0, 0, tzinfo=timezone.utc),
        "strike": float(m.group("strike")),
        "option_type": "call" if m.group("cp") == "C" else "put",
    }


@st.cache_data(ttl=20, show_spinner=False)
def bybit_get(path: str, params: tuple[tuple[str, str], ...]) -> dict[str, Any]:
    resp = requests.get(f"{BYBIT_BASE_URL}{path}", params=dict(params), timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if int(data.get("retCode", -1)) != 0:
        raise RuntimeError(f"Bybit error {data.get('retCode')}: {data.get('retMsg')}")
    return data


def _result_list(data: dict[str, Any]) -> list[Any]:
    result = data.get("result") if isinstance(data, dict) else None
    if not isinstance(result, dict):
        return []
    items = result.get("list")
    return items if isinstance(items, list) else []


@st.cache_data(ttl=20, show_spinner=False)
def fetch_option_quotes(underlying: str) -> list[OptionQuote]:
    data = bybit_get(
        "/v5/market/tickers",
        (("category", "option"), ("baseCoin", underlying.upper())),
    )
    quotes: list[OptionQuote] = []
    for item in _result_list(data):
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", ""))
        parsed = parse_bybit_symbol(symbol)
        if not parsed or parsed["underlying"] != underlying.upper():
            continue
        bid = _float(item.get("bid1Price"))
        ask = _float(item.get("ask1Price"))
        mark = _float(item.get("markPrice"))
        if mark <= 0 and bid > 0 and ask > 0:
            mark = (bid + ask) / 2.0
        quotes.append(
            OptionQuote(
                symbol=symbol,
                underlying=parsed["underlying"],
                strike=parsed["strike"],
                option_type=parsed["option_type"],
                expiry=parsed["expiry"],
                bid=bid,
                ask=ask,
                mark=mark,
                last=_float(item.get("lastPrice")),
                underlying_price=_float(item.get("underlyingPrice") or item.get("indexPrice")),
                volume_24h=_float(item.get("volume24h")),
                open_interest=_float(item.get("openInterest")),
                delta=_float(item.get("delta")),
                mark_iv=_normalize_iv(item.get("markIv") or item.get("markIV")),
            )
        )
    return quotes


@st.cache_data(ttl=20, show_spinner=False)
def fetch_spot_price(underlying: str) -> float:
    data = bybit_get(
        "/v5/market/tickers",
        (("category", "spot"), ("symbol", f"{underlying.upper()}USDT")),
    )
    items = _result_list(data)
    if not items or not isinstance(items[0], dict):
        return 0.0
    return _float(items[0].get("lastPrice") or items[0].get("markPrice") or items[0].get("indexPrice"))


@st.cache_data(ttl=20, show_spinner=False)
def fetch_perp_funding(underlying: str) -> dict[str, Any]:
    data = bybit_get(
        "/v5/market/tickers",
        (("category", "linear"), ("symbol", f"{underlying.upper()}USDT")),
    )
    items = _result_list(data)
    if not items or not isinstance(items[0], dict):
        return {"funding_rate": 0.0, "next_funding_time": ""}
    item = items[0]
    return {
        "funding_rate": _float(item.get("fundingRate")),
        "next_funding_time": str(item.get("nextFundingTime") or ""),
    }


@st.cache_data(ttl=60, show_spinner=False)
def fetch_hourly_prices(underlying: str, limit: int = 720) -> pd.DataFrame:
    data = bybit_get(
        "/v5/market/kline",
        (
            ("category", "spot"),
            ("symbol", f"{underlying.upper()}USDT"),
            ("interval", "60"),
            ("limit", str(limit)),
        ),
    )
    rows: list[dict[str, Any]] = []
    for raw in reversed(_result_list(data)):
        if not isinstance(raw, (list, tuple)) or len(raw) < 5:
            continue
        rows.append(
            {
                "timestamp": datetime.fromtimestamp(_float(raw[0]) / 1000.0, tz=timezone.utc),
                "open": _float(raw[1]),
                "high": _float(raw[2]),
                "low": _float(raw[3]),
                "close": _float(raw[4]),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_coinmetrics_daily_for_vote() -> pd.DataFrame:
    path = MARKET_DATA_DIR / "coinmetrics_btc_public_daily.csv"
    today = datetime.now(timezone.utc).date()
    if path.exists():
        cached = pd.read_csv(path, parse_dates=["date"])
        if not cached.empty and cached["date"].max().date() >= today - pd.Timedelta(days=3):
            return cached

    try:
        resp = requests.get(
            "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics",
            params={
                "assets": "btc",
                "metrics": "PriceUSD,CapMVRVCur",
                "frequency": "1d",
                "start_time": "2012-01-01",
                "end_time": today.isoformat(),
                "page_size": 10000,
            },
            timeout=20,
        )
        resp.raise_for_status()
        df = pd.DataFrame(resp.json().get("data", []))
        if df.empty:
            raise RuntimeError("empty CoinMetrics response")
        df["date"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None).dt.normalize()
        for col in ["PriceUSD", "CapMVRVCur"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[["date", "PriceUSD", "CapMVRVCur"]].sort_values("date")
        MARKET_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return df
    except Exception:
        if path.exists():
            return pd.read_csv(path, parse_dates=["date"])
        return pd.DataFrame(columns=["date", "PriceUSD", "CapMVRVCur"])


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fear_greed_daily_for_vote() -> pd.DataFrame:
    path = MARKET_DATA_DIR / "alternative_me_fear_greed.csv"
    today = datetime.now(timezone.utc).date()
    if path.exists():
        cached = pd.read_csv(path, parse_dates=["date"])
        if not cached.empty and cached["date"].max().date() >= today - pd.Timedelta(days=2):
            return cached

    try:
        resp = requests.get("https://api.alternative.me/fng/", params={"limit": 0, "format": "json"}, timeout=20)
        resp.raise_for_status()
        df = pd.DataFrame(resp.json().get("data", []))
        if df.empty:
            raise RuntimeError("empty Fear & Greed response")
        df["date"] = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="s", utc=True).dt.tz_localize(None).dt.normalize()
        df["fear_greed"] = pd.to_numeric(df["value"], errors="coerce")
        df = df[["date", "fear_greed"]].sort_values("date")
        MARKET_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return df
    except Exception:
        if path.exists():
            return pd.read_csv(path, parse_dates=["date"])
        return pd.DataFrame(columns=["date", "fear_greed"])


def rsi_daily(price: pd.Series, period: int = 14) -> pd.Series:
    diff = price.diff()
    gain = diff.clip(lower=0)
    loss = (-diff).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return 100 - 100 / (1 + avg_gain / avg_loss.replace(0, float("nan")))


def load_independent_7of6_vote_daily(spot: float) -> pd.DataFrame:
    cm = fetch_coinmetrics_daily_for_vote()
    fg = fetch_fear_greed_daily_for_vote()
    if cm.empty:
        return pd.DataFrame()

    daily = cm.merge(fg, on="date", how="outer") if not fg.empty else cm.copy()
    daily = daily.sort_values("date").reset_index(drop=True)
    for col in [c for c in daily.columns if c != "date"]:
        daily[col] = pd.to_numeric(daily[col], errors="coerce").ffill()

    today = pd.Timestamp(datetime.now(timezone.utc).date())
    if daily.empty or daily["date"].max() < today:
        last: dict[str, Any] = {str(k): v for k, v in daily.iloc[-1].to_dict().items()} if not daily.empty else {"CapMVRVCur": pd.NA, "fear_greed": pd.NA}
        last["date"] = today
        last["PriceUSD"] = spot
        daily = pd.concat([daily, pd.DataFrame([last])], ignore_index=True)
    elif spot > 0:
        daily.loc[daily["date"] == daily["date"].max(), "PriceUSD"] = spot

    daily = daily.sort_values("date").reset_index(drop=True)
    daily["price"] = pd.to_numeric(daily["PriceUSD"], errors="coerce").ffill()
    daily["mvrv_ratio"] = pd.to_numeric(daily["CapMVRVCur"], errors="coerce").ffill()
    daily["rsi_14d"] = rsi_daily(daily["price"], 14)
    daily["sma_30d"] = daily["price"].rolling(30, min_periods=30).mean()
    daily["roc_30d"] = daily["price"] / daily["price"].shift(30) - 1.0
    daily["sopr_price_proxy_155d"] = daily["price"] / daily["price"].shift(155)
    daily["roc_730d"] = daily["price"] / daily["price"].shift(730) - 1.0
    return daily


def independent_7of6_vote_signal(spot: float, drawdown: float, trigger: float) -> tuple[bool, bool, str, list[dict[str, Any]]]:
    daily = load_independent_7of6_vote_daily(spot)
    drawdown_bear = drawdown <= -abs(trigger)
    if daily.empty:
        return True, drawdown_bear, "独立7选6数据不足：默认按牛市处理", []

    latest = daily.dropna(subset=["price"]).iloc[-1]
    rows = [
        {
            "指标": "RSI14 50~80",
            "当前值": f"{float(latest.get('rsi_14d', float('nan'))):.2f}",
            "通过": bool(50 <= float(latest.get("rsi_14d", float("nan"))) <= 80),
            "含义": "趋势偏强但未极端过热",
        },
        {
            "指标": "价格 > 30日均线",
            "当前值": f"{float(latest['price']) / float(latest.get('sma_30d', float('nan'))) - 1.0:+.2%}",
            "通过": bool(float(latest["price"]) > float(latest.get("sma_30d", float("nan")))),
            "含义": "短中期趋势仍在均线上方",
        },
        {
            "指标": "30日ROC > -5%",
            "当前值": f"{float(latest.get('roc_30d', float('nan'))):+.2%}",
            "通过": bool(float(latest.get("roc_30d", float("nan"))) > -0.05),
            "含义": "近30天没有明显转弱",
        },
        {
            "指标": "SOPR代理 155日 0.95~1.5",
            "当前值": f"{float(latest.get('sopr_price_proxy_155d', float('nan'))):.3f}",
            "通过": bool(0.95 <= float(latest.get("sopr_price_proxy_155d", float("nan"))) <= 1.5),
            "含义": "用现价/155天前价格近似持有者盈亏状态",
        },
        {
            "指标": "恐惧贪婪 25~80",
            "当前值": f"{float(latest.get('fear_greed', float('nan'))):.0f}",
            "通过": bool(25 <= float(latest.get("fear_greed", float("nan"))) <= 80),
            "含义": "情绪非极端恐慌，也非极端贪婪",
        },
        {
            "指标": "730日ROC > 0%",
            "当前值": f"{float(latest.get('roc_730d', float('nan'))):+.2%}",
            "通过": bool(float(latest.get("roc_730d", float("nan"))) > 0.0),
            "含义": "两年级别长周期仍为正收益",
        },
        {
            "指标": "MVRV 1~3.5",
            "当前值": f"{float(latest.get('mvrv_ratio', float('nan'))):.3f}",
            "通过": bool(1.0 <= float(latest.get("mvrv_ratio", float("nan"))) <= 3.5),
            "含义": "链上估值不低于成本区，也未进入严重过热区",
        },
    ]
    votes = sum(1 for row in rows if row["通过"])
    threshold = 6
    bull = votes >= threshold and not drawdown_bear
    bear = (not bull) or drawdown_bear
    text = f"独立7选6：{votes}/7 通过，阈值 {threshold}/7"
    if drawdown_bear:
        text += f"；但{fmt_pct(drawdown)}回撤已触发保护"
    return bull, bear, text, rows


def vote_count_strength(votes: int) -> str:
    if votes >= 7:
        return "极强牛"
    if votes == 6:
        return "牛市"
    if votes >= 4:
        return "中性/偏弱"
    return "弱势/熊市"


def layered_call_delta_by_votes(
    vote_rows: list[dict[str, Any]],
    drawdown: float,
    trigger: float,
    delta_7: float,
    delta_6: float,
    delta_45: float,
    delta_03: float,
) -> tuple[float | None, str, int | None, str | None]:
    if not vote_rows:
        return None, "投票数据不足：使用基础趋势调整", None, None
    votes = sum(1 for row in vote_rows if bool(row.get("通过")))
    strength = vote_count_strength(votes)
    drawdown_bear = drawdown <= -abs(trigger)
    if votes >= 7:
        target = float(delta_7)
    elif votes == 6:
        target = float(delta_6)
    elif votes >= 4:
        target = float(delta_45)
    else:
        target = float(delta_03)
    target = max(0.001, target)
    text = f"{strength}：{votes}/7票，分层目标 Call Delta={target:.3f}"
    if drawdown_bear:
        text += f"；{fmt_pct(drawdown)}回撤触发保护 Put"
    return target, text, votes, strength


def choose_expiry(quotes: list[OptionQuote], target_days: float) -> datetime | None:
    now = datetime.now(timezone.utc)
    expiries = sorted({q.expiry for q in quotes if q.expiry > now and q.dte_days >= 0.25})
    if not expiries:
        return None
    return min(expiries, key=lambda exp: abs((exp - now).total_seconds() / 86400.0 - target_days))


def choose_call(quotes: list[OptionQuote], expiry: datetime, spot: float, otm_pct: float) -> OptionQuote | None:
    target = spot * (1.0 + otm_pct)
    calls = [q for q in quotes if q.option_type == "call" and q.expiry == expiry and q.strike > 0 and (q.bid > 0 or q.mark > 0)]
    if not calls:
        return None
    above = [q for q in calls if q.strike >= target]
    candidates = above or calls
    return min(candidates, key=lambda q: (abs(q.strike - target), q.strike))


def choose_call_by_delta(quotes: list[OptionQuote], expiry: datetime, target_delta: float, mode: str = "closest") -> OptionQuote | None:
    calls = [q for q in quotes if q.option_type == "call" and q.expiry == expiry and q.delta > 0 and (q.bid > 0 or q.mark > 0)]
    if not calls:
        return None
    if mode == "at_least_target":
        above = [q for q in calls if q.delta >= target_delta]
        if above:
            return min(above, key=lambda q: (q.delta - target_delta, q.strike))
    return min(calls, key=lambda q: (abs(q.delta - target_delta), q.strike))


def choose_put(quotes: list[OptionQuote], expiry: datetime, spot: float, otm_pct: float) -> OptionQuote | None:
    target = spot * (1.0 - otm_pct)
    puts = [q for q in quotes if q.option_type == "put" and q.expiry == expiry and q.strike > 0 and (q.ask > 0 or q.mark > 0)]
    if not puts:
        return None
    below = [q for q in puts if q.strike <= target]
    candidates = below or puts
    return min(candidates, key=lambda q: (abs(q.strike - target), -q.strike))


def choose_put_by_delta(quotes: list[OptionQuote], expiry: datetime, target_abs_delta: float) -> OptionQuote | None:
    puts = [q for q in quotes if q.option_type == "put" and q.expiry == expiry and q.delta < 0 and (q.ask > 0 or q.mark > 0)]
    if not puts:
        return None
    return min(puts, key=lambda q: (abs(abs(q.delta) - target_abs_delta), -q.strike))


def effective_call_otm(base_otm: float, adjust: float, spot: float, sma: float, drawdown: float, trigger: float) -> tuple[float, str]:
    bear = spot < sma or drawdown <= -abs(trigger)
    bull = spot > sma and not bear
    if bull:
        return base_otm + adjust, "牛市：call 卖远，减少上涨被盖帽"
    if bear:
        return max(0.0, base_otm - adjust), "弱势/回撤：call 卖近，增加权利金"
    return base_otm, "中性：使用基础 OTM"


def effective_call_delta(base_delta: float, adjust: float, spot: float, sma: float, drawdown: float, trigger: float) -> tuple[float, str]:
    bear = spot < sma or drawdown <= -abs(trigger)
    bull = spot > sma and not bear
    if bull:
        return max(0.001, base_delta - adjust), "牛市：call 选更低 delta，减少上涨被盖帽"
    if bear:
        return base_delta + adjust, "弱势/回撤：call 选更高 delta，增加权利金"
    return base_delta, "中性：使用基础 delta"


def _series_ref(close: pd.Series, lookback: int) -> float | None:
    if len(close) < lookback or lookback <= 0:
        return None
    value = float(close.iloc[-lookback])
    return value if math.isfinite(value) and value > 0 else None


def bull_regime_signal(spot: float, hourly: pd.DataFrame, mode_label: str, drawdown: float, trigger: float) -> tuple[bool, bool, str, list[dict[str, Any]]]:
    if mode_label == "独立7选6 多因子投票":
        return independent_7of6_vote_signal(spot, drawdown, trigger)

    if hourly.empty or "close" not in hourly.columns:
        return True, False, "行情不足：默认按牛市处理", []
    close = hourly["close"].astype(float).dropna()
    drawdown_bear = drawdown <= -abs(trigger)

    if mode_label == "30天动量 > -2%":
        ref = _series_ref(close, 24 * 30)
        if ref is None:
            return True, drawdown_bear, "30天动量数据不足：默认按牛市处理", []
        roc = spot / ref - 1.0
        bull = roc > -0.02 and not drawdown_bear
        return bull, (not bull) or drawdown_bear, f"30天动量容忍：现价/30天前={roc:+.2%}，阈值>-2%", []

    if mode_label == "30天动量":
        ref = _series_ref(close, 24 * 30)
        if ref is None:
            return True, drawdown_bear, "30天动量数据不足：默认按牛市处理", []
        bull = spot > ref and not drawdown_bear
        return bull, (not bull) or drawdown_bear, f"30天动量：现价/30天前={spot / ref - 1.0:+.2%}", []

    if mode_label == "SMA交叉 7天/30天":
        fast = float(close.tail(24 * 7).mean()) if len(close) >= 24 * 7 else float(close.mean())
        slow = float(close.tail(24 * 30).mean()) if len(close) >= 24 * 30 else float(close.mean())
        bull = fast > slow and not drawdown_bear
        return bull, (not bull) or drawdown_bear, f"SMA交叉：7天/30天={fast / slow - 1.0:+.2%}", []

    if mode_label == "SMA交叉 14天/60天":
        fast = float(close.tail(24 * 14).mean()) if len(close) >= 24 * 14 else float(close.mean())
        slow = float(close.tail(24 * 60).mean()) if len(close) >= 24 * 60 else float(close.mean())
        bull = fast > slow and not drawdown_bear
        return bull, (not bull) or drawdown_bear, f"SMA交叉：14天/60天={fast / slow - 1.0:+.2%}", []

    if mode_label == "双均线 7天/30天":
        fast = float(close.tail(24 * 7).mean()) if len(close) >= 24 * 7 else float(close.mean())
        slow = float(close.tail(24 * 30).mean()) if len(close) >= 24 * 30 else float(close.mean())
        bull = spot > slow and fast > slow and not drawdown_bear
        return bull, (not bull) or drawdown_bear, f"双均线：7天均线/30天均线={fast / slow - 1.0:+.2%}", []

    if mode_label == "7天动量":
        ref = _series_ref(close, 24 * 7)
        if ref is None:
            return True, drawdown_bear, "7天动量数据不足：默认按牛市处理", []
        bull = spot > ref and not drawdown_bear
        return bull, (not bull) or drawdown_bear, f"7天动量：现价/7天前={spot / ref - 1.0:+.2%}", []

    if mode_label == "3天动量":
        ref = _series_ref(close, 24 * 3)
        if ref is None:
            return True, drawdown_bear, "3天动量数据不足：默认按牛市处理", []
        bull = spot > ref and not drawdown_bear
        return bull, (not bull) or drawdown_bear, f"3天动量：现价/3天前={spot / ref - 1.0:+.2%}", []

    sma = float(close.tail(24 * 30).mean()) if len(close) >= 24 * 30 else float(close.mean())
    bull = spot > sma and not drawdown_bear
    return bull, (not bull) or drawdown_bear, f"30天均线：现价/均线={spot / sma - 1.0:+.2%}", []


def effective_call_delta_by_signal(base_delta: float, adjust: float, bull: bool, bear: bool) -> tuple[float, str]:
    base = max(0.001, float(base_delta))
    adjust = max(0.0, float(adjust))
    if bull:
        return max(0.001, base - adjust), "牛市：call 选更低 delta，减少上涨被盖帽"
    if bear:
        return base + adjust, "弱势/回撤：call 选更高 delta，增加权利金"
    return base, "中性：使用基础 delta"


def effective_call_otm_by_signal(base_otm: float, adjust: float, bull: bool, bear: bool) -> tuple[float, str]:
    if bull:
        return float(base_otm) + float(adjust), "牛市：call 卖远，减少上涨被盖帽"
    if bear:
        return max(0.0, float(base_otm) - float(adjust)), "弱势/回撤：call 卖近，增加权利金"
    return float(base_otm), "中性：使用基础 OTM"


def estimate_fee(option_notional: float, fee_pct: float) -> float:
    return max(option_notional, 0.0) * max(fee_pct, 0.0)


def payoff_at_expiry(
    future_spot: float,
    spot: float,
    qty: float,
    call_qty: float,
    put_qty: float,
    call: OptionQuote,
    put: OptionQuote | None,
    net_option_cash: float,
    funding_cost_usd: float = 0.0,
) -> float:
    long_btc_pnl = (future_spot - spot) * qty
    short_call_payoff = -max(future_spot - call.strike, 0.0) * call_qty
    long_put_payoff = max(put.strike - future_spot, 0.0) * put_qty if put is not None else 0.0
    return long_btc_pnl + net_option_cash + short_call_payoff + long_put_payoff - funding_cost_usd


def fmt_usd(value: float) -> str:
    return f"${value:,.0f}"


def fmt_pct(value: float) -> str:
    return f"{value * 100:,.2f}%"


def _html_escape(value: Any) -> str:
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def kpi_card(title: str, value: str, subtitle: str = "", tone: str = "neutral") -> str:
    return f"""
<div class="kpi-card {tone}">
  <div class="kpi-title">{_html_escape(title)}</div>
  <div class="kpi-value">{_html_escape(value)}</div>
  <div class="kpi-subtitle">{_html_escape(subtitle)}</div>
</div>
"""


def signal_card(title: str, body: str, tone: str = "neutral") -> str:
    return f"""
<div class="signal-card {tone}">
  <div class="signal-title">{_html_escape(title)}</div>
  <div class="signal-body">{_html_escape(body)}</div>
</div>
"""


@st.cache_data(ttl=300, show_spinner=False)
def load_historical_14otm_premiums() -> pd.DataFrame:
    path = REPO_ROOT / "reports" / "optimizations" / "btc_covered_call" / "hist_7d_14otm_call_premiums.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df


def historical_percentile(values: pd.Series, current: float) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty or not math.isfinite(current):
        return None
    return float((clean <= current).mean())


def leg_row(side: str, quote: OptionQuote, price: float, qty: float, fee_pct: float) -> dict[str, Any]:
    notional = price * qty
    return {
        "方向": side,
        "合约": quote.symbol,
        "到期": quote.expiry.strftime("%Y-%m-%d %H:%M UTC"),
        "DTE": f"{quote.dte_days:.2f}d",
        "Strike": f"{quote.strike:,.0f}",
        "Bid": f"{quote.bid:,.2f}",
        "Ask": f"{quote.ask:,.2f}",
        "Mark": f"{quote.mark:,.2f}",
        "Delta": f"{quote.delta:.3f}",
        "IV": fmt_pct(quote.mark_iv) if quote.mark_iv > 0 else "-",
        "估算成交价": f"{price:,.2f}",
        "名义现金流": fmt_usd(notional),
        "手续费估算": fmt_usd(estimate_fee(notional, fee_pct)),
    }


def _secret_or_env(secret_keys: tuple[str, ...], env_key: str, default: str) -> str:
    for key in secret_keys:
        try:
            value = st.secrets.get(key)
        except Exception:
            value = None
        if value is not None and str(value):
            return str(value)
    return os.getenv(env_key, default)


def panel_credentials() -> tuple[str, str]:
    username = _secret_or_env(("cc_panel_username", "CC_PANEL_USERNAME"), "CC_PANEL_USERNAME", DEFAULT_PANEL_USERNAME)
    password = _secret_or_env(("cc_panel_password", "CC_PANEL_PASSWORD"), "CC_PANEL_PASSWORD", DEFAULT_PANEL_PASSWORD)
    return username, password


def require_login() -> None:
    expected_username, expected_password = panel_credentials()
    if st.session_state.get("cc_panel_authenticated") is True:
        with st.sidebar:
            st.caption(f"已登录：{st.session_state.get('cc_panel_user', expected_username)}")
            if st.button("退出登录"):
                st.session_state.pop("cc_panel_authenticated", None)
                st.session_state.pop("cc_panel_user", None)
                st.rerun()
        return

    st.markdown("## 🔐 Bybit BTC Covered Call / Collar")
    st.caption("请输入账户密码后查看推荐面板。可通过环境变量 `CC_PANEL_USERNAME` / `CC_PANEL_PASSWORD` 或 Streamlit secrets 配置。")
    with st.form("cc_panel_login_form"):
        username = st.text_input("账户", value="")
        password = st.text_input("密码", value="", type="password")
        submitted = st.form_submit_button("登录", type="primary")
    if submitted:
        ok = hmac.compare_digest(username, expected_username) and hmac.compare_digest(password, expected_password)
        if ok:
            st.session_state["cc_panel_authenticated"] = True
            st.session_state["cc_panel_user"] = username
            st.rerun()
        else:
            st.error("账户或密码错误。")

    if expected_username == DEFAULT_PANEL_USERNAME and expected_password == DEFAULT_PANEL_PASSWORD:
        st.warning("当前使用默认登录凭据 admin / change-me-now。部署前请务必通过环境变量或 Streamlit secrets 修改。")
    st.stop()


st.set_page_config(page_title="Bybit BTC Covered Call 推荐", page_icon="₿", layout="wide")
st.markdown(
    """
<style>
.stApp {
    background:
        radial-gradient(circle at 16% 0%, rgba(56, 189, 248, .13), transparent 28%),
        radial-gradient(circle at 88% 6%, rgba(245, 158, 11, .10), transparent 26%),
        linear-gradient(180deg, #070b14 0%, #0b1120 46%, #090d16 100%);
}
.main .block-container {padding-top: 1.35rem; max-width: 1420px; padding-bottom: 2.6rem;}
[data-testid="stSidebar"] {background: linear-gradient(180deg, rgba(15,23,42,.98), rgba(2,6,23,.98)); border-right: 1px solid rgba(148,163,184,.14);}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {color: #e5e7eb;}
.hero {
    border: 1px solid rgba(148,163,184,.18);
    background: linear-gradient(135deg, rgba(15,23,42,.92), rgba(17,24,39,.72));
    border-radius: 24px;
    padding: 24px 28px;
    margin-bottom: 18px;
    box-shadow: 0 24px 80px rgba(0,0,0,.32);
}
.hero-top {display:flex; align-items:center; justify-content:space-between; gap: 18px; flex-wrap: wrap;}
.hero-title {font-size: 2.15rem; line-height: 1.12; font-weight: 850; letter-spacing: -.04em; color: #f8fafc;}
.hero-subtitle {margin-top: 8px; color: #94a3b8; font-size: .98rem;}
.badge-row {display:flex; gap: 8px; flex-wrap: wrap; margin-top: 16px;}
.badge {padding: 7px 10px; border-radius: 999px; border: 1px solid rgba(148,163,184,.18); color:#cbd5e1; background: rgba(15,23,42,.76); font-size: .82rem;}
.badge.live {border-color: rgba(52,211,153,.28); color:#86efac; background: rgba(6,78,59,.22);}
.kpi-card {
    height: 138px;
    border: 1px solid rgba(148,163,184,.16);
    border-radius: 22px;
    padding: 18px 18px 16px 18px;
    background: linear-gradient(145deg, rgba(15,23,42,.88), rgba(30,41,59,.62));
    box-shadow: inset 0 1px 0 rgba(255,255,255,.05), 0 18px 46px rgba(0,0,0,.22);
}
.kpi-card.good {border-color: rgba(52,211,153,.28); background: linear-gradient(145deg, rgba(6,78,59,.40), rgba(15,23,42,.78));}
.kpi-card.warn {border-color: rgba(251,191,36,.30); background: linear-gradient(145deg, rgba(113,63,18,.34), rgba(15,23,42,.78));}
.kpi-card.bad {border-color: rgba(251,113,133,.32); background: linear-gradient(145deg, rgba(127,29,29,.34), rgba(15,23,42,.78));}
.kpi-title {font-size: .78rem; color:#94a3b8; text-transform: uppercase; letter-spacing: .08em; font-weight: 700;}
.kpi-value {font-size: 1.68rem; color:#f8fafc; margin-top: 8px; font-weight: 820; letter-spacing: -.025em; white-space: nowrap;}
.kpi-subtitle {font-size: .86rem; color:#94a3b8; margin-top: 8px; line-height: 1.3;}
.signal-card {
    border-radius: 18px;
    border: 1px solid rgba(148,163,184,.16);
    padding: 15px 17px;
    margin: 13px 0 18px 0;
    background: rgba(15,23,42,.76);
}
.signal-card.good {border-color: rgba(52,211,153,.28); background: rgba(6,78,59,.22);}
.signal-card.warn {border-color: rgba(251,191,36,.36); background: rgba(113,63,18,.24);}
.signal-card.bad {border-color: rgba(251,113,133,.34); background: rgba(127,29,29,.24);}
.signal-title {font-weight: 820; color:#f8fafc; margin-bottom: 5px;}
.signal-body {color:#cbd5e1; line-height: 1.45;}
.section-title {font-size: 1.1rem; font-weight: 820; color:#f8fafc; margin: 8px 0 10px 0; letter-spacing: -.015em;}
div[data-testid="stDataFrame"] {border: 1px solid rgba(148,163,184,.14); border-radius: 16px; overflow: hidden;}
.stPlotlyChart {border: 1px solid rgba(148,163,184,.14); border-radius: 18px; background: rgba(15,23,42,.42); padding: 8px;}
hr {border-color: rgba(148,163,184,.14) !important;}
</style>
""",
    unsafe_allow_html=True,
)

require_login()

st.markdown(
        """
<div class="hero">
    <div class="hero-top">
        <div>
            <div class="hero-title">Bybit BTC Covered Call / Collar</div>
            <div class="hero-subtitle">实时行情驱动的 1x BTC + 2x 卖 Call + 2x 动态保护 Put 推荐面板</div>
        </div>
        <div class="badge live">● Live Bybit Public API</div>
    </div>
    <div class="badge-row">
        <div class="badge">Read-only</div>
        <div class="badge">No account connection</div>
        <div class="badge">No auto order</div>
        <div class="badge">USD margin view</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
)

with st.sidebar:
    st.header("策略参数")
    underlying = st.selectbox("标的", ["BTC"], index=0)
    qty = st.number_input("BTC 合约数量", min_value=0.001, max_value=100.0, value=1.0, step=0.1, format="%.3f")
    short_call_multiplier = st.number_input("卖 Call 倍数", min_value=0.0, max_value=5.0, value=2.0, step=0.25, format="%.2f")
    protective_put_multiplier = st.number_input("动态买 Put 倍数", min_value=0.0, max_value=5.0, value=2.0, step=0.25, format="%.2f")
    initial_capital = st.number_input("USD 保证金本金", min_value=0.0, value=0.0, step=100.0, help="填 0 则使用 当前BTC价格 × 数量")
    target_dte = st.number_input("目标到期天数", min_value=1.0, max_value=60.0, value=7.0, step=1.0)
    selection_mode = st.radio("选腿方式", ["OTM", "Delta"], index=1, horizontal=True)
    base_call_otm = st.number_input("基础卖 Call OTM", min_value=0.0, max_value=1.0, value=0.12, step=0.01, format="%.2f")
    base_call_delta = st.number_input("基础卖 Call Delta", min_value=0.001, max_value=0.80, value=0.10, step=0.01, format="%.3f")
    call_delta_pick_mode_label = st.radio(
        "Call Delta 匹配方式",
        ["不低于目标Delta", "绝对最接近"],
        index=0,
        horizontal=True,
        help="不低于目标Delta会优先选更近、更高权利金的一档；例如目标0.08时，会选0.111而不是0.054。",
    )
    call_adjust = st.number_input("趋势调整幅度", min_value=0.0, max_value=0.5, value=0.05, step=0.01, format="%.2f")
    st.caption("独立7因子投票模式下，Delta将由下方分层参数覆盖；基础Delta/趋势调整仅用于其它牛熊指标。")
    layered_delta_7 = st.number_input("7/7 极强牛 Call Delta", min_value=0.001, max_value=0.80, value=0.01, step=0.01, format="%.3f")
    layered_delta_6 = st.number_input("6/7 牛市 Call Delta", min_value=0.001, max_value=0.80, value=0.01, step=0.01, format="%.3f")
    layered_delta_45 = st.number_input("4~5/7 中性偏弱 Call Delta", min_value=0.001, max_value=0.80, value=0.30, step=0.01, format="%.3f")
    layered_delta_03 = st.number_input("0~3/7 弱势熊市 Call Delta", min_value=0.001, max_value=0.80, value=0.48, step=0.01, format="%.3f")
    put_otm = st.number_input("保护 Put OTM", min_value=0.0, max_value=0.8, value=0.05, step=0.01, format="%.2f")
    put_delta = st.number_input("保护 Put 绝对 Delta", min_value=0.001, max_value=0.80, value=0.08, step=0.01, format="%.3f")
    fee_pct = st.number_input("手续费估算比例", min_value=0.0, max_value=0.01, value=0.0003, step=0.0001, format="%.4f")
    include_funding = st.checkbox("计入 BTCUSDT 永续资金费率", value=True)
    funding_multiplier = st.number_input("资金费率安全倍数", min_value=0.0, max_value=5.0, value=1.0, step=0.25, format="%.2f")
    bull_indicator_label = st.selectbox(
        "牛市判断指标",
        ["独立7选6 多因子投票", "30天动量 > -2%", "SMA交叉 7天/30天", "SMA交叉 14天/60天", "30天动量", "双均线 7天/30天", "30天均线", "7天动量", "3天动量"],
        index=0,
        help="默认使用7因子投票强弱分层：7/7与6/7卖低Delta Call，4~5/7卖中高Delta Call，0~3/7卖高Delta Call并启用保护Put。",
    )
    trend_hours = st.number_input("趋势均线小时数", min_value=24, max_value=2000, value=24 * 30, step=24)
    dd_lookback_hours = st.number_input("回撤观察小时数", min_value=24, max_value=2000, value=24 * 30, step=24)
    dd_trigger = st.number_input("回撤触发阈值", min_value=0.01, max_value=0.8, value=0.10, step=0.01, format="%.2f")
    force_put = st.checkbox("强制显示买 Put", value=False)
    refresh = st.button("刷新行情", type="primary")

if refresh:
    st.cache_data.clear()

try:
    spot = fetch_spot_price(underlying)
    hourly = fetch_hourly_prices(underlying, limit=max(int(trend_hours), int(dd_lookback_hours), 720))
    quotes = fetch_option_quotes(underlying)
    funding_info = fetch_perp_funding(underlying)
except Exception as exc:
    st.error(f"获取 Bybit 行情失败：{exc}")
    st.stop()

if spot <= 0 and quotes:
    quote_spots = [q.underlying_price for q in quotes if q.underlying_price > 0]
    spot = float(pd.Series(quote_spots).median()) if quote_spots else 0.0

if spot <= 0:
    st.error("未获取到有效 BTC 现价。")
    st.stop()

if hourly.empty or len(hourly) < 24:
    sma = spot
    rolling_peak = spot
    drawdown = 0.0
else:
    close = hourly["close"].astype(float)
    sma = float(close.tail(int(trend_hours)).mean()) if len(close) >= int(trend_hours) else float(close.mean())
    rolling_peak = float(close.tail(int(dd_lookback_hours)).max()) if len(close) >= int(dd_lookback_hours) else float(close.max())
    drawdown = spot / rolling_peak - 1.0 if rolling_peak > 0 else 0.0

bull_signal, bear_signal, bull_indicator_text, vote_detail_rows = bull_regime_signal(spot, hourly, bull_indicator_label, drawdown, dd_trigger)
vote_count: int | None = None
vote_strength: str | None = None

if selection_mode == "Delta":
    layered_delta, layered_text, vote_count, vote_strength = layered_call_delta_by_votes(
        vote_detail_rows,
        drawdown,
        dd_trigger,
        layered_delta_7,
        layered_delta_6,
        layered_delta_45,
        layered_delta_03,
    )
    if bull_indicator_label == "独立7选6 多因子投票" and layered_delta is not None:
        call_target_delta, regime_text = layered_delta, layered_text
    else:
        call_target_delta, regime_text = effective_call_delta_by_signal(base_call_delta, call_adjust, bull_signal, bear_signal)
    call_otm = float("nan")
else:
    call_otm, regime_text = effective_call_otm_by_signal(base_call_otm, call_adjust, bull_signal, bear_signal)
    call_target_delta = float("nan")
buy_put = force_put or bear_signal

expiry = choose_expiry(quotes, target_dte)
if expiry is None:
    st.error("Bybit 当前没有可用 BTC 期权 ticker。")
    st.stop()

if selection_mode == "Delta":
    call_delta_pick_mode = "at_least_target" if call_delta_pick_mode_label == "不低于目标Delta" else "closest"
    call = choose_call_by_delta(quotes, expiry, call_target_delta, call_delta_pick_mode)
    put = choose_put_by_delta(quotes, expiry, put_delta) if buy_put else None
else:
    call = choose_call(quotes, expiry, spot, call_otm)
    put = choose_put(quotes, expiry, spot, put_otm) if buy_put else None

if call is None:
    st.error("未找到符合条件的 Call。")
    st.stop()

call_sell_price = call.bid if call.bid > 0 else call.mark
put_buy_price = put.ask if put is not None and put.ask > 0 else (put.mark if put is not None else 0.0)
call_qty = qty * short_call_multiplier
put_qty = qty * protective_put_multiplier if put is not None else 0.0
call_premium = call_sell_price * call_qty
put_cost = put_buy_price * put_qty if put is not None else 0.0
call_fee = estimate_fee(call_premium, fee_pct)
put_fee = estimate_fee(put_cost, fee_pct)
net_option_cash = call_premium - put_cost - call_fee - put_fee
capital = initial_capital if initial_capital > 0 else spot * qty
hours_to_expiry = max((expiry - datetime.now(timezone.utc)).total_seconds() / 3600.0, 0.0)
funding_intervals = int(math.ceil(hours_to_expiry / 8.0))
funding_rate = float(funding_info.get("funding_rate", 0.0))
funding_cost_usd = spot * qty * funding_rate * funding_intervals * funding_multiplier if include_funding else 0.0
hist_premiums = load_historical_14otm_premiums()
hist_call_premium_pctile = None
hist_call_premium_median = None
hist_call_premium_p25 = None
if not hist_premiums.empty and "sell_premium_usd" in hist_premiums.columns:
    hist_call_premium_pctile = historical_percentile(hist_premiums["sell_premium_usd"], call_premium / max(call_qty, 1e-12))
    hist_call_premium_median = float(pd.to_numeric(hist_premiums["sell_premium_usd"], errors="coerce").median())
    hist_call_premium_p25 = float(pd.to_numeric(hist_premiums["sell_premium_usd"], errors="coerce").quantile(0.25))

payoff_grid_xs = [spot * (0.6 + i * 0.01) for i in range(101)]
payoff_grid_ys = [payoff_at_expiry(x, spot, qty, call_qty, put_qty, call, put, net_option_cash, funding_cost_usd) for x in payoff_grid_xs]
max_gain_usd = max(payoff_grid_ys) if payoff_grid_ys else 0.0
min_grid_usd = min(payoff_grid_ys) if payoff_grid_ys else 0.0
call_strike_pnl = payoff_at_expiry(call.strike, spot, qty, call_qty, put_qty, call, put, net_option_cash, funding_cost_usd)
down_30_pnl = payoff_at_expiry(spot * 0.70, spot, qty, call_qty, put_qty, call, put, net_option_cash, funding_cost_usd)
up_30_pnl = payoff_at_expiry(spot * 1.30, spot, qty, call_qty, put_qty, call, put, net_option_cash, funding_cost_usd)
max_gain_pct = max_gain_usd / capital if capital > 0 else 0.0
min_grid_pct = min_grid_usd / capital if capital > 0 else 0.0

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown(kpi_card("BTC Spot", fmt_usd(spot), f"30D SMA · {fmt_usd(sma)}", "neutral"), unsafe_allow_html=True)
with kpi2:
    st.markdown(
        kpi_card("30D Drawdown", fmt_pct(drawdown), "保护触发" if bear_signal else "趋势未破", "bad" if bear_signal else "good"),
        unsafe_allow_html=True,
    )
with kpi3:
    st.markdown(
        kpi_card("Recommended Call", fmt_pct(call_otm) if selection_mode == "OTM" else f"Δ {call_target_delta:.3f}", regime_text, "warn" if bear_signal else "good"),
        unsafe_allow_html=True,
    )
with kpi4:
    funding_label = f"Funding {fmt_usd(funding_cost_usd)} · {funding_intervals} intervals"
    st.markdown(kpi_card("Net Cash After Funding", fmt_usd(net_option_cash - funding_cost_usd), funding_label, "good" if net_option_cash - funding_cost_usd >= 0 else "bad"), unsafe_allow_html=True)

if hist_call_premium_pctile is not None:
    pctile_text = fmt_pct(hist_call_premium_pctile)
    if hist_call_premium_pctile < 0.25:
        st.markdown(
            signal_card(
                "权利金偏低",
                f"当前卖 Call 约 {fmt_usd(call_premium / max(call_qty, 1e-12))}/BTC，仅处于历史 7天约14% OTM 样本的 {pctile_text} 分位。历史25分位约 {fmt_usd(hist_call_premium_p25 or 0.0)}，中位数约 {fmt_usd(hist_call_premium_median or 0.0)}。建议谨慎卖出或提高最低权利金要求。",
                "warn",
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            signal_card(
                "权利金处于可接受区间",
                f"当前卖 Call 约 {fmt_usd(call_premium / max(call_qty, 1e-12))}/BTC，处于历史 7天约14% OTM 样本的 {pctile_text} 分位。",
                "good",
            ),
            unsafe_allow_html=True,
        )

st.divider()
left, right = st.columns([1.15, 0.85])

with left:
    st.markdown('<div class="section-title">推荐组合</div>', unsafe_allow_html=True)
    rows = [leg_row(f"卖 Call ×{short_call_multiplier:g}", call, call_sell_price, call_qty, fee_pct)]
    if put is not None:
        rows.append(leg_row(f"买 Put ×{protective_put_multiplier:g}", put, put_buy_price, put_qty, fee_pct))
    else:
        rows.append({
            "方向": "不买 Put",
            "合约": "当前未触发：牛市指标未破或回撤未超过阈值时不买",
            "到期": expiry.strftime("%Y-%m-%d %H:%M UTC"),
            "DTE": f"{(expiry - datetime.now(timezone.utc)).total_seconds() / 86400.0:.2f}d",
            "Strike": "-",
            "Bid": "-",
            "Ask": "-",
            "Mark": "-",
            "Delta": "-",
            "IV": "-",
            "估算成交价": "-",
            "名义现金流": "$0",
            "手续费估算": "$0",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown('<div class="section-title">到期收益情景（相对当前账户）</div>', unsafe_allow_html=True)
    scenario_spots = sorted(
        {
            spot * 0.70,
            spot * 0.80,
            spot * 0.90,
            put.strike if put is not None else spot * 0.95,
            spot,
            call.strike,
            spot * 1.10,
            spot * 1.20,
            spot * 1.30,
        }
    )
    scenario_rows = []
    for s in scenario_spots:
        pnl = payoff_at_expiry(s, spot, qty, call_qty, put_qty, call, put, net_option_cash, funding_cost_usd)
        scenario_rows.append(
            {
                "到期BTC价格": fmt_usd(s),
                "BTC涨跌": fmt_pct(s / spot - 1.0),
                "组合PnL": fmt_usd(pnl),
                "账户收益率": fmt_pct(pnl / capital if capital > 0 else 0.0),
            }
        )
    st.dataframe(pd.DataFrame(scenario_rows), hide_index=True, use_container_width=True)

with right:
    st.markdown('<div class="section-title">潜在收益 / 风险</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(kpi_card("Call Cap", fmt_usd(call.strike), fmt_pct(call.strike / spot - 1.0), "warn"), unsafe_allow_html=True)
    if put is not None:
        with c2:
            st.markdown(kpi_card("Put Floor", fmt_usd(put.strike), fmt_pct(put.strike / spot - 1.0), "good"), unsafe_allow_html=True)
    else:
        with c2:
            st.markdown(kpi_card("Put Protection", "Off", "当前无保护", "neutral"), unsafe_allow_html=True)

    st.markdown(kpi_card("情景内最大收益", fmt_usd(max_gain_usd), fmt_pct(max_gain_pct), "good" if max_gain_usd >= 0 else "bad"), unsafe_allow_html=True)
    if put is not None:
        st.markdown(kpi_card("情景内最差PnL", fmt_usd(min_grid_usd), fmt_pct(min_grid_pct), "bad" if min_grid_usd < 0 else "good"), unsafe_allow_html=True)
    else:
        st.markdown(kpi_card("下方保护", "无", "下跌主要由 BTC 多头承担", "bad"), unsafe_allow_html=True)

    st.markdown('<div class="section-title">行情状态</div>', unsafe_allow_html=True)
    if vote_strength is not None and vote_count is not None:
        state = f"{vote_strength}（{vote_count}/7票）" + ("，启用保护 Put" if bear_signal else "，当前只卖 Call")
    else:
        state = "弱势/回撤，建议启用保护 Put" if bear_signal else "牛市指标未破，当前只卖 Call"
    st.markdown(signal_card("Regime", state, "warn" if bear_signal else "good"), unsafe_allow_html=True)
    if vote_detail_rows:
        vote_df = pd.DataFrame(vote_detail_rows)
        vote_df["通过"] = vote_df["通过"].map({True: "✅", False: "❌"})
        st.dataframe(vote_df, hide_index=True, use_container_width=True)
    st.dataframe(
        pd.DataFrame(
            [
                {"项目": "现价", "值": fmt_usd(spot)},
                {"项目": "牛市指标", "值": bull_indicator_label},
                {"项目": "指标状态", "值": bull_indicator_text},
                {"项目": "投票强弱", "值": f"{vote_strength or '-'}" + (f"（{vote_count}/7）" if vote_count is not None else "")},
                {"项目": "分层Call Delta", "值": f"{call_target_delta:.3f}" if selection_mode == "Delta" else "-"},
                {"项目": "趋势SMA", "值": fmt_usd(sma)},
                {"项目": "观察期高点", "值": fmt_usd(rolling_peak)},
                {"项目": "回撤", "值": f"{drawdown:.2%}"},
                {"项目": "当前资金费率/8h", "值": f"{funding_rate:.4%}"},
                {"项目": "预估资金费率成本", "值": fmt_usd(funding_cost_usd)},
                {"项目": "Call strike处PnL", "值": fmt_usd(call_strike_pnl)},
                {"项目": "BTC -30% PnL", "值": fmt_usd(down_30_pnl)},
                {"项目": "BTC +30% PnL", "值": fmt_usd(up_30_pnl)},
                {"项目": "实际到期", "值": expiry.strftime("%Y-%m-%d %H:%M UTC")},
                {"项目": "Bybit期权ticker数量", "值": f"{len(quotes)}"},
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown('<div class="section-title">收益曲线</div>', unsafe_allow_html=True)
    xs = payoff_grid_xs
    ys = payoff_grid_ys
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name="到期PnL",
            line=dict(color="#38bdf8", width=3),
            fill="tozeroy",
            fillcolor="rgba(56,189,248,0.10)",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(148,163,184,.75)")
    fig.add_vline(x=spot, line_dash="dot", line_color="#60a5fa", annotation_text="现价")
    fig.add_vline(x=call.strike, line_dash="dot", line_color="#f59e0b", annotation_text="Call")
    if put is not None:
        fig.add_vline(x=put.strike, line_dash="dot", line_color="#34d399", annotation_text="Put")
    fig.update_layout(
        height=390,
        margin=dict(l=10, r=10, t=22, b=10),
        xaxis_title="到期 BTC 价格",
        yaxis_title="PnL USD",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,.35)",
        font=dict(color="#cbd5e1"),
        xaxis=dict(gridcolor="rgba(148,163,184,.12)", zerolinecolor="rgba(148,163,184,.18)"),
        yaxis=dict(gridcolor="rgba(148,163,184,.12)", zerolinecolor="rgba(148,163,184,.18)"),
        hovermode="x unified",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption(
    "说明：Bybit 期权为 USDC/USD 原生报价；本面板只使用公开行情估算，资金费率按当前 BTCUSDT 永续 funding 外推；不含滑点、保证金占用变化、成交概率和交易所风控。用于决策辅助，不自动下单。"
)
