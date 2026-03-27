"""Tests for trader.binance_client — Binance 期权客户端单元测试.

覆盖:
  - 数据模型 (OptionTicker / OrderResult / AccountInfo)
  - Symbol 解析 (_parse_symbol / BINANCE_SYMBOL_RE)
  - HTTP 签名 (_sign)
  - 市场数据 (get_spot_price, get_tickers, get_mark_prices) — mock HTTP
  - 账户 (get_account) — mock HTTP + simulate 模式
  - 下单 (place_order, close_position) — mock HTTP + simulate 模式
  - USDT→币本位价格转换
  - 错误处理与边界条件
"""

from __future__ import annotations

import hashlib
import hmac
import re
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from trader.binance_client import (
    BinanceOptionsClient,
    OptionTicker,
    OrderResult,
    AccountInfo,
    BINANCE_SYMBOL_RE,
    _parse_symbol,
)
from trader.config import ExchangeConfig


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def default_config():
    """默认 testnet 模拟配置."""
    return ExchangeConfig(
        api_key="test_api_key",
        api_secret="test_api_secret",
        testnet=True,
        simulate_private=False,
    )


@pytest.fixture
def sim_config():
    """simulate_private=True 模拟配置 (不发真实请求)."""
    return ExchangeConfig(
        api_key="",
        api_secret="",
        testnet=True,
        simulate_private=True,
    )


@pytest.fixture
def client(default_config):
    return BinanceOptionsClient(default_config)


@pytest.fixture
def sim_client(sim_config):
    return BinanceOptionsClient(sim_config)


# ======================================================================
# 1. Symbol 解析
# ======================================================================


class TestParseSymbol:
    """_parse_symbol 和 BINANCE_SYMBOL_RE 测试."""

    def test_parse_eth_call(self):
        r = _parse_symbol("ETH-260321-2000-C")
        assert r is not None
        assert r["underlying"] == "ETH"
        assert r["strike"] == 2000.0
        assert r["option_type"] == "call"
        assert r["expiry"] == datetime(2026, 3, 21, 8, 0, 0, tzinfo=timezone.utc)

    def test_parse_btc_put(self):
        r = _parse_symbol("BTC-260321-85000-P")
        assert r is not None
        assert r["underlying"] == "BTC"
        assert r["strike"] == 85000.0
        assert r["option_type"] == "put"

    def test_parse_decimal_strike(self):
        r = _parse_symbol("ETH-260321-2000.5-C")
        assert r is not None
        assert r["strike"] == 2000.5

    def test_parse_different_dates(self):
        """验证不同日期的解析正确性."""
        r = _parse_symbol("ETH-250101-3000-P")
        assert r is not None
        assert r["expiry"].year == 2025
        assert r["expiry"].month == 1
        assert r["expiry"].day == 1

    def test_reject_deribit_format(self):
        """Deribit 格式不应该被解析."""
        assert _parse_symbol("ETH-28MAR26-2200-C") is None

    def test_reject_empty(self):
        assert _parse_symbol("") is None

    def test_reject_garbage(self):
        assert _parse_symbol("not-a-symbol") is None

    def test_reject_missing_cp(self):
        assert _parse_symbol("ETH-260321-2000") is None

    def test_reject_lowercase(self):
        """小写字母不应被解析."""
        assert _parse_symbol("eth-260321-2000-C") is None

    def test_reject_bad_cp(self):
        assert _parse_symbol("ETH-260321-2000-X") is None

    def test_regex_pattern_groups(self):
        """验证正则表达式分组命名."""
        m = BINANCE_SYMBOL_RE.match("ETH-260321-2000-C")
        assert m is not None
        assert m.group("ul") == "ETH"
        assert m.group("yymmdd") == "260321"
        assert m.group("strike") == "2000"
        assert m.group("cp") == "C"

    def test_expiry_at_8_utc(self):
        """Binance 期权在 08:00 UTC 结算."""
        r = _parse_symbol("ETH-260321-2000-C")
        assert r["expiry"].hour == 8
        assert r["expiry"].minute == 0
        assert r["expiry"].tzinfo == timezone.utc


# ======================================================================
# 2. 数据模型
# ======================================================================


class TestOptionTicker:
    """OptionTicker dataclass 属性测试."""

    def _make_ticker(self, **overrides):
        defaults = dict(
            symbol="ETH-260321-2000-C",
            underlying="ETH",
            strike=2000.0,
            option_type="call",
            expiry=datetime(2026, 3, 21, 8, 0, tzinfo=timezone.utc),
            bid_price=0.05,
            ask_price=0.06,
            mark_price=0.055,
            last_price=0.054,
            underlying_price=2500.0,
            volume_24h=100.0,
            open_interest=500.0,
        )
        defaults.update(overrides)
        return OptionTicker(**defaults)

    def test_mid_price_normal(self):
        t = self._make_ticker(bid_price=0.04, ask_price=0.06)
        assert t.mid_price == pytest.approx(0.05)

    def test_mid_price_fallback_to_mark(self):
        """bid 或 ask = 0 时 mid_price 回退到 mark_price."""
        t = self._make_ticker(bid_price=0.0, ask_price=0.0, mark_price=0.055)
        assert t.mid_price == pytest.approx(0.055)

    def test_spread_normal(self):
        t = self._make_ticker(bid_price=0.04, ask_price=0.06)
        assert t.spread == pytest.approx(0.02)

    def test_spread_no_quotes(self):
        t = self._make_ticker(bid_price=0.0, ask_price=0.0)
        assert t.spread == 0.0

    def test_moneyness_pct(self):
        """ETH spot=2500, strike=2000 → moneyness=(2000/2500-1)*100=-20%"""
        t = self._make_ticker(strike=2000.0, underlying_price=2500.0)
        assert t.moneyness_pct == pytest.approx(-20.0)

    def test_moneyness_pct_zero_spot(self):
        t = self._make_ticker(underlying_price=0.0)
        assert t.moneyness_pct == 0.0

    def test_dte_hours_future(self):
        """到期日在未来时 dte_hours > 0."""
        future_expiry = datetime.now(timezone.utc) + timedelta(hours=12)
        t = self._make_ticker(expiry=future_expiry)
        assert t.dte_hours > 11.0
        assert t.dte_hours < 13.0

    def test_dte_hours_past(self):
        """到期日在过去时 dte_hours = 0."""
        past_expiry = datetime.now(timezone.utc) - timedelta(hours=1)
        t = self._make_ticker(expiry=past_expiry)
        assert t.dte_hours == 0.0


class TestOrderResult:
    def test_fields(self):
        o = OrderResult(
            order_id="123",
            symbol="ETH-260321-2000-C",
            side="SELL",
            quantity=1.0,
            price=0.05,
            avg_price=0.05,
            status="FILLED",
            fee=0.001,
            raw={"test": True},
        )
        assert o.order_id == "123"
        assert o.status == "FILLED"
        assert o.fee == 0.001


class TestAccountInfo:
    def test_fields(self):
        a = AccountInfo(
            total_balance=10.0,
            available_balance=8.0,
            unrealized_pnl=0.5,
            raw={},
        )
        assert a.total_balance == 10.0
        assert a.available_balance == 8.0


# ======================================================================
# 3. Client 初始化
# ======================================================================


class TestClientInit:
    def test_testnet_url(self, default_config):
        c = BinanceOptionsClient(default_config)
        assert "testnet" in c._base

    def test_prod_url(self):
        cfg = ExchangeConfig(testnet=False)
        c = BinanceOptionsClient(cfg)
        assert "eapi.binance.com" in c._base

    def test_api_key_header(self, default_config):
        c = BinanceOptionsClient(default_config)
        assert c.session.headers.get("X-MBX-APIKEY") == "test_api_key"

    def test_no_api_key_no_header(self):
        cfg = ExchangeConfig(api_key="", testnet=True)
        c = BinanceOptionsClient(cfg)
        assert "X-MBX-APIKEY" not in c.session.headers


# ======================================================================
# 4. HMAC 签名
# ======================================================================


class TestSign:
    def test_sign_adds_timestamp_and_signature(self, client):
        params = {"symbol": "ETH-260321-2000-C"}
        signed = client._sign(params)
        assert "timestamp" in signed
        assert "signature" in signed
        assert "recvWindow" in signed
        assert signed["recvWindow"] == 5000

    def test_sign_hmac_correctness(self, client):
        """验证签名计算正确."""
        params = {"symbol": "TEST"}
        signed = client._sign(params)
        # Rebuild expected signature
        ts = signed["timestamp"]
        recv = signed["recvWindow"]
        query = f"symbol=TEST&timestamp={ts}&recvWindow={recv}"
        expected = hmac.new(
            b"test_api_secret", query.encode(), hashlib.sha256
        ).hexdigest()
        assert signed["signature"] == expected


# ======================================================================
# 5. Market Data (mocked HTTP)
# ======================================================================


class TestGetSpotPrice:
    def test_success(self, client):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"indexPrice": "2500.50"}
        mock_resp.raise_for_status = MagicMock()
        with patch.object(client.session, "get", return_value=mock_resp):
            price = client.get_spot_price("ETH")
        assert price == pytest.approx(2500.50)

    def test_error_returns_zero(self, client):
        with patch.object(client.session, "get", side_effect=Exception("timeout")):
            price = client.get_spot_price("ETH")
        assert price == 0.0

    def test_btc(self, client):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"indexPrice": "85000.0"}
        mock_resp.raise_for_status = MagicMock()
        with patch.object(client.session, "get", return_value=mock_resp) as m:
            price = client.get_spot_price("BTC")
            # 验证请求参数包含 BTCUSDT
            call_kwargs = m.call_args
            assert "BTCUSDT" in str(call_kwargs)
        assert price == pytest.approx(85000.0)


class TestGetTickers:
    """get_tickers 应返回 OptionTicker 列表，且期权价格保持 Binance 原生 USD 报价。"""

    def _mock_ticker_data(self):
        return [
            {
                "symbol": "ETH-260321-2000-C",
                "exercisePrice": "2500",
                "bidPrice": "125",    # 125 USDT
                "askPrice": "150",    # 150 USDT
                "lastPrice": "130",
                "volume": "50",
                "amount": "200",
            },
            {
                "symbol": "ETH-260321-2000-P",
                "exercisePrice": "2500",
                "bidPrice": "80",
                "askPrice": "100",
                "lastPrice": "90",
                "volume": "30",
                "amount": "150",
            },
            {
                "symbol": "BTC-260321-85000-C",  # 应被 ETH 过滤掉
                "exercisePrice": "85000",
                "bidPrice": "1000",
                "askPrice": "1200",
                "lastPrice": "1100",
                "volume": "10",
                "amount": "50",
            },
        ]

    def test_returns_only_underlying(self, client):
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._mock_ticker_data()
        mock_resp.raise_for_status = MagicMock()
        with patch.object(client.session, "get", return_value=mock_resp):
            tickers = client.get_tickers("ETH")
        # 只返回 ETH 的两个
        assert len(tickers) == 2
        assert all(t.underlying == "ETH" for t in tickers)

    def test_preserves_native_usd_prices(self, client):
        """Binance 期权盘口价格应保持原生 USD，不再除以 spot。"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._mock_ticker_data()
        mock_resp.raise_for_status = MagicMock()
        with patch.object(client.session, "get", return_value=mock_resp):
            tickers = client.get_tickers("ETH")
        call_ticker = [t for t in tickers if t.option_type == "call"][0]
        assert call_ticker.bid_price == pytest.approx(125.0)
        assert call_ticker.ask_price == pytest.approx(150.0)

    def test_mark_from_mid(self, client):
        """缺少 markPrice 时，mark_price 应取原生 USD 中间价。"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._mock_ticker_data()
        mock_resp.raise_for_status = MagicMock()
        with patch.object(client.session, "get", return_value=mock_resp):
            tickers = client.get_tickers("ETH")
        call_ticker = [t for t in tickers if t.option_type == "call"][0]
        assert call_ticker.mark_price == pytest.approx(137.5)

    def test_empty_on_error(self, client):
        with patch.object(client.session, "get", side_effect=Exception("fail")):
            tickers = client.get_tickers("ETH")
        assert tickers == []

    def test_zero_spot_still_preserves_option_prices(self, client):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{
            "symbol": "ETH-260321-2000-C",
            "exercisePrice": "0",
            "bidPrice": "100",
            "askPrice": "120",
            "lastPrice": "110",
            "volume": "10",
            "amount": "20",
        }]
        mock_resp.raise_for_status = MagicMock()
        with patch.object(client.session, "get", return_value=mock_resp):
            tickers = client.get_tickers("ETH")
        assert len(tickers) == 1
        assert tickers[0].bid_price == pytest.approx(100.0)
        assert tickers[0].underlying_price == pytest.approx(0.0)

    def test_non_list_response(self, client):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"error": "rate limit"}
        mock_resp.raise_for_status = MagicMock()
        with patch.object(client.session, "get", return_value=mock_resp):
            tickers = client.get_tickers("ETH")
        assert tickers == []


class TestGetMarkPrices:
    def test_returns_native_usd_prices(self, client):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"symbol": "ETH-260321-2000-C", "markPrice": "125", "underlyingPrice": "2500"},
            {"symbol": "ETH-260321-2000-P", "markPrice": "80", "underlyingPrice": "2500"},
            {"symbol": "BTC-260321-85000-C", "markPrice": "1000", "underlyingPrice": "85000"},
        ]
        mock_resp.raise_for_status = MagicMock()
        with patch.object(client.session, "get", return_value=mock_resp):
            prices = client.get_mark_prices("ETH")
        assert len(prices) == 2
        assert prices["ETH-260321-2000-C"] == pytest.approx(125.0)
        assert prices["ETH-260321-2000-P"] == pytest.approx(80.0)

    def test_error_returns_empty(self, client):
        with patch.object(client.session, "get", side_effect=Exception("fail")):
            prices = client.get_mark_prices("ETH")
        assert prices == {}


class TestGetGreeks:
    """get_greeks 和 enrich_greeks 测试."""

    def test_returns_delta_and_iv(self, client):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"symbol": "ETH-260321-2000-C", "markPrice": "125", "underlyingPrice": "2500",
             "delta": "0.55", "markIV": "0.72"},
            {"symbol": "ETH-260321-2000-P", "markPrice": "80", "underlyingPrice": "2500",
             "delta": "-0.45", "markIV": "0.70"},
            {"symbol": "BTC-260321-85000-C", "markPrice": "1000", "underlyingPrice": "85000",
             "delta": "0.60", "markIV": "0.50"},
        ]
        mock_resp.raise_for_status = MagicMock()
        with patch.object(client.session, "get", return_value=mock_resp):
            greeks = client.get_greeks("ETH")
        assert len(greeks) == 2
        assert greeks["ETH-260321-2000-C"]["delta"] == pytest.approx(0.55)
        assert greeks["ETH-260321-2000-C"]["mark_iv"] == pytest.approx(0.72)
        assert greeks["ETH-260321-2000-P"]["delta"] == pytest.approx(-0.45)

    def test_error_returns_empty(self, client):
        with patch.object(client.session, "get", side_effect=Exception("fail")):
            greeks = client.get_greeks("ETH")
        assert greeks == {}

    def test_enrich_greeks_populates_tickers(self, client):
        """enrich_greeks 应将 delta/mark_iv 填入 tickers."""
        tickers = [
            OptionTicker(
                symbol="ETH-260321-2000-C", underlying="ETH", strike=2000.0,
                option_type="call",
                expiry=datetime(2026, 3, 21, 8, 0, tzinfo=timezone.utc),
                bid_price=0.05, ask_price=0.06, mark_price=0.055,
                last_price=0.054, underlying_price=2500.0,
                volume_24h=100.0, open_interest=500.0,
            ),
        ]
        assert tickers[0].delta == 0.0  # before
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"symbol": "ETH-260321-2000-C", "delta": "0.55", "markIV": "0.72"},
        ]
        mock_resp.raise_for_status = MagicMock()
        with patch.object(client.session, "get", return_value=mock_resp):
            client.enrich_greeks(tickers, "ETH")
        assert tickers[0].delta == pytest.approx(0.55)
        assert tickers[0].mark_iv == pytest.approx(0.72)

    def test_enrich_greeks_no_match(self, client):
        """没有匹配的 symbol 时不修改 ticker."""
        tickers = [
            OptionTicker(
                symbol="ETH-260321-2000-C", underlying="ETH", strike=2000.0,
                option_type="call",
                expiry=datetime(2026, 3, 21, 8, 0, tzinfo=timezone.utc),
                bid_price=0.05, ask_price=0.06, mark_price=0.055,
                last_price=0.054, underlying_price=2500.0,
                volume_24h=100.0, open_interest=500.0,
            ),
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"symbol": "ETH-260321-3000-C", "delta": "0.20", "markIV": "0.80"},
        ]
        mock_resp.raise_for_status = MagicMock()
        with patch.object(client.session, "get", return_value=mock_resp):
            client.enrich_greeks(tickers, "ETH")
        assert tickers[0].delta == 0.0  # unchanged


# ======================================================================
# 6. Account
# ======================================================================


class TestGetAccount:
    def test_simulate_returns_defaults(self, sim_client):
        """simulate 模式返回 balance=1.0."""
        acct = sim_client.get_account()
        assert acct.total_balance == 1.0
        assert acct.available_balance == 1.0
        assert acct.unrealized_pnl == 0.0
        assert acct.raw.get("simulated") is True

    def test_no_credentials_returns_simulated(self):
        """无 API key 时也返回模拟数据."""
        cfg = ExchangeConfig(api_key="", api_secret="", simulate_private=False)
        c = BinanceOptionsClient(cfg)
        acct = c.get_account()
        assert acct.raw.get("simulated") is True

    def test_real_account(self, client):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "asset": [
                {"marginBalance": "10.0", "availableBalance": "8.0", "unrealizedPNL": "0.5"},
                {"marginBalance": "5.0", "availableBalance": "3.0", "unrealizedPNL": "-0.1"},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        with patch.object(client.session, "get", return_value=mock_resp) as mock_get:
            acct = client.get_account()
        assert acct.total_balance == pytest.approx(15.0)
        assert acct.available_balance == pytest.approx(11.0)
        assert acct.unrealized_pnl == pytest.approx(0.4)
        # Verify correct endpoint is called
        call_url = mock_get.call_args[1].get("url", "") if mock_get.call_args[1] else mock_get.call_args[0][0] if mock_get.call_args[0] else ""
        # URL is constructed as keyword arg
        called_url = str(mock_get.call_args)
        assert "marginAccount" in called_url, f"Should call /eapi/v1/marginAccount, got: {called_url}"

    def test_real_account_available_field(self, client):
        """Binance EAPI uses 'available' not 'availableBalance'."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "asset": [
                {"marginBalance": "10.0", "available": "8.0", "unrealizedPNL": "0.5"},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        with patch.object(client.session, "get", return_value=mock_resp):
            acct = client.get_account()
        assert acct.available_balance == pytest.approx(8.0)

    def test_account_error(self, client):
        with patch.object(client.session, "get", side_effect=Exception("conn")):
            acct = client.get_account()
        assert acct.total_balance == 0.0
        assert "error" in acct.raw


# ======================================================================
# 7. Order placement
# ======================================================================


class TestPlaceOrder:
    @staticmethod
    def _quote():
        return OptionTicker(
            symbol="ETH-260321-2000-C",
            underlying="ETH",
            strike=2000.0,
            option_type="call",
            expiry=datetime(2026, 3, 21, 8, 0, tzinfo=timezone.utc),
            bid_price=115.0,
            ask_price=120.0,
            mark_price=117.5,
            last_price=118.0,
            underlying_price=2500.0,
            volume_24h=10.0,
            open_interest=20.0,
        )

    def test_simulate_order(self, sim_client):
        result = sim_client.place_order("ETH-260321-2000-C", "SELL", 1.0)
        assert result.status == "FILLED"
        assert result.order_id.startswith("SIM-")
        assert result.raw.get("simulated") is True

    def test_real_order(self, client):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "orderId": "98765",
            "status": "FILLED",
            "executedQty": "1.0",
            "price": "125.0",
            "avgPrice": "125.5",
            "fee": "0.03",
        }
        mock_resp.raise_for_status = MagicMock()
        with patch.object(client.session, "post", return_value=mock_resp):
            with patch.object(client, "get_ticker", return_value=self._quote()):
                result = client.place_order("ETH-260321-2000-C", "SELL", 1.0)
        assert result.order_id == "98765"
        assert result.avg_price == pytest.approx(125.5)
        assert result.fee == pytest.approx(0.03)
        assert result.status == "FILLED"

    def test_limit_order_params(self, client):
        """LIMIT 订单应包含 price 和 timeInForce."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "orderId": "111",
            "status": "NEW",
            "executedQty": "0",
            "price": "100.0",
            "avgPrice": "0",
            "fee": "0",
        }
        mock_resp.raise_for_status = MagicMock()
        with patch.object(client.session, "post", return_value=mock_resp) as m:
            client.place_order(
                "ETH-260321-2000-C", "BUY", 1.0,
                order_type="LIMIT", price=100.0,
            )
            call_kwargs = m.call_args
            params = call_kwargs.kwargs.get("params") or call_kwargs[1].get("params", {})
            assert params.get("price") == "100.0" or "100.0" in str(params)
            assert params.get("timeInForce") == "GTC"

    def test_market_order_uses_synthetic_limit_ioc(self, client):
        """MARKET 订单应转成对手一档的 LIMIT IOC。"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "orderId": "222",
            "status": "FILLED",
            "executedQty": "1.0",
            "price": "120.0",
            "avgPrice": "120.0",
            "fee": "0.01",
        }
        mock_resp.raise_for_status = MagicMock()
        with patch.object(client.session, "post", return_value=mock_resp) as m:
            with patch.object(client, "get_ticker", return_value=self._quote()):
                client.place_order(
                    "ETH-260321-2000-C", "SELL", 1.0,
                    order_type="MARKET",
                )
            call_kwargs = m.call_args
            params = call_kwargs.kwargs.get("params") or call_kwargs[1].get("params", {})
            assert params.get("type") == "LIMIT"
            assert params.get("price") == "115.0"
            assert params.get("timeInForce") == "IOC"

    def test_market_order_buy_uses_ask_price(self, client):
        """BUY 的 synthetic market 应使用 ask1。"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "orderId": "333",
            "status": "FILLED",
            "executedQty": "1.0",
            "price": "120.0",
            "avgPrice": "120.0",
            "fee": "0.01",
        }
        mock_resp.raise_for_status = MagicMock()
        with patch.object(client.session, "post", return_value=mock_resp) as m:
            with patch.object(client, "get_ticker", return_value=self._quote()):
                client.place_order(
                    "ETH-260321-2000-C", "BUY", 1.0,
                    order_type="MARKET",
                )
            call_kwargs = m.call_args
            params = call_kwargs.kwargs.get("params") or call_kwargs[1].get("params", {})
            assert params.get("price") == "120.0"

    def test_market_order_without_quote_raises(self, client):
        """synthetic market 若拿不到盘口，应明确失败。"""
        with patch.object(client, "get_ticker", return_value=None):
            with pytest.raises(RuntimeError, match="No live quote available"):
                client.place_order(
                    "ETH-260321-2000-C", "BUY", 1.0,
                    order_type="MARKET",
                )

    def test_order_error_raises(self, client):
        with patch.object(client.session, "post", side_effect=Exception("failed")):
            with patch.object(client, "get_ticker", return_value=self._quote()):
                with pytest.raises(Exception, match="failed"):
                    client.place_order("ETH-260321-2000-C", "SELL", 1.0)


class TestClosePosition:
    def test_close_long(self, sim_client):
        """关闭多头 → 发 SELL 单."""
        result = sim_client.close_position("ETH-260321-2000-C", "LONG", 1.0)
        assert result.side == "SELL"

    def test_close_short(self, sim_client):
        """关闭空头 → 发 BUY 单."""
        result = sim_client.close_position("ETH-260321-2000-C", "SHORT", 1.0)
        assert result.side == "BUY"


# ======================================================================
# 8. Binance API 错误码处理
# ======================================================================


class TestBinanceErrorHandling:
    def test_api_error_code_raises(self, client):
        """Binance 返回 code < 0 应抛出异常."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"code": -1000, "msg": "Invalid parameter"}
        mock_resp.raise_for_status = MagicMock()
        with patch.object(client.session, "get", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="Binance error"):
                client._public_get("/eapi/v1/index")
