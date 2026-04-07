"""Tests for trader.bybit_client — Bybit 期权客户端单元测试."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from trader.bybit_client import (
    AccountInfo,
    BYBIT_SYMBOL_RE,
    BybitOptionsClient,
    OptionTicker,
    OrderNotFoundError,
    _parse_symbol,
)
from trader.config import ExchangeConfig


@pytest.fixture
def default_config():
    return ExchangeConfig(api_key="", api_secret="", simulate_private=True, testnet=False)


def test_parse_symbol_call():
    result = _parse_symbol("ETH-21MAR26-2000-C")
    assert result is not None
    assert result["underlying"] == "ETH"
    assert result["strike"] == 2000.0
    assert result["option_type"] == "call"


def test_parse_symbol_put():
    result = _parse_symbol("BTC-21MAR26-85000-P")
    assert result is not None
    assert result["underlying"] == "BTC"
    assert result["option_type"] == "put"


def test_symbol_regex_matches_bybit_format():
    match = BYBIT_SYMBOL_RE.match("ETH-21MAR26-2000-C")
    assert match is not None
    assert match.group("day") == "21"
    assert match.group("mon") == "MAR"
    assert match.group("yy") == "26"


def test_default_base_url(default_config):
    client = BybitOptionsClient(default_config)
    assert "api.bybit.com" in client._base


def test_testnet_base_url():
    client = BybitOptionsClient(ExchangeConfig(testnet=True, simulate_private=True))
    assert "api-testnet.bybit.com" in client._base


def test_simulated_account_returns_stub(default_config):
    client = BybitOptionsClient(default_config)
    account = client.get_account()
    assert isinstance(account, AccountInfo)
    assert account.total_balance == pytest.approx(1.0)
    assert account.raw["simulated"] is True


def test_simulated_public_market_calls_do_not_hit_network(default_config):
    client = BybitOptionsClient(default_config)
    client.session.get = MagicMock(side_effect=AssertionError("network should not be called"))

    assert client.get_spot_price("BTC") == pytest.approx(0.0)
    assert client.get_hourly_index_prices("BTC", limit=24) == []
    assert client.get_tickers("BTC") == []


def test_simulated_order_book_uses_ticker(default_config):
    client = BybitOptionsClient(default_config)
    client.get_ticker = MagicMock(
        return_value=OptionTicker(
            symbol="ETH-21MAR26-2000-C",
            underlying="ETH",
            strike=2000.0,
            option_type="call",
            expiry=_parse_symbol("ETH-21MAR26-2000-C")["expiry"],
            bid_price=50.0,
            ask_price=55.0,
            mark_price=52.5,
            last_price=52.5,
            underlying_price=2000.0,
            volume_24h=1.0,
            open_interest=1.0,
        )
    )

    book = client.get_order_book("ETH-21MAR26-2000-C")
    assert book["bids"][0][0] == pytest.approx(50.0)
    assert book["asks"][0][0] == pytest.approx(55.0)


def test_simulated_market_order_returns_filled(default_config):
    client = BybitOptionsClient(default_config)
    result = client.place_order("ETH-21MAR26-2000-C", "SELL", 1.0)
    assert result.status == "FILLED"
    assert result.raw["simulated"] is True


def test_legacy_order_type_argument_is_removed(default_config):
    client = BybitOptionsClient(default_config)
    with pytest.raises(TypeError):
        client.place_order("ETH-21MAR26-2000-C", "SELL", 1.0, order_type="LIMIT")


def test_legacy_price_argument_is_removed(default_config):
    client = BybitOptionsClient(default_config)
    with pytest.raises(TypeError):
        client.submit_order("ETH-21MAR26-2000-C", "SELL", 1.0, price=100.0)


def test_query_order_not_found_raises():
    client = BybitOptionsClient(ExchangeConfig(api_key="x", api_secret="y", simulate_private=False))
    with patch.object(client, "_private_get", return_value={"result": {"list": []}}):
        with pytest.raises(OrderNotFoundError):
            client.query_order("ETH-21MAR26-2000-C", client_order_id="CID-1")


def test_private_get_signature_uses_request_param_order():
    client = BybitOptionsClient(ExchangeConfig(api_key="key", api_secret="secret", simulate_private=False))
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"retCode": 0, "result": {"list": []}}

    with patch("trader.bybit_client.time.time", return_value=1700000000.123), patch.object(client, "_sign", wraps=client._sign) as sign_mock:
        client.session.get = MagicMock(return_value=response)
        client._private_get("/v5/position/list", {"category": "option", "baseCoin": "BTC", "cursor": None})

    sign_mock.assert_called_once_with("category=option&baseCoin=BTC", 1700000000123)
    client.session.get.assert_called_once()
    assert client.session.get.call_args.kwargs["params"] == {"category": "option", "baseCoin": "BTC"}