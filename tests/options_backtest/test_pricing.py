"""Tests for Black-76 pricing and Greeks."""

import math

import pytest
import numpy as np

from options_backtest.pricing.black76 import (
    call_price,
    put_price,
    option_price,
    option_price_btc,
    delta,
    gamma,
    vega,
    theta,
    greeks,
)
from options_backtest.pricing.iv_solver import implied_volatility


# ---------------------------------------------------------------------------
# Test parameters (typical BTC option scenario)
# ---------------------------------------------------------------------------
F = 80000.0    # BTC price
K = 85000.0    # Strike (OTM call)
T = 30 / 365   # 30 days to expiry
sigma = 0.60   # 60 % IV
r = 0.0        # zero rate


class TestBlack76Pricing:
    """Verify Black-76 prices are reasonable."""

    def test_call_price_positive(self):
        c = call_price(F, K, T, sigma, r)
        assert c > 0, "OTM call should have positive value"

    def test_put_price_positive(self):
        p = put_price(F, K, T, sigma, r)
        assert p > 0, "Put should have positive value"

    def test_put_call_parity(self):
        """C - P = e^{-rT} (F - K) for Black-76."""
        c = call_price(F, K, T, sigma, r)
        p = put_price(F, K, T, sigma, r)
        expected = math.exp(-r * T) * (F - K)
        assert abs((c - p) - expected) < 1e-6

    def test_atm_call_put_equal_at_zero_rate(self):
        """At r=0 and F=K, call = put."""
        c = call_price(F, F, T, sigma)
        p = put_price(F, F, T, sigma)
        assert abs(c - p) < 1e-6

    def test_deep_itm_call(self):
        K_itm = 50000.0
        c = call_price(F, K_itm, T, sigma)
        intrinsic = F - K_itm
        assert c >= intrinsic * 0.99  # should be at least intrinsic

    def test_expired_call(self):
        c = call_price(F, K, 0.0, sigma)
        assert c == max(F - K, 0)

    def test_expired_put(self):
        p = put_price(F, K, 0.0, sigma)
        assert p == max(K - F, 0)

    def test_coin_margined_price(self):
        c_btc = option_price_btc(F, K, T, sigma, "call")
        c_usd = call_price(F, K, T, sigma)
        assert abs(c_btc - c_usd / F) < 1e-10


class TestGreeks:
    """Verify Greeks have correct signs and magnitudes."""

    def test_call_delta_positive(self):
        d = delta(F, K, T, sigma, "call")
        assert 0 < d < 1

    def test_put_delta_negative(self):
        d = delta(F, K, T, sigma, "put")
        assert -1 < d < 0

    def test_call_put_delta_relation(self):
        """delta_call - delta_put = e^{-rT}"""
        dc = delta(F, K, T, sigma, "call")
        dp = delta(F, K, T, sigma, "put")
        assert abs((dc - dp) - math.exp(-r * T)) < 1e-6

    def test_gamma_positive(self):
        g = gamma(F, K, T, sigma)
        assert g > 0

    def test_vega_positive(self):
        v = vega(F, K, T, sigma)
        assert v > 0

    def test_call_theta_negative(self):
        """Theta for a long call should be negative (time decay)."""
        t = theta(F, K, T, sigma, "call")
        # With r=0, theta is always negative
        assert t < 0

    def test_greeks_dict(self):
        g = greeks(F, K, T, sigma, "call")
        assert set(g.keys()) == {"delta", "gamma", "theta", "vega"}


class TestIVSolver:
    """Test implied volatility inversion."""

    def test_roundtrip(self):
        """Price → IV → price should recover the original."""
        c = call_price(F, K, T, sigma)
        iv = implied_volatility(c, F, K, T, "call")
        assert abs(iv - sigma) < 1e-6

    def test_put_roundtrip(self):
        p = put_price(F, K, T, sigma)
        iv = implied_volatility(p, F, K, T, "put")
        assert abs(iv - sigma) < 1e-6

    def test_zero_price_returns_nan(self):
        iv = implied_volatility(0.0, F, K, T, "call")
        assert np.isnan(iv)

    def test_expired_returns_nan(self):
        iv = implied_volatility(100, F, K, 0.0, "call")
        assert np.isnan(iv)
