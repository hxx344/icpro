"""Black‑76 option pricing model.

Deribit uses the Black‑76 (Black futures) model for pricing options.
All prices are denominated in the underlying crypto (BTC / ETH).

Key formulas
────────────
  d1 = [ln(F/K) + σ²T/2] / (σ√T)
  d2 = d1 − σ√T

  Call = e^{-rT} [F·N(d1) − K·N(d2)]
  Put  = e^{-rT} [K·N(−d2) − F·N(−d1)]

For Deribit crypto options the risk‑free rate r is typically 0, so
  e^{-rT} = 1

The prices returned by this module are in USD; divide by the underlying
price F to convert to coin‑margined (BTC/ETH) denomination.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr as _ndtr


class _NormCompat:
    """Thin wrapper providing norm.cdf / norm.pdf via faster ndtr."""
    @staticmethod
    def cdf(x):
        return _ndtr(x)

    @staticmethod
    def pdf(x):
        return np.exp(-0.5 * np.asarray(x) ** 2) * (1.0 / np.sqrt(2.0 * np.pi))


norm = _NormCompat()


# ---------------------------------------------------------------------------
# Core pricing
# ---------------------------------------------------------------------------

def _d1_d2(F: float, K: float, T: float, sigma: float, r: float = 0.0):
    """Compute d1, d2 for Black‑76."""
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive")
    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def call_price(F: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Black‑76 call price (USD).

    Parameters
    ----------
    F : forward / index price (USD)
    K : strike price
    T : time to expiry in years
    sigma : implied volatility (decimal, e.g. 0.80 for 80 %)
    r : risk‑free rate (default 0)
    """
    if T <= 1e-10:
        return max(F - K, 0.0)
    d1, d2 = _d1_d2(F, K, T, sigma, r)
    df = np.exp(-r * T)
    return float(df * (F * norm.cdf(d1) - K * norm.cdf(d2)))


def put_price(F: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Black‑76 put price (USD)."""
    if T <= 1e-10:
        return max(K - F, 0.0)
    d1, d2 = _d1_d2(F, K, T, sigma, r)
    df = np.exp(-r * T)
    return float(df * (K * norm.cdf(-d2) - F * norm.cdf(-d1)))


def option_price(F: float, K: float, T: float, sigma: float,
                 option_type: str = "call", r: float = 0.0) -> float:
    """Convenience – dispatches to call or put."""
    if option_type.lower().startswith("c"):
        return call_price(F, K, T, sigma, r)
    return put_price(F, K, T, sigma, r)


def option_price_btc(F: float, K: float, T: float, sigma: float,
                     option_type: str = "call", r: float = 0.0) -> float:
    """Return the option price in BTC (= USD price / F)."""
    usd = option_price(F, K, T, sigma, option_type, r)
    return usd / F if F > 0 else 0.0


# ---------------------------------------------------------------------------
# Greeks (analytical, Black‑76)
# ---------------------------------------------------------------------------

def delta(F: float, K: float, T: float, sigma: float,
          option_type: str = "call", r: float = 0.0) -> float:
    """Option delta (d price / d F)."""
    if T <= 1e-10:
        if option_type.lower().startswith("c"):
            return 1.0 if F > K else 0.0
        return -1.0 if F < K else 0.0
    d1, _ = _d1_d2(F, K, T, sigma, r)
    df = np.exp(-r * T)
    if option_type.lower().startswith("c"):
        return float(df * norm.cdf(d1))
    return float(df * (norm.cdf(d1) - 1))


def gamma(F: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Option gamma (d² price / d F²). Same for call and put."""
    if T <= 1e-10:
        return 0.0
    d1, _ = _d1_d2(F, K, T, sigma, r)
    df = np.exp(-r * T)
    return float(df * norm.pdf(d1) / (F * sigma * np.sqrt(T)))


def vega(F: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Option vega (d price / d sigma). Same for call and put.

    Returned in USD per 1 unit of sigma (not per 1 %).
    """
    if T <= 1e-10:
        return 0.0
    d1, _ = _d1_d2(F, K, T, sigma, r)
    df = np.exp(-r * T)
    return float(df * F * norm.pdf(d1) * np.sqrt(T))


def theta(F: float, K: float, T: float, sigma: float,
          option_type: str = "call", r: float = 0.0) -> float:
    """Option theta (d price / d T, per year).

    To get daily theta divide by 365.
    """
    if T <= 1e-10:
        return 0.0
    d1, d2 = _d1_d2(F, K, T, sigma, r)
    df = np.exp(-r * T)
    sqrt_T = np.sqrt(T)

    # Time decay component (same for calls and puts)
    time_decay = -(df * F * norm.pdf(d1) * sigma) / (2 * sqrt_T)

    if option_type.lower().startswith("c"):
        cost_of_carry = r * df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        cost_of_carry = r * df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

    return float(time_decay - cost_of_carry)


def greeks(F: float, K: float, T: float, sigma: float,
           option_type: str = "call", r: float = 0.0) -> dict[str, float]:
    """Return all Greeks as a dict."""
    return {
        "delta": delta(F, K, T, sigma, option_type, r),
        "gamma": gamma(F, K, T, sigma, r),
        "theta": theta(F, K, T, sigma, option_type, r),
        "vega": vega(F, K, T, sigma, r),
    }


# ---------------------------------------------------------------------------
# Vectorised helpers (NumPy)
# ---------------------------------------------------------------------------

def call_price_vec(F, K, T, sigma, r=0.0):
    """Vectorised Black‑76 call price."""
    F, K, T, sigma = np.asarray(F, float), np.asarray(K, float), np.asarray(T, float), np.asarray(sigma, float)
    sqrt_T = np.sqrt(np.maximum(T, 1e-10))
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    df = np.exp(-r * T)
    price = df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    # For expired options set intrinsic
    expired = T <= 1e-10
    price = np.where(expired, np.maximum(F - K, 0), price)
    return price


def put_price_vec(F, K, T, sigma, r=0.0):
    """Vectorised Black‑76 put price."""
    F, K, T, sigma = np.asarray(F, float), np.asarray(K, float), np.asarray(T, float), np.asarray(sigma, float)
    sqrt_T = np.sqrt(np.maximum(T, 1e-10))
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    df = np.exp(-r * T)
    price = df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    expired = T <= 1e-10
    price = np.where(expired, np.maximum(K - F, 0), price)
    return price


def delta_vec(F, K, T, sigma, option_type_is_call=True, r=0.0):
    """Vectorised delta."""
    F, K, T, sigma = np.asarray(F, float), np.asarray(K, float), np.asarray(T, float), np.asarray(sigma, float)
    sqrt_T = np.sqrt(np.maximum(T, 1e-10))
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    df = np.exp(-r * T)
    is_call = np.asarray(option_type_is_call)
    d = np.where(is_call, df * norm.cdf(d1), df * (norm.cdf(d1) - 1))
    return d
