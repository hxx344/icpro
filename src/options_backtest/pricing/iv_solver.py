"""Implied volatility solver using Black‑76.

Uses `scipy.optimize.brentq` (Brent's method) to invert the Black‑76
pricing formula and recover the implied volatility from a market price.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from options_backtest.pricing.black76 import option_price


def implied_volatility(
    market_price_usd: float,
    F: float,
    K: float,
    T: float,
    option_type: str = "call",
    r: float = 0.0,
    lower: float = 0.01,
    upper: float = 10.0,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> float:
    """Solve for implied volatility (decimal) given a market price (USD).

    Parameters
    ----------
    market_price_usd : observed option price in USD
    F : forward / index price (USD)
    K : strike price
    T : time to expiry (years)
    option_type : "call" or "put"
    r : risk-free rate
    lower, upper : IV search bounds (decimal)
    tol : solver tolerance
    max_iter : max iterations

    Returns
    -------
    Implied volatility as a decimal (e.g. 0.80 = 80 %).

    Raises
    ------
    ValueError
        If the solver cannot bracket a root.
    """
    if T <= 1e-10 or market_price_usd <= 0:
        return np.nan

    def objective(sigma: float) -> float:
        return option_price(F, K, T, sigma, option_type, r) - market_price_usd

    try:
        return float(brentq(objective, lower, upper, xtol=tol, maxiter=max_iter))
    except ValueError:
        # Root not bracketed – price may be outside valid range
        return np.nan


def implied_volatility_btc(
    market_price_btc: float,
    F: float,
    K: float,
    T: float,
    option_type: str = "call",
    r: float = 0.0,
    **kwargs,
) -> float:
    """Same as `implied_volatility` but accepts a coin‑margined price.

    Converts BTC‑denominated price to USD via ``price_usd = price_btc × F``.
    """
    return implied_volatility(market_price_btc * F, F, K, T, option_type, r, **kwargs)
