"""
src/models/black_scholes.py
---------------------------
Objective 1 -- Classical Black-Scholes option pricing.

Implements the standard Black-Scholes-Merton model for European call and put
options and prices the observations produced by the preprocessing module.

Follows the methodology of:
  "Pricing options with a new hybrid neural network model"
  Shvimer & Zhu, Expert Systems With Applications, 251 (2024), 123979.

Model inputs, per the preprocessing handoff schema:
  S      -- underlying asset price
  strike -- strike price K
  T      -- time to expiration in years
  sigma  -- previous trading day's VIX close (in VIX POINTS, i.e. percent)
  r      -- maturity-matched risk-free rate (already a decimal, e.g. 0.0556)

Important unit convention
-------------------------
The VIX is quoted in percentage points (e.g. 17.2 == 17.2% annualised
volatility). Black-Scholes requires sigma as a decimal, so this module divides
the VIX level by 100 (sigma_decimal = VIX / 100). The risk-free rate `r`
supplied by the preprocessing module is already a decimal and is used as-is.

Dividends
---------
No dividend yield is available in the dataset, so the standard non-dividend
Black-Scholes form is used (q = 0). This is documented as the least-invasive
choice consistent with the available features.
"""

import numpy as np
from scipy.stats import norm

# VIX is quoted in percentage points; Black-Scholes needs a decimal.
VIX_TO_DECIMAL = 100.0


def _as_array(x):
    """Return a float numpy array (scalars become 0-d -> 1-d friendly)."""
    return np.asarray(x, dtype=float)


def d1_d2(S, K, T, sigma, r):
    """
    Compute the Black-Scholes d1 and d2 terms.

        d1 = [ln(S/K) + (r + sigma^2 / 2) * T] / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

    `sigma` here is the volatility as a DECIMAL (already converted from VIX).
    Arrays are supported element-wise. Where sigma*sqrt(T) == 0 the result is
    undefined; callers should handle T <= 0 or sigma <= 0 separately.
    """
    S = _as_array(S)
    K = _as_array(K)
    T = _as_array(T)
    sigma = _as_array(sigma)
    r = _as_array(r)

    vol = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / vol
        d2 = d1 - vol
    return d1, d2


def bs_call_price(S, K, T, sigma, r):
    """
    European call price under Black-Scholes (no dividends).

        C = S * N(d1) - K * exp(-r * T) * N(d2)

    `sigma` is a DECIMAL volatility. Use `price_from_vix` to price directly
    from the VIX-point `sigma` column of the datasets.
    """
    S = _as_array(S)
    K = _as_array(K)
    T = _as_array(T)
    r = _as_array(r)
    d1, d2 = d1_d2(S, K, T, sigma, r)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, T, sigma, r):
    """
    European put price under Black-Scholes (no dividends).

        P = K * exp(-r * T) * N(-d2) - S * N(-d1)

    `sigma` is a DECIMAL volatility.
    """
    S = _as_array(S)
    K = _as_array(K)
    T = _as_array(T)
    r = _as_array(r)
    d1, d2 = d1_d2(S, K, T, sigma, r)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_price(S, K, T, sigma, r, option_type):
    """
    Price calls or puts element-wise based on `option_type`.

    `option_type` may be a scalar or array of strings; values are matched
    case-insensitively against 'CALL'/'C' and 'PUT'/'P'. `sigma` is a DECIMAL.
    """
    opt = np.asarray(option_type)
    is_call = np.char.upper(opt.astype(str))
    call_mask = np.isin(is_call, ["CALL", "C"])
    put_mask = np.isin(is_call, ["PUT", "P"])

    if not np.all(call_mask | put_mask):
        bad = np.unique(opt[~(call_mask | put_mask)])
        raise ValueError(f"Unrecognised option_type value(s): {bad.tolist()}")

    call = bs_call_price(S, K, T, sigma, r)
    put = bs_put_price(S, K, T, sigma, r)
    return np.where(call_mask, call, put)


def price_from_vix(S, K, T, vix_points, r, option_type):
    """
    Price options directly from the dataset columns, converting the VIX-point
    `sigma` column to a decimal volatility (VIX / 100) before pricing.

    This is the convenience entry point for the evaluation pipeline: pass the
    raw `sigma`, `r`, and `option_type` columns straight from the CSVs.
    """
    sigma_decimal = _as_array(vix_points) / VIX_TO_DECIMAL
    return bs_price(S, K, T, sigma_decimal, r, option_type)
