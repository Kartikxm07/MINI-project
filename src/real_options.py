"""
Real Indian option data from NSE's official F&O Bhavcopy (end-of-day settlement),
downloaded via the `jugaad-data` library's access to NSE archives.

This replaces the synthetic option dataset with **observed market prices** for a
chosen NSE underlying (e.g. RELIANCE, or an index like NIFTY), so Objective 1
("evaluate Black-Scholes against observed market option prices") and the ML
models can be run on real contracts.

Caveats: Bhavcopy is end-of-day *settlement* data (not live bid/ask); we filter
out illiquid/zero-volume strikes. The risk-free rate is a constant assumption
(no rate feed). Pulling several recent trading days gives a cross-section across
strikes/expiries (and a little cross-day variation), not a long out-of-time test.
"""

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.pricing import black_scholes_call

# NSE UDiFF bhavcopy instrument types: STO=stock option, IDO=index option.
OPTION_TYPES = ("STO", "IDO")
# Index underlyings need their Yahoo index symbol for historical volatility.
INDEX_YF = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "FINNIFTY": "^CNXFIN"}


def download_bhavcopies(days_back: int = 10, cache_dir: str = "data/real/bhavcopy"):
    """Download the most recent `days_back` trading days of F&O bhavcopy. Returns
    a concatenated DataFrame (cached per day; weekends/holidays skipped)."""
    from jugaad_data.nse import bhavcopy_fo_save
    os.makedirs(cache_dir, exist_ok=True)
    frames, got = [], 0
    probe = 0
    while got < days_back and probe < days_back * 3 + 10:
        day = date.today() - timedelta(days=probe)
        probe += 1
        if day.weekday() >= 5:
            continue
        try:
            path = bhavcopy_fo_save(day, cache_dir)
        except Exception:
            continue  # holiday or not yet published
        try:
            df = pd.read_csv(path)
            frames.append(df)
            got += 1
        except Exception:
            continue
    if not frames:
        raise RuntimeError("No F&O bhavcopy could be downloaded (NSE archive unreachable?).")
    return pd.concat(frames, ignore_index=True)


def _historical_vol(underlying_yf: str, as_of: pd.Timestamp, window: int = 60) -> float:
    """Trailing annualised volatility of the underlying up to `as_of`."""
    try:
        import yfinance as yf
        hist = yf.download(underlying_yf, period="6mo", auto_adjust=True, progress=False)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        close = hist["Close"].dropna()
        close = close[close.index <= as_of]
        rets = np.log(close / close.shift(1)).dropna().to_numpy()[-window:]
        if len(rets) >= 20:
            return float(np.std(rets, ddof=1) * np.sqrt(252))
    except Exception:
        pass
    return 0.25  # fallback


def build_option_dataset(symbol: str = "RELIANCE", days_back: int = 10,
                         rate: float = 0.065, min_volume: int = 1,
                         max_contracts: int = None) -> pd.DataFrame:
    """
    Build a real call-option dataset for one NSE underlying from recent bhavcopies.

    Returns a DataFrame with the same core columns the rest of the project uses:
    S, K, T, r, sigma_hist, C_market, moneyness, plus bookkeeping (expiry, OI, vol).
    """
    raw = download_bhavcopies(days_back=days_back)
    df = raw[(raw["TckrSymb"] == symbol) & (raw["OptnTp"] == "CE") &
             (raw["FinInstrmTp"].isin(OPTION_TYPES))].copy()
    if df.empty:
        raise ValueError(f"No call options found for '{symbol}'. Check the symbol (e.g. RELIANCE, NIFTY).")

    df["TradDt"] = pd.to_datetime(df["TradDt"])
    df["XpryDt"] = pd.to_datetime(df["XpryDt"])
    df["T"] = (df["XpryDt"] - df["TradDt"]).dt.days / 365.0
    # Prefer settlement price; fall back to last/close.
    price = df["SttlmPric"].where(df["SttlmPric"] > 0, df["LastPric"])
    price = price.where(price > 0, df["ClsPric"])
    df["C_market"] = pd.to_numeric(price, errors="coerce")
    df["S"] = pd.to_numeric(df["UndrlygPric"], errors="coerce")
    df["K"] = pd.to_numeric(df["StrkPric"], errors="coerce")
    df["volume"] = pd.to_numeric(df["TtlTradgVol"], errors="coerce").fillna(0)
    df["open_interest"] = pd.to_numeric(df["OpnIntrst"], errors="coerce").fillna(0)

    # Clean: positive, liquid, dated, and within no-arbitrage call bounds.
    df = df[(df["T"] > 1 / 365.0) & (df["C_market"] > 0) & (df["S"] > 0) & (df["K"] > 0)]
    df = df[df["volume"] >= min_volume]
    disc_intrinsic = np.maximum(0.0, df["S"] - df["K"] * np.exp(-rate * df["T"]))
    df = df[(df["C_market"] >= disc_intrinsic - 0.5) & (df["C_market"] <= df["S"] + 1e-6)]
    df["moneyness"] = df["S"] / df["K"]
    df = df[(df["moneyness"] > 0.5) & (df["moneyness"] < 2.0)]
    df["r"] = rate

    # Historical volatility of the underlying (one value per trade date).
    und_yf = INDEX_YF.get(symbol, symbol + ".NS")
    sigma_by_date = {d: _historical_vol(und_yf, d) for d in df["TradDt"].unique()}
    df["sigma_hist"] = df["TradDt"].map(sigma_by_date)

    df["moneyness_cat"] = np.where(df["moneyness"] < 0.97, "OTM",
                                   np.where(df["moneyness"] < 1.03, "ATM", "ITM"))
    cols = ["TradDt", "XpryDt", "S", "K", "moneyness", "moneyness_cat", "T", "r",
            "sigma_hist", "C_market", "volume", "open_interest"]
    df = df[cols].dropna().reset_index(drop=True)
    if max_contracts and len(df) > max_contracts:
        df = df.sample(max_contracts, random_state=42).reset_index(drop=True)
    return df


def evaluate_black_scholes(df: pd.DataFrame) -> dict:
    """Objective 1 on real data: BS (historical vol) vs observed market price."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    c_bs = black_scholes_call(df["S"].to_numpy(), df["K"].to_numpy(), df["T"].to_numpy(),
                              df["r"].to_numpy(), df["sigma_hist"].to_numpy())
    y = df["C_market"].to_numpy()
    out = {"n_contracts": len(df),
           "RMSE": float(np.sqrt(mean_squared_error(y, c_bs))),
           "MAE": float(mean_absolute_error(y, c_bs)),
           "R2": float(r2_score(y, c_bs))}
    return out, c_bs


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Real NSE option data (F&O bhavcopy) + Black-Scholes eval")
    p.add_argument("--symbol", default="RELIANCE")
    p.add_argument("--days-back", type=int, default=10)
    p.add_argument("--rate", type=float, default=0.065)
    p.add_argument("--min-volume", type=int, default=1)
    args = p.parse_args()
    data = build_option_dataset(args.symbol, days_back=args.days_back,
                                rate=args.rate, min_volume=args.min_volume)
    print(f"Real {args.symbol} call contracts: {len(data)}")
    print(data.head().to_string())
    metrics, _ = evaluate_black_scholes(data)
    print("\nObjective 1 — Black-Scholes vs REAL market prices:")
    print(metrics)
