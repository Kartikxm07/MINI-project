"""
src/preprocessing/vix_preprocessing.py
---------------------------------------
Step 7 -- VIX Preprocessing

Follows methodology from:
  "Pricing options with a new hybrid neural network model"
  Shvimer & Zhu, Expert Systems With Applications, 251 (2024), 123979.

The paper uses the PREVIOUS TRADING DAY's VIX closing level as sigma (sigma).
This is implemented by:
  1. Sorting VIX data by date ascending.
  2. Shifting VIX_close by +1 row so each date's "prev_VIX_close"
     corresponds to the actual previous trading day's closing value.
  3. Renaming prev_VIX_close -> sigma.

Column mapping:
  DATE  -> date
  CLOSE -> VIX_close -> sigma (previous trading day)

Raw dataset:  data/raw/vix/vix_daily.csv
Output:       data/processed/intermediate/vix_cleaned.csv
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_THIS_DIR, "..", "..")

RAW_PATH = os.path.join(
    _PROJECT_ROOT, "data", "raw", "vix", "vix_daily.csv"
)
OUT_PATH = os.path.join(
    _PROJECT_ROOT, "data", "processed", "intermediate", "vix_cleaned.csv"
)


def run(raw_path: str = RAW_PATH, out_path: str = OUT_PATH, verbose: bool = True) -> pd.DataFrame:
    """
    VIX preprocessing pipeline.
    Returns cleaned DataFrame with columns [date, VIX_close, sigma].
    Saves to out_path.
    """

    if verbose:
        print("=" * 70)
        print("STEP 7 -- VIX PREPROCESSING")
        print("=" * 70)

    # -----------------------------------------------------------------------
    # Load raw VIX
    # -----------------------------------------------------------------------
    df = pd.read_csv(raw_path)

    if verbose:
        print(f"Raw VIX shape : {df.shape}")
        print(f"Raw columns   : {df.columns.tolist()}")
        print(f"First 5 rows  :\n{df.head()}")
        print(f"Missing values:\n{df.isnull().sum()}")

    # -----------------------------------------------------------------------
    # Column mapping
    # -----------------------------------------------------------------------
    # DATE -> date, CLOSE -> VIX_close
    df.rename(columns={"DATE": "date", "CLOSE": "VIX_close"}, inplace=True)

    # Keep only what we need
    df = df[["date", "VIX_close"]].copy()

    # -----------------------------------------------------------------------
    # Date conversion
    # -----------------------------------------------------------------------
    df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True, errors="coerce")
    df.dropna(subset=["date"], inplace=True)

    # -----------------------------------------------------------------------
    # Numeric conversion of VIX_close
    # -----------------------------------------------------------------------
    df["VIX_close"] = pd.to_numeric(df["VIX_close"], errors="coerce")

    # Remove rows where VIX_close is missing or invalid
    before = len(df)
    df.dropna(subset=["VIX_close"], inplace=True)
    df = df[df["VIX_close"] >= 0].copy()
    if verbose:
        print(f"Dropped {before - len(df)} rows with invalid VIX values.")

    # -----------------------------------------------------------------------
    # Sort by date ascending (required for shift to work correctly)
    # -----------------------------------------------------------------------
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # -----------------------------------------------------------------------
    # Create sigma = PREVIOUS trading day's VIX close
    # The shift(1) operation moves each VIX_close value one row forward,
    # so a given date gets the VIX_close from the prior trading day.
    # The first row will have NaN sigma (no previous day available).
    # -----------------------------------------------------------------------
    df["sigma"] = df["VIX_close"].shift(1)

    # The sigma column is in VIX index points (e.g. 13.56).
    # Per the paper, this is used directly as the sigma parameter.
    # Note: some implementations divide by 100 to get decimal form;
    # however the paper does not explicitly specify conversion --
    # we preserve the raw VIX level here and let the user document
    # if conversion is needed for Black-Scholes (sigma=VIX/100).
    # DOCUMENTED DECISION: sigma = raw VIX close of previous trading day.

    if verbose:
        print(f"\nPrevious-day sigma assignment:")
        print(f"  Method   : VIX_close.shift(1) on date-sorted data")
        print(f"  Example  :\n{df[['date', 'VIX_close', 'sigma']].head(6)}")
        print(f"\nDate range: {df['date'].min().date()} -> {df['date'].max().date()}")
        print(f"Shape     : {df.shape}")
        print(f"Missing sigma (first row expected): {df['sigma'].isnull().sum()}")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    if verbose:
        print(f"\n[OK] Saved vix_cleaned.csv -> {out_path}")
        print(f"  Columns: {df.columns.tolist()}")

    return df


if __name__ == "__main__":
    run()
