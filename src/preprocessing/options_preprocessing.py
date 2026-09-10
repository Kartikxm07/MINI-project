"""
src/preprocessing/options_preprocessing.py
------------------------------------------
Step 1 - Raw data inspection
Step 2 - Underlying Options Cleaning
Step 3 - Options Data Filtering
Step 4 - Time to Expiration (T)
Step 5 - Moneyness
Step 6 - Moneyness Category

Follows methodology from:
  "Pricing options with a new hybrid neural network model"
  Shvimer & Zhu, Expert Systems With Applications, 251 (2024), 123979.

Column Mapping (Raw -> Standardized):
  quote_date                        -> date
  expiration                        -> expiration
  option_type (C/P)                 -> option_type (CALL/PUT)
  (underlying_bid_1545 +
   underlying_ask_1545) / 2         -> S
  strike                            -> strike
  bid_1545                          -> bid
  ask_1545                          -> ask
  trade_volume                      -> volume
  open_interest                     -> open_interest

Raw dataset: data/raw/underlying_options/UnderlyingOptionsEODCalcs_2023-08-25_cgi_or_historical.csv
Output:      data/processed/intermediate/options_cleaned.csv
"""

import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_THIS_DIR, "..", "..")

RAW_PATH = os.path.join(
    _PROJECT_ROOT,
    "data", "raw", "underlying_options",
    "UnderlyingOptionsEODCalcs_2023-08-25_cgi_or_historical.csv",
)
OUT_PATH = os.path.join(
    _PROJECT_ROOT, "data", "processed", "intermediate", "options_cleaned.csv"
)

# ---------------------------------------------------------------------------
# Column mapping: raw -> standardized
# ---------------------------------------------------------------------------
COLUMN_MAPPING = {
    "quote_date": "date",
    "expiration": "expiration",
    "option_type": "option_type",
    "strike": "strike",
    "bid_1545": "bid",
    "ask_1545": "ask",
    "trade_volume": "volume",
    "open_interest": "open_interest",
}

# Columns to keep from raw dataset before renaming
KEEP_RAW = list(COLUMN_MAPPING.keys()) + [
    "underlying_bid_1545",
    "underlying_ask_1545",
]

# ---------------------------------------------------------------------------
# Filtering thresholds (per paper)
# ---------------------------------------------------------------------------
MIN_MARKET_PRICE = 0.125   # 1/8 dollar
MAX_DTE = 120              # days
MONEYNESS_LOW = 0.90
MONEYNESS_HIGH = 1.10


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
def run(raw_path: str = RAW_PATH, out_path: str = OUT_PATH, verbose: bool = True) -> pd.DataFrame:
    """
    Full options preprocessing pipeline.
    Returns the cleaned DataFrame and saves it to out_path.
    """

    quality_report = {}

    # -----------------------------------------------------------------------
    # STEP 1 -- Raw data inspection
    # -----------------------------------------------------------------------
    if verbose:
        print("=" * 70)
        print("STEP 1 -- RAW DATA INSPECTION")
        print("=" * 70)

    df_raw = pd.read_csv(raw_path, low_memory=False)
    quality_report["1_original_rows"] = len(df_raw)

    if verbose:
        print(f"File : {raw_path}")
        print(f"Shape: {df_raw.shape}")
        print(f"\nColumns ({len(df_raw.columns)}):")
        for c in df_raw.columns:
            print(f"  {c}")
        print(f"\nData types:\n{df_raw.dtypes}")
        print(f"\nFirst 5 rows:\n{df_raw.head()}")
        print(f"\nMissing values:\n{df_raw.isnull().sum().sort_values(ascending=False).head(20)}")
        print(f"\nDuplicate rows: {df_raw.duplicated().sum()}")
        print(f"\nOption types: {df_raw['option_type'].unique()}")

    # -----------------------------------------------------------------------
    # STEP 2.0 -- Select relevant columns & rename
    # -----------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 2 -- OPTIONS CLEANING")
        print("=" * 70)

    # Keep only needed raw columns
    missing_raw = [c for c in KEEP_RAW if c not in df_raw.columns]
    if missing_raw:
        raise ValueError(f"Missing expected raw columns: {missing_raw}")

    df = df_raw[KEEP_RAW].copy()

    # Derive S = midpoint of underlying bid/ask at 15:45
    df["S"] = (df["underlying_bid_1545"] + df["underlying_ask_1545"]) / 2

    # Drop the raw underlying price columns -- S is now standardized
    df.drop(columns=["underlying_bid_1545", "underlying_ask_1545"], inplace=True)

    # Apply column renaming
    df.rename(columns=COLUMN_MAPPING, inplace=True)

    # -----------------------------------------------------------------------
    # STEP 2.1 -- Date conversion
    # -----------------------------------------------------------------------
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["expiration"] = pd.to_datetime(df["expiration"], errors="coerce")

    if verbose:
        print(f"Date range (raw): {df['date'].min()} -> {df['date'].max()}")

    # -----------------------------------------------------------------------
    # STEP 2.2 -- Numerical conversion
    # -----------------------------------------------------------------------
    numeric_cols = ["S", "strike", "bid", "ask", "volume", "open_interest"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # -----------------------------------------------------------------------
    # STEP 2.3 -- Option type standardization (C->CALL, P->PUT)
    # -----------------------------------------------------------------------
    type_map = {
        "C": "CALL", "CALL": "CALL",
        "P": "PUT",  "PUT": "PUT",
    }
    df["option_type"] = (
        df["option_type"].astype(str).str.strip().str.upper().map(type_map)
    )

    # -----------------------------------------------------------------------
    # STEP 2.4 -- Drop rows missing critical fields
    # -----------------------------------------------------------------------
    critical = ["date", "expiration", "option_type", "S", "strike", "bid", "ask", "volume", "open_interest"]
    before = len(df)
    df.dropna(subset=critical, inplace=True)
    quality_report["2_after_missing_drop"] = len(df)
    if verbose:
        print(f"Dropped {before - len(df)} rows with missing critical fields. Remaining: {len(df)}")

    # -----------------------------------------------------------------------
    # Remove unknown option types
    # -----------------------------------------------------------------------
    before = len(df)
    df = df[df["option_type"].isin(["CALL", "PUT"])].copy()
    quality_report["3_after_invalid_type"] = len(df)
    if verbose:
        print(f"Dropped {before - len(df)} rows with unknown option type. Remaining: {len(df)}")

    # -----------------------------------------------------------------------
    # Remove duplicate rows
    # -----------------------------------------------------------------------
    before = len(df)
    df.drop_duplicates(inplace=True)
    quality_report["4_after_duplicates"] = len(df)
    if verbose:
        print(f"Dropped {before - len(df)} duplicate rows. Remaining: {len(df)}")

    # -----------------------------------------------------------------------
    # STEP 3 -- OPTIONS DATA FILTERING
    # -----------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 3 -- OPTIONS DATA FILTERING")
        print("=" * 70)

    # 8.1 -- Remove options with no open interest
    before = len(df)
    df = df[df["open_interest"] > 0].copy()
    quality_report["5_after_open_interest"] = len(df)
    if verbose:
        print(f"[OI>0] Dropped {before - len(df)} rows. Remaining: {len(df)}")

    # 8.2 -- Remove options with no volume
    before = len(df)
    df = df[df["volume"] > 0].copy()
    quality_report["6_after_volume"] = len(df)
    if verbose:
        print(f"[volume>0] Dropped {before - len(df)} rows. Remaining: {len(df)}")

    # 8.3 -- Calculate market price (bid-ask midpoint)
    df["market_price"] = (df["bid"] + df["ask"]) / 2

    # 8.4 -- Remove invalid quotes
    before = len(df)
    df = df[(df["ask"] >= df["bid"]) & (df["market_price"] > 0)].copy()
    quality_report["7_after_invalid_quotes"] = len(df)
    if verbose:
        print(f"[ask>=bid & market_price>0] Dropped {before - len(df)} rows. Remaining: {len(df)}")

    # 8.5 -- Minimum option price (1/8 dollar = 0.125)
    before = len(df)
    df = df[df["market_price"] >= MIN_MARKET_PRICE].copy()
    quality_report["8_after_min_price"] = len(df)
    if verbose:
        print(f"[market_price>=0.125] Dropped {before - len(df)} rows. Remaining: {len(df)}")

    # 8.6 -- Calculate days to expiration (calendar days)
    df["days_to_expiration"] = (df["expiration"] - df["date"]).dt.days

    # 8.7 -- Remove expired options
    before = len(df)
    df = df[df["days_to_expiration"] > 0].copy()
    quality_report["9_after_expired"] = len(df)
    if verbose:
        print(f"[DTE>0] Dropped {before - len(df)} rows. Remaining: {len(df)}")

    # 8.8 -- Maximum maturity (120 days)
    before = len(df)
    df = df[df["days_to_expiration"] <= MAX_DTE].copy()
    quality_report["10_after_max_dte"] = len(df)
    if verbose:
        print(f"[DTE<=120] Dropped {before - len(df)} rows. Remaining: {len(df)}")

    # -----------------------------------------------------------------------
    # STEP 4 -- TIME TO EXPIRATION T = DTE / 365
    # -----------------------------------------------------------------------
    df["T"] = df["days_to_expiration"] / 365.0
    # Safety: remove any T <= 0
    df = df[df["T"] > 0].copy()

    # -----------------------------------------------------------------------
    # STEP 5 -- MONEYNESS = S / strike
    # -----------------------------------------------------------------------
    df["moneyness"] = df["S"] / df["strike"]

    before = len(df)
    df = df[
        (df["moneyness"] >= MONEYNESS_LOW) & (df["moneyness"] <= MONEYNESS_HIGH)
    ].copy()
    quality_report["11_after_moneyness"] = len(df)
    if verbose:
        print(f"[0.90<=moneyness<=1.10] Dropped {before - len(df)} rows. Remaining: {len(df)}")

    # -----------------------------------------------------------------------
    # STEP 6 -- MONEYNESS CATEGORY (OTM / ATM / ITM per paper)
    # -----------------------------------------------------------------------
    def _categorize(m):
        if m < 0.97:
            return "OTM"
        elif m < 1.03:
            return "ATM"
        else:
            return "ITM"

    df["moneyness_category"] = df["moneyness"].apply(_categorize)

    # -----------------------------------------------------------------------
    # Final column ordering
    # -----------------------------------------------------------------------
    col_order = [
        "date", "expiration", "option_type",
        "S", "strike",
        "bid", "ask", "market_price",
        "volume", "open_interest",
        "days_to_expiration", "T",
        "moneyness", "moneyness_category",
    ]
    df = df[col_order].reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    if verbose:
        print(f"\n[OK] Saved options_cleaned.csv -> {out_path}")
        print(f"  Final shape: {df.shape}")
        print(f"  Date range : {df['date'].min().date()} -> {df['date'].max().date()}")
        print(f"  Option types:\n{df['option_type'].value_counts()}")
        print(f"  Moneyness categories:\n{df['moneyness_category'].value_counts()}")
        print(f"\nData Quality Report (Options):")
        for k, v in quality_report.items():
            print(f"  {k}: {v}")

    return df, quality_report


if __name__ == "__main__":
    run()
