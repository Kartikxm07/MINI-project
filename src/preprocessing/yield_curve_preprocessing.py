"""
src/preprocessing/yield_curve_preprocessing.py
------------------------------------------------
Step 8 -- Yield Curve Preprocessing

Follows methodology from:
  "Pricing options with a new hybrid neural network model"
  Shvimer & Zhu, Expert Systems With Applications, 251 (2024), 123979.

The paper matches the risk-free rate to the option's expiration period using
the US Government Par Yield Curve. Since the yield curve provides only standard
maturities (1m, 2m, 3m, 4m, 6m, 1y, 2y, 3y, 5y, 7y, 10y, 20y, 30y),
linear interpolation is performed on the days-to-expiration axis.

Method:
  1. Load and clean the yield curve CSV.
  2. Convert maturity labels to days (approximate calendar days).
  3. For each option observation, given its days_to_expiration,
     linearly interpolate between the two nearest maturity points
     to obtain the rate r.
  4. Rates are stored as percentages in the raw data -> convert to decimal
     (divide by 100) so they are compatible with Black-Scholes.

DOCUMENTED DECISION:
  - Linear interpolation between adjacent standard maturities.
  - Rates converted from percentage to decimal (÷ 100).
  - For DTE below 1-month (~30 days), the 1-month rate is used (floor).
  - For DTE above 30-year (~10950 days), the 30-year rate is used (ceiling).

Raw dataset:  data/raw/yield_curve/treasury_yield_curve.csv
Output:       data/processed/intermediate/yield_curve_cleaned.csv
              (long-format table with [date, days, rate] for all maturities)
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_THIS_DIR, "..", "..")

RAW_PATH = os.path.join(
    _PROJECT_ROOT, "data", "raw", "yield_curve", "treasury_yield_curve.csv"
)
OUT_PATH = os.path.join(
    _PROJECT_ROOT, "data", "processed", "intermediate", "yield_curve_cleaned.csv"
)

# ---------------------------------------------------------------------------
# Maturity label -> approximate calendar days mapping
# ---------------------------------------------------------------------------
MATURITY_DAYS = {
    "1 mo":  30,
    "2 mo":  60,
    "3 mo":  91,
    "4 mo":  122,
    "6 mo":  182,
    "1 yr":  365,
    "2 yr":  730,
    "3 yr":  1095,
    "5 yr":  1825,
    "7 yr":  2555,
    "10 yr": 3650,
    "20 yr": 7300,
    "30 yr": 10950,
}


def clean_yield_curve(raw_path: str = RAW_PATH, out_path: str = OUT_PATH, verbose: bool = True) -> pd.DataFrame:
    """
    Clean the yield curve dataset.
    Returns a wide-format DataFrame with columns:
      [date, 1 mo, 2 mo, 3 mo, ..., 30 yr]  (rates as decimals)
    Saves a long-format version to out_path.
    """

    if verbose:
        print("=" * 70)
        print("STEP 8 -- YIELD CURVE PREPROCESSING")
        print("=" * 70)

    # -----------------------------------------------------------------------
    # Load raw
    # -----------------------------------------------------------------------
    df = pd.read_csv(raw_path)

    if verbose:
        print(f"Raw shape  : {df.shape}")
        print(f"Columns    : {df.columns.tolist()}")
        print(f"First 5    :\n{df.head()}")
        print(f"Missing    :\n{df.isnull().sum()}")

    # -----------------------------------------------------------------------
    # Date conversion
    # -----------------------------------------------------------------------
    df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True, errors="coerce")
    before = len(df)
    df.dropna(subset=["date"], inplace=True)
    if verbose:
        print(f"Dropped {before - len(df)} rows with invalid dates.")

    # -----------------------------------------------------------------------
    # Convert rate columns to numeric; rates are in percentage -> divide by 100
    # -----------------------------------------------------------------------
    maturity_cols = [c for c in df.columns if c in MATURITY_DAYS]
    for col in maturity_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0

    # -----------------------------------------------------------------------
    # Sort by date
    # -----------------------------------------------------------------------
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if verbose:
        print(f"\nAvailable maturity columns: {maturity_cols}")
        print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
        print(f"Cleaned shape: {df[['date'] + maturity_cols].shape}")
        print(f"Sample rates (first 3 rows):\n{df[['date'] + maturity_cols].head(3)}")

    # -----------------------------------------------------------------------
    # Save wide-format (date + all maturity rates as decimals)
    # -----------------------------------------------------------------------
    keep_cols = ["date"] + maturity_cols
    df_clean = df[keep_cols].copy()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_clean.to_csv(out_path, index=False)

    if verbose:
        print(f"\n[OK] Saved yield_curve_cleaned.csv -> {out_path}")

    return df_clean


def interpolate_r(date: pd.Timestamp, days_to_expiration: int,
                  yield_df: pd.DataFrame) -> float:
    """
    Given an option's trading date and DTE, find the appropriate yield curve
    row (exact date match) and linearly interpolate between the two nearest
    standard maturity points to obtain r.

    Parameters
    ----------
    date               : Option trading date (Timestamp)
    days_to_expiration : Option DTE in calendar days
    yield_df           : Cleaned yield curve DataFrame (wide format)

    Returns
    -------
    float : Interpolated risk-free rate (decimal, e.g. 0.0525 = 5.25%)
            or np.nan if the date is not found in the yield curve.
    """
    row = yield_df[yield_df["date"] == date]
    if row.empty:
        return np.nan

    row = row.iloc[0]
    maturity_cols = [c for c in yield_df.columns if c != "date"]

    # Build lists of (days, rate) for available (non-NaN) maturities
    points = []
    for col in maturity_cols:
        if col in MATURITY_DAYS and not pd.isna(row[col]):
            points.append((MATURITY_DAYS[col], row[col]))

    if not points:
        return np.nan

    points.sort(key=lambda x: x[0])
    days_arr = np.array([p[0] for p in points])
    rate_arr = np.array([p[1] for p in points])

    # Clamp DTE to [min_maturity, max_maturity]
    dte = float(days_to_expiration)
    dte = max(dte, days_arr[0])
    dte = min(dte, days_arr[-1])

    # Linear interpolation
    r = float(np.interp(dte, days_arr, rate_arr))
    return r


def assign_r_to_options(options_df: pd.DataFrame, yield_df: pd.DataFrame,
                        verbose: bool = True) -> pd.DataFrame:
    """
    Merge risk-free rate r onto the options DataFrame.
    Uses date-matched, DTE-interpolated yield curve rates.

    Parameters
    ----------
    options_df : Cleaned options DataFrame (must have 'date', 'days_to_expiration')
    yield_df   : Cleaned yield curve DataFrame (wide format, decimal rates)

    Returns
    -------
    options_df with new column 'r'
    """
    if verbose:
        print("Assigning interpolated risk-free rate r to each option observation...")

    # Vectorised: apply row-by-row interpolation
    def _get_r(row):
        return interpolate_r(row["date"], row["days_to_expiration"], yield_df)

    options_df = options_df.copy()
    options_df["r"] = options_df.apply(_get_r, axis=1)

    missing_r = options_df["r"].isnull().sum()
    if verbose:
        print(f"  Rows with r assigned     : {(~options_df['r'].isnull()).sum()}")
        print(f"  Rows missing r (no date match): {missing_r}")
        print(f"  r stats:\n{options_df['r'].describe()}")

    return options_df


if __name__ == "__main__":
    clean_yield_curve()
