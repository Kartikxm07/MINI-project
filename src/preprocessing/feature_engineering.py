"""
src/preprocessing/feature_engineering.py
------------------------------------------
Step 9  -- Data Integration
Step 10 -- Final Data Validation
Step 11 -- Remove Rows Missing Model Inputs
Step 12 -- Feature Engineering (market_price_over_K)
Step 13 -- Call / Put Dataset Separation
Step 14 -- Chronological 80/20 Train/Test Split
         + Data Quality Report

Follows methodology from:
  "Pricing options with a new hybrid neural network model"
  Shvimer & Zhu, Expert Systems With Applications, 251 (2024), 123979.

Inputs:
  data/processed/intermediate/options_cleaned.csv
  data/processed/intermediate/vix_cleaned.csv
  data/processed/intermediate/yield_curve_cleaned.csv

Outputs:
  data/processed/final/train/call_train.csv
  data/processed/final/train/put_train.csv
  data/processed/final/test/call_test.csv
  data/processed/final/test/put_test.csv
  outputs/reports/data_quality_report.txt
"""

import os
import pandas as pd
import numpy as np

from yield_curve_preprocessing import assign_r_to_options

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_THIS_DIR, "..", "..")

OPTIONS_PATH    = os.path.join(_PROJECT_ROOT, "data", "processed", "intermediate", "options_cleaned.csv")
VIX_PATH        = os.path.join(_PROJECT_ROOT, "data", "processed", "intermediate", "vix_cleaned.csv")
YIELD_PATH      = os.path.join(_PROJECT_ROOT, "data", "processed", "intermediate", "yield_curve_cleaned.csv")

CALL_TRAIN_PATH = os.path.join(_PROJECT_ROOT, "data", "processed", "final", "train", "call_train.csv")
PUT_TRAIN_PATH  = os.path.join(_PROJECT_ROOT, "data", "processed", "final", "train", "put_train.csv")
CALL_TEST_PATH  = os.path.join(_PROJECT_ROOT, "data", "processed", "final", "test",  "call_test.csv")
PUT_TEST_PATH   = os.path.join(_PROJECT_ROOT, "data", "processed", "final", "test",  "put_test.csv")
REPORT_PATH     = os.path.join(_PROJECT_ROOT, "outputs", "reports", "data_quality_report.txt")

# Final schema column order
FINAL_COLUMNS = [
    "date", "expiration", "option_type",
    "S", "strike",
    "bid", "ask", "market_price",
    "volume", "open_interest",
    "days_to_expiration", "T",
    "moneyness", "moneyness_category",
    "sigma", "r",
    "market_price_over_K",
]

TRAIN_RATIO = 0.80


def run(verbose: bool = True) -> dict:
    """
    Full feature engineering and integration pipeline.
    Returns quality report dict.
    """

    quality = {}
    report_lines = ["=" * 70, "DATA QUALITY REPORT", "=" * 70, ""]

    def log(msg):
        if verbose:
            print(msg)
        report_lines.append(msg)

    # -----------------------------------------------------------------------
    # STEP 9 -- DATA INTEGRATION
    # -----------------------------------------------------------------------
    log("=" * 70)
    log("STEP 9 -- DATA INTEGRATION")
    log("=" * 70)

    # Load cleaned intermediates
    opts = pd.read_csv(OPTIONS_PATH, parse_dates=["date", "expiration"])
    vix  = pd.read_csv(VIX_PATH,     parse_dates=["date"])
    yield_df = pd.read_csv(YIELD_PATH, parse_dates=["date"])

    quality["A_options_cleaned_rows"] = len(opts)
    log(f"Options cleaned rows  : {len(opts)}")
    log(f"VIX cleaned rows      : {len(vix)}")
    log(f"Yield curve rows      : {len(yield_df)}")

    # -----------------------------------------------------------------------
    # Merge sigma (previous trading day's VIX close) onto options
    # The vix_cleaned.csv already has sigma = shift(+1) of VIX_close,
    # so we simply left-join on date.
    # -----------------------------------------------------------------------
    log("\nMerging sigma (prev-day VIX) onto options by date ...")
    vix_sigma = vix[["date", "sigma"]].dropna(subset=["sigma"])
    df = opts.merge(vix_sigma, on="date", how="left")

    before = len(df)
    df.dropna(subset=["sigma"], inplace=True)
    quality["B_after_vix_merge"] = len(df)
    log(f"  Rows after VIX sigma merge  : {len(df)}  (dropped {before - len(df)} with no sigma)")

    # -----------------------------------------------------------------------
    # Merge r (interpolated yield curve) onto options
    # -----------------------------------------------------------------------
    log("\nAssigning interpolated risk-free rate r ...")
    df = assign_r_to_options(df, yield_df, verbose=verbose)

    before = len(df)
    df.dropna(subset=["r"], inplace=True)
    quality["C_after_yield_merge"] = len(df)
    log(f"  Rows after r assignment     : {len(df)}  (dropped {before - len(df)} with no r)")

    # -----------------------------------------------------------------------
    # STEP 12 -- FEATURE ENGINEERING
    # market_price_over_K = market_price / strike
    # -----------------------------------------------------------------------
    df["market_price_over_K"] = df["market_price"] / df["strike"]

    # -----------------------------------------------------------------------
    # STEP 10 -- FINAL DATA VALIDATION
    # -----------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("STEP 10 -- FINAL DATA VALIDATION")
    log("=" * 70)

    critical_no_null = ["S", "strike", "T", "sigma", "r", "market_price", "moneyness"]
    for col in critical_no_null:
        n = df[col].isnull().sum()
        log(f"  Missing {col:25s}: {n}")
        assert n == 0, f"Validation FAILED: {col} has {n} missing values"

    # Invalid value checks
    checks = [
        ("S > 0",              (df["S"] > 0).all()),
        ("strike > 0",         (df["strike"] > 0).all()),
        ("T > 0",              (df["T"] > 0).all()),
        ("sigma >= 0",         (df["sigma"] >= 0).all()),
        ("market_price > 0",   (df["market_price"] > 0).all()),
        ("0.90 <= moneyness <= 1.10",
         ((df["moneyness"] >= 0.90) & (df["moneyness"] <= 1.10)).all()),
        ("bid >= 0",           (df["bid"] >= 0).all()),
        ("ask >= 0",           (df["ask"] >= 0).all()),
        ("ask >= bid",         (df["ask"] >= df["bid"]).all()),
        ("option_type in CALL/PUT",
         df["option_type"].isin(["CALL", "PUT"]).all()),
        ("moneyness_category in OTM/ATM/ITM",
         df["moneyness_category"].isin(["OTM", "ATM", "ITM"]).all()),
    ]
    for desc, result in checks:
        status = "PASS" if result else "FAIL"
        log(f"  [{status}] {desc}")

    # -----------------------------------------------------------------------
    # STEP 11 -- REMOVE UNNECESSARY ROWS (final drop of any remaining nulls)
    # -----------------------------------------------------------------------
    required_model_inputs = ["S", "strike", "T", "sigma", "r", "market_price"]
    before = len(df)
    df.dropna(subset=required_model_inputs, inplace=True)
    quality["D_final_integrated_rows"] = len(df)
    log(f"\nFinal integrated dataset rows: {len(df)}  (dropped {before - len(df)} missing model inputs)")

    # -----------------------------------------------------------------------
    # Enforce final column order
    # -----------------------------------------------------------------------
    df = df[FINAL_COLUMNS].copy()
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # -----------------------------------------------------------------------
    # STEP 13 -- CALL / PUT SPLIT
    # -----------------------------------------------------------------------
    call_data = df[df["option_type"] == "CALL"].copy().reset_index(drop=True)
    put_data  = df[df["option_type"] == "PUT"].copy().reset_index(drop=True)

    quality["E_call_count"] = len(call_data)
    quality["F_put_count"]  = len(put_data)
    log(f"\nCall observations : {len(call_data)}")
    log(f"Put  observations : {len(put_data)}")

    # -----------------------------------------------------------------------
    # STEP 14 -- CHRONOLOGICAL 80/20 TRAIN/TEST SPLIT
    # Applied separately to CALL and PUT datasets
    # -----------------------------------------------------------------------
    def chronological_split(data: pd.DataFrame, train_ratio: float = TRAIN_RATIO):
        """Sort by date, take earliest 80% as train, latest 20% as test."""
        data = data.sort_values("date").reset_index(drop=True)
        split_idx = int(len(data) * train_ratio)
        train = data.iloc[:split_idx].copy()
        test  = data.iloc[split_idx:].copy()
        return train, test

    call_train, call_test = chronological_split(call_data)
    put_train,  put_test  = chronological_split(put_data)

    quality["G_call_train"] = len(call_train)
    quality["H_call_test"]  = len(call_test)
    quality["I_put_train"]  = len(put_train)
    quality["J_put_test"]   = len(put_test)

    log(f"\nCall train : {len(call_train)} rows  [{call_train['date'].min().date()} -> {call_train['date'].max().date()}]")
    log(f"Call test  : {len(call_test)}  rows  [{call_test['date'].min().date()}  -> {call_test['date'].max().date()}]")
    log(f"Put  train : {len(put_train)} rows  [{put_train['date'].min().date()} -> {put_train['date'].max().date()}]")
    log(f"Put  test  : {len(put_test)}  rows  [{put_test['date'].min().date()}  -> {put_test['date'].max().date()}]")

    # -----------------------------------------------------------------------
    # STEP 29 -- FINAL SANITY CHECK
    # -----------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("STEP 29 -- FINAL SANITY CHECK")
    log("=" * 70)
    log(f"1. Final dataset shape        : {df.shape}")
    log(f"2. Number of Call observations: {len(call_data)}")
    log(f"3. Number of Put observations : {len(put_data)}")
    log(f"4. OTM/ATM/ITM distribution:")
    for cat, cnt in df["moneyness_category"].value_counts().items():
        log(f"     {cat}: {cnt}")
    log(f"5. Full date range            : {df['date'].min().date()} -> {df['date'].max().date()}")
    log(f"6. Call train date range      : {call_train['date'].min().date()} -> {call_train['date'].max().date()}")
    log(f"7. Call test date range       : {call_test['date'].min().date()} -> {call_test['date'].max().date()}")
    log(f"8. Missing values in final:\n{df[critical_no_null].isnull().sum().to_string()}")
    log(f"9. Numerical summary:\n{df[['S','strike','T','sigma','r','market_price','moneyness']].describe().to_string()}")
    log(f"10. Observations removed at each stage:")
    prev = quality.get("A_options_cleaned_rows", 0)
    for k, v in quality.items():
        if k.startswith("A_"):
            continue
        removed = prev - v if v < prev else 0
        log(f"    {k}: {v}  (removed {removed})")
        prev = v

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    for path in [CALL_TRAIN_PATH, PUT_TRAIN_PATH, CALL_TEST_PATH, PUT_TEST_PATH]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    call_train.to_csv(CALL_TRAIN_PATH, index=False)
    put_train.to_csv(PUT_TRAIN_PATH,   index=False)
    call_test.to_csv(CALL_TEST_PATH,   index=False)
    put_test.to_csv(PUT_TEST_PATH,     index=False)

    log(f"\n[OK] Saved: {CALL_TRAIN_PATH}")
    log(f"[OK] Saved: {PUT_TRAIN_PATH}")
    log(f"[OK] Saved: {CALL_TEST_PATH}")
    log(f"[OK] Saved: {PUT_TEST_PATH}")

    # Save quality report
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    log(f"[OK] Saved report: {REPORT_PATH}")

    log("\n" + "=" * 70)
    log("PREPROCESSING COMPLETE -- HANDOFF READY")
    log("=" * 70)

    return quality


if __name__ == "__main__":
    # Must run from the src/preprocessing directory or adjust sys.path
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    run()
