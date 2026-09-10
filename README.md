# Hybrid Option Pricing Project
## Module: End-to-End Data Preprocessing & Feature Engineering

This module is responsible ONLY for:

1. Raw data inspection
2. Data cleaning
3. Data preprocessing
4. Data integration
5. Feature engineering
6. Data quality validation
7. Chronological train/test splitting
8. Preparing final datasets for the Black–Scholes model and Hybrid Neural Network models

This module MUST NOT implement or train the Black–Scholes model or Neural Network models.

The final output of this module will be handed over to another team member who will implement:

- Black–Scholes Call pricing
- Black–Scholes Put pricing
- Hybrid Neural Network Model 1
- Hybrid Neural Network Model 2
- Model evaluation
- SHAP analysis
- Sensitivity analysis
- Put–Call parity analysis


# 1. RESEARCH PAPER TO FOLLOW

The preprocessing and feature engineering must follow the methodology described in:

"Pricing options with a new hybrid neural network model"

Authors:
Yossi Shvimer
Song-Ping Zhu

Published in:
Expert Systems With Applications, 251 (2024), 123979.

The paper proposes a hybrid option pricing framework combining the traditional Black–Scholes model with neural networks.

The preprocessing pipeline must preserve all information required by this methodology.

Do NOT invent additional features unnecessarily.

If a preprocessing decision is not explicitly specified by the paper, choose the least invasive approach and document it clearly.


# 2. RAW DATASETS

Three raw datasets are already available in the project.

They are:

1. Underlying Options Data
2. VIX History
3. Par Yield Curve Rates

The raw datasets MUST NOT be modified or overwritten.

Expected conceptual role of each dataset:

--------------------------------------------------
UNDERLYING OPTIONS
--------------------------------------------------

Provides:

- Trading date
- Expiration date
- Strike price K
- Underlying price S
- Option type
- Bid
- Ask
- Volume
- Open interest

The dataset may contain additional columns.


--------------------------------------------------
VIX HISTORY
--------------------------------------------------

Used to obtain the volatility input σ.

The research paper uses the previous day's VIX closing level as the standard deviation parameter.

Therefore the preprocessing pipeline must preserve the temporal relationship between:

Option trading date

and

Previous trading day's VIX close.


--------------------------------------------------
PAR YIELD CURVE RATES
--------------------------------------------------

Used to obtain the risk-free interest rate r.

The research paper uses the real-time US Government Bond yield curve and matches the risk-free parameter to the option expiration period.

The implementation must create a clean risk-free-rate feature that can be joined to the option observations.


# 3. PROJECT DIRECTORY

Maintain the following structure:

Option_Pricing_Project/

├── data/
│   ├── raw/
│   │   ├── underlying_options/
│   │   ├── vix/
│   │   └── yield_curve/
│   │
│   └── processed/
│       ├── intermediate/
│       └── final/
│           ├── train/
│           └── test/
│
├── notebooks/
│   ├── 01_options_preprocessing.ipynb
│   ├── 02_vix_preprocessing.ipynb
│   ├── 03_yield_curve_preprocessing.ipynb
│   └── 04_feature_engineering_and_integration.ipynb
│
├── src/
│   └── preprocessing/
│       ├── options_preprocessing.py
│       ├── vix_preprocessing.py
│       ├── yield_curve_preprocessing.py
│       └── feature_engineering.py
│
├── outputs/
│   ├── figures/
│   └── reports/
│
├── README.md
└── requirements.txt


# 4. IMPORTANT PRINCIPLE

The pipeline must be:

RAW DATA
    ↓
INDIVIDUAL DATASET CLEANING
    ↓
DATA ALIGNMENT
    ↓
FEATURE ENGINEERING
    ↓
DATA QUALITY CHECKS
    ↓
CHRONOLOGICAL 80/20 SPLIT
    ↓
FINAL TRAIN/TEST DATASETS
    ↓
HANDOFF TO BLACK–SCHOLES + NN TEAM


# 5. STEP 1 — INSPECT ALL RAW DATA

Before preprocessing:

- Identify all files
- Identify file formats
- Display dataset shapes
- Display column names
- Display data types
- Display first 5–10 rows
- Check missing values
- Check duplicate rows
- Check date ranges
- Check unique option types
- Check numerical ranges
- Check whether timestamps are available
- Identify the actual column names used in the uploaded datasets

DO NOT assume column names.

Create a mapping between actual column names and standardized names.

The pipeline must document this mapping.


# 6. STANDARD COLUMN NAMES

After preprocessing, use these standardized names wherever applicable:

date
expiration
option_type
S
strike
bid
ask
volume
open_interest

Create:

days_to_expiration
T
market_price
moneyness
moneyness_category

After integrating VIX:

sigma

After integrating yield curve:

r


# 7. STEP 2 — UNDERLYING OPTIONS CLEANING

Process the underlying options dataset first.


## 7.1 Date conversion

Convert trading date and expiration date to proper datetime format.

Required:

date
expiration


## 7.2 Numerical conversion

Convert the following columns to numeric:

S
strike
bid
ask
volume
open_interest

Handle malformed numeric values appropriately.


## 7.3 Option type

Standardize option type into:

CALL
PUT

Support common representations such as:

C
P
CALL
PUT

Remove unknown option types.


## 7.4 Missing values

Inspect missing values.

Rows missing critical pricing/model information should not be passed to the final dataset.

Critical fields include:

date
expiration
option_type
S
strike
bid
ask

Volume/open interest are also required for the paper's filtering step.


# 8. STEP 3 — OPTIONS DATA FILTERING

Apply the filtering methodology used by the paper.


## 8.1 Remove options with no open interest

Keep:

open_interest > 0


## 8.2 Remove options with no volume

Keep:

volume > 0


## 8.3 Calculate market price

The paper defines the market option price as the Bid–Ask midpoint.

Formula:

market_price = (bid + ask) / 2


## 8.4 Remove invalid quotes

Keep only observations satisfying:

ask >= bid

and:

market_price > 0


## 8.5 Minimum option price

The paper removes options whose midpoint price is less than:

1/8 dollar = 0.125

Therefore:

market_price >= 0.125


## 8.6 Calculate days to expiration

Formula:

days_to_expiration =
expiration - date

Use calendar days unless the source data methodology explicitly requires otherwise.


## 8.7 Remove expired options

Keep:

days_to_expiration > 0


## 8.8 Maximum maturity

The paper excludes options with more than 120 days to expiration.

Keep:

days_to_expiration <= 120


# 9. STEP 4 — TIME TO EXPIRATION

Create:

T

using:

T = days_to_expiration / 365

T is required later by the Black–Scholes implementation.

DO NOT remove T.

DO NOT replace T with another representation without documenting it.


# 10. STEP 5 — MONEyness

Calculate:

moneyness = S / strike

This is one of the important features in the research paper.

The paper filters the data using:

0.90 <= S/K <= 1.10

Therefore:

moneyness >= 0.90

and:

moneyness <= 1.10


# 11. STEP 6 — MONEyness CATEGORY

Create:

moneyness_category

using exactly:

if S/K < 0.97:
    OTM

elif 0.97 <= S/K < 1.03:
    ATM

else:
    ITM

Do NOT use a different classification.

This categorical feature is required for reproducing the paper's OTM/ATM/ITM analysis.


# 12. STEP 7 — VIX PREPROCESSING

Process the VIX history independently.

Identify:

date
VIX close

Standardize the columns to:

date
VIX_close


## Critical temporal rule

The paper uses the PREVIOUS DAY'S VIX closing level as the standard deviation parameter.

Therefore, for an option traded on date D:

sigma must correspond to the VIX closing value from the previous trading day.

Do NOT simply merge same-day VIX without checking the paper's methodology.

Use a time-aware merge.

Do not introduce future information.

Create:

sigma

where:

sigma = previous trading day's VIX close

Document exactly how the previous trading day was determined.


# 13. STEP 8 — YIELD CURVE PREPROCESSING

Process the Par Yield Curve Rates dataset independently.

Clean:

- Date
- Treasury maturity columns
- Missing values
- Invalid values

The final objective is to obtain:

r

for each option observation.

The paper matches the risk-free rate to the option expiration period using the US Government yield curve.

If the available yield curve contains standard maturities rather than the exact option maturity:

- identify the appropriate maturity points
- perform the appropriate maturity matching/interpolation
- document the method used

Do NOT silently choose an arbitrary Treasury rate.

The selected rate must be appropriate for the option's remaining maturity.

The final dataset must contain:

r

as the risk-free rate used later by Black–Scholes.


# 14. STEP 9 — DATA INTEGRATION

After independently cleaning:

1. Underlying Options
2. VIX
3. Yield Curve

integrate them into one dataset.

The final integrated observation must contain:

date
expiration
option_type
S
strike
bid
ask
volume
open_interest
market_price
days_to_expiration
T
moneyness
moneyness_category
sigma
r


# 15. TEMPORAL DATA LEAKAGE RULE

This project is a financial forecasting/pricing project.

Therefore:

DO NOT use future information.

Examples of invalid processing:

- Using future VIX
- Using future option prices
- Randomly mixing future observations into training
- Calculating features using future data
- Randomly splitting the complete dataset before chronological ordering

Every feature must be available at the option pricing date.


# 16. STEP 10 — FINAL DATA VALIDATION

Before splitting the data, perform comprehensive validation.


Check:

### Missing values

No missing values in:

S
strike
T
sigma
r
market_price
moneyness


### Invalid values

Verify:

S > 0
strike > 0
T > 0
sigma >= 0
market_price > 0

Also verify:

0.90 <= moneyness <= 1.10


### Quote consistency

Verify:

bid >= 0
ask >= 0
ask >= bid


### Option types

Only:

CALL
PUT


### Moneyness categories

Only:

OTM
ATM
ITM


# 17. STEP 11 — REMOVE UNNECESSARY ROWS

After integration:

Remove observations where required model inputs cannot be obtained.

Required final fields:

S
strike
T
sigma
r
market_price

Do NOT fill critical financial variables with arbitrary constants.

If interpolation or a documented financial-data method is used, record it.

Otherwise remove the affected observation.


# 18. STEP 12 — FEATURE ENGINEERING FOR MODEL HANDOFF

The final preprocessing pipeline should create the following features:

--------------------------------------------------
CORE FEATURES
--------------------------------------------------

S
strike
T
sigma
r


--------------------------------------------------
MARKET TARGET
--------------------------------------------------

market_price


--------------------------------------------------
MARKET INFORMATION
--------------------------------------------------

bid
ask
volume
open_interest


--------------------------------------------------
OPTION INFORMATION
--------------------------------------------------

option_type
date
expiration
days_to_expiration


--------------------------------------------------
ENGINEERED FEATURES
--------------------------------------------------

moneyness
moneyness_category


--------------------------------------------------
NORMALIZED MARKET TARGET
--------------------------------------------------

market_price_over_K

Formula:

market_price_over_K = market_price / strike


# 19. IMPORTANT — DO NOT CREATE BS PRICE

DO NOT calculate:

BS_price

inside this preprocessing module.

The Black–Scholes teammate will calculate:

BS_call_price

and:

BS_put_price

from:

S
strike
sigma
r
T


# 20. IMPORTANT — BS NORMALIZED PRICE

After the Black–Scholes teammate calculates:

BS_price

they will create:

BS_price_over_K = BS_price / strike

DO NOT create BS_price_over_K during preprocessing because BS_price does not exist yet.

The preprocessing output must simply be ready for this operation.


# 21. WHY THESE FEATURES ARE REQUIRED

The final dataset must not hamper either the Black–Scholes implementation or the Hybrid Neural Network.

Black–Scholes requires:

S
K
sigma
r
T

where:

K = strike

The Hybrid Model requires the Black–Scholes price as an additional input.

The paper's Hybrid Model 1 uses:

BS price
S
K
sigma
r
T

The paper's Hybrid Model 2 separates the nonparametric component and combines it with the BS price.

Therefore the preprocessing must preserve:

S
K
sigma
r
T
market_price
moneyness


# 22. STEP 13 — CALL AND PUT DATASETS

Separate the final dataset into:

CALL
PUT

Create:

call_data
put_data

Do not combine Call and Put into separate unrelated preprocessing logic.

They should use the same preprocessing rules.

The difference is only:

option_type


# 23. STEP 14 — CHRONOLOGICAL TRAIN/TEST SPLIT

The paper uses:

80% → training
20% → out-of-sample testing

The split must be chronological.

FIRST:

sort by date.

THEN:

take earliest 80% as training.

Take latest 20% as test.

DO NOT randomly shuffle before this split.

The purpose is to simulate out-of-sample prediction on future market observations.


# 24. CALL/PUT SPLIT

Perform the chronological split separately for Call and Put datasets.

Create:

call_train.csv
call_test.csv

put_train.csv
put_test.csv


# 25. NN VALIDATION SPLIT

DO NOT create the final NN 75/25 validation split in this module.

The research paper further divides the training data into:

75% training
25% cross-validation

This should be handled by the Neural Network teammate.

Your module should provide:

80% chronological training dataset
20% chronological test dataset


# 26. FINAL OUTPUT STRUCTURE

The final directory should contain:

data/

├── raw/

│   ├── underlying_options/
│   ├── vix/
│   └── yield_curve/

│
└── processed/

    ├── intermediate/

    │   ├── options_cleaned.csv
    │   ├── vix_cleaned.csv
    │   └── yield_curve_cleaned.csv

    │
    └── final/

        ├── train/

        │   ├── call_train.csv
        │   └── put_train.csv

        │
        └── test/

            ├── call_test.csv
            └── put_test.csv


# 27. FINAL DATASET SCHEMA

The final CSV files should contain at least:

date
expiration
option_type

S
strike

bid
ask
market_price

volume
open_interest

days_to_expiration
T

moneyness
moneyness_category

sigma
r

market_price_over_K


# 28. COLUMN DEFINITIONS

| Column | Meaning |
|---|---|
| date | Option trading date |
| expiration | Option expiration date |
| option_type | CALL or PUT |
| S | Underlying asset price |
| strike | Strike price K |
| bid | Bid option price |
| ask | Ask option price |
| market_price | Bid–Ask midpoint |
| volume | Option trading volume |
| open_interest | Open interest |
| days_to_expiration | Days remaining until expiration |
| T | Time to expiration in years |
| moneyness | S/K |
| moneyness_category | OTM / ATM / ITM |
| sigma | Previous trading day's VIX close |
| r | Maturity-matched risk-free rate |
| market_price_over_K | Market price divided by strike |


# 29. FINAL SANITY CHECK

Before declaring preprocessing complete, print:

1. Final dataset shape
2. Number of Call observations
3. Number of Put observations
4. OTM/ATM/ITM distribution
5. Date range
6. Train date range
7. Test date range
8. Missing values
9. Numerical summary
10. Number of observations removed at each filtering stage


# 30. DATA QUALITY REPORT

Generate a report containing:

- Original row count
- Rows after duplicate removal
- Rows after missing-value filtering
- Rows after volume/open-interest filtering
- Rows after price filtering
- Rows after DTE filtering
- Rows after moneyness filtering
- Rows after VIX integration
- Rows after risk-free-rate integration
- Final Call count
- Final Put count
- Final Train count
- Final Test count


# 31. REPRODUCIBILITY

The preprocessing must be reproducible.

Use:

- Fixed code
- Explicit column mappings
- Explicit filtering conditions
- Explicit formulas
- Explicit merge rules
- Explicit train/test split logic

Do not manually edit CSV files.

Do not overwrite raw datasets.


# 32. NOTEBOOK RESPONSIBILITIES

Notebook 01:

01_options_preprocessing.ipynb

Should handle:

- Options inspection
- Cleaning
- Filtering
- Market price
- DTE
- T
- Moneyness


Notebook 02:

02_vix_preprocessing.ipynb

Should handle:

- VIX cleaning
- Date handling
- Previous-day VIX alignment
- sigma creation


Notebook 03:

03_yield_curve_preprocessing.ipynb

Should handle:

- Yield curve cleaning
- Maturity handling
- Risk-free rate creation
- r creation


Notebook 04:

04_feature_engineering_and_integration.ipynb

Should handle:

- Merge all datasets
- Final feature engineering
- Validation
- Call/Put separation
- Chronological 80/20 split
- Final CSV generation


# 33. STOP CONDITION

THIS IS VERY IMPORTANT.

Once the following final files have been successfully generated:

call_train.csv
call_test.csv
put_train.csv
put_test.csv

and they contain:

S
strike
sigma
r
T
market_price
moneyness
moneyness_category

plus the required metadata columns,

THE PREPROCESSING MODULE IS COMPLETE.

DO NOT:

- Implement Black–Scholes
- Train a Neural Network
- Implement Hybrid Model 1
- Implement Hybrid Model 2
- Perform hyperparameter tuning
- Perform SHAP
- Perform sensitivity analysis
- Perform Put–Call parity analysis

Those are downstream tasks.


# 34. HANDOFF TO NEXT TEAM MEMBER

The next teammate should be able to load:

call_train.csv
call_test.csv
put_train.csv
put_test.csv

and immediately start Black–Scholes calculation.

For each observation they should have:

S
K = strike
sigma
r
T

They will calculate:

BS_price

and then:

BS_price_over_K = BS_price / K


# 35. FINAL PIPELINE

The complete preprocessing pipeline must therefore be:

RAW OPTIONS
      ↓
OPTIONS CLEANING
      ↓
OPTIONS FILTERING
      ↓
MARKET MID PRICE
      ↓
DTE
      ↓
T
      ↓
MONEYNESS
      ↓
OTM / ATM / ITM
      ↓
      ├──────── VIX CLEANING
      │              ↓
      │       PREVIOUS-DAY VIX
      │              ↓
      │             sigma
      │
      └──────── YIELD CURVE CLEANING
                     ↓
              MATURITY MATCHING
                     ↓
                     r
                     ↓
              DATA INTEGRATION
                     ↓
             FINAL FEATURE SET
                     ↓
             DATA VALIDATION
                     ↓
             CALL / PUT SPLIT
                     ↓
          CHRONOLOGICAL 80/20 SPLIT
                     ↓
        ┌────────────┴────────────┐
        ↓                         ↓
   TRAIN DATA                 TEST DATA
        │                         │
        └────────────┬────────────┘
                     ↓
              FINAL CSV FILES
                     ↓
              HANDOFF TO TEAM
                     ↓
             BLACK–SCHOLES
                     ↓
              BS PRICE / K
                     ↓
             HYBRID NEURAL NN