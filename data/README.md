# Data Directory

This folder contains all datasets for the project, cleanly separated into raw and processed stages.

## Folder Structure

```text
data/
├── raw/
│   ├── vix_daily.csv             # Historical Daily VIX Volatility Index (1990 - 2026)
│   └── treasury_yield_curve.csv  # US Treasury Yield Curve Rates (1990 - 2023)
├── processed/                    # Cleaned, merged, or normalized datasets output from preprocessing
└── README.md
```

## Datasets Overview

### 1. `vix_daily.csv` (CBOE Volatility Index - Daily)
- **Rows**: 9,266 records
- **Period**: January 2, 1990 to September 4, 2026
- **Features**:
  - `DATE`: Trading date (`MM/DD/YYYY`)
  - `OPEN`: Opening index level
  - `HIGH`: Highest level of the trading session
  - `LOW`: Lowest level of the trading session
  - `CLOSE`: Closing index level

### 2. `treasury_yield_curve.csv` (U.S. Treasury Yield Curve Rates)
- **Rows**: 8,507 records
- **Period**: January 2, 1990 to December 29, 2023
- **Features**:
  - `date`: Daily observation date (`MM/DD/YYYY`)
  - Yields across standard maturities: `1 mo`, `2 mo`, `3 mo`, `4 mo`, `6 mo`, `1 yr`, `2 yr`, `3 yr`, `5 yr`, `7 yr`, `10 yr`, `20 yr`, `30 yr`

> [!NOTE]
> When you provide the 3rd dataset, it will be placed in `data/raw/` alongside these files.
