# StockSight and Deep Learning Option Pricing

StockSight combines Indian stock forecasting with stock-specific European call pricing comparisons. The search page runs TensorFlow LSTM/GRU stock forecasts and a separate PyTorch option comparison for the searched symbol. The original synthetic benchmark remains available at `/options`.

## Stock-specific option comparison

Enter `SBIN`, `INFY`, `RELIANCE` or another supported NSE symbol on the homepage. **Generate forecast + option comparison** runs both tasks independently; **Compare options only** skips the slower stock forecast. Each option comparison trains Direct ANN, Residual ANN and Option LSTM, then compares them with Black-Scholes on identical held-out contracts. The page shows RMSE, MAE, R², premium and error graphs, a contract table and a CSV download.

The homepage uses stock-calibrated simulation automatically, with a default 5% annual interest rate and 0% dividend yield. These settings are not shown in the form. Results highlight the model with the lowest test RMSE, rank all four methods, and compare the winner with Black-Scholes. A short simulation label stays visible. The detailed methodology below documents the assumptions.

The backend supports two data modes (observed-data uploads remain available through the API):

- **Stock-calibrated simulation (default):** downloads three years of adjusted daily stock prices from Yahoo Finance. At up to 150 historical valuation dates, the preceding 126 returns calibrate an empirical-bootstrap Monte Carlo reference (1,024 paths). Returns are recentered to risk-neutral drift using the entered annual interest and dividend assumptions. Nine hypothetical calls per date use strike/spot ratios 0.9, 1.0 and 1.1 and horizons of 21, 42 and 63 trading sessions. These are simulated premiums, not exchange quotes, and the stock need not have listed options. Historical adjusted prices define the simulated spot scale. Monte Carlo standard errors are shown alongside predictions; small model differences may be sampling noise.
- **Upload observed call premiums:** supply a CSV of 60–4,000 end-of-day observations spanning at least 20 distinct valuation dates for the selected symbol. Actual uploaded premiums become the targets; there is no silent simulation fallback. Yahoo history supplies past-only volatility and LSTM features. The uploaded `spot` supplies the contemporaneous underlying price in the same units as strike and premium. The app does not independently verify uploaded quote provenance or download an NSE option archive.

Required CSV columns (a header-only template is available at `/static/option_contracts_template.csv`):

```csv
symbol,date,expiry,spot,strike,call_price
```

Use `YYYY-MM-DD` dates, rupees per share (not contract-lot totals), European calls, and one closing observation per date/expiry/strike. All symbols must match the search. Expiry must follow valuation by 1–366 days. At least 126 preceding trading sessions must be available for every valuation date. The upload limit is 2 MB. A one-day option-chain snapshot is not sufficient to train and test chronologically. Normalize provider exports to this schema first; arbitrary CSV layouts are not automatically mapped. Rates and dividend yields are constant user assumptions for the entire run (defaults 5% and 0%, not live market values); premiums outside theoretical bounds, allowing small quote-rounding tolerance, are rejected.

All contracts for a valuation date stay together in a chronological 70% training / 15% validation / 15% test split. Scalers fit training rows only, and neural networks restore their best validation checkpoint. Features use trailing 60-session annualized volatility; the LSTM also receives the preceding 20-session stock sequence. No future prices or target-derived implied volatility enter the features. Network targets are normalized by strike, and every prediction is clipped to European call bounds. Results are held-out pricing estimates using information through each valuation day's close, not future stock forecasts. Overlapping trailing histories are intentional; this is evaluation on later dates for the same stock, not a test on entirely unseen underlying assets or contract identities.

`POST /api/stock-options` accepts JSON for simulation or multipart form data for uploads: `symbol`, `mode` (`simulation`/`observed`), `rate` and `dividend` as decimal annual rates, plus `contracts` for the CSV. Results are returned to the requesting browser without overwriting the saved benchmark or sharing symbol-specific artifacts between users. The server trains at most one option model suite at a time. There is no model persistence or request cache for this feature.

## Stock forecasting dashboard

`app.py` downloads daily NSE stock prices from Yahoo Finance. Each request uses 60 trading days of Open, High, Low, Close and Volume to train TensorFlow LSTM and GRU networks that predict the next session's open and close.

Data is split chronologically: 70% training, 10% validation and 20% testing. The scaler is fitted only on training data. Early stopping and model selection use validation data; displayed test errors use the held-out test period. RMSE is in scaled units, not rupees. Predictions are converted back to rupees for the charts and forecast.

The dashboard accepts symbols such as `RELIANCE`, `TCS.NS` and `M&M`, adds `.NS` when necessary, and displays historical prices, test predictions, forecasts and company fundamentals. Unavailable fundamentals display `N/A`.

## Option pricing experiment

`train.py` generates synthetic European option contracts, adds volatility smile/skew and price noise, and compares five call pricing methods:

- Black-Scholes: the mathematical baseline using historical volatility.
- Direct ANN: a feedforward network predicting the option price.
- Residual ANN: predicts the difference between the synthetic price and Black-Scholes, then adds that correction to the baseline.
- Option LSTM: combines simulated historical sequences and contract features.
- DL-Vol Hybrid: a neural network predicts a contract-specific volatility, which is fed into Black-Scholes to price the option (see "Project objectives" below).

Contracts are split 70%/15%/15% for training, validation and testing. Network targets are divided by strike before standardization and multiplied by strike after prediction, so scale-free inputs can represent different contract sizes. Metrics include RMSE, MAE and R², with breakdowns for out-of-the-money, at-the-money and in-the-money calls. Lower RMSE/MAE means less pricing error.

### Project objectives

The four project objectives map onto the code as follows:

| # | Objective | Implementation |
| --- | --- | --- |
| 1 | Evaluate the classical Black–Scholes model against observed prices | `src/pricing.py` (BS price, Greeks, implied vol) evaluated with RMSE/MAE/R² and moneyness breakdowns in `train.py` and `src/stock_options.py`. |
| 2 | Deep-learning dynamic volatility model, σ_DL = f_θ(X) | `VolatilityNet` in `model.py`, trained in `train.py` (`train_vol_hybrid`). A `volatility_metrics.csv` report compares σ_DL against the true market volatility. |
| 3 | Hybrid model, C = BS(S, K, T, r, σ_DL) | `VolatilityNet.hybrid_price` + `black_scholes_call_torch` (a differentiable Black-Scholes) in `model.py`. The network is trained end-to-end through Black-Scholes, so it only needs the market price as a target, and it relaxes BS' constant-volatility assumption. Reported as "DL-Vol Hybrid". |
| 4 | Residual learning, E = C_market − C_BS, C = C_BS + Ê | `ResidualOptionANN` in `model.py` (reported as "Residual ANN"). |

These are simulated prices and histories, not observed market options. Strong synthetic results do not establish performance on real contracts. Live option chain downloading is not implemented. `HybridShvimerSubModel1` is an additional architecture defined in `model.py`; the training pipeline does not train it.

## News + cross/co-attention experiments

Two experiments fuse **news text** with **numerical data** using attention, then
compare against numeric-only baselines. Both share the same building blocks
(`src/news/` and `src/attention/`):

- **News ingestion** (`src/news/providers.py`): pluggable live providers —
  `finnhub` and `alphavantage` return historical, date-ranged company news (set
  `FINNHUB_API_KEY` or `ALPHAVANTAGE_API_KEY`); `yfinance` needs no key but
  returns only a few recent headlines. Failures degrade gracefully to "no news".
- **Embeddings** (`src/news/embeddings.py`): pretrained **FinBERT**
  (`ProsusAI/finbert`) sentence embeddings with on-disk caching, plus an offline
  `hashing` fallback for tests / no-internet runs.
- **Fusion** (`src/attention/`): a GRU encodes the numeric sequence into tokens,
  then the modalities are fused by one of three mechanisms before an MLP head:
  **cross-attention** (numeric tokens attend to news), **co-attention** (parallel
  numeric↔news both ways), or **co+cross** (`cocross`: co-attention both ways,
  then a cross-attention stage over the two co-attended streams). Each is
  compared against a **numeric-only** baseline (no news).

Task A — stock next-day return (`train_stock_news.py`):

```powershell
python train_stock_news.py --symbol RELIANCE.NS --epochs 40 --news-mode auto
```

Predicts the next day's log-return from a price-feature window fused with news.
Compares numeric-only, cross-attention and co-attention by RMSE and directional
accuracy. `--news-mode synthetic` (the offline default fallback) fabricates
news correlated with the target so the pipeline runs without an API key;
`--news-mode real` uses a live provider + FinBERT.

Task B — option pricing with sentiment (`train_option_news.py`):

```powershell
python train_option_news.py --samples 6000 --epochs 40
```

Fuses news with the option contract features. Because synthetic options have no
real news feed, this injects an **exogenous sentiment premium** into the traded
price that the numeric features cannot see but the news can, so news-fused models
can beat the numeric-only and Black-Scholes baselines.

Hyperparameter tuning — Optuna (`tune_news.py`):

```powershell
python tune_news.py --task stock  --model co    --trials 25
python tune_news.py --task option --model cross --trials 25
```

Runs a TPE search over architecture/optimiser hyperparameters (minimising
validation loss), retrains the best configuration, and prints the held-out test
metrics of the tuned model versus an untuned default.

**Important caveats.** Live news APIs mostly return *recent* headlines, so
training an attention model over years of history needs a keyed provider with
historical depth (Finnhub/Alpha Vantage) — otherwise news coverage is sparse.
The synthetic news modes generate signal *by construction* to validate the
architecture and the comparison; they are demonstrations, not evidence that real
news improves real forecasts.

## Real market data (replacing the synthetic datasets)

The objectives can be run on **real observed market data**, not just synthetic:

- **Real NSE option contracts** (`src/real_options.py`, `train_real_options.py`):
  downloads NSE's official end-of-day **F&O Bhavcopy** via `jugaad-data` and
  builds a clean call-option dataset (real spot, strikes, settlement prices,
  liquidity-filtered). `train_real_options.py` runs Objectives 1–4 against these
  **observed market prices**.

  ```powershell
  python train_real_options.py --symbol RELIANCE --days-back 18 --epochs 40
  python train_real_options.py --symbol NIFTY    --days-back 12 --epochs 40
  ```

  Honest findings on real data: on **RELIANCE** stock options the **DL-Vol Hybrid**
  (Obj 2 & 3) beats Black–Scholes (e.g. RMSE ₹9.7 vs ₹12.1, R² 0.95 vs 0.93); on
  the highly-liquid **NIFTY index**, Black–Scholes with historical vol is already
  near-perfect (R² ≈ 0.995) and the ML models do not beat it. So deep learning
  helps where there is exploitable smile structure BS misses, less so for liquid
  indices. Caveats: Bhavcopy is end-of-day *settlement* data (not live bid/ask),
  the split is largely cross-sectional over a few recent days, and the risk-free
  rate is a constant assumption.

- **Real Indian stock prices**: yfinance `.NS` symbols (e.g. `RELIANCE.NS`) work
  out of the box in `train_stock_news.py` (omit `--synthetic-prices`).

- **Real Indian news**: the **Marketaux** provider (`--provider marketaux`) covers
  NSE/BSE names with sentiment; set `MARKETAUX_API_KEY` in `.env` (free tier, so
  coverage is sparse). US news uses Finnhub.

Not freely available here: **NSE option chains with live bid/ask** (only Bhavcopy
settlement data) and deep historical **Indian news** (Marketaux free tier is thin).
Real US option *chains* are available via yfinance (`AAPL`, `SPY`) if US focus is
preferred.

## Faithful reference-paper implementations

Two papers are reproduced directly:

- **Shvimer & Zhu (2024), "Pricing options with a new hybrid neural network model"**
  — the BS + NN hybrid. `model.py` implements both sub-models exactly:
  `HybridShvimerSubModel1` (one FC net over `[features, C_BS]`, two 40-unit sigmoid
  layers, 10% dropout, linear output) and `HybridShvimerSubModel2` (a non-parametric
  branch φ then fused with `C_BS` in a final layer, Eq. 8). Both are trained and
  compared against Black–Scholes in `train_real_options.py` on real NSE contracts.
  Note: the paper trained on ~1.5M SPX options; on our few-thousand-contract real
  samples these sigmoid hybrids underfit and do not beat BS, whereas the DL-Vol
  Hybrid does — an honest small-data caveat.

- **Zhang et al. (2026), "Integrative stock price trend prediction … patch-based
  transformer with co-attention"** — `src/attention/patch_coattn.py`
  (`PatchCoAttentionNet`) implements the pipeline: cross-attention alignment
  (price=Query, text=Key/Value) → overlapping **patch** partitioning + per-stream
  self-attention → **bidirectional co-attention fusion** (text↔price, concat,
  linear, residual+LayerNorm, FFN) → up/down **classification**. `train_stock_patch.py`
  trains it and reports **directional accuracy and MCC** (the paper's metrics),
  against a news-ablated baseline.

  ```powershell
  python train_stock_patch.py --symbol RELIANCE.NS --news-mode auto --provider marketaux --embedder finbert
  ```

  (In place of the paper's hierarchical-LLM daily summary we use FinBERT/Marketaux
  headline embeddings; the attention architecture is reproduced faithfully.)

## Files

| File | Purpose |
| --- | --- |
| `app.py`, `templates/index.html` | Flask API and stock dashboard |
| `model_final.ipynb` | Separate stock forecasting notebook |
| `model.py` | PyTorch architectures: direct ANN, residual ANN, LSTM, `VolatilityNet` (σ_DL), and a differentiable Black-Scholes (`black_scholes_call_torch`) for the hybrid pricer |
| `src/stock_options.py` | Stock-calibrated simulation, observed CSV validation, chronological comparison and training |
| `static/stock_options.js` | Stock-specific option tables, graphs and prediction downloads |
| `train.py` | Synthetic option training, evaluation and artifact saving |
| `src/pricing.py` | Black-Scholes call/put prices, Greeks, implied volatility |
| `src/data_loader.py` | Synthetic contracts and simulated price sequences |
| `deep_learning_option_models.ipynb` | Runs the option pricing experiment |
| `src/news/` | News providers (Finnhub, Alpha Vantage, Marketaux, yfinance), FinBERT/hashing embeddings, alignment, synthetic news |
| `src/attention/` | Cross-attention & co-attention modules, fusion models (incl. `cocross`), shared trainer |
| `src/real_options.py`, `train_real_options.py` | Real NSE option data (F&O Bhavcopy); Objectives 1–4 + Shvimer sub-models (PDF 2) on real prices |
| `src/attention/patch_coattn.py`, `train_stock_patch.py` | PDF 1: patch-based cross/co-attention up/down classifier (accuracy, MCC) |
| `train_stock_news.py` | Task A: stock-return prediction with news cross/co-attention |
| `train_option_news.py` | Task B: option pricing with news cross/co-attention |
| `tune_news.py` | Optuna hyperparameter tuning for the fusion models |
| `tests/` | Formula, API, regression and short training checks |
| `outputs/` | Locally generated metrics, weights and plots (not committed) |

## Setup and run

Use Python 3.12. In PowerShell, from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000/. Internet access is required for stock data; each request trains both models and can take several minutes. Charts use native SVG, without an external JavaScript CDN.

The **View the separate saved synthetic benchmark** link opens http://127.0.0.1:5000/options.
It displays all four methods, a metric table and five downloadable comparison
graphs from `outputs/comparison`. These are independent synthetic option results,
not predictions for the stock selected on the home page. A fresh clone has no saved benchmark; generate it with the command below. The homepage's stock-specific comparison works without these saved files.

```powershell
python train.py --samples 3000 --epochs 30 --output-dir outputs/comparison
```

Run the option experiment separately (no market download needed):

```powershell
python train.py --samples 8000 --epochs 50
```

To write a fresh run without replacing previous results:

```powershell
python train.py --samples 100 --epochs 1 --output-dir outputs/smoke
```

New CLI runs save three model weight files, `preprocessing.joblib` (feature order, scalers and target normalization), metric CSVs and five plots. Old weights and metrics were produced before the normalization fix; retrain them together with the preprocessing artifact before reuse. Only load trusted pickle/joblib files. Stock-specific web comparisons train fresh models and do not reuse those saved weights.

Run all tests, including one-epoch training on synthetic/local data:

```powershell
python -m unittest discover -s tests -v
```

For notebooks, select the `.venv` kernel in a Jupyter-capable editor (install `ipykernel` if needed), restart the kernel and run cells in order. Notebook outputs are cleared to keep the files small; rerun cells to regenerate them. The stock notebook is a separate experiment, not the web app's evaluation implementation.

Generated caches, the entire `outputs/` directory, local screenshots and the reference PDF are ignored by Git. Keep `outputs/comparison/` locally to populate the saved benchmark page. The `.venv/` directory holds the installed dependencies; `.uv-cache/` is a disposable package download cache.

## Current limitations

- Forecast models retrain on every web request; there is no job queue or cache.
- Yahoo availability and data quality affect forecasts and fundamentals.
- The app is a local Flask demo, without production deployment infrastructure.
- Simulated option sequences do not demonstrate that real historical sequences improve pricing. Models enforce nonnegative prices, not all no-arbitrage bounds.
- Greeks require strictly positive maturity and volatility; prices support expiry and zero volatility. Implied volatility is undefined at expiry.
- This is an educational experiment, not validated investment guidance.
