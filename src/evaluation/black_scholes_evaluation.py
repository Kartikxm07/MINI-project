"""
src/evaluation/black_scholes_evaluation.py
-------------------------------------------
Objective 1 -- Evaluate the classical Black-Scholes model against observed
market option prices.

Loads the preprocessing handoff datasets, prices every observation with the
standard Black-Scholes model (src/models/black_scholes.py), and evaluates the
pricing error versus the market bid-ask midpoint (`market_price`).

Outputs
-------
  outputs/predictions/call_train_bs.csv   (+ put_train, call_test, put_test)
  outputs/reports/black_scholes_evaluation.txt
  outputs/figures/bs_pred_vs_market.png
  outputs/figures/bs_error_by_moneyness.png
  outputs/figures/bs_error_distribution.png

Error metrics reported (per group):
  RMSE     -- root mean squared error (price units, $)
  MAE      -- mean absolute error ($)
  Bias     -- mean signed error (BS - market); >0 = BS overprices
  MAPE     -- mean absolute percentage error (%)
  MdAPE    -- median absolute percentage error (%), robust to deep-OTM blow-up
  R2       -- coefficient of determination vs market_price

Metrics are reported on the raw price scale and, additionally overall, on the
normalised price/K scale used by the downstream hybrid model.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt
import seaborn as sns

# Make src/ importable when run as a script.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from models.black_scholes import price_from_vix  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FINAL_DIR = os.path.join(_PROJECT_ROOT, "data", "processed", "final")
PRED_DIR = os.path.join(_PROJECT_ROOT, "outputs", "predictions")
REPORT_DIR = os.path.join(_PROJECT_ROOT, "outputs", "reports")
FIG_DIR = os.path.join(_PROJECT_ROOT, "outputs", "figures")

DATASETS = {
    "call_train": os.path.join(FINAL_DIR, "train", "call_train.csv"),
    "put_train": os.path.join(FINAL_DIR, "train", "put_train.csv"),
    "call_test": os.path.join(FINAL_DIR, "test", "call_test.csv"),
    "put_test": os.path.join(FINAL_DIR, "test", "put_test.csv"),
}

MONEYNESS_ORDER = ["OTM", "ATM", "ITM"]


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------
def price_dataset(df):
    """Add BS_price, BS_price_over_K and error columns to a copy of `df`."""
    df = df.copy()
    df["BS_price"] = price_from_vix(
        S=df["S"].to_numpy(),
        K=df["strike"].to_numpy(),
        T=df["T"].to_numpy(),
        vix_points=df["sigma"].to_numpy(),
        r=df["r"].to_numpy(),
        option_type=df["option_type"].to_numpy(),
    )
    df["BS_price_over_K"] = df["BS_price"] / df["strike"]
    df["error"] = df["BS_price"] - df["market_price"]          # signed
    df["abs_error"] = df["error"].abs()
    df["pct_error"] = 100.0 * df["error"] / df["market_price"]
    df["abs_pct_error"] = df["pct_error"].abs()
    return df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(pred, actual):
    """Return a dict of error metrics comparing `pred` to `actual` (arrays)."""
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    err = pred - actual
    abs_err = np.abs(err)
    ape = 100.0 * abs_err / actual

    ss_res = np.sum(err**2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "N": int(len(actual)),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAE": float(np.mean(abs_err)),
        "Bias": float(np.mean(err)),
        "MAPE": float(np.mean(ape)),
        "MdAPE": float(np.median(ape)),
        "R2": float(r2),
    }


def metrics_table(df, group_cols=None):
    """Build a tidy metrics DataFrame, optionally grouped by columns."""
    rows = []
    if group_cols is None:
        m = compute_metrics(df["BS_price"], df["market_price"])
        m = {"group": "ALL", **m}
        rows.append(m)
    else:
        for keys, g in df.groupby(group_cols, sort=False, observed=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            label = " | ".join(str(k) for k in keys)
            m = compute_metrics(g["BS_price"], g["market_price"])
            rows.append({"group": label, **m})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def _fmt_table(df):
    disp = df.copy()
    for c in ["RMSE", "MAE", "Bias"]:
        disp[c] = disp[c].map(lambda v: f"{v:8.4f}")
    for c in ["MAPE", "MdAPE"]:
        disp[c] = disp[c].map(lambda v: f"{v:7.2f}%")
    disp["R2"] = disp["R2"].map(lambda v: f"{v:7.4f}")
    return disp.to_string(index=False)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(full):
    sns.set_theme(style="whitegrid")

    # 1. Predicted vs market, faceted by option type, coloured by split.
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, otype in zip(axes, ["CALL", "PUT"]):
        sub = full[full["option_type"] == otype]
        for split, color in [("train", "#4C72B0"), ("test", "#DD8452")]:
            s = sub[sub["split"] == split]
            ax.scatter(s["market_price"], s["BS_price"], s=10, alpha=0.4,
                       label=split, color=color)
        lim = [0, max(sub["market_price"].max(), sub["BS_price"].max()) * 1.02]
        ax.plot(lim, lim, "k--", lw=1, label="perfect")
        ax.set(xlim=lim, ylim=lim, xlabel="Market price ($)",
               ylabel="Black-Scholes price ($)", title=f"{otype}")
        ax.legend()
    fig.suptitle("Black-Scholes price vs market price", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "bs_pred_vs_market.png"), dpi=130)
    plt.close(fig)

    # 2. Signed error by moneyness category (box), split by option type.
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=full, x="moneyness_category", y="error", hue="option_type",
        order=MONEYNESS_ORDER, showfliers=False, ax=ax,
    )
    ax.axhline(0, color="k", lw=1)
    ax.set(xlabel="Moneyness category", ylabel="Signed error: BS - market ($)",
           title="Black-Scholes pricing error by moneyness")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "bs_error_by_moneyness.png"), dpi=130)
    plt.close(fig)

    # 3. Error distribution.
    fig, ax = plt.subplots(figsize=(10, 6))
    for otype, color in [("CALL", "#4C72B0"), ("PUT", "#DD8452")]:
        sub = full[full["option_type"] == otype]["error"]
        ax.hist(sub, bins=80, alpha=0.5, label=otype, color=color)
    ax.axvline(0, color="k", lw=1)
    ax.set(xlabel="Signed error: BS - market ($)", ylabel="Count",
           title="Distribution of Black-Scholes pricing errors")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "bs_error_distribution.png"), dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    priced = {}
    for name, path in DATASETS.items():
        df = pd.read_csv(path)
        pdf = price_dataset(df)
        pdf.to_csv(os.path.join(PRED_DIR, f"{name}_bs.csv"), index=False)
        otype, split = name.split("_")
        pdf["split"] = split
        pdf["option_type_key"] = otype.upper()
        priced[name] = pdf

    full = pd.concat(priced.values(), ignore_index=True)

    make_figures(full)

    # ---- Build the report ----
    lines = []
    w = lines.append
    w("=" * 78)
    w("OBJECTIVE 1 -- CLASSICAL BLACK-SCHOLES: PRICING EVALUATION")
    w("=" * 78)
    w("")
    w("Model  : standard Black-Scholes-Merton, no dividends (q = 0)")
    w("Target : market_price (bid-ask midpoint)")
    w("sigma  : VIX close converted to decimal volatility (VIX / 100)")
    w("r      : maturity-matched risk-free rate (decimal, used as-is)")
    w(f"Data   : trading date(s) {full['date'].min()} -> {full['date'].max()}")
    w(f"Total observations priced: {len(full)}")
    w("")
    w("Metric key: Bias = mean(BS - market); >0 means BS overprices.")
    w("            MdAPE = median abs % error (robust to deep-OTM blow-up).")
    w("")

    w("-" * 78)
    w("1. OVERALL (all observations)")
    w("-" * 78)
    w(_fmt_table(metrics_table(full)))
    w("")

    w("-" * 78)
    w("2. BY SPLIT x OPTION TYPE")
    w("-" * 78)
    w(_fmt_table(metrics_table(full, ["split", "option_type"])))
    w("")

    w("-" * 78)
    w("3. BY OPTION TYPE x MONEYNESS CATEGORY (all data)")
    w("-" * 78)
    cat = full.copy()
    cat["moneyness_category"] = pd.Categorical(
        cat["moneyness_category"], categories=MONEYNESS_ORDER, ordered=True)
    cat = cat.sort_values(["option_type", "moneyness_category"])
    w(_fmt_table(metrics_table(cat, ["option_type", "moneyness_category"])))
    w("")

    w("-" * 78)
    w("4. TEST SET ONLY (out-of-sample) BY OPTION TYPE x MONEYNESS")
    w("-" * 78)
    test = cat[cat["split"] == "test"]
    w(_fmt_table(metrics_table(test, ["option_type", "moneyness_category"])))
    w("")

    # Normalised price/K scale (what the hybrid model consumes).
    w("-" * 78)
    w("5. NORMALISED SCALE: price / K  (overall, by option type)")
    w("-" * 78)
    norm_rows = []
    for otype, g in full.groupby("option_type"):
        m = compute_metrics(g["BS_price_over_K"], g["market_price_over_K"])
        norm_rows.append({"group": otype, **m})
    ndf = pd.DataFrame(norm_rows)
    # price/K values are tiny; show RMSE/MAE/Bias with more precision.
    disp = ndf.copy()
    for c in ["RMSE", "MAE", "Bias"]:
        disp[c] = disp[c].map(lambda v: f"{v:10.6f}")
    for c in ["MAPE", "MdAPE"]:
        disp[c] = disp[c].map(lambda v: f"{v:7.2f}%")
    disp["R2"] = disp["R2"].map(lambda v: f"{v:7.4f}")
    w(disp.to_string(index=False))
    w("")

    w("-" * 78)
    w("NOTES")
    w("-" * 78)
    w("- MAPE is inflated by deep out-of-the-money options whose market price")
    w("  is near the $0.125 floor; a small $ error is a large % error there.")
    w("  MdAPE and RMSE give a more stable picture of typical accuracy.")
    w("- A single flat VIX volatility is used for every strike/maturity, so")
    w("  Black-Scholes cannot capture the volatility smile; systematic error")
    w("  across moneyness categories is expected and is the motivation for the")
    w("  downstream hybrid neural-network correction.")
    w("")

    report = "\n".join(lines)
    report_path = os.path.join(REPORT_DIR, "black_scholes_evaluation.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"\nReport written to : {report_path}")
    print(f"Predictions in    : {PRED_DIR}")
    print(f"Figures in        : {FIG_DIR}")


if __name__ == "__main__":
    main()
