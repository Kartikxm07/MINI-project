"""
Task A -- Stock movement prediction with news + cross/co-attention.

Predicts the next day's log-return for a stock from a trailing window of price
features, fused with news-headline embeddings. Three models are compared on the
same chronological test set:

    * numeric  -- price window only (baseline, no news)
    * cross    -- price tokens cross-attend to news
    * co       -- parallel co-attention (price <-> news)

News source:
    --news-mode real       live provider (Finnhub/AlphaVantage/yfinance) + FinBERT
    --news-mode synthetic  fabricated headlines/embeddings correlated with the
                           target (offline demo; see src/news/aggregate.py)
    --news-mode auto       real when a keyed provider returns news, else synthetic

Example:
    python train_stock_news.py --symbol RELIANCE.NS --epochs 40 --news-mode auto
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.attention.fusion_models import build_model, ModelConfig
from src.attention.trainer import set_seed, train_model, predict, evaluate_regression
from src.news.providers import get_provider
from src.news.embeddings import get_embedder
from src.news.aggregate import group_by_date, build_news_tensor, synthetic_news_embeddings

MODEL_KINDS = ["numeric", "cross", "co", "cocross"]
MODEL_LABELS = {"numeric": "Numeric-only (no news)",
                "cross": "Cross-Attention (news)",
                "co": "Co-Attention (news)",
                "cocross": "Co+Cross-Attention (news)"}
PALETTE = ["gray", "steelblue", "orange", "seagreen", "crimson"]


# ─── Data ────────────────────────────────────────────────────────────────────

def load_prices(symbol, period="3y", synthetic=False, seed=42):
    """Return a daily OHLCV DataFrame; fall back to synthetic GBM if needed."""
    if not synthetic:
        try:
            import yfinance as yf
            df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            if len(df) >= 200:
                print(f"[data] Downloaded {len(df)} daily bars for {symbol}.")
                return df[["Open", "High", "Low", "Close", "Volume"]]
            print(f"[data] Only {len(df)} bars for {symbol}; using synthetic prices.")
        except Exception as exc:
            print(f"[data] Price download failed ({exc}); using synthetic prices.")
    # Synthetic geometric Brownian motion with mild autocorrelation.
    rng = np.random.default_rng(seed)
    n = 760
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    rets = 0.0003 + 0.015 * rng.standard_normal(n)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(0.01 * rng.standard_normal(n)))
    low = close * (1 - np.abs(0.01 * rng.standard_normal(n)))
    openp = close * (1 + 0.005 * rng.standard_normal(n))
    vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
    print(f"[data] Generated {n} synthetic daily bars.")
    return pd.DataFrame({"Open": openp, "High": high, "Low": low, "Close": close, "Volume": vol},
                        index=dates)


def build_samples(df, window=20):
    """
    Build windowed numeric features and next-day-return targets.

    Feature per day t: [log-return, (High-Low)/Close, log volume change].
    numeric[i] is the window of features ending at day t; target[i] is the
    log-return from t to t+1 (strictly future). sample_date[i] = date of day t.
    """
    close = df["Close"].to_numpy()
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    vol = df["Volume"].to_numpy().astype(float)

    log_ret = np.zeros(len(close))
    log_ret[1:] = np.log(close[1:] / close[:-1])
    hl_range = (high - low) / np.clip(close, 1e-8, None)
    vol_chg = np.zeros(len(vol))
    vol_chg[1:] = np.log(np.clip(vol[1:], 1.0, None) / np.clip(vol[:-1], 1.0, None))
    feats = np.column_stack([log_ret, hl_range, vol_chg]).astype(np.float32)

    numeric, target, dates = [], [], []
    for t in range(window, len(df) - 1):
        numeric.append(feats[t - window + 1:t + 1])
        target.append(np.log(close[t + 1] / close[t]))
        dates.append(df.index[t])
    return (np.asarray(numeric, dtype=np.float32),
            np.asarray(target, dtype=np.float32).reshape(-1, 1),
            pd.to_datetime(dates))


def standardize_numeric(numeric, n_train):
    """Per-feature standardization using training rows only."""
    flat = numeric[:n_train].reshape(-1, numeric.shape[-1])
    mu = flat.mean(0, keepdims=True)
    sd = flat.std(0, keepdims=True) + 1e-8
    return ((numeric - mu) / sd).astype(np.float32)


# ─── News ────────────────────────────────────────────────────────────────────

def assemble_news(mode, symbol, sample_dates, target, args):
    """Return (news_emb, news_mask, label) aligned to sample_dates."""
    if mode in ("auto", "real"):
        provider = get_provider(args.provider)
        start, end = sample_dates.min(), sample_dates.max()
        print(f"[news] Fetching from provider '{provider.name}' ({start.date()}..{end.date()})...")
        items = provider.fetch(symbol, start, end)
        print(f"[news] Retrieved {len(items)} headlines.")
        if items:
            embedder = get_embedder(args.embedder)
            by_date = group_by_date(items)
            emb, mask = build_news_tensor(sample_dates, by_date, embedder,
                                          max_headlines=args.max_headlines,
                                          lookback_days=args.lookback_days)
            cov = mask.any(1).mean()
            print(f"[news] Embedded with '{args.embedder}' (dim={embedder.dim}); "
                  f"news coverage {cov:.1%} of samples.")
            return emb, mask, f"real:{provider.name}+{args.embedder}"
        if mode == "real":
            print("[news] No live news returned; proceeding with empty news tensors.")
            dim = get_embedder(args.embedder).dim
            n = len(sample_dates)
            return (np.zeros((n, args.max_headlines, dim), np.float32),
                    np.zeros((n, args.max_headlines), bool), f"real-empty:{provider.name}")
        print("[news] No live news; falling back to synthetic news.")

    emb, mask, _ = synthetic_news_embeddings(
        target.reshape(-1), dim=args.synthetic_dim, max_headlines=args.max_headlines,
        strength=args.news_strength, noise=1.0, coverage=args.news_coverage,
        seed=args.seed, symbol=symbol.replace(".NS", ""))
    print(f"[news] Synthetic news (dim={args.synthetic_dim}, strength={args.news_strength}, "
          f"coverage={args.news_coverage}).")
    return emb, mask, "synthetic"


# ─── Pipeline ────────────────────────────────────────────────────────────────

def slice_data(numeric, news_emb, news_mask, y, idx):
    return {"numeric": numeric[idx], "news_emb": news_emb[idx],
            "news_mask": news_mask[idx], "y": y[idx]}


def prepare_data(args):
    """Assemble train/val/test splits and metadata (shared by run and tuning)."""
    set_seed(args.seed)
    df = load_prices(args.symbol, period=args.period, synthetic=args.synthetic_prices, seed=args.seed)
    numeric, target, dates = build_samples(df, window=args.window)
    n = len(numeric)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    numeric = standardize_numeric(numeric, n_train)

    news_emb, news_mask, news_label = assemble_news(args.news_mode, args.symbol, dates, target, args)

    idx_tr = np.arange(0, n_train)
    idx_va = np.arange(n_train, n_train + n_val)
    idx_te = np.arange(n_train + n_val, n)
    return {
        "task": "stock",
        "train": slice_data(numeric, news_emb, news_mask, target, idx_tr),
        "val": slice_data(numeric, news_emb, news_mask, target, idx_va),
        "test": slice_data(numeric, news_emb, news_mask, target, idx_te),
        "num_feat_dim": numeric.shape[-1],
        "news_dim": news_emb.shape[-1],
        "news_label": news_label,
        "splits": (n_train, n_val, len(idx_te)),
        "n": n,
    }


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    data = prepare_data(args)
    train_d, val_d, test_d = data["train"], data["val"], data["test"]
    num_feat_dim, news_dim = data["num_feat_dim"], data["news_dim"]
    n_train, n_val, n_test = data["splits"]
    cfg = ModelConfig(d_model=args.d_model, num_heads=args.num_heads,
                      mlp_hidden=args.mlp_hidden, dropout=args.dropout, out_dim=1)

    print(f"\nTask A: stock next-day return | samples={data['n']} "
          f"(train {n_train}/val {n_val}/test {n_test}) | news={data['news_label']}\n")

    rows, histories, preds = [], {}, {}
    for kind in MODEL_KINDS:
        set_seed(args.seed)
        model = build_model(kind, num_feat_dim, news_dim, cfg)
        model, hist, best_val = train_model(
            model, train_d, val_d, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, weight_decay=args.weight_decay,
            device=device, verbose=args.verbose)
        yhat = predict(model, test_d, device=device)
        metrics = evaluate_regression(test_d["y"], yhat, MODEL_LABELS[kind])
        metrics["BestVal"] = best_val
        rows.append(metrics)
        histories[kind] = hist
        preds[kind] = yhat
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"stock_{kind}.pth"))
        print(f"[{kind:7s}] RMSE {metrics['RMSE']:.5f} | DirAcc {metrics['DirAcc']:.3f}")

    results = pd.DataFrame(rows)
    print("\n--- TASK A: STOCK RETURN PREDICTION (held-out test) ---")
    print(results.to_string(index=False))
    results.to_csv(os.path.join(args.output_dir, "stock_news_metrics.csv"), index=False)
    _plot(results, histories, args.output_dir)
    return results


def _plot(results, histories, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    colors = PALETTE[:len(results)]
    axes[0].bar(results["Model"], results["RMSE"], color=colors)
    axes[0].set_title("Test RMSE (lower better)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(results["Model"], results["DirAcc"], color=colors)
    axes[1].axhline(0.5, color="red", ls="--", label="chance")
    axes[1].set_title("Directional accuracy (higher better)")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend()
    for kind, hist in histories.items():
        axes[2].plot(hist["val"], label=MODEL_LABELS[kind])
    axes[2].set_title("Validation loss")
    axes[2].set_xlabel("epoch")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "stock_news_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"[artifacts] saved to {output_dir}/")


def build_arg_parser():
    p = argparse.ArgumentParser(description="Stock return prediction with news cross/co-attention")
    p.add_argument("--symbol", default="RELIANCE.NS")
    p.add_argument("--period", default="3y")
    p.add_argument("--synthetic-prices", action="store_true", help="skip yfinance, use synthetic prices")
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--news-mode", choices=["auto", "real", "synthetic"], default="auto")
    p.add_argument("--provider", default=None, help="news provider (default: auto by API key)")
    p.add_argument("--embedder", choices=["finbert", "hashing"], default="finbert")
    p.add_argument("--max-headlines", type=int, default=8)
    p.add_argument("--lookback-days", type=int, default=3)
    p.add_argument("--synthetic-dim", type=int, default=64)
    p.add_argument("--news-strength", type=float, default=0.6)
    p.add_argument("--news-coverage", type=float, default=0.8)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--mlp-hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="outputs/news_stock")
    p.add_argument("--verbose", action="store_true")
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
