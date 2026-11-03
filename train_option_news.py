"""
Task B -- Option pricing with news + cross/co-attention.

Extends the synthetic option experiment: each contract's numeric features (a
15-step simulated price path plus static contract terms [S/K, T, r, sigma_hist,
C_bs/K]) are fused with news embeddings via cross- and co-attention, and the
models predict the (strike-normalised) call price. Compared against:

    * Black-Scholes      -- constant-vol baseline (reference)
    * numeric-only       -- attention model, no news
    * cross / co         -- news-fused attention models

Options have no natural per-contract live news feed, so this task uses SYNTHETIC
news whose sentiment correlates with the Black-Scholes mispricing (C_market -
C_bs) -- i.e. news stands in for the volatility-smile information constant-vol BS
misses. It demonstrates the fusion architecture on the option numerical data;
it is not a claim about real option news.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data_loader import generate_market_options_data, generate_time_series_option_sequences
from src.attention.fusion_models import build_model, ModelConfig
from src.attention.trainer import set_seed, train_model, predict
from src.news.aggregate import synthetic_news_embeddings

MODEL_KINDS = ["numeric", "cross", "co", "cocross"]
MODEL_LABELS = {"numeric": "Numeric-only (no news)",
                "cross": "Cross-Attention (news)",
                "co": "Co-Attention (news)",
                "cocross": "Co+Cross-Attention (news)"}
PALETTE = ["dimgray", "gray", "steelblue", "orange", "seagreen"]


def price_metrics(y_true, y_pred, name):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return {"Model": name,
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred))}


def build_numeric(df, seq_len, seed):
    """Numeric tokens = per-step [path S/K, return] ++ broadcast static features."""
    X_seq, X_static, _, _ = generate_time_series_option_sequences(df, seq_len=seq_len, seed=seed)
    static_rep = np.repeat(X_static[:, None, :], X_seq.shape[1], axis=1)
    numeric = np.concatenate([X_seq, static_rep], axis=-1).astype(np.float32)  # (N, seq_len, 7)
    return numeric


def standardize_numeric(numeric, n_train):
    flat = numeric[:n_train].reshape(-1, numeric.shape[-1])
    mu = flat.mean(0, keepdims=True)
    sd = flat.std(0, keepdims=True) + 1e-8
    return ((numeric - mu) / sd).astype(np.float32)


def slice_data(numeric, news_emb, news_mask, y, lo, hi):
    sl = slice(lo, hi)
    return {"numeric": numeric[sl], "news_emb": news_emb[sl],
            "news_mask": news_mask[sl], "y": y[sl]}


def prepare_data(args):
    """Assemble train/val/test splits and metadata (shared by run and tuning)."""
    set_seed(args.seed)
    df = generate_market_options_data(n_samples=args.samples, seed=args.seed)
    numeric = build_numeric(df, seq_len=args.seq_len, seed=args.seed)
    K = df["K"].to_numpy()
    C_market = df["C_market"].to_numpy()
    C_bs = df["C_bs"].to_numpy()

    # Exogenous sentiment premium: a latent factor z that shifts the traded price
    # beyond what fundamentals (the numeric features) explain. The numeric models
    # CANNOT see z; the news is generated from z, so news-fused models can recover
    # part of this premium. This models the real-world idea that sentiment moves
    # prices, and is what makes the news comparison meaningful on synthetic data.
    rng = np.random.default_rng(args.seed + 1)
    z = rng.standard_normal(len(df))
    intrinsic = np.maximum(0.0, df["S"].to_numpy() - K * np.exp(-df["r"].to_numpy() * df["T"].to_numpy()))
    premium = args.sentiment_shock * z * K
    C_adj = np.maximum(intrinsic, C_market + premium)           # observed/traded price
    target = (C_adj / K).reshape(-1, 1).astype(np.float32)      # strike-normalised price

    n = len(df)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    numeric = standardize_numeric(numeric, n_train)

    # News sentiment correlates with the latent premium z.
    news_emb, news_mask, _ = synthetic_news_embeddings(
        z, dim=args.synthetic_dim, max_headlines=args.max_headlines,
        strength=args.news_strength, noise=1.0, coverage=args.news_coverage, seed=args.seed)

    test_lo = n_train + n_val
    return {
        "task": "option",
        "train": slice_data(numeric, news_emb, news_mask, target, 0, n_train),
        "val": slice_data(numeric, news_emb, news_mask, target, n_train, n_train + n_val),
        "test": slice_data(numeric, news_emb, news_mask, target, test_lo, n),
        "num_feat_dim": numeric.shape[-1],
        "news_dim": news_emb.shape[-1],
        "K_test": K[test_lo:n],
        "y_test_price": C_adj[test_lo:n],
        "C_bs_test": C_bs[test_lo:n],
        "splits": (n_train, n_val, n - test_lo),
        "n": n,
    }


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    data = prepare_data(args)
    train_d, val_d, test_d = data["train"], data["val"], data["test"]
    K_test, y_test_price, C_bs_test = data["K_test"], data["y_test_price"], data["C_bs_test"]
    num_feat_dim, news_dim = data["num_feat_dim"], data["news_dim"]
    n_train, n_val, n_test = data["splits"]
    n = data["n"]

    cfg = ModelConfig(d_model=args.d_model, num_heads=args.num_heads,
                      mlp_hidden=args.mlp_hidden, dropout=args.dropout, out_dim=1)

    print(f"\nTask B: option price (strike-normalised) | contracts={n} "
          f"(train {n_train}/val {n_val}/test {n_test}) | news=synthetic\n")

    # Reference: Black-Scholes constant-vol baseline.
    rows = [price_metrics(y_test_price, C_bs_test, "Classical Black-Scholes")]
    histories, preds = {}, {}
    for kind in MODEL_KINDS:
        set_seed(args.seed)
        model = build_model(kind, num_feat_dim, news_dim, cfg)
        model, hist, best_val = train_model(
            model, train_d, val_d, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, weight_decay=args.weight_decay,
            device=device, verbose=args.verbose)
        yhat_norm = predict(model, test_d, device=device).reshape(-1)
        yhat_price = np.maximum(0.0, yhat_norm * K_test)
        rows.append(price_metrics(y_test_price, yhat_price, MODEL_LABELS[kind]))
        histories[kind] = hist
        preds[kind] = yhat_price
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"option_{kind}.pth"))
        print(f"[{kind:7s}] RMSE {rows[-1]['RMSE']:.4f} | R2 {rows[-1]['R2']:.4f}")

    results = pd.DataFrame(rows)
    print("\n--- TASK B: OPTION PRICE PREDICTION (held-out test) ---")
    print(results.to_string(index=False))
    results.to_csv(os.path.join(args.output_dir, "option_news_metrics.csv"), index=False)
    _plot(results, histories, args.output_dir)
    return results


def _plot(results, histories, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].bar(results["Model"], results["RMSE"], color=PALETTE[:len(results)])
    axes[0].set_title("Test RMSE on option price (lower better)")
    axes[0].tick_params(axis="x", rotation=20)
    for kind, hist in histories.items():
        axes[1].plot(hist["val"], label=MODEL_LABELS[kind])
    axes[1].set_title("Validation loss")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "option_news_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"[artifacts] saved to {output_dir}/")


def build_arg_parser():
    p = argparse.ArgumentParser(description="Option pricing with news cross/co-attention")
    p.add_argument("--samples", type=int, default=6000)
    p.add_argument("--seq-len", type=int, default=15)
    p.add_argument("--max-headlines", type=int, default=8)
    p.add_argument("--synthetic-dim", type=int, default=64)
    p.add_argument("--news-strength", type=float, default=1.2)
    p.add_argument("--news-coverage", type=float, default=0.9)
    p.add_argument("--sentiment-shock", type=float, default=0.006,
                   help="std of the latent sentiment premium, in units of strike")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--mlp-hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="outputs/news_option")
    p.add_argument("--verbose", action="store_true")
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
