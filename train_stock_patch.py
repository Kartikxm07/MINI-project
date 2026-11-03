"""
Stock movement (up/down) prediction with the patch + cross/co-attention model of
Zhang et al. (2026, PDF 1). Metrics follow the paper: directional accuracy and
Matthews Correlation Coefficient (MCC).

Compares the full model (price + news) against a news-ablated baseline (same
architecture, news zeroed) to isolate the contribution of the text modality.

    python train_stock_patch.py --symbol RELIANCE.NS --news-mode auto --provider marketaux
"""

import argparse
import copy
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score

import train_stock_news as A
from src.attention.patch_coattn import PatchCoAttentionNet
from src.attention.trainer import set_seed


def _loaders(d, labels, bs, shuffle, zero_news=False):
    news = np.zeros_like(d["news_emb"]) if zero_news else d["news_emb"]
    mask = np.zeros_like(d["news_mask"]) if zero_news else d["news_mask"]
    ds = TensorDataset(torch.tensor(d["numeric"], dtype=torch.float32),
                       torch.tensor(news, dtype=torch.float32),
                       torch.tensor(mask, dtype=torch.bool),
                       torch.tensor(labels, dtype=torch.float32))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)


def train_one(data, labels, price_dim, news_dim, args, device, zero_news):
    set_seed(args.seed)
    model = PatchCoAttentionNet(price_dim, news_dim, d_model=args.d_model, num_heads=args.num_heads,
                                patch_len=args.patch_len, stride=args.stride, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.BCEWithLogitsLoss()
    tl = _loaders(data["train"], labels["train"], args.batch_size, True, zero_news)
    vl = _loaders(data["val"], labels["val"], 256, False, zero_news)
    best, best_state = float("inf"), copy.deepcopy(model.state_dict())
    for _ in range(args.epochs):
        model.train()
        for num, news, mask, y in tl:
            num, news, mask, y = num.to(device), news.to(device), mask.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(num, news, mask), y); loss.backward(); opt.step()
        model.eval(); vrun = 0.0
        with torch.no_grad():
            for num, news, mask, y in vl:
                num, news, mask, y = num.to(device), news.to(device), mask.to(device), y.to(device)
                vrun += crit(model(num, news, mask), y).item() * len(num)
        vrun /= len(vl.dataset)
        if vrun < best:
            best, best_state = vrun, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate(model, data, labels, device, zero_news):
    model.eval()
    loader = _loaders(data["test"], labels["test"], 256, False, zero_news)
    logits = []
    for num, news, mask, _ in loader:
        logits.append(model(num.to(device), news.to(device), mask.to(device)).cpu().numpy())
    prob = 1 / (1 + np.exp(-np.concatenate(logits).reshape(-1)))
    pred = (prob >= 0.5).astype(int)
    y = labels["test"].reshape(-1).astype(int)
    return {"Accuracy": accuracy_score(y, pred), "MCC": matthews_corrcoef(y, pred),
            "F1": f1_score(y, pred, zero_division=0)}


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    base = A.build_arg_parser().parse_args([])
    for k in ["symbol", "period", "synthetic_prices", "news_mode", "provider", "embedder",
              "lookback_days", "max_headlines", "synthetic_dim", "news_strength",
              "news_coverage", "window", "seed"]:
        setattr(base, k, getattr(args, k))
    data = A.prepare_data(base)

    # Up/down labels from the next-day return target.
    labels = {sp: (data[sp]["y"].reshape(-1) > 0).astype(np.float32).reshape(-1, 1)
              for sp in ["train", "val", "test"]}
    price_dim, news_dim = data["num_feat_dim"], data["news_dim"]
    base_rate = max(labels["test"].mean(), 1 - labels["test"].mean())
    print(f"\nPDF1 patch co-attention | {args.symbol} | samples {data['n']} "
          f"| news={data['news_label']} | test up-rate {labels['test'].mean():.2f}\n")

    rows = []
    for zero_news, name in [(True, "Numeric-only (no news)"), (False, "Patch Cross+Co-Attention (news)")]:
        model = train_one(data, labels, price_dim, news_dim, args, device, zero_news)
        m = evaluate(model, data, labels, device, zero_news)
        m = {"Model": name, **m}
        rows.append(m)
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"patch_{'base' if zero_news else 'full'}.pth"))
        print(f"[{name:34s}] Acc {m['Accuracy']:.3f} | MCC {m['MCC']:.3f} | F1 {m['F1']:.3f}")

    res = pd.DataFrame(rows)
    print("\n--- PDF1: STOCK MOVEMENT (up/down) — held-out test ---")
    print(res.to_string(index=False))
    res.to_csv(os.path.join(args.output_dir, "patch_coattn_metrics.csv"), index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(res["Model"], res["Accuracy"], color=["gray", "orange"]); ax[0].axhline(base_rate, color="red", ls="--", label="majority class"); ax[0].set_title("Directional accuracy"); ax[0].tick_params(axis="x", rotation=15); ax[0].legend()
    ax[1].bar(res["Model"], res["MCC"], color=["gray", "orange"]); ax[1].axhline(0, color="red", ls="--"); ax[1].set_title("MCC (0 = random)"); ax[1].tick_params(axis="x", rotation=15)
    fig.tight_layout(); fig.savefig(os.path.join(args.output_dir, "patch_coattn_comparison.png"), dpi=140); plt.close(fig)
    print(f"[artifacts] saved to {args.output_dir}/")
    return res


def build_arg_parser():
    p = argparse.ArgumentParser(description="PDF1 patch cross/co-attention stock movement classifier")
    p.add_argument("--symbol", default="RELIANCE.NS")
    p.add_argument("--period", default="2y")
    p.add_argument("--synthetic-prices", action="store_true")
    p.add_argument("--news-mode", choices=["auto", "real", "synthetic"], default="synthetic")
    p.add_argument("--provider", default="marketaux")
    p.add_argument("--embedder", choices=["finbert", "hashing"], default="finbert")
    p.add_argument("--lookback-days", type=int, default=7)
    p.add_argument("--max-headlines", type=int, default=8)
    p.add_argument("--synthetic-dim", type=int, default=64)
    p.add_argument("--news-strength", type=float, default=0.6)
    p.add_argument("--news-coverage", type=float, default=0.8)
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--patch-len", type=int, default=4)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="outputs/patch_coattn")
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
