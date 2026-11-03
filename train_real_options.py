"""
Objectives 1-4 on REAL NSE option contracts (from F&O Bhavcopy).

Replaces the synthetic option experiment with observed market prices for a chosen
NSE underlying (e.g. NIFTY, RELIANCE). Trains and compares on a held-out set:

    1. Classical Black-Scholes (historical vol)        [Objective 1]
    2. Direct ANN         (X -> price)
    3. Residual ANN       (X -> C_market - C_bs)        [Objective 4]
    4. DL-Vol Hybrid      (X -> sigma_DL, BS(sigma_DL)) [Objectives 2 & 3]

The split is by trade date (latest day(s) held out) for a mild out-of-time test.
Option LSTM is omitted: Bhavcopy has no per-contract price sequence.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

from src.real_options import build_option_dataset
from src.pricing import black_scholes_call, implied_volatility
from model import HybridShvimerSubModel1, HybridShvimerSubModel2
from train import (train_ann_direct, train_ann_residual, train_vol_hybrid,
                   evaluate_predictions, evaluate_by_moneyness)

ANN_FEATURES = ["moneyness", "T", "r", "sigma_hist"]


def train_shvimer(kind, X, cbs, y, Xv, cbsv, yv, epochs=40, lr=1e-3, bs=64, device="cpu"):
    """
    Train a Shvimer & Zhu (2024) hybrid sub-model (PDF 2).
    kind='sub1': HybridShvimerSubModel1 on [features, C_bs/K] -> C_market/K.
    kind='sub2': HybridShvimerSubModel2 on features + C_bs/K  -> C_market/K.
    X: standardised features (N,4); cbs: C_bs/K (N,1); y: C_market/K (N,1).
    The BS-price input and target are standardised internally (they are tiny
    fractions) so the sigmoid network converges; returns (model, cbs_scaler, y_scaler).
    """
    import copy
    sc_c, sc_y = StandardScaler().fit(cbs), StandardScaler().fit(y)
    cbs_s, y_s = sc_c.transform(cbs).astype(np.float32), sc_y.transform(y).astype(np.float32)
    cbsv_s, yv_s = sc_c.transform(cbsv).astype(np.float32), sc_y.transform(yv).astype(np.float32)

    model = (HybridShvimerSubModel1(input_dim=X.shape[1] + 1) if kind == "sub1"
             else HybridShvimerSubModel2(input_dim=X.shape[1])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    tl = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32),
                                  torch.tensor(cbs_s), torch.tensor(y_s)), batch_size=bs, shuffle=True)

    def fwd(m, x, c):
        return m(torch.cat([x, c], -1)) if kind == "sub1" else m(x, c)

    best, best_state = float("inf"), copy.deepcopy(model.state_dict())
    Xv_t = torch.tensor(Xv, dtype=torch.float32).to(device)
    cbsv_t = torch.tensor(cbsv_s).to(device); yv_t = torch.tensor(yv_s).to(device)
    for _ in range(epochs):
        model.train()
        for bx, bc, by in tl:
            bx, bc, by = bx.to(device), bc.to(device), by.to(device)
            opt.zero_grad(); loss = crit(fwd(model, bx, bc), by); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(fwd(model, Xv_t, cbsv_t), yv_t).item()
        if vl < best:
            best, best_state = vl, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, sc_c, sc_y


def predict_shvimer(kind, model, sc_c, sc_y, X, cbs, K, device="cpu"):
    """Predict real option prices (₹) from a trained Shvimer sub-model."""
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    ct = torch.tensor(sc_c.transform(cbs).astype(np.float32)).to(device)
    model.eval()
    with torch.no_grad():
        out = model(torch.cat([Xt, ct], -1)) if kind == "sub1" else model(Xt, ct)
    pred_norm = sc_y.inverse_transform(out.cpu().numpy()).flatten()
    return np.maximum(0.0, pred_norm * K)


def prepare(df):
    df = df.sort_values("TradDt").reset_index(drop=True)
    df["C_bs"] = black_scholes_call(df["S"].to_numpy(), df["K"].to_numpy(), df["T"].to_numpy(),
                                    df["r"].to_numpy(), df["sigma_hist"].to_numpy())
    df["C_error"] = df["C_market"] - df["C_bs"]
    n = len(df)
    n_tr, n_va = int(0.70 * n), int(0.15 * n)
    return df, n_tr, n_va


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Downloading real NSE option data for {args.symbol} ...")
    df = build_option_dataset(args.symbol, days_back=args.days_back, rate=args.rate,
                              min_volume=args.min_volume, max_contracts=args.max_contracts)
    df, n_tr, n_va = prepare(df)
    n = len(df)
    print(f"Real {args.symbol} call contracts: {n} "
          f"(train {n_tr}/val {n_va}/test {n - n_tr - n_va}); "
          f"dates {df['TradDt'].min().date()}..{df['TradDt'].max().date()}")

    tr, va = df.iloc[:n_tr], df.iloc[n_tr:n_tr + n_va]
    te = df.iloc[n_tr + n_va:]
    K_te = te["K"].to_numpy()

    scaler_x = StandardScaler()
    Xtr = scaler_x.fit_transform(tr[ANN_FEATURES].to_numpy())
    Xva = scaler_x.transform(va[ANN_FEATURES].to_numpy())
    Xte = scaler_x.transform(te[ANN_FEATURES].to_numpy())

    sy_dir = StandardScaler()
    ytr_dir = sy_dir.fit_transform((tr["C_market"] / tr["K"]).to_numpy()[:, None])
    yva_dir = sy_dir.transform((va["C_market"] / va["K"]).to_numpy()[:, None])
    sy_res = StandardScaler()
    ytr_res = sy_res.fit_transform((tr["C_error"] / tr["K"]).to_numpy()[:, None])
    yva_res = sy_res.transform((va["C_error"] / va["K"]).to_numpy()[:, None])

    raw_tr = tr[["S", "K", "T", "r"]].to_numpy().astype(np.float32)
    raw_va = va[["S", "K", "T", "r"]].to_numpy().astype(np.float32)
    ytr_px = tr["C_market"].to_numpy()[:, None].astype(np.float32)
    yva_px = va["C_market"].to_numpy()[:, None].astype(np.float32)

    direct, *_ = train_ann_direct(Xtr, ytr_dir, Xva, yva_dir, epochs=args.epochs, device=device)
    resid, *_ = train_ann_residual(Xtr, ytr_res, Xva, yva_res, epochs=args.epochs, device=device)
    volm, *_ = train_vol_hybrid(Xtr, raw_tr, ytr_px, Xva, raw_va, yva_px, epochs=args.epochs, device=device)

    # Shvimer & Zhu (2024) hybrid sub-models (PDF 2), on strike-normalised prices.
    cbs_tr = (tr["C_bs"] / tr["K"]).to_numpy()[:, None].astype(np.float32)
    cbs_va = (va["C_bs"] / va["K"]).to_numpy()[:, None].astype(np.float32)
    cbs_te = (te["C_bs"] / te["K"]).to_numpy()[:, None].astype(np.float32)
    ynorm_tr = (tr["C_market"] / tr["K"]).to_numpy()[:, None].astype(np.float32)
    ynorm_va = (va["C_market"] / va["K"]).to_numpy()[:, None].astype(np.float32)
    shv1, c1, y1 = train_shvimer("sub1", Xtr, cbs_tr, ynorm_tr, Xva, cbs_va, ynorm_va, epochs=args.epochs, device=device)
    shv2, c2, y2 = train_shvimer("sub2", Xtr, cbs_tr, ynorm_tr, Xva, cbs_va, ynorm_va, epochs=args.epochs, device=device)

    t_xte = torch.tensor(Xte, dtype=torch.float32).to(device)
    pred_bs = te["C_bs"].to_numpy()
    direct.eval(); resid.eval(); volm.eval()
    with torch.no_grad():
        pd_dir = sy_dir.inverse_transform(direct(t_xte).cpu().numpy()).flatten() * K_te
        pd_res = sy_res.inverse_transform(resid(t_xte).cpu().numpy()).flatten() * K_te
        sigma_dl = volm(t_xte).cpu().numpy().flatten()
    pred_direct = np.maximum(0.0, pd_dir)
    pred_resid = np.maximum(0.0, pred_bs + pd_res)
    pred_hybrid = np.maximum(0.0, black_scholes_call(te["S"].to_numpy(), K_te, te["T"].to_numpy(),
                                                     te["r"].to_numpy(), sigma_dl))
    pred_shv1 = predict_shvimer("sub1", shv1, c1, y1, Xte, cbs_te, K_te, device=device)
    pred_shv2 = predict_shvimer("sub2", shv2, c2, y2, Xte, cbs_te, K_te, device=device)

    y_true = te["C_market"].to_numpy()
    rows = [evaluate_predictions(y_true, pred_bs, "Classical Black-Scholes"),
            evaluate_predictions(y_true, pred_direct, "Direct ANN"),
            evaluate_predictions(y_true, pred_resid, "Residual ANN (Obj 4)"),
            evaluate_predictions(y_true, pred_hybrid, "DL-Vol Hybrid (Obj 2&3)"),
            evaluate_predictions(y_true, pred_shv1, "Shvimer Sub-model 1 (PDF2)"),
            evaluate_predictions(y_true, pred_shv2, "Shvimer Sub-model 2 (PDF2)")]
    overall = pd.DataFrame(rows)
    print("\n--- OBJECTIVES 1-4 ON REAL NSE OPTIONS (held-out test) ---")
    print(overall.to_string(index=False))

    te = te.copy()
    for col, p in [("pred_bs", pred_bs), ("pred_direct", pred_direct),
                   ("pred_resid", pred_resid), ("pred_hybrid", pred_hybrid)]:
        te[col] = p
    mon = pd.DataFrame(sum((evaluate_by_moneyness(te.rename(columns={"C_market": "C_market"}), c, nm)
                            for c, nm in [("pred_bs", "Black-Scholes"), ("pred_resid", "Residual ANN"),
                                          ("pred_hybrid", "DL-Vol Hybrid")]), []))

    # Objective 2 diagnostic on real data: sigma_DL vs market-implied vol.
    iv = np.array([implied_volatility(mp, s, k, t, r, "call")
                   for mp, s, k, t, r in zip(y_true, te["S"], K_te, te["T"], te["r"])])
    mask = np.isfinite(iv)
    vol_rows = None
    if mask.sum() > 5:
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        vol_rows = pd.DataFrame([
            {"Volatility": "Constant historical (BS)",
             "RMSE_vs_IV": float(np.sqrt(mean_squared_error(iv[mask], te["sigma_hist"].to_numpy()[mask]))),
             "MAE_vs_IV": float(mean_absolute_error(iv[mask], te["sigma_hist"].to_numpy()[mask]))},
            {"Volatility": "DL VolatilityNet (sigma_DL)",
             "RMSE_vs_IV": float(np.sqrt(mean_squared_error(iv[mask], sigma_dl[mask]))),
             "MAE_vs_IV": float(mean_absolute_error(iv[mask], sigma_dl[mask]))}])
        print("\n--- OBJECTIVE 2: DL vol vs market-implied vol (real) ---")
        print(vol_rows.to_string(index=False))

    overall.to_csv(os.path.join(args.output_dir, "real_overall_metrics.csv"), index=False)
    mon.to_csv(os.path.join(args.output_dir, "real_moneyness_metrics.csv"), index=False)
    if vol_rows is not None:
        vol_rows.to_csv(os.path.join(args.output_dir, "real_volatility_metrics.csv"), index=False)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].bar(overall["Model"], overall["RMSE"],
              color=["gray", "steelblue", "seagreen", "orange", "crimson", "purple"][:len(overall)])
    ax[0].set_title(f"Real {args.symbol} options — test RMSE (₹)"); ax[0].tick_params(axis="x", rotation=30)
    order = te.sort_values("moneyness")
    ax[1].scatter(order["moneyness"], np.abs(order["C_market"] - order["pred_bs"]), s=8, alpha=.4, label="BS error", color="gray")
    ax[1].scatter(order["moneyness"], np.abs(order["C_market"] - order["pred_hybrid"]), s=8, alpha=.4, label="DL-Vol Hybrid error", color="orange")
    ax[1].set_xlabel("Moneyness S/K"); ax[1].set_ylabel("|error| (₹)"); ax[1].set_title("Error across moneyness"); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(args.output_dir, "real_option_comparison.png"), dpi=140); plt.close(fig)
    print(f"\n[artifacts] saved to {args.output_dir}/")
    return overall, mon


def build_arg_parser():
    p = argparse.ArgumentParser(description="Objectives 1-4 on real NSE option data")
    p.add_argument("--symbol", default="NIFTY")
    p.add_argument("--days-back", type=int, default=12)
    p.add_argument("--rate", type=float, default=0.065)
    p.add_argument("--min-volume", type=int, default=1)
    p.add_argument("--max-contracts", type=int, default=6000)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--output-dir", default="outputs/real_options")
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
