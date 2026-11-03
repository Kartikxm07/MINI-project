"""
Hyperparameter tuning (Optuna) for the news + cross/co-attention fusion models.

Runs a TPE search over architecture and optimiser hyperparameters, minimising
validation loss, then retrains the best configuration and reports the held-out
test metrics against an untuned default -- showing the gain from tuning.

Examples:
    python tune_news.py --task stock  --model co   --trials 25
    python tune_news.py --task option --model cross --trials 25 --samples 4000
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import optuna
from sklearn.metrics import mean_squared_error, r2_score

import train_stock_news
import train_option_news
from src.attention.fusion_models import build_model, ModelConfig
from src.attention.trainer import set_seed, train_model, predict, evaluate_regression


def make_base_args(args):
    """Build a task-module args namespace with tuning-relevant overrides."""
    if args.task == "stock":
        base = train_stock_news.build_arg_parser().parse_args([])
        base.symbol = args.symbol
        base.synthetic_prices = args.synthetic_prices
        base.news_mode = args.news_mode
        base.embedder = args.embedder
    else:
        base = train_option_news.build_arg_parser().parse_args([])
        base.samples = args.samples
        base.sentiment_shock = args.sentiment_shock
    base.epochs = args.epochs
    base.seed = args.seed
    return base


def evaluate_on_test(task, model, data, device):
    if task == "stock":
        yhat = predict(model, data["test"], device=device)
        m = evaluate_regression(data["test"]["y"], yhat, "x")
        return {"RMSE": m["RMSE"], "DirAcc": m["DirAcc"], "R2": m["R2"]}
    yhat = predict(model, data["test"], device=device).reshape(-1)
    price = np.maximum(0.0, yhat * data["K_test"])
    yt = data["y_test_price"]
    return {"RMSE": float(np.sqrt(mean_squared_error(yt, price))),
            "R2": float(r2_score(yt, price))}


def cfg_and_optim_from_params(p):
    d_model = p["num_heads"] * p["head_dim"]
    cfg = ModelConfig(d_model=d_model, num_heads=p["num_heads"], gru_layers=p["gru_layers"],
                      mlp_hidden=p["mlp_hidden"], dropout=p["dropout"], out_dim=1)
    return cfg, p["lr"], p["weight_decay"], p["batch_size"]


def run(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = train_stock_news if args.task == "stock" else train_option_news
    base = make_base_args(args)

    data = module.prepare_data(base)
    nfd, nd = data["num_feat_dim"], data["news_dim"]
    print(f"[tune] task={args.task} model={args.model} | "
          f"num_feat_dim={nfd} news_dim={nd} | trials={args.trials} epochs={args.epochs}")

    def objective(trial):
        set_seed(args.seed)
        num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
        head_dim = trial.suggest_categorical("head_dim", [8, 16, 32])
        cfg = ModelConfig(
            d_model=num_heads * head_dim, num_heads=num_heads,
            gru_layers=trial.suggest_int("gru_layers", 1, 2),
            mlp_hidden=trial.suggest_categorical("mlp_hidden", [32, 64, 128]),
            dropout=trial.suggest_float("dropout", 0.0, 0.3), out_dim=1)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        bs = trial.suggest_categorical("batch_size", [32, 64, 128])
        model = build_model(args.model, nfd, nd, cfg)
        _, _, best_val = train_model(model, data["train"], data["val"], epochs=args.epochs,
                                     lr=lr, batch_size=bs, weight_decay=wd, device=device)
        return best_val

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    print(f"\n[tune] best validation loss: {study.best_value:.6f}")
    print(f"[tune] best params: {study.best_params}")

    # Default (untuned) baseline for the same model kind.
    set_seed(args.seed)
    default_model = build_model(args.model, nfd, nd, ModelConfig(out_dim=1))
    default_model, _, _ = train_model(default_model, data["train"], data["val"],
                                      epochs=args.epochs, device=device)
    default_metrics = evaluate_on_test(args.task, default_model, data, device)

    # Retrain with best params and evaluate on test.
    tuned_cfg, lr, wd, bs = cfg_and_optim_from_params(study.best_params)
    set_seed(args.seed)
    tuned_model = build_model(args.model, nfd, nd, tuned_cfg)
    tuned_model, _, _ = train_model(tuned_model, data["train"], data["val"], epochs=args.epochs,
                                    lr=lr, batch_size=bs, weight_decay=wd, device=device)
    tuned_metrics = evaluate_on_test(args.task, tuned_model, data, device)

    results = pd.DataFrame([{"Config": "default", **default_metrics},
                            {"Config": "tuned (optuna)", **tuned_metrics}])
    print(f"\n--- {args.task.upper()} / {args.model} : default vs Optuna-tuned (held-out test) ---")
    print(results.to_string(index=False))

    os.makedirs(args.output_dir, exist_ok=True)
    tag = f"{args.task}_{args.model}"
    results.to_csv(os.path.join(args.output_dir, f"tune_{tag}.csv"), index=False)
    with open(os.path.join(args.output_dir, f"best_params_{tag}.json"), "w") as f:
        json.dump({"task": args.task, "model": args.model,
                   "best_value": study.best_value, "best_params": study.best_params}, f, indent=2)
    torch.save(tuned_model.state_dict(), os.path.join(args.output_dir, f"tuned_{tag}.pth"))
    print(f"[artifacts] saved to {args.output_dir}/")
    return results


def build_arg_parser():
    p = argparse.ArgumentParser(description="Optuna tuning for news+attention fusion models")
    p.add_argument("--task", choices=["stock", "option"], required=True)
    p.add_argument("--model", choices=["numeric", "cross", "co", "cocross"], default="cocross")
    p.add_argument("--trials", type=int, default=25)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    # stock options
    p.add_argument("--symbol", default="RELIANCE.NS")
    p.add_argument("--synthetic-prices", action="store_true")
    p.add_argument("--news-mode", choices=["auto", "real", "synthetic"], default="synthetic")
    p.add_argument("--embedder", choices=["finbert", "hashing"], default="hashing")
    # option options
    p.add_argument("--samples", type=int, default=3000)
    p.add_argument("--sentiment-shock", type=float, default=0.006)
    p.add_argument("--output-dir", default="outputs/news_tuning")
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
