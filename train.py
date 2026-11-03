"""
Training and Evaluation Pipeline for Deep Learning Option Pricing Models.
Compares:
1. Classical Black-Scholes (Baseline)                         [Objective 1]
2. Simple ANN Direct (X -> C_market)
3. Simple ANN Residual (X -> E, C_pred = C_bs + E_hat)        [Objective 4: residual learning]
4. Option LSTM (Sequential dynamics + static contract features)
5. DL-Volatility + Hybrid BS (X -> sigma_DL, C = BS(.,sigma_DL)) [Objectives 2 & 3]
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.data_loader import generate_market_options_data, generate_time_series_option_sequences
from src.pricing import black_scholes_call
from model import SimpleOptionANN, ResidualOptionANN, OptionLSTM, VolatilityNet


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_predictions(y_true, y_pred, name="Model"):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2}


def evaluate_by_moneyness(df_test, y_pred_col, pred_name="Model"):
    results = []
    for cat in ['OTM', 'ATM', 'ITM']:
        sub = df_test[df_test['moneyness_cat'] == cat]
        if len(sub) == 0:
            continue
        rmse = np.sqrt(mean_squared_error(sub['C_market'], sub[y_pred_col]))
        mae = mean_absolute_error(sub['C_market'], sub[y_pred_col])
        results.append({
            'Model': pred_name,
            'Moneyness': cat,
            'Count': len(sub),
            'RMSE': rmse,
            'MAE': mae
        })
    return results


def train_ann_direct(X_train, y_train, X_val, y_val, epochs=60, batch_size=64, lr=0.001, device='cpu'):
    print("\n" + "="*60)
    print(" [Step 1] Training Simple ANN Direct: X -> C_market")
    print("="*60)
    
    input_dim = X_train.shape[1]
    model = SimpleOptionANN(input_dim=input_dim, hidden_dim1=64, hidden_dim2=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(bx)
        
        train_loss = running_loss / len(train_ds)

        model.eval()
        running_vloss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                vloss = criterion(pred, by)
                running_vloss += vloss.item() * len(bx)
        val_loss = running_vloss / len(val_ds)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 15 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

    return model, train_losses, val_losses


def train_ann_residual(X_train, y_err_train, X_val, y_err_val, epochs=60, batch_size=64, lr=0.001, device='cpu'):
    print("\n" + "="*60)
    print(" [Step 2] Training Research Model (Residual ANN): X -> E (E = C_market - C_bs)")
    print("="*60)
    
    input_dim = X_train.shape[1]
    model = ResidualOptionANN(input_dim=input_dim, hidden_dim1=64, hidden_dim2=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_err_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_err_val, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(bx)
        
        train_loss = running_loss / len(train_ds)

        model.eval()
        running_vloss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                vloss = criterion(pred, by)
                running_vloss += vloss.item() * len(bx)
        val_loss = running_vloss / len(val_ds)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 15 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Error Loss: {train_loss:.6f} | Val Error Loss: {val_loss:.6f}")

    return model, train_losses, val_losses


def train_lstm(X_seq_tr, X_st_tr, y_tr, X_seq_val, X_st_val, y_val, epochs=50, batch_size=64, lr=0.001, device='cpu'):
    print("\n" + "="*60)
    print(" [Step 3] Training Option LSTM: (X_seq, X_static) -> C_market")
    print("="*60)
    
    seq_input_dim = X_seq_tr.shape[2]
    static_input_dim = X_st_tr.shape[1]
    
    model = OptionLSTM(
        seq_input_dim=seq_input_dim,
        static_input_dim=static_input_dim,
        lstm_hidden=64,
        dense_hidden=32,
        dropout_prob=0.1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(
        torch.tensor(X_seq_tr, dtype=torch.float32),
        torch.tensor(X_st_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(X_seq_val, dtype=torch.float32),
        torch.tensor(X_st_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for b_seq, b_st, by in train_loader:
            b_seq, b_st, by = b_seq.to(device), b_st.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(b_seq, b_st)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(b_seq)
        
        train_loss = running_loss / len(train_ds)

        model.eval()
        running_vloss = 0.0
        with torch.no_grad():
            for b_seq, b_st, by in val_loader:
                b_seq, b_st, by = b_seq.to(device), b_st.to(device), by.to(device)
                pred = model(b_seq, b_st)
                vloss = criterion(pred, by)
                running_vloss += vloss.item() * len(b_seq)
        val_loss = running_vloss / len(val_ds)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 15 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

    return model, train_losses, val_losses


def train_vol_hybrid(X_train, raw_train, y_train, X_val, raw_val, y_val,
                     epochs=60, batch_size=64, lr=0.001, device='cpu'):
    """
    Objective 2 + 3: Deep-Learning Volatility model trained end-to-end through a
    differentiable Black-Scholes (the Hybrid pricer).

    The network predicts sigma_DL = f_theta(X); that volatility is fed into
    Black-Scholes to price the contract; the loss compares the resulting hybrid
    price against the observed market price. The *only* supervision is the market
    price, so the network must discover the volatility that best reprices each
    contract -- a data-driven relaxation of BS' constant-volatility assumption.

    raw_* hold the unscaled columns [S, K, T, r]; y_* hold C_market. Prices are
    normalised by strike (price / K) inside the loss to stay scale-free, matching
    the other models.
    """
    print("\n" + "="*60)
    print(" [Step 4] Training DL Volatility + Hybrid BS: X -> sigma_DL -> BS(sigma_DL)")
    print("="*60)

    input_dim = X_train.shape[1]
    model = VolatilityNet(input_dim=input_dim, hidden_dim1=64, hidden_dim2=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(raw_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(raw_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    def batch_loss(bx, braw, by):
        S, K, T, r = braw[:, 0], braw[:, 1], braw[:, 2], braw[:, 3]
        price = model.hybrid_price(bx, S, K, T, r)          # (batch,)
        pred = (price / K).unsqueeze(-1)                    # price / strike
        target = (by.squeeze(-1) / K).unsqueeze(-1)
        return criterion(pred, target)

    train_losses, val_losses = [], []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for bx, braw, by in train_loader:
            bx, braw, by = bx.to(device), braw.to(device), by.to(device)
            optimizer.zero_grad()
            loss = batch_loss(bx, braw, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(bx)
        train_loss = running_loss / len(train_ds)

        model.eval()
        running_vloss = 0.0
        with torch.no_grad():
            for bx, braw, by in val_loader:
                bx, braw, by = bx.to(device), braw.to(device), by.to(device)
                running_vloss += batch_loss(bx, braw, by).item() * len(bx)
        val_loss = running_vloss / len(val_ds)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 15 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Price/K MSE: {train_loss:.6f} | Val Price/K MSE: {val_loss:.6f}")

    return model, train_losses, val_losses


def plot_results(df_test, train_losses_dict, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Training Loss Curves
    plt.figure(figsize=(10, 5))
    for name, (tr_loss, val_loss) in train_losses_dict.items():
        plt.plot(val_loss, label=f'{name} (Val Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss (MSE)')
    plt.title('Training & Validation Convergence Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=300)
    plt.close()

    # 2. Predicted vs. Actual Scatter (Sample of 300 points)
    sample_df = df_test.sample(min(300, len(df_test)), random_state=42).sort_values('C_market')
    plt.figure(figsize=(12, 6))
    plt.plot(sample_df['C_market'].values, sample_df['C_market'].values, 'k--', label='Ideal 1:1 Parity', alpha=0.7)
    plt.scatter(sample_df['C_market'].values, sample_df['pred_bs'].values, color='gray', alpha=0.5, label='Black-Scholes (Constant Vol)')
    plt.scatter(sample_df['C_market'].values, sample_df['pred_ann_direct'].values, color='blue', alpha=0.6, label='ANN Direct (X -> C)')
    plt.scatter(sample_df['C_market'].values, sample_df['pred_ann_resid'].values, color='green', alpha=0.6, label='ANN Residual (X -> E)')
    plt.scatter(sample_df['C_market'].values, sample_df['pred_lstm'].values, color='purple', alpha=0.6, label='Option LSTM')
    plt.scatter(sample_df['C_market'].values, sample_df['pred_hybrid'].values, color='orange', alpha=0.6, label='DL-Vol Hybrid BS(sigma_DL)')
    plt.xlabel('Actual Market Price ($)')
    plt.ylabel('Predicted Option Price ($)')
    plt.title('Option Pricing Accuracy: Predicted vs. Actual')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'), dpi=300)
    plt.close()

    # 3. Pricing Error vs Moneyness (The Volatility Smile Effect)
    plt.figure(figsize=(12, 6))
    sort_idx = df_test.sort_values('moneyness')
    plt.scatter(sort_idx['moneyness'], np.abs(sort_idx['C_market'] - sort_idx['pred_bs']), color='red', alpha=0.3, label='Black-Scholes Error |C_market - C_bs|')
    plt.scatter(sort_idx['moneyness'], np.abs(sort_idx['C_market'] - sort_idx['pred_ann_direct']), color='blue', alpha=0.3, label='ANN Direct Absolute Error')
    plt.scatter(sort_idx['moneyness'], np.abs(sort_idx['C_market'] - sort_idx['pred_ann_resid']), color='green', alpha=0.3, label='ANN Residual Absolute Error')
    plt.scatter(sort_idx['moneyness'], np.abs(sort_idx['C_market'] - sort_idx['pred_lstm']), color='purple', alpha=0.3, label='Option LSTM Absolute Error')
    plt.scatter(sort_idx['moneyness'], np.abs(sort_idx['C_market'] - sort_idx['pred_hybrid']), color='orange', alpha=0.3, label='DL-Vol Hybrid Absolute Error')
    plt.axvline(0.97, color='black', linestyle=':', label='OTM/ATM Boundary')
    plt.axvline(1.03, color='black', linestyle='--', label='ATM/ITM Boundary')
    plt.xlabel('Moneyness (S / K)')
    plt.ylabel('Absolute Pricing Error ($)')
    plt.title('Pricing Discrepancy Across Moneyness (Smile / Skew Impact)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_across_moneyness.png'), dpi=300)
    plt.close()

    # Comparable held-out metrics for every method, including the baseline.
    columns = ['pred_bs', 'pred_ann_direct', 'pred_ann_resid', 'pred_lstm', 'pred_hybrid']
    names = ['Black-Scholes', 'Direct ANN', 'Residual ANN', 'Option LSTM', 'DL-Vol Hybrid']
    colors = ['gray', 'blue', 'green', 'purple', 'orange']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, metric in zip(axes, ['RMSE', 'MAE']):
        values = [evaluate_predictions(df_test['C_market'], df_test[col])[metric] for col in columns]
        axis.bar(names, values, color=colors, alpha=.8)
        axis.set_ylabel(f'{metric} (price units)')
        axis.tick_params(axis='x', rotation=20)
        axis.grid(axis='y', alpha=.2)
    fig.suptitle('Held-out Option Pricing Error — Lower Is Better')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'overall_comparison.png'), dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 5))
    categories = ['OTM', 'ATM', 'ITM']
    x = np.arange(len(categories))
    width = 0.8 / len(columns)
    offset0 = (len(columns) - 1) / 2.0
    for i, (col, name, color) in enumerate(zip(columns, names, colors)):
        values = []
        for category in categories:
            subset = df_test[df_test['moneyness_cat'] == category]
            values.append(np.sqrt(mean_squared_error(subset['C_market'], subset[col])) if len(subset) else np.nan)
        axis.bar(x + (i - offset0) * width, values, width, label=name, color=color, alpha=.8)
    axis.set_xticks(x, categories)
    axis.set_ylabel('RMSE (price units)')
    axis.set_title('Held-out Error by Call Moneyness')
    axis.legend()
    axis.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'moneyness_comparison.png'), dpi=180)
    plt.close(fig)

    print(f"\n[Artifacts] Visual plots successfully saved to '{output_dir}/'")


def run_pipeline(n_samples=10000, epochs=50, batch_size=64, lr=0.001, output_dir="outputs"):
    if n_samples < 20 or epochs < 1 or batch_size < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("Use at least 20 samples, positive epochs/batch size, and a positive finite learning rate")
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Compute Device: {device}")
    
    # 1. Generate realistic market dataset
    print(f"Generating synthetic options dataset with {n_samples} contracts...")
    df = generate_market_options_data(n_samples=n_samples, seed=42)
    
    # Train / Validation / Test split (70% train, 15% val, 15% test)
    n = len(df)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    
    df_train = df.iloc[:n_train].copy()
    df_val = df.iloc[n_train:n_train+n_val].copy()
    df_test = df.iloc[n_train+n_val:].copy()

    # Prepare features for ANN Direct: [S, K, T, r, sigma_hist] or normalized [S/K, T, r, sigma_hist, C_bs/K]
    ann_features = ['moneyness', 'T', 'r', 'sigma_hist']
    
    scaler_x = StandardScaler()
    X_train_ann = scaler_x.fit_transform(df_train[ann_features].values)
    X_val_ann = scaler_x.transform(df_val[ann_features].values)
    X_test_ann = scaler_x.transform(df_test[ann_features].values)

    scaler_y_direct = StandardScaler()
    # Inputs are scale-free, so targets must also be normalized by strike.
    y_train_direct = scaler_y_direct.fit_transform((df_train['C_market'] / df_train['K']).values[:, None])
    y_val_direct = scaler_y_direct.transform((df_val['C_market'] / df_val['K']).values[:, None])

    # Prepare targets for ANN Residual: E = C_market - C_bs
    scaler_y_resid = StandardScaler()
    y_train_resid = scaler_y_resid.fit_transform((df_train['C_error'] / df_train['K']).values[:, None])
    y_val_resid = scaler_y_resid.transform((df_val['C_error'] / df_val['K']).values[:, None])

    # Prepare inputs for the DL-Volatility + Hybrid BS model.
    # Network input is the same scaled feature matrix; Black-Scholes inside the
    # hybrid needs the *raw* contract parameters [S, K, T, r] and the raw market
    # price as the only training target.
    raw_cols = ['S', 'K', 'T', 'r']
    raw_train = df_train[raw_cols].values.astype(np.float32)
    raw_val = df_val[raw_cols].values.astype(np.float32)
    y_train_price = df_train['C_market'].values[:, None].astype(np.float32)
    y_val_price = df_val['C_market'].values[:, None].astype(np.float32)

    # 2. Step 1: Train ANN Direct
    ann_direct, tr_loss_dir, val_loss_dir = train_ann_direct(
        X_train_ann, y_train_direct, X_val_ann, y_val_direct,
        epochs=epochs, batch_size=batch_size, lr=lr, device=device
    )

    # 3. Step 2: Train ANN Residual
    ann_resid, tr_loss_res, val_loss_res = train_ann_residual(
        X_train_ann, y_train_resid, X_val_ann, y_val_resid,
        epochs=epochs, batch_size=batch_size, lr=lr, device=device
    )

    # 4. Step 3: Train Option LSTM
    print("\nPreparing temporal sequences for LSTM...")
    X_seq, X_static, _, _ = generate_time_series_option_sequences(df, seq_len=15, seed=42)
    
    X_seq_tr = X_seq[:n_train]
    X_st_tr = X_static[:n_train]
    
    X_seq_val = X_seq[n_train:n_train+n_val]
    X_st_val = X_static[n_train:n_train+n_val]
    
    X_seq_te = X_seq[n_train+n_val:]
    X_st_te = X_static[n_train+n_val:]

    lstm_model, tr_loss_lstm, val_loss_lstm = train_lstm(
        X_seq_tr, X_st_tr, y_train_direct,
        X_seq_val, X_st_val, y_val_direct,
        epochs=epochs, batch_size=batch_size, lr=lr, device=device
    )

    # 4b. Step 4: Train DL Volatility model + Hybrid BS (Objectives 2 & 3)
    vol_model, tr_loss_vol, val_loss_vol = train_vol_hybrid(
        X_train_ann, raw_train, y_train_price,
        X_val_ann, raw_val, y_val_price,
        epochs=epochs, batch_size=batch_size, lr=lr, device=device
    )

    # 5. Evaluate All Models on Unseen Test Set
    print("\n" + "="*60)
    print(" Evaluating Models on Out-of-Sample Test Set")
    print("="*60)

    # Baseline BS
    pred_bs = df_test['C_bs'].values

    # ANN Direct
    ann_direct.eval()
    with torch.no_grad():
        t_x_test = torch.tensor(X_test_ann, dtype=torch.float32).to(device)
        pred_scaled_dir = ann_direct(t_x_test).cpu().numpy()
        pred_ann_direct = scaler_y_direct.inverse_transform(pred_scaled_dir).flatten() * df_test['K'].values
        pred_ann_direct = np.maximum(0.0, pred_ann_direct)

    # ANN Residual
    ann_resid.eval()
    with torch.no_grad():
        pred_scaled_res = ann_resid(t_x_test).cpu().numpy()
        pred_error = scaler_y_resid.inverse_transform(pred_scaled_res).flatten() * df_test['K'].values
        # Final price = C_bs + E_hat
        pred_ann_resid = np.maximum(0.0, df_test['C_bs'].values + pred_error)

    # LSTM
    lstm_model.eval()
    with torch.no_grad():
        t_seq = torch.tensor(X_seq_te, dtype=torch.float32).to(device)
        t_st = torch.tensor(X_st_te, dtype=torch.float32).to(device)
        pred_scaled_lstm = lstm_model(t_seq, t_st).cpu().numpy()
        pred_lstm = scaler_y_direct.inverse_transform(pred_scaled_lstm).flatten() * df_test['K'].values
        pred_lstm = np.maximum(0.0, pred_lstm)

    # DL-Volatility + Hybrid BS: predict sigma_DL, then price with classical BS
    vol_model.eval()
    with torch.no_grad():
        sigma_dl = vol_model(t_x_test).cpu().numpy().flatten()
    # Price with the analytic (numpy) Black-Scholes for consistency with the
    # baseline and proper no-arbitrage/intrinsic clamping (Objective 3).
    pred_hybrid = black_scholes_call(
        df_test['S'].values, df_test['K'].values,
        df_test['T'].values, df_test['r'].values, sigma_dl
    )
    pred_hybrid = np.maximum(0.0, pred_hybrid)

    df_test['pred_bs'] = pred_bs
    df_test['pred_ann_direct'] = pred_ann_direct
    df_test['pred_ann_resid'] = pred_ann_resid
    df_test['pred_lstm'] = pred_lstm
    df_test['sigma_dl'] = sigma_dl
    df_test['pred_hybrid'] = pred_hybrid

    y_test_true = df_test['C_market'].values

    # Overall Metrics Table
    m_bs = evaluate_predictions(y_test_true, pred_bs, "Classical Black-Scholes")
    m_dir = evaluate_predictions(y_test_true, pred_ann_direct, "ANN Direct (X -> C)")
    m_res = evaluate_predictions(y_test_true, pred_ann_resid, "ANN Residual (X -> E)")
    m_lstm = evaluate_predictions(y_test_true, pred_lstm, "Option LSTM")
    m_hyb = evaluate_predictions(y_test_true, pred_hybrid, "DL-Vol Hybrid BS(sigma_DL)")

    overall_df = pd.DataFrame([m_bs, m_dir, m_res, m_lstm, m_hyb])
    print("\n--- OVERALL TEST PERFORMANCE ---")
    print(overall_df.to_string(index=False))

    # Moneyness Stratified Metrics Table (Matching the Research Paper)
    b_bs = evaluate_by_moneyness(df_test, 'pred_bs', "Classical Black-Scholes")
    b_dir = evaluate_by_moneyness(df_test, 'pred_ann_direct', "ANN Direct (X -> C)")
    b_res = evaluate_by_moneyness(df_test, 'pred_ann_resid', "ANN Residual (X -> E)")
    b_lstm = evaluate_by_moneyness(df_test, 'pred_lstm', "Option LSTM")
    b_hyb = evaluate_by_moneyness(df_test, 'pred_hybrid', "DL-Vol Hybrid BS(sigma_DL)")

    money_df = pd.DataFrame(b_bs + b_dir + b_res + b_lstm + b_hyb)
    print("\n--- PERFORMANCE STRATIFIED BY MONEYNESS (OTM / ATM / ITM) ---")
    print(money_df.to_string(index=False))

    # Objective 2 diagnostic: how well does the DL volatility recover the true
    # (smile/skew) market volatility, versus BS' single constant historical vol?
    sigma_true = df_test['sigma_market'].values
    vol_rows = [
        {"Volatility Estimate": "Constant historical (BS)",
         "RMSE": float(np.sqrt(mean_squared_error(sigma_true, df_test['sigma_hist'].values))),
         "MAE": float(mean_absolute_error(sigma_true, df_test['sigma_hist'].values))},
        {"Volatility Estimate": "DL VolatilityNet (sigma_DL)",
         "RMSE": float(np.sqrt(mean_squared_error(sigma_true, sigma_dl))),
         "MAE": float(mean_absolute_error(sigma_true, sigma_dl))},
    ]
    vol_df = pd.DataFrame(vol_rows)
    print("\n--- DL VOLATILITY RECOVERY vs TRUE MARKET VOL (Objective 2) ---")
    print(vol_df.to_string(index=False))

    # Save metrics
    os.makedirs(output_dir, exist_ok=True)
    overall_df.to_csv(os.path.join(output_dir, "overall_metrics.csv"), index=False)
    money_df.to_csv(os.path.join(output_dir, "moneyness_metrics.csv"), index=False)
    vol_df.to_csv(os.path.join(output_dir, "volatility_metrics.csv"), index=False)
    joblib.dump({"features": ann_features, "scaler_x": scaler_x,
                 "scaler_y_direct": scaler_y_direct, "scaler_y_resid": scaler_y_resid,
                 "target_normalization": "divide_by_strike", "sequence_length": 15,
                 "vol_sigma_bounds": (vol_model.sigma_min, vol_model.sigma_max)},
                os.path.join(output_dir, "preprocessing.joblib"))

    # Save models
    torch.save(ann_direct.state_dict(), os.path.join(output_dir, "ann_direct.pth"))
    torch.save(ann_resid.state_dict(), os.path.join(output_dir, "ann_resid.pth"))
    torch.save(lstm_model.state_dict(), os.path.join(output_dir, "lstm_model.pth"))
    torch.save(vol_model.state_dict(), os.path.join(output_dir, "vol_model.pth"))

    # Plot graphs
    loss_dict = {
        'ANN Direct': (tr_loss_dir, val_loss_dir),
        'ANN Residual': (tr_loss_res, val_loss_res),
        'Option LSTM': (tr_loss_lstm, val_loss_lstm),
        'DL-Vol Hybrid': (tr_loss_vol, val_loss_vol)
    }
    plot_results(df_test, loss_dict, output_dir=output_dir)

    return overall_df, money_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Option Pricing Models (ANN vs. LSTM & Residual)")
    parser.add_argument('--samples', type=int, default=8000, help="Number of option contracts to generate")
    parser.add_argument('--epochs', type=int, default=50, help="Training epochs")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--output-dir', default='outputs', help='Directory for model and evaluation artifacts')
    args = parser.parse_args()

    run_pipeline(n_samples=args.samples, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, output_dir=args.output_dir)
