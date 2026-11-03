"""
Shared training / evaluation loop for the fusion models.

Keeps the comparison fair: numeric-only, cross-attention and co-attention models
are all trained with the same loop, loss, optimiser and early-stopping-by-best-
validation-checkpoint, differing only in architecture.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_tensor(a, dtype=torch.float32):
    return torch.as_tensor(a, dtype=dtype)


def make_loader(numeric, news_emb, news_mask, y, batch_size, shuffle):
    ds = TensorDataset(
        _to_tensor(numeric),
        _to_tensor(news_emb),
        _to_tensor(news_mask, dtype=torch.bool),
        _to_tensor(y),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_model(model, train_data, val_data, epochs=40, lr=1e-3, batch_size=64,
                weight_decay=0.0, device="cpu", verbose=False):
    """
    Train one fusion model. ``*_data`` are dicts with keys
    ``numeric, news_emb, news_mask, y``. Returns (model, history, best_val_loss)
    with the best-validation weights restored.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    train_loader = make_loader(**train_data, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(**val_data, batch_size=batch_size, shuffle=False)

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history = {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        model.train()
        run = 0.0
        for numeric, news_emb, news_mask, y in train_loader:
            numeric, news_emb = numeric.to(device), news_emb.to(device)
            news_mask, y = news_mask.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(numeric, news_emb, news_mask)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            run += loss.item() * len(numeric)
        train_loss = run / len(train_loader.dataset)

        model.eval()
        vrun = 0.0
        with torch.no_grad():
            for numeric, news_emb, news_mask, y in val_loader:
                numeric, news_emb = numeric.to(device), news_emb.to(device)
                news_mask, y = news_mask.to(device), y.to(device)
                vrun += criterion(model(numeric, news_emb, news_mask), y).item() * len(numeric)
        val_loss = vrun / len(val_loader.dataset)

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  epoch {epoch:03d}/{epochs} | train {train_loss:.5f} | val {val_loss:.5f}")

    model.load_state_dict(best_state)
    return model, history, best_val


@torch.no_grad()
def predict(model, data, device="cpu", batch_size=256):
    model = model.to(device).eval()
    loader = make_loader(**data, batch_size=batch_size, shuffle=False)
    preds = []
    for numeric, news_emb, news_mask, _ in loader:
        preds.append(model(numeric.to(device), news_emb.to(device),
                           news_mask.to(device)).cpu().numpy())
    return np.concatenate(preds, axis=0)


def evaluate_regression(y_true, y_pred, name="Model"):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    # Directional accuracy is meaningful when the target is a return/change.
    dir_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    return {"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2, "DirAcc": dir_acc}
