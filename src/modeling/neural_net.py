"""PyTorch MLP for churn — moved out of NN_MLP_experiments.ipynb into src/.

Architecture and training mirror the validated notebook: Input → 64 → 32 → 1
with ReLU + Dropout(0.3), BCEWithLogitsLoss, Adam, and early stopping. Kept
deliberately small and dependency-light so it can be imported by train.py for
MLflow logging without dragging in the rest of the notebook.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import SEED


class ChurnMLP(nn.Module):
    """Two-hidden-layer MLP (64 → 32) with ReLU + dropout, single logit output."""

    def __init__(self, input_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    max_epochs: int = 150,
    patience: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    dropout: float = 0.3,
) -> tuple[ChurnMLP, dict]:
    """Train the MLP with early stopping; return the best model + run params.

    Inputs are already-scaled dense arrays (the sklearn preprocessor runs
    upstream). Returns the model restored to its best-validation-loss weights.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_train_t = torch.tensor(np.asarray(X_train), dtype=torch.float32)
    y_train_t = torch.tensor(np.asarray(y_train), dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(np.asarray(X_val), dtype=torch.float32)
    y_val_t = torch.tensor(np.asarray(y_val), dtype=torch.float32).unsqueeze(1)

    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    model = ChurnMLP(X_train_t.shape[1], dropout=dropout)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    epochs_run = 0

    for epoch in range(max_epochs):
        epochs_run = epoch + 1
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    params = {
        "architecture": "64-32",
        "dropout": dropout,
        "lr": lr,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "patience": patience,
        "epochs_run": epochs_run,
        "best_val_loss": best_val_loss,
    }
    return model, params


def predict_proba_mlp(model: ChurnMLP, X: np.ndarray) -> np.ndarray:
    """Return churn probabilities for already-scaled dense input."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(np.asarray(X), dtype=torch.float32))
        return torch.sigmoid(logits).numpy().squeeze()
