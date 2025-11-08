### inspired by Transformer Networks for Energy Time-Series Forecasting
### GitHub: @KIT-IAI

import os
import math
import numpy as np

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from typing import Tuple

class ValueEmbedding(nn.Module):
    def __init__(self, d_model: int, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        return self.pe[:, :x.size(1), :]

class TotalEmbedding(nn.Module):
    def __init__(self, d_model: int, n_features: int, dropout: float):
        super().__init__()
        self.value_embedding = ValueEmbedding(d_model, n_features)
        self.positional_encoding = PositionalEncoding(d_model)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        value_emb = self.value_embedding(x)
        pe = self.positional_encoding(x).repeat(x.size(0), 1, 1)
        return self.dropout(self.alpha * value_emb + self.beta * pe)

class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = TotalEmbedding(d_model, input_size, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
            activation='relu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)  # prever apenas 1 valor
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        emb = self.embedding(x)
        out = self.encoder(emb)
        out = self.relu(out[:, -1, :])   # usa o último token
        return self.fc_out(out)

def training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    RESULTS_DIR: str,
    PATIENCE: int = 10,
    EPOCHS: int = 100,
    device: torch.device = torch.device("cpu")
) -> None:
    """
    Trains and validates a GRU (or other PyTorch model) with early stopping, saving the best checkpoint.

    Args:
        model (nn.Module): PyTorch model to be trained.
        train_loader (DataLoader): Dataloader containing the training batches.
        val_loader (DataLoader): Dataloader containing the validation batches.
        optimizer (torch.optim.Optimizer): Optimization algorithm (e.g., Adam, SGD).
        criterion (nn.Module): Loss function (e.g., MSELoss, L1Loss).
        RESULTS_DIR (str): Directory where the best model checkpoint will be saved.
        PATIENCE (int): Number of epochs without validation improvement allowed before stopping. Default = 10.
        EPOCHS (int): Maximum number of training epochs. Default = 100.
        device (torch.device): Device used for training ('cpu' or 'cuda'). Default = 'cpu'.
    """
    best_val = float("inf")
    wait = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Validation phase
        model.eval()
        val_losses = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(criterion(pred, yb).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Check for improvement
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_transformer.pth"))
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping.")
                break


def testing(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device = torch.device("cpu")
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluates a trained GRU model, returning predictions and true targets.

    Args:
        model (nn.Module): Trained PyTorch model.
        val_loader (DataLoader): DataLoader for the validation/test set.
        device (torch.device): Device used for inference ('cpu' or 'cuda'). Default = 'cpu'.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - preds_val_scaled: Model predictions (scaled), shape (N, horizon * n_targets)
            - y_val_scaled: Ground truth values (scaled), shape (N, 1)
    """
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())
            trues.append(yb.numpy())

    preds_val_scaled = np.concatenate(preds, axis=0)
    y_val_scaled = np.concatenate(trues, axis=0).reshape(-1, 1)

    return preds_val_scaled, y_val_scaled
