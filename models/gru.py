import os
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from typing import Tuple

class GRU(nn.Module):
    """
    A GRU model for time series forecasting.

    Architecture:
        - GRU layers with configurable input size, hidden size, number of layers, and dropout.
        - Fully connected layer mapping the last hidden state to the forecasted outputs.

    Args:
        input_size (int): Number of input features per timestep.
        hidden_size (int): Number of hidden units in each GRU layer.
        num_layers (int): Number of stacked GRU layers.
        horizon (int): Number of future timesteps to predict.
        n_targets (int): Number of target variables to predict.
        dropout (float, optional): Dropout probability between GRU layers (default=0.0).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        horizon: int,
        n_targets: int,
        dropout: float = 0.0
    ) -> None:
        super().__init__()

        self.gru: nn.GRU = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc: nn.Linear = nn.Linear(hidden_size, horizon * n_targets)
        self.horizon: int = horizon
        self.n_targets: int = n_targets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GRU model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, horizon * n_targets).
        """
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # Use the last timestep’s hidden state
        return out


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
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_gru.pth"))
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
