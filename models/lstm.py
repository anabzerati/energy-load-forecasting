import os
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from typing import Tuple

class LSTM(nn.Module):
    """
    A LSTM model for time series forecasting.

    Architecture:
        - LSTM layers with configurable input size, hidden size, number of layers, and dropout.
        - Fully connected layer mapping the last hidden state to the forecasted outputs.

    Args:
        input_size (int): Number of input features per timestep.
        hidden_size (int): Number of hidden units in each LSTM layer.
        num_layers (int): Number of stacked LSTM layers.
        horizon (int): Number of future timesteps to predict.
        n_targets (int): Number of target variables to predict.
        dropout (float, optional): Dropout probability between LSTM layers (default=0.0). 
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 horizon: int, n_targets: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.lstm: nn.LSTM = nn.LSTM(
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
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])                     
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
    device: torch.device = 'cpu'
) -> None:        
    """
    Trains and validates a PyTorch model with early stopping, saving the best checkpoint.

    Args:
        model (nn.Module): PyTorch model to be trained.
        train_loader (DataLoader): Dataloader containing the training batches.
        val_loader (DataLoader): Dataloader containing the validation batches.
        optimizer (optim.Optimizer): Optimization algorithm (e.g., Adam, SGD).
        criterion (nn.Module): Loss function (e.g., MSELoss, L1Loss).
        RESULTS_DIR (str): Where to save the best model.
        PATIENCE (int): Number of epochs without improvement accepted. Default value = 10.
        EPOCHS (int): Number of training epochs. Deafult value = 100
        device (torch.device): Device on which the model and data are placed ('cpu' or 'cuda').

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - preds_val_scaled: Predicted values on the validation set (scaled).
            - y_val_scaled: Ground truth validation targets (scaled).
    """
    best_val = 1e12
    wait = 0

    for epoch in range(1, EPOCHS+1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            pred = model(xb)
            loss = criterion(pred, yb)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb)

                val_losses.append(criterion(pred, yb).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        print(f"Epoch {epoch:03d} Train Loss {train_loss:.6f} Val Loss {val_loss:.6f}")
        
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
        
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_lstm.pth"))
        
        else:
            wait += 1
        
            if wait >= PATIENCE:
                print("Early stopping.")
                break

def testing(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device = 'cpu'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tests a PyTorch model, returning prediction values and ground truth.

    Args:
        model (nn.Module): PyTorch model to be trained.
        val_loader (DataLoader): Dataloader containing the validation batches.
        device (torch.device): Device on which the model and data are placed ('cpu' or 'cuda').

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - preds_val_scaled: Predicted values on the validation set (scaled).
            - y_val_scaled: Ground truth validation targets (scaled).
    """
    model.eval()

    preds, trues = [], []
    
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)

            out = model(xb).cpu().numpy()
                
            preds.append(out)
            trues.append(yb.numpy())
        
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0).reshape(-1, 1)
        
    return preds, trues