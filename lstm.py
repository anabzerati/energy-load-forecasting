import os
import random
import numpy as np
import pandas as pd
from typing import Tuple, List

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# reproducibility
SEED = 59
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# configs
CSV_PATH = "Data Morocco - Laayoune.csv"
RESULTS_DIR = "results"

DATETIME_COL = "DateTime"
TARGET_COLS = ["zone1"]        
RESAMPLE_RULE = "H"            # '10T' = 10min, 'H' = 1 hour, '24H' = 1 day
SEQ_LEN = 6                    # lagged input (6h)
HORIZON = 1                    # how many steps ahead we are predicting (1h)

BATCH_SIZE = 64
EPOCHS = 40
PATIENCE = 10
LR = 1e-5
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3 
VAL_SPLIT = 0.2

def load_and_prepare(csv_path: str, dt_col: str, decimal: str = ',', resample_rule: str = 'H') -> pd.DataFrame:    
    """
    Load a CSV file, parse the datetime column and resample the data.

    Args:
        csv_path (str): Path to the CSV file.
        dt_col (str): Name of the column containing datetime information.
        decimal (str, optional): Decimal separator used in the CSV (default is ',').
        resample_rule (str, optional): Pandas offset alias to resample the data (e.g., 'H' for hourly). 
                                       If None, no resampling is applied. Default is 'H'.

    Returns:
        pd.DataFrame: DataFrame with datetime index, numeric values, and resampled according to the given rule.
    """

    df = pd.read_csv(csv_path, decimal=decimal, quotechar='"', parse_dates=[dt_col])
    df.index = pd.to_datetime(df[dt_col])
    df = df.drop(columns=[dt_col])

    # resampling to desired frequency
    df = df.resample(resample_rule).sum()

    # interpolate gaps 
    df = df.interpolate(method='time').dropna()

    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich the DataFrame with cyclical and categorical time-based features

    Includes:
        - Daily and weekly cycles (sin, cos)
        - Day of the year and week of the year (seasonality)
        - Weekend indicator
        - Time of day segmentation (Night, Morning, Afternoon, Evening)

    Args:
        df (pd.DataFrame): DataFrame with a datetime index.

    Returns:
        pd.DataFrame: DataFrame with the new features added.
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a datetime index.")

    hour = df.index.hour

    # seasonality
    dayofweek = df.index.dayofweek          # 0=monday, 6=sunday
    dayofyear = df.index.dayofyear
    weekofyear = df.index.isocalendar().week.astype(int)

    # cycles
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dayofweek / 7)
    df['doy_sin'] = np.sin(2 * np.pi * dayofyear / 365)
    df['doy_cos'] = np.cos(2 * np.pi * dayofyear / 365)
    df['woy_sin'] = np.sin(2 * np.pi * weekofyear / 52)
    df['woy_cos'] = np.cos(2 * np.pi * weekofyear / 52)

    # weekend indicatior
    df['is_weekend'] = (dayofweek >= 5).astype(int)

    # time of day
    def get_time_of_day(h):
        if 0 <= h < 6:
            return "Night"
        elif 6 <= h < 12:
            return "Morning"
        elif 12 <= h < 18:
            return "Afternoon"
        else:
            return "Evening"

    df['time_of_day'] = [get_time_of_day(h) for h in hour]

    # One-hot encoding
    tod_dummies = pd.get_dummies(df['time_of_day'], prefix='tod')
    df = pd.concat([df.drop(columns='time_of_day'), tod_dummies], axis=1)

    return df

def create_lag_windows(df: pd.DataFrame, target_col: str, lag: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input (X) and output (y) windows for time-series model training.

    Args:
        df (pd.DataFrame): DataFrame with a datetime index and feature columns
        target_col (str): Name of the column to be predicted.
        lag (int, optional): Number of past timesteps used as input. Default is 1.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            X: NumPy array of shape (num_samples, lag, num_features)
            y: NumPy array of shape (num_samples, 1)
    """
    values = df.values
    target_idx = df.columns.get_loc(target_col)
    
    X_list, y_list = [], []
    for i in range(len(df) - lag):
        X_list.append(values[i:i+lag, :])              
        y_list.append(values[i+lag, target_idx])      
    
    X = np.array(X_list)
    y = np.array(y_list).reshape(-1, 1)               
    
    return X, y

def train_val_split(df: pd.DataFrame, val_split: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into training and validation sets based on a given ratio.

    Args:
        df (pd.DataFrame): DataFrame to be split.
        val_split (float): Fraction of the data to be used for validation.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
            df_train: Training subset of the DataFrame.
            df_val: Validation subset of the DataFrame.
    """
    split_idx = int(len(df) * (1 - val_split))

    df_train = df.iloc[:split_idx]
    df_val = df.iloc[split_idx:]

    return df_train, df_val

def evaluation_metrics(y_true: np.ndarray | list, y_pred: np.ndarray | list) -> pd.DataFrame:    
    """
    Compute global evaluation metrics (MAE, RMSE, MAPE).

    Args:
        y_true (np.array): True target values, shape (N*H, 1) or (N*H,).
        y_pred (np.array): Predicted values, shape (N*H, 1) or (N*H,).

    Returns:
        pd.DataFrame: DataFrame containing the calculated metrics:
                      - 'MAE'  : Mean Absolute Error
                      - 'RMSE' : Root Mean Squared Error
                      - 'MAPE' : Mean Absolute Percentage Error
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

    metrics = pd.DataFrame([{
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }])
    
    return metrics

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

def run_pipeline():
    # load data
    df = load_and_prepare(CSV_PATH, DATETIME_COL, resample_rule=RESAMPLE_RULE)

    df = df.drop(columns=['zone2', 'zone3', 'zone4', 'zone5'])

    print(f"=== Original === \n{df.head(10)}")

    df = add_time_features(df)

    print(f"\n\n=== With time features === \n{df.head(10)}")

    # split (avoid data leakage)
    df_train, df_val = train_val_split(df, VAL_SPLIT)

    # scaling
    feature_cols = list(df.columns)
    print(feature_cols)

    scaler = MinMaxScaler()
    scaler.fit(df_train[feature_cols])  

    df_train_scaled = pd.DataFrame(scaler.transform(df_train[feature_cols]),
                                index=df_train.index, columns=feature_cols)
    df_val_scaled = pd.DataFrame(scaler.transform(df_val[feature_cols]),
                                index=df_val.index, columns=feature_cols)

    # lag windows
    X_train, Y_train = create_lag_windows(df_train_scaled, lag=SEQ_LEN)
    X_val, Y_val = create_lag_windows(df_val_scaled, lag=SEQ_LEN)

    # data loaders
    train_ds = Dataset(X_train, Y_train)
    val_ds = Dataset(X_val, Y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    print(f"Train shape: X={X_train.shape}, y={Y_train.shape}")
    print(f"Val shape:   X={X_val.shape}, y={Y_val.shape}")

    # lstm
    n_features = X_train.shape[2]
    try:
        n_targets = Y_train.shape[2]
    except:
        n_targets = 1
    model = LSTM(n_features, HIDDEN_SIZE, NUM_LAYERS, HORIZON, n_targets, DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # training
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

    # load best model
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "best_lstm.pth"), map_location=device))
    model.eval()

    def predict_all(loader):
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)

                out = model(xb).cpu().numpy()
                
                preds.append(out)
                trues.append(yb.numpy())
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        return preds, trues

    preds_val_scaled, y_val_scaled = predict_all(val_loader)  
    y_val_scaled = y_val_scaled.reshape(-1, 1)

    print(preds_val_scaled.shape)
    print(y_val_scaled.shape)

    def inverse_scale_targets(y_scaled: np.ndarray,  scaler: MinMaxScaler, feature_cols: List[str], 
        target_idx: List[int] = [0]) -> np.ndarray:
        """
        Reconstructs the original scale of the target variable after it was scaled with a fitted MinMaxScaler. 
        Only the specified target column is inverted, while other columns are filled with zeros.

        Args:
            y_scaled (np.ndarray): Scaled predictions, shape (N,) or (N, 1).
            scaler (MinMaxScaler): Fitted scaler object used to scale the original data.
            feature_cols (List[str]): List of original column names used when fitting the scaler.
            target_idx (List[int], optional): Index of the target column to inverse scale. Default is [0].

        Returns:
            np.ndarray: Inverse-scaled target values, shape (N,).
        """
        y_scaled = np.array(y_scaled).reshape(-1, 1)
        full = np.zeros((len(y_scaled), len(feature_cols)))

        # insert scaled values only in the target column
        full[:, target_idx[0]] = y_scaled[:, 0]

        # apply inverse transformation
        inv = scaler.inverse_transform(full)

        # return only the target column
        y_real = inv[:, target_idx[0]]
        return y_real

    y_pred_real = inverse_scale_targets(preds_val_scaled, scaler, feature_cols)
    y_true_real = inverse_scale_targets(y_val_scaled, scaler, feature_cols)

    # evaluation
    df_metrics = evaluation_metrics(y_pred_real, y_true_real)

    print(df_metrics)

    df_metrics.to_csv(os.path.join(RESULTS_DIR, "lstm_metrics_by_horizon.csv"), index=False)
    
    print("LSTM metrics saved to", os.path.join(RESULTS_DIR, "lstm_metrics_by_horizon.csv"))

if __name__ == "__main__":
    run_pipeline()
