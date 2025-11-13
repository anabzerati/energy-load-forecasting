import os
import numpy as np
import pandas as pd
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.preprocessing import MinMaxScaler

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from data.dataset import TimeSeriesDataset
from models.lstm import LSTM, training, testing
# from models.gru import GRU, training, testing
# # from models.transformer import TimeSeriesTransformer, training, testing

from utils.data_prep import load_and_prepare, add_time_features, create_lag_windows, train_val_split, interquartile_range
from utils.evaluation import evaluation_metrics, plot_predictions

def run_pipeline(
    csv_path: str,
    resample_rule: str,
    seq_len: int,
    horizon: int,
    device: torch.device,
    tag: str,
    target_col: str = "consumo",
):
    """
    Runs the full forecasting pipeline for given parameters.
    """
    # load data
    df = load_and_prepare(csv_path, DATETIME_COL, resample_rule=resample_rule) 
    # df = df.drop(columns=['zone2', 'zone3', 'zone4', 'zone5', 'zone6', 'zone7'])
    df = df.drop(columns=['precipitation', 'rain', 'surface_pressure', 'wind_speed_10m', 'weather_code'])

    # remove outliers
    for col in df.columns:
        print(col)
        interquartile_range(df, col)

    print(f"=== Original === \n{df.head(10)}")

    # df = add_time_features(df)

    print(f"\n\n=== With time features === \n{df.head(10)}")

    # split (avoid data leakage)
    df_train, df_val = train_val_split(df, VAL_SPLIT)

    # scaling
    feature_cols = list(df.columns)

    scaler = MinMaxScaler()
    scaler.fit(df_train[feature_cols])  

    df_train_scaled = pd.DataFrame(scaler.transform(df_train[feature_cols]),
                                index=df_train.index, columns=feature_cols)
    df_val_scaled = pd.DataFrame(scaler.transform(df_val[feature_cols]),
                                index=df_val.index, columns=feature_cols)

    # lag windows
    X_train, Y_train = create_lag_windows(df_train_scaled, target_col=target_col, lag=seq_len, horizon=horizon)
    X_val, Y_val = create_lag_windows(df_val_scaled, target_col=target_col, lag=seq_len, horizon=horizon)

    # data loaders
    train_ds = TimeSeriesDataset(X_train, Y_train)
    val_ds = TimeSeriesDataset(X_val, Y_val)

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

    # horizon is always 1, because we make only 1 prediction per input data
    model = LSTM(input_size=n_features, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, horizon=1, 
                 n_targets=n_targets, dropout=DROPOUT).to(device)
        
    # model = TimeSeriesTransformer(
    #     input_size=n_features,
    #     d_model=HIDDEN_SIZE, # dimensão interna
    #     num_layers=NUM_LAYERS, # num camadas encoder
    #     num_heads=8,  # cabeças de atenção
    #     dim_feedforward=256, # feedforward interno
    #     dropout=DROPOUT
    # ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    training(model, train_loader, val_loader, optimizer, criterion, RESULTS_DIR, PATIENCE, EPOCHS, device, tag=tag)

    # load and test best model
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, f"best_lstm_{tag}.pth"), map_location=device))
    preds_val_scaled, y_val_scaled = testing(model, val_loader, device)

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
    df_metrics = evaluation_metrics(y_true_real, y_pred_real, tag)

    print(df_metrics)

    df_metrics.to_csv(os.path.join(RESULTS_DIR, "lstm_metrics_by_horizon_foum_eloued.csv"), mode='a', index=False)
    
    plot_predictions(df_val, y_true_real, y_pred_real, save_path=os.path.join(RESULTS_DIR, f"plot_lstm_{tag}.pdf"))

    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"best_lstm_{tag}.pth"))

    print("LSTM metrics saved to", os.path.join(RESULTS_DIR, "lstm_metrics_by_horizon_foum_eloued.csv"))


# configs
# CSV_PATH = "../data/Data Morocco - Foum eloued.csv"
CSV_PATH = "../data/weather-data/consumption_weather_hourly_foum_eloued.csv"

RESULTS_DIR = "../results"

# DATETIME_COL = "DateTime"
# TARGET_COLS = "zone1"        

DATETIME_COL = "date"
TARGET_COLS = "consumption"

BATCH_SIZE = 64
EPOCHS = 500
PATIENCE = 20
LR = 1e-3
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3 
VAL_SPLIT = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experiments = [
    # Horizon = 1h
    {"RESAMPLE_RULE": "h", "SEQ_LEN": 3,  "HORIZON": 1, "TAG": "1h_lag3"},
    {"RESAMPLE_RULE": "h", "SEQ_LEN": 6,  "HORIZON": 1, "TAG": "1h_lag6"},
    {"RESAMPLE_RULE": "h", "SEQ_LEN": 12, "HORIZON": 1, "TAG": "1h_lag12"},
    {"RESAMPLE_RULE": "h", "SEQ_LEN": 24, "HORIZON": 1, "TAG": "1h_lag24"},

    # Horizon = 6h
    {"RESAMPLE_RULE": "h", "SEQ_LEN": 6,  "HORIZON": 6, "TAG": "6h_lag6"},
    {"RESAMPLE_RULE": "h", "SEQ_LEN": 12, "HORIZON": 6, "TAG": "6h_lag12"},
    {"RESAMPLE_RULE": "h", "SEQ_LEN": 24, "HORIZON": 6, "TAG": "6h_lag24"},

    # Horizon = 1 day
    {"RESAMPLE_RULE": "d", "SEQ_LEN": 1, "HORIZON": 1, "TAG": "1d_lag1"},
    {"RESAMPLE_RULE": "d", "SEQ_LEN": 2, "HORIZON": 1, "TAG": "1d_lag2"},
]

if __name__ == "__main__":
    all_metrics = []

    for exp in experiments:
        print("\n" + "=" * 60)
        print(f"Running experiment: {exp['TAG']}")
        print("=" * 60)

        run_pipeline(
            csv_path=str(CSV_PATH),
            resample_rule=exp["RESAMPLE_RULE"],
            seq_len=exp["SEQ_LEN"],
            horizon=exp["HORIZON"],
            device=device,
            tag=f"{exp['TAG']}_onlyweather",
            target_col=TARGET_COLS
        )

    print("\nAll experiments finished.")