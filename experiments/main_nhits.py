import os
import torch
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from utils.data_prep import load_and_prepare, add_time_features, interquartile_range
from utils.evaluation import evaluation_metrics

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

lags = {
    "3h": 3,
    "6h": 6,
    "12h": 12,
    "24h": 24,
}

futr_cols = [
    "temperature_2m", 
    "relative_humidity",
    "Hour",
    "DayOfWeek",
    "DayOfYear",
    "Month",
    "WeekOfYear",
    "Weekend"
]

horizons = [1, 6]    


BATCH_SIZE = 64
EPOCHS = 500
PATIENCE = 20
LR = 1e-3
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3 
VAL_SPLIT = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_feature_set(feature_mode):
    """
    feature_mode:
        - 'consumption'
        - 'consumption+temporal'
        - 'consumption+weather'
        - 'all'
    """

    if feature_mode == "consumption":
        futr_cols = []

    elif feature_mode == "consumption+temporal":
        futr_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'doy_sin', 'doy_cos', 'month_sin', 'month_cos', 'woy_sin', 'woy_cos',
            'is_weekend', 'tod_Afternoon', 'tod_Evening', 'tod_Morning',
            'tod_Night'
            ]

    elif feature_mode == "consumption+weather":
        futr_cols = ['temperature_2m', 'relative_humidity_2m']

    elif feature_mode == "all":
        futr_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'doy_sin', 'doy_cos', 'month_sin', 'month_cos', 'woy_sin', 'woy_cos',
            'is_weekend', 'tod_Afternoon', 'tod_Evening', 'tod_Morning',
            'tod_Night', 'temperature_2m', 'relative_humidity_2m'
            ]
        
    else:
        raise ValueError("Modo de features inválido.")

    return futr_cols


def train_nhits_model(df, lag_size, horizon, futr_cols, df_val):
    model = NHITS(
        input_size=lag_size,
        h=horizon,
        max_steps=EPOCHS,
        early_stop_patience_steps=PATIENCE,
        n_freq_downsample=[2, 2, 1],
        futr_exog_list=futr_cols if futr_cols else []
    )

    nf = NeuralForecast(
        models=[model],
        freq='h'
    )

    print("oiii vou treinar\n")
    nf.fit(df=df, val_size=len(df_val))
    print("oiii treinei, agora vou predizer")

    forecast = nf.predict(
        h=len(df_val), 
        futr_df=df_val if futr_cols else None
    ).reset_index()
    print("feitoooo")

    return nf.models[0], forecast

def run_pipeline_nhits(feature_mode: str):
    df = load_and_prepare(CSV_PATH, DATETIME_COL, resample_rule='h')
    # df = df.drop(columns=['zone2','zone3','zone4','zone5'])
    df = df.drop(columns=['precipitation', 'rain', 'surface_pressure', 'wind_speed_10m', 'weather_code'])
    
    for col in df.columns:
        interquartile_range(df, col)

    df = add_time_features(df)
    df = df.reset_index()

    futr_cols = prepare_feature_set(feature_mode)

    print(f"\n=== Rodando NHITS com feature set: {feature_mode} ===")
    print("Future columns:", futr_cols)

    print(df.columns)

    # NeuralForecast 
    df_nf = df.rename(columns={DATETIME_COL: "ds", TARGET_COLS: "y"})
    df_nf["unique_id"] = "series1"
    df_nf = df_nf.reset_index(drop=True)

    VAL_SPLIT = 0.2
    split_idx = int(len(df) * (1 - VAL_SPLIT))
    df_nf_train = df_nf.iloc[:split_idx].copy()
    df_nf_val = df_nf.iloc[split_idx:].copy()

    print("DADOS DE TREINO")
    print(df_nf_train)
    print()

    for horizon in horizons:
        for lag_name, lag_value in lags.items():

            tag = f"{feature_mode}_{horizon}h_lag{lag_name}"
            print(f"\n=== Treinando {tag} ===")

            model, forecast = train_nhits_model(
                df=df_nf_train,
                lag_size=lag_value,
                horizon=horizon,
                futr_cols=futr_cols,
                df_val=df_nf_val
            )

            print("DADOS PREDITOS")
            print(forecast)
            print()

            print("DADOS DE TESTE")
            print(df_nf_val)
            print()

            y_true = df_nf_val["y"].values
            y_pred = forecast["NHITS"].values

            df_metrics = evaluation_metrics(y_true, y_pred, tag)
            print(df_metrics)

            df_metrics.to_csv(
                os.path.join(RESULTS_DIR, "nhits_metrics.csv"),
                mode='a',
                index=False
            )

            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"nhits_{tag}.pth"))

            print(f"Salvo modelo: nhits_{tag}")


RESULTS_DIR = "../results"

CSV_PATH = "../data/weather-data/consumption_weather_hourly.csv"
DATETIME_COL = "date"
TARGET_COLS = "consumo"

# CSV_PATH = "../data/Data Morocco - Laayoune.csv"
# DATETIME_COL = "DateTime"
# TARGET_COLS = "zone1"

run_pipeline_nhits('all')