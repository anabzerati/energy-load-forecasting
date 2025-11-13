import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score
from pathlib import Path

def evaluation_metrics(y_true: np.ndarray | list, y_pred: np.ndarray | list, tag) -> pd.DataFrame:    
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
    
    print(y_true)
    print()
    print(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = pd.DataFrame([{
        'exp': tag,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }])
    
    return metrics

def plot_predictions(
    df: pd.DataFrame,
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    save_path: str | None = None
):
    """
    Plot and optionally save forecast vs. real consumption.

    Args:
        df (pd.DataFrame): Input dataframe
        y_true (array-like): True target values
        y_pred (array-like): Predicted values
        save_path (str | None): Optional path to save the plot as PDF.
    """
    plt.figure(figsize=(12, 5))

    plt.plot(df.index[-len(y_true):], y_true, label='Real')
    plt.plot(df.index[-len(y_pred):], y_pred, label='Prediction')

    plt.title("Energy load forescast")
    plt.xlabel("Time")
    plt.ylabel("Consumption [Ampere]")

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, format="pdf")
        print(f"Plot saved as: {save_path}")
    else:
        plt.show()