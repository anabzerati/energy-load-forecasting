import os
import numpy as np
import pandas as pd
from typing import Tuple

def interquartile_range(df: pd.DataFrame, target_col: str, interpolate: bool = True) -> pd.DataFrame:
    """
    Detects and handles outliers using the Interquartile Range (IQR) method.

    Args:
        df (pd.DataFrame): input DataFrame containing the target column.
        target_col (str): name of the column on which to perform the outlier detection.
        interpolate : bool, optional (default=True)
            If True: interpolates outliers over time.
            If False: removes rows containing outlier values entirely.

    Returns
        pd.DataFrame: cleaned DataFrame
    """
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # outliers
    mask_outlier = (df[target_col] < lower_bound) | (df[target_col] > upper_bound)

    if interpolate:
        df.loc[mask_outlier, target_col] = np.nan
        df[target_col] = df[target_col].interpolate(method='time').bfill().ffill()
    else:
        # remove lines
        df = df[~mask_outlier]

    return df

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
    df.index = pd.to_datetime(df[dt_col], utc=True)

    # remove timezone info (optional, for simplicity)
    df.index = df.index.tz_convert(None)

    df = df.drop(columns=[dt_col])

    # force numeric conversion (important!)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

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
    dayofweek = df.index.dayofweek # 0=monday, 6=sunday
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

    # one-hot encoding
    tod_dummies = pd.get_dummies(df['time_of_day'], prefix='tod')
    df = pd.concat([df.drop(columns='time_of_day'), tod_dummies], axis=1)

    return df

def create_lag_windows(
    df: pd.DataFrame,
    target_col: str,
    lag: int,
    horizon: int
):
    """
    Creates lagged input sequences (X) and target values (y) for forecasting.

    Args:
        df: DataFrame containing the time series data.
        target_col: Name of the target column.
        lag: Number of past time steps to use as input.
        horizon: Number of steps ahead to predict (e.g., 5 means predict t+5).

    Returns:
        X: np.ndarray of shape (n_samples, lag, n_features)
        y: np.ndarray of shape (n_samples, 1)
    """
    X, y = [], []
    data = df.values
    target_idx = df.columns.get_loc(target_col)

    for i in range(len(df) - lag - horizon + 1):
        X.append(data[i:i+lag, :]) # lag steps of all features
        y.append(data[i+lag+horizon-1, target_idx]) # value at t+lag+horizon-1

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
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