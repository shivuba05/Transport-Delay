"""
Feature Engineering Module for Transit Delay Forecasting
Builds temporal, cyclical, weather, lag, and rolling window features.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict


def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts temporal, cyclical, and rush-hour features from DatetimeIndex.
    """
    df_feat = df.copy()
    if not isinstance(df_feat.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")
        
    dt = df_feat.index
    
    # Standard calendar features
    df_feat["hour"] = dt.hour
    df_feat["minute"] = dt.minute
    df_feat["dayofweek"] = dt.dayofweek
    df_feat["dayofmonth"] = dt.day
    df_feat["month"] = dt.month
    df_feat["is_weekend"] = (dt.dayofweek >= 5).astype(int)
    
    # Continuous fractional hour
    fractional_hour = dt.hour + dt.minute / 60.0
    
    # Cyclical Sine/Cosine encodings
    df_feat["sin_hour"] = np.sin(2 * np.pi * fractional_hour / 24.0)
    df_feat["cos_hour"] = np.cos(2 * np.pi * fractional_hour / 24.0)
    df_feat["sin_dow"] = np.sin(2 * np.pi * dt.dayofweek / 7.0)
    df_feat["cos_dow"] = np.cos(2 * np.pi * dt.dayofweek / 7.0)
    df_feat["sin_month"] = np.sin(2 * np.pi * (dt.month - 1) / 12.0)
    df_feat["cos_month"] = np.cos(2 * np.pi * (dt.month - 1) / 12.0)
    
    # Rush Hour Indicators (Morning 07:00-09:30, Evening 16:30-19:30 on weekdays)
    is_weekday = (df_feat["is_weekend"] == 0)
    is_morning_rush = is_weekday & (fractional_hour >= 7.0) & (fractional_hour <= 9.5)
    is_evening_rush = is_weekday & (fractional_hour >= 16.5) & (fractional_hour <= 19.5)
    
    df_feat["is_morning_rush"] = is_morning_rush.astype(int)
    df_feat["is_evening_rush"] = is_evening_rush.astype(int)
    df_feat["is_rush_hour"] = (is_morning_rush | is_evening_rush).astype(int)
    
    # Continuous rush hour intensity curve
    morning_bell = np.exp(-((fractional_hour - 8.25) ** 2) / (2 * (0.8 ** 2)))
    evening_bell = np.exp(-((fractional_hour - 17.5) ** 2) / (2 * (1.0 ** 2)))
    df_feat["rush_intensity"] = np.where(
        is_weekday,
        0.8 * morning_bell + 1.0 * evening_bell,
        0.2 * morning_bell + 0.3 * evening_bell
    )
    
    return df_feat


def merge_weather_features(
    transit_ts_df: pd.DataFrame,
    weather_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges synchronized weather observations into the transit time-series.
    """
    df_feat = transit_ts_df.copy()
    w_df = weather_df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(w_df["timestamp"]):
        w_df["timestamp"] = pd.to_datetime(w_df["timestamp"])
        
    w_df.set_index("timestamp", inplace=True)
    
    # Reindex or interpolate weather to match transit time-series resolution
    # Resample weather to transit grid via forward-fill / linear interpolation
    w_resampled = w_df.reindex(df_feat.index.union(w_df.index))
    w_resampled["temperature_c"] = w_resampled["temperature_c"].interpolate(method="time").ffill().bfill()
    w_resampled["precipitation_mm"] = w_resampled["precipitation_mm"].interpolate(method="time").ffill().bfill()
    w_resampled["wind_speed_kmh"] = w_resampled["wind_speed_kmh"].interpolate(method="time").ffill().bfill()
    w_resampled["weather_condition"] = w_resampled["weather_condition"].ffill().bfill()
    
    # Subset back to transit timestamps
    w_aligned = w_resampled.loc[df_feat.index]
    
    df_feat["temperature_c"] = w_aligned["temperature_c"]
    df_feat["precipitation_mm"] = w_aligned["precipitation_mm"]
    df_feat["wind_speed_kmh"] = w_aligned["wind_speed_kmh"]
    df_feat["weather_condition"] = w_aligned["weather_condition"]
    
    # Derived weather indicators
    df_feat["is_raining"] = (df_feat["precipitation_mm"] > 0.1).astype(int)
    df_feat["is_heavy_rain"] = (df_feat["precipitation_mm"] > 4.0).astype(int)
    df_feat["is_freezing"] = (df_feat["temperature_c"] <= 0.0).astype(int)
    
    # Weather severity composite score [0, 1]
    norm_precip = np.clip(df_feat["precipitation_mm"] / 10.0, 0, 1)
    norm_wind = np.clip(df_feat["wind_speed_kmh"] / 40.0, 0, 1)
    norm_cold = np.clip((5.0 - df_feat["temperature_c"]) / 15.0, 0, 1)
    df_feat["weather_severity_index"] = 0.5 * norm_precip + 0.3 * norm_cold + 0.2 * norm_wind
    
    return df_feat


def create_lag_and_rolling_features(
    df: pd.DataFrame,
    target_col: str = "mean_delay",
    lags: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Generates historical lag features and rolling window aggregates.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Time-series dataframe with DatetimeIndex.
    target_col : str
        Target column for lag generation (default: 'mean_delay').
    lags : List[int]
        Steps to lag (e.g., [1, 2, 3, 4, 96]).
    rolling_windows : List[int]
        Window lengths for rolling statistics (e.g. [4, 12, 24]).
    """
    df_feat = df.copy()
    
    if lags is None:
        # Default for 15-min series: 1 (15m), 2 (30m), 3 (45m), 4 (1h), 96 (24h)
        lags = [1, 2, 3, 4, 8, 96]
        
    if rolling_windows is None:
        # 4 steps (1 hour), 12 steps (3 hours), 24 steps (6 hours)
        rolling_windows = [4, 12, 24]
        
    # Lag features
    for lag in lags:
        df_feat[f"lag_{lag}"] = df_feat[target_col].shift(lag)
        
    # Rolling features (shifted by 1 to prevent target leakage at time t)
    target_shifted = df_feat[target_col].shift(1)
    
    for window in rolling_windows:
        df_feat[f"rolling_mean_{window}"] = target_shifted.rolling(window=window, min_periods=1).mean()
        df_feat[f"rolling_std_{window}"] = target_shifted.rolling(window=window, min_periods=1).std().fillna(0.0)
        df_feat[f"rolling_max_{window}"] = target_shifted.rolling(window=window, min_periods=1).max()
        df_feat[f"rolling_min_{window}"] = target_shifted.rolling(window=window, min_periods=1).min()
        
    # Exponential moving average
    df_feat["ewma_4"] = target_shifted.ewm(span=4, adjust=False).mean()
    df_feat["ewma_12"] = target_shifted.ewm(span=12, adjust=False).mean()
    
    return df_feat


def prepare_feature_pipeline(
    resampled_transit_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    target_col: str = "mean_delay",
    drop_na_lags: bool = True
) -> pd.DataFrame:
    """
    Executes the end-to-end contextual feature engineering pipeline.
    """
    df = build_temporal_features(resampled_transit_df)
    df = merge_weather_features(df, weather_df)
    df = create_lag_and_rolling_features(df, target_col=target_col)
    
    if drop_na_lags:
        df = df.dropna()
        
    return df


def split_temporal_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits time-series chronologically into train, validation, and test sets.
    Preserves strict temporal ordering to prevent lookahead bias.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    return train_df, val_df, test_df
