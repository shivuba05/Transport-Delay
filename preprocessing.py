"""
Preprocessing and Time-Series Analysis Module
Handles resampling, missing value imputation, stationarity testing (ADF), and seasonal decomposition.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose, STL


def resample_transit_data(
    df: pd.DataFrame,
    freq: str = "15min",
    route_id: Optional[str] = None,
    target_col: str = "delay_minutes"
) -> pd.DataFrame:
    """
    Resamples irregular transit arrival logs into a regular time-series grid.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw transit dataframe with 'timestamp' column.
    freq : str
        Resampling frequency (e.g., '15min', '30min', '1h').
    route_id : Optional[str]
        If specified, filters logs for a specific route.
    target_col : str
        Target delay column name.
        
    Returns:
    --------
    pd.DataFrame:
        Regular time-series dataframe indexed by timestamp with mean, std, min, max, count.
    """
    df_copy = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_copy["timestamp"]):
        df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"])
        
    if route_id is not None:
        df_copy = df_copy[df_copy["route_id"] == route_id]
        
    df_copy.set_index("timestamp", inplace=True)
    
    # Resample and aggregate
    resampled = df_copy[target_col].resample(freq).agg(["mean", "max", "min", "std", "count"]).rename(
        columns={
            "mean": "mean_delay",
            "max": "max_delay",
            "min": "min_delay",
            "std": "std_delay",
            "count": "trip_count"
        }
    )
    
    # Include weekend/holiday flags if present
    if "is_weekend" in df_copy.columns:
        resampled["is_weekend"] = df_copy["is_weekend"].resample(freq).max()
    if "is_holiday" in df_copy.columns:
        resampled["is_holiday"] = df_copy["is_holiday"].resample(freq).max()
        
    return resampled


def impute_missing_series(
    series_df: pd.DataFrame,
    method: str = "time_interpolation",
    service_start_hour: int = 5,
    service_end_hour: int = 24
) -> pd.DataFrame:
    """
    Imputes missing values resulting from GPS dropouts or sparse time slots.
    
    Parameters:
    -----------
    series_df : pd.DataFrame
        Resampled time-series dataframe.
    method : str
        Imputation method: 'time_interpolation', 'forward_fill', 'spline', or 'hybrid'.
    service_start_hour : int
        Start of daily transit operations (0-23).
    service_end_hour : int
        End of daily transit operations (0-23).
        
    Returns:
    --------
    pd.DataFrame:
        Dataframe with imputed missing values.
    """
    df_imputed = series_df.copy()
    
    # Identify overnight / non-operational periods (where trip_count == 0 and hours outside service)
    hours = df_imputed.index.hour
    is_operational = (hours >= service_start_hour) & (hours < service_end_hour)
    
    # For operational hours, if mean_delay is NaN, interpolate
    if method == "time_interpolation":
        df_imputed["mean_delay"] = df_imputed["mean_delay"].interpolate(method="time")
    elif method == "forward_fill":
        df_imputed["mean_delay"] = df_imputed["mean_delay"].ffill().bfill()
    elif method == "spline":
        df_imputed["mean_delay"] = df_imputed["mean_delay"].interpolate(method="spline", order=2)
    elif method == "hybrid":
        # Forward fill first for short gaps, then time interpolation for longer
        df_imputed["mean_delay"] = df_imputed["mean_delay"].interpolate(method="linear").ffill().bfill()
    else:
        df_imputed["mean_delay"] = df_imputed["mean_delay"].interpolate(method="linear").ffill().bfill()
        
    # For non-operational overnight hours, delay is essentially 0 or baseline
    df_imputed.loc[~is_operational, "mean_delay"] = df_imputed.loc[~is_operational, "mean_delay"].fillna(0.0)
    
    # Remaining NaNs in auxiliary columns
    df_imputed["mean_delay"] = df_imputed["mean_delay"].fillna(0.0)
    df_imputed["trip_count"] = df_imputed["trip_count"].fillna(0)
    df_imputed["std_delay"] = df_imputed["std_delay"].fillna(0.0)
    df_imputed["max_delay"] = df_imputed["max_delay"].fillna(df_imputed["mean_delay"])
    df_imputed["min_delay"] = df_imputed["min_delay"].fillna(df_imputed["mean_delay"])
    
    if "is_weekend" in df_imputed.columns:
        df_imputed["is_weekend"] = (df_imputed.index.weekday >= 5).astype(int)
    if "is_holiday" in df_imputed.columns:
        df_imputed["is_holiday"] = df_imputed["is_holiday"].ffill().fillna(0).astype(int)
        
    return df_imputed


def check_stationarity_adf(
    series: pd.Series,
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    Performs Augmented Dickey-Fuller (ADF) test to evaluate stationarity.
    
    Null Hypothesis (H0): The series has a unit root (is non-stationary).
    Alternative Hypothesis (H1): The series has no unit root (is stationary).
    
    Parameters:
    -----------
    series : pd.Series
        Time-series data to test.
    significance_level : float
        P-value threshold for rejecting the null hypothesis (default 0.05).
        
    Returns:
    --------
    Dict[str, Any]:
        Dictionary of ADF test statistics, p-value, critical values, and stationarity conclusion.
    """
    clean_series = series.dropna()
    adf_result = adfuller(clean_series, autolag="AIC")
    
    test_stat = adf_result[0]
    p_value = adf_result[1]
    used_lags = adf_result[2]
    n_obs = adf_result[3]
    critical_values = adf_result[4]
    ic_best = adf_result[5]
    
    is_stationary = p_value < significance_level
    
    conclusion = (
        f"Reject H0 at alpha={significance_level}: The series is STATISTICAL STATIONARY (p={p_value:.4e} < {significance_level})."
        if is_stationary else
        f"Fail to reject H0 at alpha={significance_level}: The series is NON-STATIONARY (p={p_value:.4e} >= {significance_level})."
    )
    
    return {
        "test_statistic": float(test_stat),
        "p_value": float(p_value),
        "used_lags": int(used_lags),
        "n_obs": int(n_obs),
        "critical_values": critical_values,
        "is_stationary": bool(is_stationary),
        "conclusion": conclusion
    }


def decompose_time_series(
    series: pd.Series,
    period: int = 96,  # 96 15-min intervals in 24 hours
    model: str = "additive"
) -> Any:
    """
    Decomposes time-series into Trend, Seasonal, and Residual components.
    
    Parameters:
    -----------
    series : pd.Series
        Time-series values with DatetimeIndex.
    period : int
        Seasonal period length (e.g. 96 for 15-min daily cycle, 24 for hourly).
    model : str
        'additive' or 'multiplicative'.
        
    Returns:
    --------
    DecomposeResult:
        statsmodels decomposition result object.
    """
    clean_series = series.dropna()
    # Add small epsilon if multiplicative to avoid zero errors
    if model == "multiplicative":
        clean_series = clean_series.clip(lower=0.01)
        
    decomposition = seasonal_decompose(
        clean_series,
        model=model,
        period=period,
        extrapolate_trend="freq"
    )
    return decomposition


def compute_acf_pacf(
    series: pd.Series,
    nlags: int = 48
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Autocorrelation (ACF) and Partial Autocorrelation (PACF).
    """
    clean_series = series.dropna()
    acf_vals = acf(clean_series, nlags=nlags)
    pacf_vals = pacf(clean_series, nlags=nlags, method="yw")
    return acf_vals, pacf_vals
