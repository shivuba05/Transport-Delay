"""
Evaluation and Visualization Module for Transit Delay Forecasting
Computes benchmark metrics (RMSE, MAE, MAPE, R2) and generates publication-quality plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard time-series forecasting error metrics.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    # Filter out NaNs if any
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_t = y_true[valid_mask]
    y_p = y_pred[valid_mask]
    
    if len(y_t) == 0:
        return {"rmse": 0.0, "mae": 0.0, "mape": 0.0, "wape": 0.0, "r2": 0.0, "max_err": 0.0}
        
    mse = mean_squared_error(y_t, y_p)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_t, y_p)
    
    # Avoid division by zero in MAPE by adding small epsilon or filtering zeros
    nonzero_mask = np.abs(y_t) > 0.1
    if np.sum(nonzero_mask) > 0:
        mape = np.mean(np.abs((y_t[nonzero_mask] - y_p[nonzero_mask]) / y_t[nonzero_mask])) * 100.0
    else:
        mape = 0.0
        
    # WAPE (Weighted Absolute Percentage Error) = sum(|y_t - y_p|) / sum(|y_t|)
    sum_actual = np.sum(np.abs(y_t))
    wape = (np.sum(np.abs(y_t - y_p)) / sum_actual * 100.0) if sum_actual > 0 else 0.0
    
    r2 = r2_score(y_t, y_p)
    max_err = np.max(np.abs(y_t - y_p))
    
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "wape": float(wape),
        "r2": float(r2),
        "max_err": float(max_err)
    }


def plot_actual_vs_predicted_24h(
    timestamps: pd.DatetimeIndex,
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    save_path: Optional[str] = "reports/figures/actual_vs_predicted_24h.png",
    title: str = "Transit Delay Forecasts: 24-Hour Horizon Comparison (Actual vs. Predicted)"
):
    """
    Plots Actual vs. Predicted curves for all models over a 24-hour test period.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 7), dpi=300)
    
    # Ground truth
    plt.plot(timestamps, y_true, label="Ground Truth (Actual Delay)", color="#1F2937", linewidth=2.5, zorder=5)
    
    colors = {
        "ARIMA": "#EF4444",
        "SARIMAX": "#F59E0B",
        "Prophet": "#10B981",
        "LSTM": "#3B82F6",
        "GRU": "#8B5CF6"
    }
    
    styles = {
        "ARIMA": "--",
        "SARIMAX": "-.",
        "Prophet": ":",
        "LSTM": "-",
        "GRU": "-"
    }
    
    for name, preds in predictions_dict.items():
        c = colors.get(name, "#6B7280")
        ls = styles.get(name, "-")
        lw = 2.0 if "LSTM" in name or "GRU" in name else 1.6
        plt.plot(timestamps, preds, label=f"Predicted ({name})", color=c, linestyle=ls, linewidth=lw, alpha=0.9)
        
    plt.title(title, fontsize=15, fontweight="bold", pad=15)
    plt.xlabel("Timestamp", fontsize=12, fontweight="semibold")
    plt.ylabel("Delay (Minutes)", fontsize=12, fontweight="semibold")
    plt.legend(loc="upper left", frameon=True, fontsize=11, framealpha=0.95)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved 24-hour forecast plot to: {save_path}")
    plt.close()


def plot_model_comparison_bar(
    metrics_df: pd.DataFrame,
    save_path: Optional[str] = "reports/figures/model_comparison_bar.png"
):
    """
    Plots comparative bar charts of RMSE, MAE, and R2 across models.
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    
    palette = sns.color_palette("mako", len(metrics_df))
    
    # 1. RMSE
    sns.barplot(data=metrics_df, x="Model", y="RMSE", ax=axes[0], palette=palette)
    axes[0].set_title("Root Mean Squared Error (RMSE) - Lower is Better", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("RMSE (Minutes)", fontsize=11)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha="center", va="center", xytext=(0, 5), textcoords="offset points", fontsize=10, fontweight="bold")
                         
    # 2. MAE
    sns.barplot(data=metrics_df, x="Model", y="MAE", ax=axes[1], palette=palette)
    axes[1].set_title("Mean Absolute Error (MAE) - Lower is Better", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("MAE (Minutes)", fontsize=11)
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha="center", va="center", xytext=(0, 5), textcoords="offset points", fontsize=10, fontweight="bold")
                         
    # 3. R2 Score
    sns.barplot(data=metrics_df, x="Model", y="R2", ax=axes[2], palette=palette)
    axes[2].set_title("Coefficient of Determination ($R^2$) - Higher is Better", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("$R^2$ Score", fontsize=11)
    for p in axes[2].patches:
        axes[2].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha="center", va="center", xytext=(0, 5), textcoords="offset points", fontsize=10, fontweight="bold")
                         
    for ax in axes:
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)
        
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved model comparison bar chart to: {save_path}")
    plt.close()


def plot_residual_diagnostics(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    save_path: Optional[str] = "reports/figures/residual_diagnostics.png"
):
    """
    Plots residual distribution and error density curves across models.
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    
    colors = {
        "ARIMA": "#EF4444",
        "SARIMAX": "#F59E0B",
        "Prophet": "#10B981",
        "LSTM": "#3B82F6",
        "GRU": "#8B5CF6"
    }
    
    for name, preds in predictions_dict.items():
        residuals = y_true - preds
        c = colors.get(name, "#6B7280")
        sns.kdeplot(residuals, ax=axes[0], label=name, color=c, linewidth=2.0)
        axes[1].scatter(preds, residuals, label=name, color=c, alpha=0.3, s=18)
        
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1.2)
    axes[0].set_title("Residual Error Density Distribution", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Residual (Actual - Predicted) [Minutes]", fontsize=11)
    axes[0].set_ylabel("Density", fontsize=11)
    axes[0].legend()
    
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1.2)
    axes[1].set_title("Residuals vs. Predicted Values (Homoscedasticity Check)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Predicted Delay (Minutes)", fontsize=11)
    axes[1].set_ylabel("Residual (Actual - Predicted) [Minutes]", fontsize=11)
    axes[1].legend()
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved residual diagnostics plot to: {save_path}")
    plt.close()
