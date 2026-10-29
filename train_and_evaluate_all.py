"""
End-to-End Master Pipeline for Public Transport Delay Forecasting
Executes Data Generation -> Preprocessing -> Feature Engineering -> Statistical & DL Modeling -> Evaluation & Plotting.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from src.data_generator import generate_synthetic_transit_dataset, save_datasets
from src.preprocessing import (
    resample_transit_data,
    impute_missing_series,
    check_stationarity_adf,
    decompose_time_series,
    compute_acf_pacf
)
from src.feature_engineering import (
    prepare_feature_pipeline,
    split_temporal_dataset
)
from src.statistical_models import (
    StatisticalModelTrainer,
    ProphetDelayForecaster
)
from src.deep_learning_models import (
    set_seed,
    SlidingWindowProcessor,
    TimeSeriesSequenceDataset,
    LSTMDelayModel,
    GRUDelayModel,
    DeepLearningTrainer
)
from src.evaluation import (
    calculate_metrics,
    plot_actual_vs_predicted_24h,
    plot_model_comparison_bar,
    plot_residual_diagnostics
)


def run_pipeline():
    set_seed(42)
    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    print("=" * 70)
    print("STEP 1: DATA INGESTION & GENERATION")
    print("=" * 70)
    transit_df, weather_df = generate_synthetic_transit_dataset(
        start_date="2026-01-01",
        days=90,
        frequency_minutes=15,
        gps_dropout_rate=0.04,
        random_seed=42
    )
    t_path, w_path = save_datasets(transit_df, weather_df)
    print(f"Generated {len(transit_df):,} transit logs and {len(weather_df):,} weather records.")
    
    print("\n" + "=" * 70)
    print("STEP 2: PREPROCESSING, RESAMPLING & IMPUTATION")
    print("=" * 70)
    resampled_df = resample_transit_data(transit_df, freq="15min", target_col="delay_minutes")
    imputed_df = impute_missing_series(resampled_df, method="time_interpolation")
    print(f"Resampled series shape: {imputed_df.shape} (15-min intervals over 90 days)")
    
    # Stationarity & Decomposition
    adf_results = check_stationarity_adf(imputed_df["mean_delay"])
    print(f"ADF Test Statistic: {adf_results['test_statistic']:.4f}")
    print(f"ADF p-value: {adf_results['p_value']:.4e}")
    print(f"Conclusion: {adf_results['conclusion']}")
    
    # Save ADF results to JSON
    with open("reports/adf_results.json", "w") as f:
        json.dump(adf_results, f, indent=4)
        
    # Seasonal Decomposition Plot
    decomp = decompose_time_series(imputed_df["mean_delay"], period=96, model="additive")
    fig = decomp.plot()
    fig.set_size_inches(14, 8)
    fig.suptitle("Seasonal Decomposition of Transit Delay (Trend, Seasonality, Residual)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("reports/figures/seasonal_decomposition.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved seasonal decomposition plot to reports/figures/seasonal_decomposition.png")
    
    # ACF & PACF Plot
    acf_vals, pacf_vals = compute_acf_pacf(imputed_df["mean_delay"], nlags=96)
    fig, axes = plt.subplots(1, 2, figsize=(15, 4), dpi=300)
    axes[0].stem(range(len(acf_vals)), acf_vals)
    axes[0].set_title("Autocorrelation Function (ACF) - 96 Lags (24 Hours)", fontweight="bold")
    axes[0].set_xlabel("Lag (15-min intervals)")
    axes[0].set_ylabel("Autocorrelation")
    axes[0].axhline(0, color="black", linestyle="--", alpha=0.5)
    axes[0].axhline(1.96 / np.sqrt(len(imputed_df)), color="red", linestyle="--", alpha=0.5)
    axes[0].axhline(-1.96 / np.sqrt(len(imputed_df)), color="red", linestyle="--", alpha=0.5)
    
    axes[1].stem(range(len(pacf_vals)), pacf_vals)
    axes[1].set_title("Partial Autocorrelation Function (PACF) - 96 Lags", fontweight="bold")
    axes[1].set_xlabel("Lag (15-min intervals)")
    axes[1].set_ylabel("Partial Autocorrelation")
    axes[1].axhline(0, color="black", linestyle="--", alpha=0.5)
    axes[1].axhline(1.96 / np.sqrt(len(imputed_df)), color="red", linestyle="--", alpha=0.5)
    axes[1].axhline(-1.96 / np.sqrt(len(imputed_df)), color="red", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("reports/figures/acf_pacf.png", dpi=300)
    plt.close()
    print("Saved ACF/PACF plot to reports/figures/acf_pacf.png")
    
    print("\n" + "=" * 70)
    print("STEP 3: CONTEXTUAL FEATURE ENGINEERING")
    print("=" * 70)
    feature_df = prepare_feature_pipeline(imputed_df, weather_df, target_col="mean_delay")
    feature_df.to_csv("data/processed/featured_transit_data.csv")
    print(f"Engineered feature set: {feature_df.shape[1]} features across {len(feature_df):,} timestamps.")
    
    train_df, val_df, test_df = split_temporal_dataset(feature_df, train_ratio=0.70, val_ratio=0.15)
    print(f"Train set: {len(train_df):,} rows ({train_df.index.min()} to {train_df.index.max()})")
    print(f"Val set:   {len(val_df):,} rows ({val_df.index.min()} to {val_df.index.max()})")
    print(f"Test set:  {len(test_df):,} rows ({test_df.index.min()} to {test_df.index.max()})")
    
    # Common test ground truth and timestamps
    y_test_true = test_df["mean_delay"].values
    test_timestamps = test_df.index
    
    metrics_records = []
    predictions_dict = {}
    
    # -------------------------------------------------------------
    # 4.1 Statistical Baseline: ARIMA (p, d, q)
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4.1: TRAINING ARIMA BASELINE")
    print("=" * 70)
    t0 = time.time()
    # Fast grid search / optimal order
    best_order, arima_model, _ = StatisticalModelTrainer.grid_search_arima(
        train_df["mean_delay"],
        p_values=[1, 2, 3],
        d_values=[0, 1],
        q_values=[1, 2]
    )
    print(f"Optimal ARIMA order selected: {best_order}")
    
    # Fit full train+val for final out-of-sample forecast or forecast from train
    # For fair benchmarking, forecast steps equal to test set length
    arima_train_val = pd.concat([train_df["mean_delay"], val_df["mean_delay"]])
    from statsmodels.tsa.arima.model import ARIMA
    fitted_arima = ARIMA(arima_train_val, order=best_order).fit()
    arima_forecast = fitted_arima.forecast(steps=len(test_df))
    arima_time = time.time() - t0
    
    predictions_dict["ARIMA"] = arima_forecast.values
    arima_metrics = calculate_metrics(y_test_true, arima_forecast.values)
    arima_metrics["Model"] = "ARIMA"
    arima_metrics["Training_Time_Sec"] = round(arima_time, 2)
    metrics_records.append(arima_metrics)
    print(f"ARIMA Results -> RMSE: {arima_metrics['rmse']:.3f} | MAE: {arima_metrics['mae']:.3f} | R2: {arima_metrics['r2']:.3f}")
    
    # -------------------------------------------------------------
    # 4.2 Statistical Model: SARIMAX with Exogenous Features
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4.2: TRAINING SARIMAX WITH EXOGENOUS REGRESSORS")
    print("=" * 70)
    t0 = time.time()
    exog_cols = ["rush_intensity", "precipitation_mm", "is_weekend", "temperature_c"]
    exog_train_val = pd.concat([train_df[exog_cols], val_df[exog_cols]])
    exog_test = test_df[exog_cols]
    
    sarimax_model = StatisticalModelTrainer.train_sarimax(
        train_series=arima_train_val,
        order=(2, 0, 1),
        seasonal_order=(0, 0, 0, 0),
        exog_train=exog_train_val
    )
    sarimax_preds, sarimax_conf = StatisticalModelTrainer.forecast_sarimax(
        sarimax_model,
        steps=len(test_df),
        exog_test=exog_test
    )
    sarimax_time = time.time() - t0
    
    predictions_dict["SARIMAX"] = sarimax_preds.values
    sarimax_metrics = calculate_metrics(y_test_true, sarimax_preds.values)
    sarimax_metrics["Model"] = "SARIMAX"
    sarimax_metrics["Training_Time_Sec"] = round(sarimax_time, 2)
    metrics_records.append(sarimax_metrics)
    print(f"SARIMAX Results -> RMSE: {sarimax_metrics['rmse']:.3f} | MAE: {sarimax_metrics['mae']:.3f} | R2: {sarimax_metrics['r2']:.3f}")
    
    # -------------------------------------------------------------
    # 4.3 Statistical Model: Facebook Prophet
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4.3: TRAINING FACEBOOK PROPHET")
    print("=" * 70)
    t0 = time.time()
    prophet_forecaster = ProphetDelayForecaster(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        extra_regressors=["rush_intensity", "precipitation_mm", "is_weekend"]
    )
    train_val_df = pd.concat([train_df, val_df])
    prophet_forecaster.fit(train_val_df, target_col="mean_delay")
    prophet_forecast = prophet_forecaster.predict(test_df)
    prophet_preds = prophet_forecast["yhat"].values
    prophet_time = time.time() - t0
    
    predictions_dict["Prophet"] = prophet_preds
    prophet_metrics = calculate_metrics(y_test_true, prophet_preds)
    prophet_metrics["Model"] = "Prophet"
    prophet_metrics["Training_Time_Sec"] = round(prophet_time, 2)
    metrics_records.append(prophet_metrics)
    print(f"Prophet Results -> RMSE: {prophet_metrics['rmse']:.3f} | MAE: {prophet_metrics['mae']:.3f} | R2: {prophet_metrics['r2']:.3f}")
    
    # -------------------------------------------------------------
    # 4.4 Deep Learning: Sliding Window Preparation & PyTorch Models
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4.4: PREPARING SLIDING WINDOWS & TRAINING LSTM / GRU")
    print("=" * 70)
    dl_feature_cols = [
        "mean_delay", "sin_hour", "cos_hour", "sin_dow", "cos_dow",
        "rush_intensity", "is_weekend", "is_holiday",
        "temperature_c", "precipitation_mm", "wind_speed_kmh", "weather_severity_index",
        "lag_1", "lag_2", "lag_3", "lag_4", "rolling_mean_4", "rolling_std_4", "ewma_4"
    ]
    
    lookback = 24  # 6 hours lookback (24 * 15min)
    processor = SlidingWindowProcessor(
        feature_cols=dl_feature_cols,
        target_col="mean_delay",
        lookback_window=lookback,
        scaler_type="minmax"
    )
    processor.fit(train_df)
    
    X_train, y_train, _ = processor.transform_and_create_windows(train_df)
    X_val, y_val, _ = processor.transform_and_create_windows(val_df)
    
    # For test data, to predict from the very first test step, we prepend the last `lookback` rows of val_df
    test_with_prefix = pd.concat([val_df.iloc[-lookback:], test_df])
    X_test, y_test_dl, dl_test_ts = processor.transform_and_create_windows(test_with_prefix)
    
    print(f"Sliding Window Shapes -> X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    
    train_dataset = TimeSeriesSequenceDataset(X_train, y_train)
    val_dataset = TimeSeriesSequenceDataset(X_val, y_val)
    test_dataset = TimeSeriesSequenceDataset(X_test, y_test_dl)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Deep Learning Models on Device: {device}")
    
    # 4.4.1 Stacked LSTM
    t0 = time.time()
    lstm_model = LSTMDelayModel(input_dim=len(dl_feature_cols), hidden_dim=64, num_layers=2, dropout=0.2)
    lstm_trainer = DeepLearningTrainer(lstm_model, device=device, learning_rate=0.002, weight_decay=1e-4)
    lstm_history = lstm_trainer.fit(train_loader, val_loader, epochs=45, early_stopping_patience=10, verbose=True)
    
    lstm_preds_scaled = lstm_trainer.predict(test_loader)
    lstm_preds = processor.inverse_transform_target(lstm_preds_scaled)
    lstm_time = time.time() - t0
    
    predictions_dict["LSTM"] = lstm_preds
    lstm_metrics = calculate_metrics(y_test_true, lstm_preds)
    lstm_metrics["Model"] = "LSTM (Deep Learning)"
    lstm_metrics["Training_Time_Sec"] = round(lstm_time, 2)
    metrics_records.append(lstm_metrics)
    print(f"LSTM Results -> RMSE: {lstm_metrics['rmse']:.3f} | MAE: {lstm_metrics['mae']:.3f} | R2: {lstm_metrics['r2']:.3f}")
    
    # 4.4.2 GRU Model
    t0 = time.time()
    gru_model = GRUDelayModel(input_dim=len(dl_feature_cols), hidden_dim=64, num_layers=2, dropout=0.2)
    gru_trainer = DeepLearningTrainer(gru_model, device=device, learning_rate=0.002, weight_decay=1e-4)
    gru_history = gru_trainer.fit(train_loader, val_loader, epochs=45, early_stopping_patience=10, verbose=True)
    
    gru_preds_scaled = gru_trainer.predict(test_loader)
    gru_preds = processor.inverse_transform_target(gru_preds_scaled)
    gru_time = time.time() - t0
    
    predictions_dict["GRU"] = gru_preds
    gru_metrics = calculate_metrics(y_test_true, gru_preds)
    gru_metrics["Model"] = "GRU (Deep Learning)"
    gru_metrics["Training_Time_Sec"] = round(gru_time, 2)
    metrics_records.append(gru_metrics)
    print(f"GRU Results  -> RMSE: {gru_metrics['rmse']:.3f} | MAE: {gru_metrics['mae']:.3f} | R2: {gru_metrics['r2']:.3f}")
    
    # Save PyTorch Model Checkpoints
    torch.save(lstm_model.state_dict(), "reports/lstm_delay_model.pt")
    torch.save(gru_model.state_dict(), "reports/gru_delay_model.pt")
    
    # Plot Training Loss Curves
    plt.figure(figsize=(12, 5), dpi=300)
    plt.plot(lstm_history["train_loss"], label="LSTM Train Loss", color="#3B82F6", linewidth=2.0)
    plt.plot(lstm_history["val_loss"], label="LSTM Val Loss", color="#3B82F6", linestyle="--", linewidth=2.0)
    plt.plot(gru_history["train_loss"], label="GRU Train Loss", color="#8B5CF6", linewidth=2.0)
    plt.plot(gru_history["val_loss"], label="GRU Val Loss", color="#8B5CF6", linestyle="--", linewidth=2.0)
    plt.title("Deep Learning Training & Validation Loss Curves (MSE)", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss (Scaled MSE)", fontsize=11)
    plt.legend(frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("reports/figures/training_loss_curves.png", dpi=300)
    plt.close()
    print("Saved training loss curves to reports/figures/training_loss_curves.png")
    
    # -------------------------------------------------------------
    # 5. VISUALIZATIONS & METRICS COMPILATION
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: VISUALIZATIONS & SUMMARY METRICS")
    print("=" * 70)
    
    # Summary DataFrame
    metrics_df = pd.DataFrame(metrics_records)
    metrics_df.rename(columns={
        "rmse": "RMSE",
        "mae": "MAE",
        "mape": "MAPE (%)",
        "wape": "WAPE (%)",
        "r2": "R2",
        "max_err": "Max Error"
    }, inplace=True)
    
    metrics_df = metrics_df[["Model", "RMSE", "MAE", "MAPE (%)", "WAPE (%)", "R2", "Max Error", "Training_Time_Sec"]]
    metrics_df.to_csv("reports/metrics_summary.csv", index=False)
    print("\n--- PERFORMANCE SUMMARY TABLE ---")
    print(metrics_df.to_string(index=False))
    
    # 24-Hour Horizon Plot (first 96 intervals of test set = 24 hours)
    horizon_24h = 96
    test_24h_ts = test_timestamps[:horizon_24h]
    y_true_24h = y_test_true[:horizon_24h]
    preds_24h = {k: v[:horizon_24h] for k, v in predictions_dict.items()}
    
    plot_actual_vs_predicted_24h(
        timestamps=test_24h_ts,
        y_true=y_true_24h,
        predictions_dict=preds_24h,
        save_path="reports/figures/actual_vs_predicted_24h.png",
        title="Public Transit Delay Forecasts: 24-Hour Test Period (Actual vs. Predicted)"
    )
    
    # 7-Day Extended Test Window Plot
    horizon_7d = min(len(test_timestamps), 96 * 7)
    test_7d_ts = test_timestamps[:horizon_7d]
    y_true_7d = y_test_true[:horizon_7d]
    preds_7d = {k: v[:horizon_7d] for k, v in predictions_dict.items()}
    
    plot_actual_vs_predicted_24h(
        timestamps=test_7d_ts,
        y_true=y_true_7d,
        predictions_dict=preds_7d,
        save_path="reports/figures/actual_vs_predicted_7days.png",
        title="Public Transit Delay Forecasts: 7-Day Extended Test Horizon"
    )
    
    # Bar Chart Comparisons
    plot_model_comparison_bar(metrics_df, save_path="reports/figures/model_comparison_bar.png")
    
    # Residual Diagnostics
    plot_residual_diagnostics(y_test_true, predictions_dict, save_path="reports/figures/residual_diagnostics.png")
    
    print("\n" + "=" * 70)
    print("MASTER PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
