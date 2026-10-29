"""
Unit and Integration Tests for Public Transport Delay Forecasting Pipeline
"""

import os
import unittest
import numpy as np
import pandas as pd
import torch

from src.data_generator import generate_synthetic_transit_dataset
from src.preprocessing import (
    resample_transit_data,
    impute_missing_series,
    check_stationarity_adf,
    compute_acf_pacf
)
from src.feature_engineering import (
    build_temporal_features,
    merge_weather_features,
    create_lag_and_rolling_features,
    prepare_feature_pipeline,
    split_temporal_dataset
)
from src.deep_learning_models import (
    SlidingWindowProcessor,
    LSTMDelayModel,
    GRUDelayModel
)
from src.evaluation import calculate_metrics


class TestTransportDelayPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Generate small dataset for testing
        cls.transit_df, cls.weather_df = generate_synthetic_transit_dataset(
            start_date="2026-01-01",
            days=5,
            frequency_minutes=15,
            gps_dropout_rate=0.05,
            random_seed=42
        )
        
    def test_data_generation(self):
        self.assertGreater(len(self.transit_df), 100)
        self.assertGreater(len(self.weather_df), 50)
        self.assertIn("timestamp", self.transit_df.columns)
        self.assertIn("delay_minutes", self.transit_df.columns)
        self.assertIn("temperature_c", self.weather_df.columns)
        
    def test_resampling_and_imputation(self):
        resampled = resample_transit_data(self.transit_df, freq="15min")
        self.assertIn("mean_delay", resampled.columns)
        self.assertIn("trip_count", resampled.columns)
        
        imputed = impute_missing_series(resampled)
        self.assertEqual(imputed["mean_delay"].isna().sum(), 0)
        self.assertEqual(len(imputed), len(resampled))
        
    def test_stationarity_adf(self):
        resampled = resample_transit_data(self.transit_df, freq="15min")
        imputed = impute_missing_series(resampled)
        adf_res = check_stationarity_adf(imputed["mean_delay"])
        self.assertIn("test_statistic", adf_res)
        self.assertIn("p_value", adf_res)
        self.assertIn("is_stationary", adf_res)
        
    def test_feature_engineering_pipeline(self):
        resampled = resample_transit_data(self.transit_df, freq="15min")
        imputed = impute_missing_series(resampled)
        featured = prepare_feature_pipeline(imputed, self.weather_df)
        
        expected_cols = [
            "sin_hour", "cos_hour", "sin_dow", "cos_dow",
            "is_rush_hour", "rush_intensity", "precipitation_mm",
            "temperature_c", "lag_1", "lag_2", "rolling_mean_4"
        ]
        for col in expected_cols:
            self.assertIn(col, featured.columns)
            
        train, val, test = split_temporal_dataset(featured, train_ratio=0.6, val_ratio=0.2)
        self.assertEqual(len(train) + len(val) + len(test), len(featured))
        self.assertLess(train.index.max(), val.index.min())
        self.assertLess(val.index.max(), test.index.min())
        
    def test_sliding_window_and_dl_models(self):
        resampled = resample_transit_data(self.transit_df, freq="15min")
        imputed = impute_missing_series(resampled)
        featured = prepare_feature_pipeline(imputed, self.weather_df)
        
        feature_cols = ["mean_delay", "sin_hour", "cos_hour", "rush_intensity", "lag_1"]
        processor = SlidingWindowProcessor(
            feature_cols=feature_cols,
            target_col="mean_delay",
            lookback_window=10
        )
        processor.fit(featured)
        X, y, ts = processor.transform_and_create_windows(featured)
        
        self.assertEqual(len(X), len(featured) - 10)
        self.assertEqual(X.shape[1], 10)
        self.assertEqual(X.shape[2], len(feature_cols))
        
        # Test LSTM & GRU forward pass
        lstm = LSTMDelayModel(input_dim=len(feature_cols), hidden_dim=16, num_layers=1)
        gru = GRUDelayModel(input_dim=len(feature_cols), hidden_dim=16, num_layers=1)
        
        dummy_input = torch.tensor(X[:4], dtype=torch.float32)
        out_lstm = lstm(dummy_input)
        out_gru = gru(dummy_input)
        
        self.assertEqual(out_lstm.shape, (4, 1))
        self.assertEqual(out_gru.shape, (4, 1))
        
    def test_metrics_calculation(self):
        y_true = np.array([2.0, 4.0, 6.0, 8.0])
        y_pred = np.array([2.5, 3.5, 6.0, 9.0])
        metrics = calculate_metrics(y_true, y_pred)
        
        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("r2", metrics)
        self.assertGreater(metrics["r2"], 0.8)
        self.assertAlmostEqual(metrics["mae"], 0.5, places=2)


if __name__ == "__main__":
    unittest.main()
