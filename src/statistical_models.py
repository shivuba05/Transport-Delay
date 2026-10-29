"""
Statistical Modeling Module for Transit Delay Forecasting
Implements ARIMA, SARIMAX (with exogenous variables), and Facebook Prophet.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from prophet import Prophet

warnings.filterwarnings("ignore")


class StatisticalModelTrainer:
    """
    Wrapper for training, tuning, and evaluating statistical time-series models.
    """
    
    @staticmethod
    def grid_search_arima(
        train_series: pd.Series,
        p_values: List[int] = [0, 1, 2, 3],
        d_values: List[int] = [0, 1],
        q_values: List[int] = [0, 1, 2, 3],
        exog_train: Optional[pd.DataFrame] = None
    ) -> Tuple[Tuple[int, int, int], Any, pd.DataFrame]:
        """
        Performs grid search to find the optimal (p, d, q) order based on AIC.
        """
        best_aic = float("inf")
        best_order = None
        best_model = None
        results_list = []
        
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    try:
                        if exog_train is not None:
                            model = ARIMA(
                                train_series,
                                order=(p, d, q),
                                exog=exog_train
                            ).fit()
                        else:
                            model = ARIMA(
                                train_series,
                                order=(p, d, q)
                            ).fit()
                            
                        aic = model.aic
                        bic = model.bic
                        results_list.append({
                            "order": (p, d, q),
                            "aic": aic,
                            "bic": bic,
                            "converged": True
                        })
                        
                        if aic < best_aic:
                            best_aic = aic
                            best_order = (p, d, q)
                            best_model = model
                    except Exception as e:
                        continue
                        
        results_df = pd.DataFrame(results_list).sort_values(by="aic").reset_index(drop=True)
        return best_order, best_model, results_df
        
    @staticmethod
    def train_sarimax(
        train_series: pd.Series,
        order: Tuple[int, int, int] = (2, 0, 1),
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
        exog_train: Optional[pd.DataFrame] = None
    ) -> Any:
        """
        Fits a SARIMAX model with optional seasonal components and exogenous regressors.
        """
        model = SARIMAX(
            train_series,
            order=order,
            seasonal_order=seasonal_order,
            exog=exog_train,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted_model = model.fit(disp=False, maxiter=200)
        return fitted_model
        
    @staticmethod
    def forecast_sarimax(
        fitted_model: Any,
        steps: int,
        exog_test: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Generates out-of-sample forecasts and 95% confidence intervals from SARIMAX.
        """
        forecast_res = fitted_model.get_forecast(steps=steps, exog=exog_test)
        predicted_mean = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int(alpha=0.05)
        return predicted_mean, conf_int

    @staticmethod
    def diagnose_residuals(fitted_model: Any, lags: int = 24) -> Dict[str, Any]:
        """
        Performs residual diagnostic checks including Ljung-Box test for white noise.
        """
        residuals = fitted_model.resid
        lb_test = acorr_ljungbox(residuals.dropna(), lags=[lags], return_df=True)
        lb_pvalue = lb_test["lb_pvalue"].values[0]
        lb_stat = lb_test["lb_stat"].values[0]
        
        return {
            "mean_residual": float(residuals.mean()),
            "std_residual": float(residuals.std()),
            "ljung_box_stat": float(lb_stat),
            "ljung_box_pvalue": float(lb_pvalue),
            "is_white_noise": bool(lb_pvalue > 0.05)
        }


class ProphetDelayForecaster:
    """
    Wrapper for Facebook Prophet forecasting with external regressors.
    """
    
    def __init__(
        self,
        growth: str = "linear",
        daily_seasonality: bool = True,
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = False,
        seasonality_mode: str = "additive",
        extra_regressors: Optional[List[str]] = None
    ):
        self.growth = growth
        self.daily_seasonality = daily_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.seasonality_mode = seasonality_mode
        self.extra_regressors = extra_regressors or []
        self.model = None
        
    def fit(self, train_df: pd.DataFrame, target_col: str = "mean_delay"):
        """
        Fits Prophet on time-series dataframe.
        """
        prophet_df = pd.DataFrame()
        prophet_df["ds"] = train_df.index
        prophet_df["y"] = train_df[target_col].values
        
        for reg in self.extra_regressors:
            if reg in train_df.columns:
                prophet_df[reg] = train_df[reg].values
                
        self.model = Prophet(
            growth=self.growth,
            daily_seasonality=self.daily_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality,
            seasonality_mode=self.seasonality_mode
        )
        
        for reg in self.extra_regressors:
            if reg in train_df.columns:
                self.model.add_regressor(reg)
                
        self.model.fit(prophet_df)
        return self
        
    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates predictions for the test time-series dataframe.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
            
        future_df = pd.DataFrame()
        future_df["ds"] = test_df.index
        
        for reg in self.extra_regressors:
            if reg in test_df.columns:
                future_df[reg] = test_df[reg].values
                
        forecast = self.model.predict(future_df)
        forecast.set_index("ds", inplace=True)
        return forecast
