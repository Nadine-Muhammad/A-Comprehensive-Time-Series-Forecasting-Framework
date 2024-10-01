from .base_model import BaseModel
import pandas as pd
import numpy as np
from pmdarima import auto_arima
from pmdarima.arima import ADFTest
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.express as px
from sklearn.linear_model import LinearRegression
from typing import Tuple
from constants import constants as consts
from timeseries_data import TimeSeriesObject
from utils.utils import validate_inputs
import warnings
import logging
import time
import contextlib
import os
import sys

# TODO BLACK FORMATTER


class ARIMAModel(BaseModel):
    def __init__(
        self,
        time_series_obj: TimeSeriesObject,
        end_date: str = None,
        develop_mode: bool = False,
    ) -> None:
        """
        Initialize the ARIMAModel with data.

        Args:
        - data (pd.DataFrame): Input time series data with a 'value' column.

        Attributes:
        - model (pmdarima.auto_arima): ARIMA model fitted to the residuals.
        - trend (pd.Series): Trend component from seasonal decomposition.
        - seasonal (pd.Series): Seasonal component from seasonal decomposition.
        - residual (pd.Series): Residual component from seasonal decomposition.
        """
        super().__init__(time_series_obj, end_date, develop_mode)
        self.params = self.config[consts.ARIMA]
        self.model = None
        self.trend = None
        self.seasonal = None
        self.residual = None
        self.data = time_series_obj.data
        self.end_date = end_date
        self.seasonal_component = None
        self._assign_seasonal_component()
        self.future_forecast_range = None
        if self.develop_mode:
            self.train, self.test = time_series_obj.split(self.horizon)
            start_time = time.time()

            print("Arima starts training .. ")
            with open(os.devnull, "w") as fnull:
                with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(
                    fnull
                ):
                    self.fit()
            end_time = time.time()
            print("Arima training is done ")
            self.train_duration = end_time - start_time
            start_time = time.time()
            self.test_predictions = self.predict()
            end_time = time.time()
            self.predict_duration = end_time - start_time
            self.future_forecast_range = self._calculate_forecast_range(
                self.data.index[-1]
            )

    def _calculate_forecast_range(self, start_date: pd.Timestamp) -> pd.DatetimeIndex:
        """
        Calculate the future forecast range.

        Returns:
        - pd.DatetimeIndex: Range of future dates for forecasting.
        """
        return pd.date_range(
            start=start_date, periods=self.horizon + 1, freq=self.freq
        )[
            1:
        ]  # Exclude the starting date

    def _assign_seasonal_component(self) -> str:
        """ """
        self.seasonal_component = "H_week" if self.freq == "H" else "15T_week"

    def _should_difference(self) -> int:
        """
        Determine if differencing is needed based on ADF test.

        Args:
        - data (pd.Series): Time series data to be tested.

        Returns:
        - int: 1 if differencing is needed, 0 otherwise.
        """
        diff = ADFTest(alpha=self.params["ADF_alpha"]).should_diff(self.data["value"])[
            1
        ]
        return 1 if diff else 0

    def _seasonal_decompose(self, df: pd.DataFrame) -> None:
        """
        Perform seasonal decomposition on the time series data.

        Decomposes the data into trend, seasonal, and residual components
        and stores them as attributes of the class.
        """

        decomposition = seasonal_decompose(
            df["value"],
            model=self.params["seasonal_decompose"],
            period=self.params["period"][self.seasonal_component],
            two_sided=False,
            extrapolate_trend=consts.FREQUENCY,
        )
        self.trend = decomposition.trend
        self.seasonal = decomposition.seasonal
        self.residual = decomposition.resid

    def fit(self) -> None:
        """seasonal_decompose
        Fit the ARIMA model to the residuals.

        Calls the seasonal_decompose method to get residuals and uses auto_arima
        to fit an ARIMA model to these residuals.
        """
        # Check whether we should difference
        d_value = self._should_difference()
        if self.develop_mode:
            self._seasonal_decompose(self.train)

        else:
            self._seasonal_decompose(self.data)

        # Fit Autoarima model
        self.model = auto_arima(
            self.residual,
            start_p=self.params["start_p"],
            m=self.params["period"][self.seasonal_component],
            d=d_value,
            seasonal=False,
            stepwise=True,
            trace=True,
            verbose=False,
        )

    def _trend_seas_forecast(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forecast future values of trend and seasonal components.

        Args:
        - horizon (int): Number of periods to forecast.

        Returns:
        - forecast_trend (np.ndarray): Forecasted trend values.
        - forecast_seasonal (np.ndarray): Forecasted seasonal values.
        """
        trend_index = np.arange(len(self.trend)).reshape(-1, 1)

        # Fit a linear model to the trend
        trend_model = LinearRegression()
        trend_model.fit(
            trend_index[-self.params["period"][self.seasonal_component] :],
            self.trend[-self.params["period"][self.seasonal_component] :],
        )

        # Predict future trend
        future_indices = np.arange(
            len(self.trend), len(self.trend) + self.horizon
        ).reshape(-1, 1)
        forecast_trend = trend_model.predict(future_indices)
        forecast_seasonal = self.seasonal[-self.horizon :]

        return forecast_trend, forecast_seasonal

    def predict(self) -> np.array:
        """
        Make forecasts for the specified horizon
        Forecasts trend, seasonality, residual separately
        Then combine results for final forecast

        Args:
        - horizon (int): Number of periods to forecast.
        - start_date (pd.Timestamp): Starting date for the forecast index.

        Returns:
        - np.array : Forecasted values.
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted yet")

        forecast_trend, forecast_seasonal = self._trend_seas_forecast()
        forecast_residuals = self.model.predict(n_periods=self.horizon)

        if self.future_forecast_range is not None:

            forecast_index = self._calculate_forecast_range(
                start_date=self.test.index[1]
            )

        else:
            forecast_index = self._calculate_forecast_range(
                start_date=self.data.index[1]
            )

        # Combine Forecast Components
        forecast = (
            np.array(forecast_trend)
            + np.array(forecast_seasonal)
            + np.array(forecast_residuals)
        )

        forecast_df = pd.DataFrame(
            {
                "Forecast": forecast,
                "Lower CI": forecast
                - self.params["conf_interval"] * np.std(forecast_residuals),
                "Upper CI": forecast
                + self.params["conf_interval"] * np.std(forecast_residuals),
            },
            index=forecast_index,
        )

        return np.array(forecast_df["Forecast"])
