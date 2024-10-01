import numpy as np
import pandas as pd
from prophet import Prophet
from .base_model import BaseModel
from constants import constants as consts
from timeseries_data import TimeSeriesObject
import contextlib
import os
import time


class ProphetModel(BaseModel):
    def __init__(
        self,
        time_series_obj: TimeSeriesObject,
        end_date: str,
        develop_mode: bool = False,
    ) -> None:
        super().__init__(time_series_obj, end_date, develop_mode)

        # Model parameters get it from config file
        params = self.config["prophet"]

        # Model Attributes
        self.data = time_series_obj.data
        self.data = self._rename_columns(self.data, "ds", "y")
        self.future_forecast_dates = None

        self.model = Prophet(
            seasonality_prior_scale=params[consts.SEASONALITY_PRIOR_SCALE][self.freq],
            weekly_seasonality=params[consts.WEEKLY_SEASONALITY][self.freq],
            daily_seasonality=params[consts.DAILY_SEASONALITY][self.freq],
            changepoint_prior_scale=params[consts.CHANGEPOINT_PRIOR_SCALE][self.freq],
            seasonality_mode=params[consts.SEASONALITY_MODE][self.freq],
        )

        if self.develop_mode:
            self.train, self.test = time_series_obj.split(self.horizon)
            self.train = self._rename_columns(self.train, "ds", "y")
            self.test = self._rename_columns(self.test, "ds", "y")
            start_time = time.time()
            with open(os.devnull, "w") as fnull:
                with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(
                    fnull
                ):
                    self.fit()
            end_time = time.time()
            self.train_duration = end_time - start_time
            self.future_forecast_dates = self._calculate_forecast_range()
            start_time = time.time()
            self.test_predictions = self.predict()
            end_time = time.time()
            self.predict_duration = end_time - start_time


    def fit(self) -> None:
        """
        Fit the Prophet model.
        """
        if self.develop_mode:
            self.model.fit(self.train)

        else:
            self.model.fit(self.data)

    def predict(self) -> np.array:
        """
        Make predictions using the fitted Prophet model.

        Returns:
        - np.array: Forecasted values.
        """

        if self.future_forecast_dates is not None:
            test_ds_df = pd.DataFrame({"ds": self.test["ds"]})
            predictions = self.model.predict(test_ds_df)["yhat"]
            return np.array(predictions)

        else:
            # Create future points to predict their values
            test_df = self._calculate_forecast_range()
            predictions = self.model.predict(test_df)["yhat"]
            return np.array(predictions)

    def _rename_columns(self, df: pd.DataFrame, c1: str, c2: str) -> pd.DataFrame:
        """
        Rename the columns of the DataFrame to the specified names.

        Args:
        - df (pd.DataFrame): Dataframe to change its columns.
        - c1 (str): New name for the 'timestamp' column.
        - c2 (str): New name for the 'value' column.

        Returns:
        - renamed_data (pd.DataFrame): DataFrame with columns renamed.
        """
        if not isinstance(c1, str) or not isinstance(c2, str):
            raise TypeError("Column names must be strings")

        if df.index.name == "timestamp":
           df = df.reset_index()

        # Rename columns
        renamed_data = df.rename(columns={"timestamp": c1, "value": c2})
        return renamed_data

    def _calculate_forecast_range(self) -> pd.DataFrame:
        """
        Calculate the forecast range for the time series model.

        Returns:
            pd.DataFrame: A DataFrame containing the forecast range based on the frequency.
        """
        return self.model.make_future_dataframe(self.horizon, self.freq).tail(
            self.horizon
        )

    def plot_components(self, df) -> None:
        """
        Plot the components of the time series data.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the time series data.

        """
        predictions = self.model.predict(df)
        self.model.plot_components(predictions)
