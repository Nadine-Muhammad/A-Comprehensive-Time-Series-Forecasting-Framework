from abc import ABC, abstractmethod
from timeseries_data import TimeSeriesObject
import pandas as pd
from config import config_handler
from utils import utils


class BaseModel(ABC):
    def __init__(
        self,
        time_series_obj: TimeSeriesObject,
        end_date: str,
        develop_mode: bool = False,
    ) -> None:
        """
        Initialize the BaseModel.

        Args:
        - time_series_obj (TimeSeriesObject): The time series data object.
        - end_date (str): The end date to forecast upto it.
        - develop_mode (bool, optional): If False the model fit on all the data, else split the data and fit on a subset of it.
        """
        # self.time_series_obj = time_series_obj
        self.config = config_handler.load_config()
        self.freq = time_series_obj.show_frequency()
        self.horizon = utils.validate_inputs(time_series_obj.data, self.freq, end_date)
        self.develop_mode = develop_mode

    @abstractmethod
    def fit(self) -> None:
        """
        Abstract method for fitting the model. Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def predict(self) -> pd.Series:
        """
        Abstract method for making predictions. Must be implemented by subclasses.

        Returns:
        - pd.Series: Forecasted values.
        """
        pass
