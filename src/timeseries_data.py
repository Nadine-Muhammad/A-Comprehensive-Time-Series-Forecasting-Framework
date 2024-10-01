import pandas as pd
from pandas import DataFrame
from typing import Tuple


class TimeSeriesObject:
    def __init__(self, data: DataFrame) -> None:
        """
        Initialize the TimeSeriesObject.

        Args:
        - data (DataFrame): DataFrame containing time series data with at least two columns.
        """
        self.data = data
        self._standard_form()
        self.frequency = self.infer_freq()
        self._handle_missing_values()

    def _standard_form(self) -> None:
        """
        Convert the DataFrame to a standard format with two columns: 'timestamp' and 'value'.
        """

        # Check if there is id column at the begigning and drop it
        if len(self.data.columns) == 3:
            self.data.drop(self.data.columns[0], axis=1, inplace=True)

        if self.data.columns[0] != "timestamp" or self.data.columns[1] != "value":
            self.data.columns = ["timestamp", "value"]

    def infer_freq(self) -> str:
        """
        Infer the frequency of the time series data.

        Returns:
        - str: Inferred frequency of the time series.
        """
        self.data["timestamp"] = pd.to_datetime(self.data["timestamp"])
        self.data.set_index("timestamp", inplace=True)

        # Infer frequency
        freq = pd.infer_freq(self.data.index)
        if freq is None:
            raise ValueError("Frequency could not be inferred")

        return freq

    def _handle_missing_values(self) -> None:
        """
        Resample the data to the inferred frequency and fill missing values.
        """
        # Resample to the inferred frequency
        self.data = self.data.resample(self.frequency).mean()  # Mean for aggregation
        self.data["value"] = self.data["value"].interpolate(method="linear")

    def show_frequency(self) -> str:
        """
        Print the inferred frequency.
        """
        return self.frequency

    def split(self, horizon: int) -> Tuple[DataFrame, DataFrame]:
        """
        Split the dataset into training and test sets based on the horizon.
        Train is the whole dataset except for the last horizon.
        Test is the last horizon

        Args:
        - horizon (int): Number of timestamps to forecast.

        Returns:
        - tuple: A tuple containing:(train, test)
        """
        train = self.data.iloc[: len(self.data) - int(horizon)]
        test = self.data.iloc[len(self.data) - int(horizon) :]
        return train, test


def convert_datetime_format(
    df: pd.DataFrame, formats=["%d/%m/%Y %H:%M"]
) -> pd.DataFrame:
    """
    Convert datetime strings in a specified column of a DataFrame from various formats to a standard format.

    Parameters:
    - df: pd.DataFrame - The DataFrame containing the datetime strings.
    - column: str - The column name containing the datetime strings.
    - formats: List[str] - A list of possible input datetime formats.
    - set_index: bool - Whether to set the first column as the index of the DataFrame.

    Returns:
    - pd.DataFrame - The DataFrame with the datetime column converted to the standard format '%y-%m-%d %H:%M'.
    """

    # Function to convert using multiple formats
    def parse_dates(date_series, formats):
        for fmt in formats:
            try:
                return pd.to_datetime(date_series, format=fmt, errors="raise")
            except (ValueError, TypeError):
                continue
        return pd.NaT

    # Apply the function to the specified column
    df[df.columns[0]] = df[df.columns[0]].apply(lambda x: parse_dates(x, formats))

    # Drop rows where conversion failed (optional)
    df.dropna(subset=[df.columns[0]], inplace=True)

    # Ensure the column is in datetime format
    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])

    # Format to the desired output format
    df[df.columns[0]] = df[df.columns[0]].dt.strftime("%Y-%m-%d %H:%M")

    return df
