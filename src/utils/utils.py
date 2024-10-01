from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from tabulate import tabulate
import numpy as np
import pandas as pd
from dateutil import parser
from typing import Tuple


def calculate_errors(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """
    Calculate and return error metrics: RMSE, MAE, and MAPE.

    Args:
    - y_true (np.ndarray): Actual values.
    - y_pred (np.ndarray): Predicted values.

    Returns:
    - tuple: RMSE, MAE, and MAPE.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return rmse, mae, mape


def print_predictions(
    start_date, freq: str, horizon: int, y_predict: np.ndarray
) -> pd.DataFrame:
    """
    Prints prediction results in a table with their timestamps.

    Args:
    - start_date (Timestamp): Last date value in data.
    - freq (str): Frequency of data.
    - horizon (int): Number of points to forecast.
    - y_predict (NumPy array): Predicted vaues.

    Returns:
    - df (pd.DataFrame): Dataframe with date as index and a values column.
    """

    # Generate the date range, starting from the timestamp after start_date
    datetime_column = pd.date_range(start=start_date, periods=horizon + 1, freq=freq)[
        1:
    ]

    # Create a DataFrame
    df = pd.DataFrame({"Value": y_predict}, index=datetime_column)

    headers = ["Date", "Values"]
    table = tabulate(df, headers, tablefmt="pretty")

    # Display the table
    print(table)
    return df


def get_horizon(user_num_points: int, valid_num_points: int, freq: str) -> int:
    """
    Calculate and return the horizon for predictions based on user requirements and data frequency.

    Parameters:
        user_num_points (int): Number of points the user wants to predict.
        valid_num_points (int): Number of valid points available for prediction (25% of the data).
        freq (str): Frequency of the time series data.

    Returns:
        horizon (int): The calculated horizon for predictions.
    """

    # Determine max horizon according to the freq
    if freq == "15T":
        max_horizon = 1344
    elif freq == "H":
        max_horizon = 336

    min_horizon = 1
    horizon = min(
        user_num_points, valid_num_points
    )  # To check weither the user required horizon greater than 25% of history or not
    horizon = max(horizon, min_horizon)  # The horizon must not be less from min_horizon
    horizon = min(horizon, max_horizon)  # The max horizon to predict is 2 weeks

    if horizon > max_horizon:
        print("The model can predict upto 2 weeks maximum.")
    elif horizon < min_horizon:
        print("The model can predict at least 1 point.")
    return horizon


def get_seq_len(horizon: int, freq: str) -> int:
    """
    Calculate and return the sequence length based on the horizon and data frequency.

    Parameters:
        horizon (int): The horizon for predictions.
        freq (str): Frequency of the time series data.

    Returns:
        seq_len (int): The calculated sequence length.
    """

    if freq == "15T":
        seq_len = max(96, 4 * horizon)

    elif freq == "H":
        seq_len = max(24, 4 * horizon)

    return seq_len


def validate_inputs(
    data: pd.DataFrame, freq: str, end_date: str, percentage: float = 0.25
) -> int:
    """
    Args:
        data (pd.DataFrame): The time series data.
        freq (str): The frequency of the time series data.
        end_date (str): The end date to forecast upto it.
        percentage (float, optional): A percentage (between 0 and 1) to calculate a proportion of the data for validation
                                      or forecasting. Default is 0.25 (i.e., 25%).

    Returns:
        horizon (int): number of points to forecat on
        input_seq (int): input sequence length for neural network models
    """

    valid = False
    while not valid:
        end_date = parser.parse(end_date).strftime("%Y-%m-%d %H:%M")
        datetime_period = pd.date_range(
            start=data[-1:].index[0], end=end_date, freq=freq
        )[1:]
        user_num_points = len(
            datetime_period
        )  # horizon required from the user as points
        valid_num_points = int(len(data) * percentage)  # 25% of the data

        horizon = get_horizon(user_num_points, valid_num_points, freq)
        seq_len = get_seq_len(horizon, freq)

        if seq_len > len(data) - horizon + 1:
            print("The model cannot forecast on this horizon decrease it")
            end_date = input("Enter end date again: ")
        else:
            valid = True

    return horizon
