from pandas import DataFrame
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from typing import Tuple

def train_val_test_split(data: DataFrame, trian_ratio: float, val_ratio: float) -> tuple:
        """
        Split the dataset into training, validation, and test sets based on the specified ratios.
 
        Args:
        - data (pandas.DataFrame): The DataFrame containing the time series data to be split.
        - train_ratio (float): The proportion of the data to be used for the training set (e.g., 0.7 for 70%).
        - val_ratio (float): The proportion of the data to be used for the validation set (e.g., 0.15 for 15%).
        - test_ratio (float): The proportion of the data to be used for the test set (e.g., 0.15 for 15%).
 
        Returns:
        - tuple: A tuple containing:(train, val, test)
        """
        train_size = int(len(data) * trian_ratio)
        val_size = int(len(data) * val_ratio)
        #test_size = int(len(data) * test_ratio)
        train = data.iloc[:train_size]
        val = data.iloc[train_size:train_size+val_size]
        test = data.iloc[train_size+val_size:]
        return train, val, test
        # train_size = int(len(data) * trian_ratio)
        # val_size = int(len(data) * val_ratio)
        # test_size = int(len(data) * test_ratio)
        # train = data.iloc[:train_size]
        # val = data.iloc[train_size:train_size+val_size]
        # test = data.iloc[train_size+val_size:train_size + val_size + test_size]
        # return train, val, test

def scale(train, val=None, test=None) -> tuple:
    """
    Scales the training, validation, and test data using Min-Max scaling.

    Args:
    - train (pandas.DataFrame): The training data to be scaled.
    - val (pandas.DataFrame, optional): The validation data to be scaled. Defaults to None.
    - test (pandas.DataFrame, optional): The test data to be scaled. Defaults to None.

    Returns:
    - tuple: A tuple containing:
        - scaler (sklearn.preprocessing.MinMaxScaler): The scaler fitted to the training data.
        - scaled_train (NumPy array): The scaled training data.
        - scaled_val (NumPy array, optional): The scaled validation data, returned only if `val` is provided.
        - scaled_test (NumPy array, optional): The scaled test data, returned only if `test` is provided.

    """
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train.values.reshape(-1, 1))
    if val is not None and test is not None:
        scaled_val = scaler.transform(val.values.reshape(-1, 1))
        scaled_test = scaler.transform(test.values.reshape(-1, 1))
        return scaler, scaled_train, scaled_val, scaled_test

    elif val is not None and test is None:
        scaled_val = scaler.transform(val.values.reshape(-1, 1))
        return scaler, scaled_train, scaled_val

    else:
        scaled_val = None
        scaled_test = None
        return scaler, scaled_train


def inverse(scaler, y_pred):
    """
    Reverses the predicted data back to its original scale.

    Args:
    - scaler (sklearn.preprocessing.MinMaxScaler): The scaler previously used to transform the data.
    - y_pred (NumPy array): Predicted values.

    Returns:
    - original_pred (NumPy array): Forecasted values in original scale.
    """

    return scaler.inverse_transform(y_pred)


def ann_generator(scaled_train, n_input, batch_size, scaled_val=None, scaled_test=None):
    """
      Creates data generators for training, validation and testing data.

    Args:
    - scaled_train (NumPy array): The scaled training data.
    - scaled_val (NumPy array): The scaled validation data.
    - scaled_test (NumPy array): The scaled test data.
    - n_input (int): Input sequence length.
    - batch_size (int): Training batch size.

    Returns:
    - train_generator (PyDataset): Training data generator.
    - val_generator (PyDataset): Validation data generator.
    - test_generator (PyDataset): Testing data generator.
    """

    train_generator = TimeseriesGenerator(
        scaled_train.flatten(),
        scaled_train.flatten(),
        length=n_input,
        batch_size=batch_size,
    )
    if scaled_val is not None and scaled_test is not None:
        val_generator = TimeseriesGenerator(
            scaled_val.flatten(),
            scaled_val.flatten(),
            length=n_input,
            batch_size=batch_size,
        )
        test_generator = TimeseriesGenerator(
            scaled_test.flatten(),
            scaled_test.flatten(),
            length=n_input,
            batch_size=batch_size,
        )
        return train_generator, val_generator, test_generator
    else:
        val_generator = None
        test_generator = None
        return train_generator


def rnn_generator(
    scaled_data: np.ndarray, seq_len: int, forecast_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create dataset from scaled data for sequence-to-sequence forecasting.

    Args:
    - scaled_data (np.ndarray): Scaled data to create dataset from.
    - seq_len (int): Length of the input sequence.
    - forecast_len (int): Number of future time steps to forecast.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: Tuple of features and labels.
    """
    X = []
    y = []
    for i in range(seq_len, len(scaled_data) - forecast_len + 1):
        X.append(scaled_data[i - seq_len : i, 0])
        y.append(scaled_data[i : i + forecast_len, 0])

    X = np.array(X)
    y = np.array(y)

    # Reshaping
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Shape: (samples, seq_len, 1)
    y = np.reshape(
        y, (y.shape[0], forecast_len, 1)
    )  # Shape: (samples, forecast_len, 1)

    return X, y
