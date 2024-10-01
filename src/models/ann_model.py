from .base_model import BaseModel
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import load_model
from models import model_utils
from timeseries_data import TimeSeriesObject
from utils import utils
from constants import constants as const
import time


class ANNModel(BaseModel):
    def __init__(
        self,
        time_series_obj: TimeSeriesObject,
        end_date: str,
        develop_mode: bool = False,
    ):
        """
        Initialize the ANN model.

        Args:
        - time_series_obj (TimeSeriesObject): The time series data object.
        - develop_mode (bool, optional): If False the model fit on all the data, else split the data and fit on a subset of it.
        """
        super().__init__(time_series_obj, end_date, develop_mode)
        self.batch_size = self.config[const.ANN][const.BATCH_SIZE]
        self.epochs = self.config[const.ANN][const.EPOCHS]
        self.seq_len = utils.get_seq_len(self.horizon, self.freq)
        self.x_forecast = None

        if develop_mode == False:
            self.scaler, self.scaled = model_utils.scale(time_series_obj.data)
            self.x_train, self.y_train = model_utils.rnn_generator(
                self.scaled, self.seq_len, self.horizon
            )
            self.x_test = self.scaled[-self.seq_len :].T
            self.model = self.build()

        else:
            self.train, self.test = time_series_obj.split(self.horizon)
            self.scaler, self.scaled_train, self.scaled_test = model_utils.scale(
                self.train, self.test
            )
            self.x_train, self.y_train = model_utils.rnn_generator(
                self.scaled_train, self.seq_len, self.horizon
            )
            self.x_test = self.scaled_train[
                -self.seq_len :
            ].T  # last sequence in scaled train data
            self.model = self.build()
            start_time = time.time()
            self.history = self.fit()
            end_time = time.time()
            self.train_duration = end_time - start_time
            start_time = time.time()
            self.predictions = self.predict().flatten()
            end_time = time.time()
            self.predict_duration = end_time - start_time
            self.x_forecast = np.concatenate((self.scaled_train, self.scaled_test))[
                -self.seq_len :
            ].T

    def build(self) -> Sequential:
        """
        Builds and compiles a one layer neural network with a linear activation function.

         Returns:
        - keras.models.Sequential: The compiled Keras Sequential model.
        """
        model = Sequential()
        model.add(Dense(self.horizon, input_dim=self.seq_len, activation="linear"))
        model.compile(optimizer=Adam(), loss="mean_squared_error")
        return model

    def fit(self, val_generator=None):
        """
        Fits and trains the neural network on train and validation data for a number of epochs.

        Args:
        - val_generator (PyDataset, optional): Validation data generator.

         Returns:
        - keras.callbacks.History: The history object containing training metrics for each epoch.
        """
        early_stopping = EarlyStopping(
            monitor=self.config[const.ANN][const.EARLY_STOPPING][const.MONITOR],
            patience=self.config[const.ANN][const.EARLY_STOPPING][const.PATIENCE],
            min_delta=self.config[const.ANN][const.EARLY_STOPPING][const.MIN_DELTA],
            restore_best_weights=True,
        )

        # In case of splitting data into train and validation to monitor val_loss during training
        if val_generator is not None:
            history = self.model.fit(
                x=self.x_train,
                y=self.y_train,
                epochs=self.epochs,
                verbose=1,
                validation_data=val_generator,
            )
        else:
            history = self.model.fit(
                x=self.x_train,
                y=self.y_train,
                epochs=self.epochs,
                callbacks=[early_stopping],
                verbose=0,
            )

        return history

    def rolling_forecast(self) -> np.ndarray:
        """
        Makes predictions using the trained model.

        Returns:
        - y_pred (np.ndarray): Predicted values.
        """
        y_pred = []
        if self.seq_len > 1:
            for _ in range(self.seq_len):
                pred = self.model.predict(self.x_test, verbose=0)
                y_pred.append(pred[0])

                self.x_test[0, :-1] = self.x_test[
                    0, 1:
                ]  # Shift all elements except the last one
                self.x_test[0, -1] = pred[
                    0
                ]  # Replace the last value with the new prediction

        elif self.seq_len == 1:
            pred = self.model.predict(self.x_test, verbose=0)
            y_pred.append(pred[0])

        return y_pred

    def predict(self) -> np.ndarray:
        """
        Makes predictions using the trained model.

        Returns:
        - y_pred (np.ndarray): Predicted values.
        """
        if self.x_forecast is None:
            y_pred = self.model.predict(self.x_test, verbose=0)
        else:
            y_pred = self.model.predict(self.x_forecast, verbose=0)

        return model_utils.inverse(self.scaler, y_pred).flatten()

    def load(self, directory: str, name: str):
        """
        Saves and loads the trained model into specified path.

        Args:
        - directory (str): A string containing an existing path to a directory to save the model to (e.g. '/workspaces/ds-internship-2024/models/').
        - name (str): A string containing the name of file to save to (e.g. 'my_model').

         Returns:
         - keras.Model: The loaded Keras model.
        """
        path = os.path.join(directory, name, "_tf")
        self.model.save(path)
        loaded = load_model(path)
        return loaded
