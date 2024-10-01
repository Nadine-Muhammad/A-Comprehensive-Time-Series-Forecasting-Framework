from .base_model import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
import os
from models.model_utils import scale,rnn_generator,inverse
from timeseries_data import TimeSeriesObject
from utils.utils import get_seq_len
from config import config_handler
from constants import constants as const
import time
from keras.callbacks import EarlyStopping






class RNNModel(BaseModel):
    def __init__(self, time_series_obj: TimeSeriesObject, end_date: str, develop_mode: bool = False) -> None:
        """
        Initialize the RNN model.

        Args:
        - time_series_obj (TimeSeriesObject): The TimeSeriesObject containing the data and other attributes.
        - end_date (str): The end date of the time series in 'YYYY-MM-DD' format.
        - develop_mode (bool, optional): If True, split the data into training and testing sets. Defaults to False.
        """
        super().__init__(time_series_obj, end_date, develop_mode)
        
        # Load configuration settings
        self.config = config_handler.load_config()
        self.learning_rate = self.config[const.RNN][const.LR]  # Learning rate from config
        self.batch_size = self.config[const.RNN][const.BATCH_SIZE]  # Batch size from config
        self.epochs = self.config[const.RNN][const.EPOCHS]  # Number of epochs from config
        self.x_forecast = None
        # Get sequence length for RNN input
        self.seq = get_seq_len(self.horizon, self.freq)

        # Conditional data preparation
        if not develop_mode:
            # Use the entire dataset for training
            self.scaler, self.scaled_data = scale(time_series_obj.data)
            self.X_train, self.y_train = rnn_generator(self.scaled_data, self.seq, self.horizon)
            self.X_test, self.y_test = None, None  
            self.X_test = self.scaled_data[-self.seq:].T  # Prepare the last sequence for prediction
            self.model= self.build()  # Build the RNN model upon initialization

            
        else:
            # Split the data into training and testing sets
            self.train_data, self.test_data = time_series_obj.split(self.horizon)
            self.scaler, self.scaled_train_data, self.scaled_test_data = scale(self.train_data, self.test_data)
            self.X_train, self.y_train = rnn_generator(self.scaled_train_data, self.seq, self.horizon)
            self.X_test = self.scaled_train_data[-self.seq:].T  # Prepare the last sequence for testing
            self.model= self.build()
            start_time = time.time()
            print("RNN starts training .. ")
            self.history = self.fit()
            print("RNN training is done ")
            end_time = time.time()
            self.train_duration = end_time - start_time
            start_time = time.time()
            self.predictions = self.predict().flatten() 
            end_time = time.time()
            self.predict_duration = end_time - start_time
            self.x_forecast = np.concatenate((self.scaled_train_data, self.scaled_test_data))[-self.seq:].T

      
    def _create_datasets(self):
        """
        Create training and test datasets.

        This method generates the input features and target values for both training and testing.
        """
        self.X_train, self.y_train = rnn_generator(self.scaled_train_data, self.seq, self.horizon)
        self.X_test, self.y_test = rnn_generator(self.scaler.transform(self.test_data), self.seq, self.horizon)

    def build(self):
        """
        Build and compile the RNN model.

        This method constructs the RNN model architecture and compiles it with the optimizer and loss function.
        """
        model = Sequential()
        model.add(SimpleRNN(units=50, activation="tanh", return_sequences=True, input_shape=(self.seq, 1)))
        model.add(SimpleRNN(units=50, activation="tanh", return_sequences=True))
        model.add(SimpleRNN(units=50))
        model.add(Dense(units=self.horizon, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mean_squared_error')

        return model

    def fit(self) -> dict:
        """
        Fit the RNN model.

        This method trains the model on the training dataset.

        Returns:
        - dict: Training history containing loss values.
        
        """
        early_stopping = EarlyStopping(
            monitor=self.config[const.RNN][const.EARLY_STOPPING][const.MONITOR],
            patience=self.config[const.RNN][const.EARLY_STOPPING][const.PATIENCE],
            min_delta=self.config[const.RNN][const.EARLY_STOPPING][const.MIN_DELTA],
            mode = 'min',
        )

        return self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size,callbacks=[early_stopping])

    def predict(self) -> np.ndarray:
        """
        Makes predictions using the trained model.

        Returns:
        - y_pred (np.ndarray): Predicted values.
        """
        if self.x_forecast is None:
            y_pred = self.model.predict(self.X_test, verbose=0) 
        else:
           y_pred = self.model.predict(self.x_forecast, verbose=0)

        return inverse(self.scaler, y_pred).flatten()


    def rolling_forecast(self) -> np.ndarray:
        """
        Perform rolling forecast.

        This method generates forecasts iteratively by updating the last observed window with the most recent forecast.

        Returns:
        - np.ndarray: Forecasted values.
        """
        forecasts = []
        last_window = self.data.iloc[-self.seq:]['value'].values.reshape(-1, 1)
        last_window = np.expand_dims(last_window, axis=0)  # Add batch dimension

        for _ in range(self.horizon):
            forecast = self.model.predict(last_window)
            forecasts.append(forecast[0, 0])
            
            # Update the window with the latest forecast
            last_window = np.roll(last_window, shift=-1, axis=1)
            last_window[0, -1, 0] = forecast
            
        forecast_values = self.scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()
        return forecast_values

    def load(self, directory: str, name: str) -> tf.keras.Model:
        """
        Save and load the trained model.

        Args:   
        - directory (str): Path to save the model.
        - name (str): Name of the file to save.

        Returns:
        - keras.Model: The loaded Keras model.
        """
        path = os.path.join(directory, name + '_tf')
        self.model.save(path)
        loaded_model = load_model(path)
        return loaded_model
