import numpy as np
import pandas as pd
from typing import Tuple
from utils.utils import calculate_errors
from timeseries_data import TimeSeriesObject
from models import ARIMAModel, ANNModel, BaseModel, ProphetModel, RNNModel
import multiprocessing


class AutoModel(BaseModel):
    def __init__(
        self,
        time_series_obj: TimeSeriesObject,
        end_date: str,
        develop_mode: bool = True,
    ) -> None:
        """
        Initialize the model with instances of ARIMA, ANN, Prophet, and RNN models running in multiprocessing.

        Parameters:
        time_series_obj (TimeSeriesObject): An object containing the time series data.

        Attributes:
        arima (ARIMAModel): Instance of the ARIMA model.
        ann (ANNModel): Instance of the ANN model.
        prophet (ProphetModel): Instance of the Prophet model.
        rnn (RNNModel): Instance of the RNN model.
        """

        super().__init__(time_series_obj, end_date, develop_mode)

        manager = multiprocessing.Manager()
        self.models = manager.dict()

        # Define a function to initialize models in parallel
        def initialize_model(key, model_class):
            self.models[key] = model_class(time_series_obj, end_date, develop_mode)

        # List of processes for initializing each model
        processes = [
            multiprocessing.Process(
                target=initialize_model, args=("arima", ARIMAModel)
            ),
            multiprocessing.Process(target=initialize_model, args=("ann", ANNModel)),
            multiprocessing.Process(
                target=initialize_model, args=("prophet", ProphetModel)
            ),
            multiprocessing.Process(target=initialize_model, args=("rnn", RNNModel)),
        ]

        # Start all processes
        print(f'Models are training now ..............')
        for process in processes:
            process.start()

        for process in processes:
            process.join()
        print(f"models finished training ..............")

        # Extract the model instances from the manager dictionary
        self.arima = self.models["arima"]
        self.ann = self.models["ann"]
        self.prophet = self.models["prophet"]
        self.rnn = self.models["rnn"]

    def fit(self) -> None:
        pass

    def predict(self) -> np.array:
        """
        Returns the predictions of the best model based on MAPE values.

        Returns:
            np.array: Predictions of the best-performing model.
        """
        model_name = self.get_best_model_name()
        print(f"The best model is: {model_name}")
        best_model = self.get_the_best_model()
        return best_model.predict()

    def get_models_predictions(self) -> np.array:
        """
        Generates predictions using each model (ARIMA, ANN, Prophet, RNN).

        Returns:
            Tuple containing predictions from each model:
            - ARIMA predictions as a pandas DataFrame.
            - ANN predictions as a NumPy array.
            - Prophet predictions as a pandas Series.
            - RNN predictions as a NumPy array.
        """

        # Models Predictions
        arima_predicions = self.arima.predict()
        ann_predictions = self.ann.predict()
        prophet_predictins = self.prophet.predict()
        rnn_predictions = self.rnn.predict()

        return arima_predicions, ann_predictions, prophet_predictins, rnn_predictions

    def order_models(self) -> pd.DataFrame:
        """
        Orders models based on Mean Absolute Percentage Error (MAPE) values and returns a DataFrame.

        Returns:
            df (pd.DataFrame): containing model names and corresponding MAPE values sorted by MAPE.
        """

        # Calculate MAPE for all the models
        _, _, arima_mape = calculate_errors(
            self.arima.test.values, self.arima.test_predictions
        )
        _, _, ann_mape = calculate_errors(self.ann.test, self.ann.predictions)
        _, _, prophet_mape = calculate_errors(
            self.prophet.test["y"], self.prophet.test_predictions
        )
        _, _, rnn_mape = calculate_errors(self.rnn.test_data, self.rnn.predictions)

        df = pd.DataFrame(
            {
                "Model": ["ARIMA", "ANN", "Prophet", "RNN"],
                "MAPE": [arima_mape, ann_mape, prophet_mape, rnn_mape],
                "Train_Duration(Sec)": [
                    round(self.arima.train_duration,2),
                    round(self.ann.train_duration,2),
                    round(self.prophet.train_duration,2),
                    round(self.rnn.train_duration,2),
                ],
                "Predict_Duration(Sec)": [
                    round(self.arima.predict_duration,2),
                    round(self.ann.predict_duration,2),
                    round(self.prophet.predict_duration,2),
                    round(self.rnn.predict_duration,2),
                ],
            }
        )
        df.sort_values(inplace=True, by="MAPE")
        df.reset_index(inplace = True)
        df.drop("index", inplace = True, axis = 1)

        return df

    def get_best_model_name(self) -> str:
        """
        Returns the best model name based on MAPE values.

        Returns:
            BaseModel: The name of the best-performing model.
        """
        df = self.order_models()
        model_name = df.iloc[0]["Model"].lower()
        return model_name

    def get_the_best_model(self) -> BaseModel:
        """
        Returns the best model based on MAPE values.

        Returns:
            BaseModel: The instance of the best-performing model.
        """
        model_name = self.get_best_model_name()
        return getattr(self, model_name)
    
    def get_model_by_order(self, order: int) -> object:
        """
        Retrieves a model by its order from the ordered models.

        Args:
            order (int): The order of the model to retrieve. This should correspond to the index in the ordered models dataframe.

        Returns:
            object: The model instance corresponding to the specified order.
        """
        df_models = self.order_models()
        return getattr(self, df_models.loc[order, 'Model'])

