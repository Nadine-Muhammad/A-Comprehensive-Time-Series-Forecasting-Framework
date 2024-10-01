import warnings
import os

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from utils import utils, plot_utils
from models.prophet_model import ProphetModel
from models.ann_model import ANNModel
from models.rnn_model import RNNModel
from models.arima_model import ARIMAModel
from models.auto_model import AutoModel
from timeseries_data import TimeSeriesObject
import pandas as pd


def run_forecast(data_path: str, model_name: str, end_date: str) -> None:
    """
     Creates a model object based on the chosen model name, prints predictions, and plots history data with predicted data.

     Args:
    - data (str): Path to the CSV file containing time series data.
    - model_name (str): The name of the model to be used ('ann', 'rnn', 'arima', 'prophet', 'auto').
    - end_date (str): The date up to which predictions are to be made.
    """
    data = pd.read_csv(data_path, parse_dates=True)
    time_series_object = TimeSeriesObject(data)

    model_classes = {
        "ann": ANNModel,
        "rnn": RNNModel,
        "arima": ARIMAModel,
        "prophet": ProphetModel,
    }
    if model_name == "auto":

        model = AutoModel(time_series_object, end_date)
        y_predict = model.predict()

    else:
        model_class = model_classes[model_name]
        model = model_class(time_series_object, end_date)
        model.fit()
        y_predict = model.predict()

    dataframe = time_series_object.data
    df = utils.print_predictions(
        dataframe.index[-1], model.freq, model.horizon, y_predict
    )
    # To plot the data and predictions with the model name and dataset name as title
    dataset_name = data_path.split("/")[-1].split(".")[0]
    title = f"{model_name.upper()} Model on {dataset_name} Dataset "
    fig = plot_utils.plot_data_vs_forecast(
        dataframe.index, dataframe.values.flatten(), df.index, df["Value"], title
    )
    fig.show()


def validate_model(model_name: str) -> str:
    """
    Validates that user input is a valid model name.

    Args:
    - model_name (str): User's input model name.

    Returns:
    - model_name (str): The valid names of the models to be used ('ann', 'rnn', 'arima', 'prophet', 'auto').
    """
    valid_models = {"ann", "rnn", "prophet", "arima", "auto"}
    while model_name not in valid_models:
        print("The model name is incorrect.")
        model_name = (
            input(
                "Choose the model to use in forecasting:\n  - ANN \n  - RNN\n  - ARIMA\n  - Prophet\n  - Auto\n"
            )
            .lower()
            .strip()
        )
    return model_name


def main():

    data = input("Enter the path of the data: ").strip()
    model_name = (
        input(
            "Choose the model to use in forecasting:\n  - ANN \n  - RNN\n  - ARIMA\n  - Prophet\n  - Auto\n"
        )
        .lower()
        .strip()
    )
    model_name = validate_model(model_name)
    end_date = input("Enter the end date to forecast upto it: ")

    run_forecast(data, model_name, end_date)


if __name__ == "__main__":
    main()
