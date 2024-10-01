import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt


def plot_raw_data(data: pd.DataFrame) -> None:
    """
    Plot the raw time series data.

    Args:
    - data (pd.DataFrame): DataFrame containing time series data with 'timestamp' and 'value' columns.
    """
    fig = go.Figure(
        [go.Scatter(x=data.index, y=data["value"], mode="lines", name="Raw Data")]
    )
    fig.update_layout(
        title="Raw Time Series Data", xaxis_title="Timestamp", yaxis_title="Data Values"
    )
    fig.show()


def plot_train_test_validation(
    train_data: pd.DataFrame, test_data: pd.DataFrame, validation_data: pd.DataFrame
) -> None:
    """
    Plot training, test, and validation data.

    Args:
    - train_data (pd.DataFrame): Training data with 'timestamp' and 'value' columns.
    - test_data (pd.DataFrame): Test data with 'timestamp' and 'value' columns.
    - validation_data (pd.DataFrame): Validation data with 'timestamp' and 'value' columns.
    """
    fig = go.Figure()

    # Add training data trace
    fig.add_trace(
        go.Scatter(
            x=train_data.index, y=train_data["value"], mode="lines", name="Train Data"
        )
    )

    # Add test data trace
    fig.add_trace(
        go.Scatter(
            x=test_data.index, y=test_data["value"], mode="lines", name="Test Data"
        )
    )

    # Add validation data trace
    fig.add_trace(
        go.Scatter(
            x=validation_data.index,
            y=validation_data["value"],
            mode="lines",
            name="Validation Data",
        )
    )

    # Edit the layout
    fig.update_layout(
        title="Train, Test, and Validation Data",
        xaxis_title="Timestamp",
        yaxis_title="Data Values",
    )
    fig.show()


def plot_actual_vs_predicted(
    original_data: pd.DataFrame,
    actual: np.ndarray,
    predicted: np.ndarray,
    window_size: int,
) -> None:
    """
    Plot actual vs. predicted values.

    Args:
    - actual (pd.Ser): Series of actual values.
    - predicted (pd.Series): Series of predicted values.
    - window_size (int): The size of the window used for the model.
    """
    fig = go.Figure()

    # Plot actual values
    fig.add_trace(
        go.Scatter(
            x=original_data[window_size:].index,
            y=actual.flatten(),
            mode="lines",
            name="Actual",
        )
    )

    # Plot predicted values
    fig.add_trace(
        go.Scatter(
            x=original_data[window_size:].index,
            y=predicted.flatten(),
            mode="lines",
            name="Predicted",
        )
    )

    # Edit the layout
    fig.update_layout(
        title="Actual vs. Predicted Values",
        xaxis_title="Timestamp",
        yaxis_title="Data Values",
    )
    fig.show()


def plot_data_vs_forecast(
    ds_train: pd.Series,
    y_true: pd.Series,
    ds_test: pd.Series,
    y_pred: pd.Series,
    title: str,
) -> None:
    """
    Plot the original data versus the forecasted data using Plotly Express.

    This function takes in the training dates and true values, as well as the test dates and predicted values,
    and creates a line plot to compare the original data with the forecasted data.

    Parameters:
    -----------
    ds_train : pd.Series
        Represents the dates of the training data.
    y_true : pd.Series
        Represents the actual values of the training data.
    ds_test : pd.Series
        Represents the dates of the test data (forecasted period).
    y_pred : pd.Series
        Represents the predicted values for the forecasted period.
    title : str
        Represents the title for the plot

    Returns:
    --------
    None
        The function does not return any value but displays a plot.
    """

    # Combine original and forecasted data into a single DataFrame
    original_df = pd.DataFrame(
        {"Date": ds_train, "Value": y_true, "Type": "Original Data"}
    )
    forecast_df = pd.DataFrame(
        {"Date": ds_test, "Value": np.array(y_pred), "Type": "Forecasted Data"}
    )
    df = pd.concat([original_df, forecast_df])

    fig = px.line(df, x="Date", y="Value", color="Type", title=title)
    return fig


def plot_loss(hist) -> None:
    """
    Plots the training and validation loss over epochs.

    Args:
      - hist (keras.callbacks.History): The history object returned by the `fit` method of a Keras model.
        This object contains the training and validation loss values for each epoch.

    """
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper right")
