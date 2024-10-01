A framework for time series forecasting that supports multiple models, which are: ARIMA, ANN, Prophet, RNN, and AUTOModel.

AUTOARIMA: Automatically selects the best parameters for ARIMA models to forecast time series data by combining autoregression, differencing, and moving average. Ideal for non-stationary data with trends.​

PROPHET: Developed by Facebook, this model is tailored for time series data with strong seasonality and outliers. It decomposes data into trend, seasonality, and holiday effects, making it user-friendly and customizable.​

ANN (Artificial Neural Networks): Machine learning models inspired by the human brain, capable of learning complex patterns. Useful for modeling non-linear relationships in time series but requires large data and careful tuning.​

RNN (Recurrent Neural Networks): Designed for sequential data, RNNs maintain a memory of previous inputs, making them suitable for time series forecasting. Variants like LSTM and GRU are used to capture long-term dependencies.

AUTOModel: evaluates and ranks the four time series models based on their Mean Absolute Percentage Error (MAPE) and identifies the most accurate model for different time series datasets, enhancing forecasting reliability.
