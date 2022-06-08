import pandas_datareader as pdr
import datetime as datetime
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    # drop rows with prev
    to_drop = [6, 7, 8, 9, 10]
    # for x in range(data.shape[1], agg.shape[1]):
    #     if (x) % n_vars == 0:
    #         to_drop.append(x-1)
    agg.drop(agg.columns[to_drop], axis=1, inplace=True)
    return agg


def get_ticker_data(ticker_symbol, start_datetime, end_datetime):
    df_ticker_full = pdr.DataReader(ticker_symbol, data_source='yahoo', start=start_datetime, end=end_datetime)
    return df_ticker_full.drop(['High', 'Low', 'Open', 'Volume', 'Adj Close'], axis=1)


def get_sentiment_data(file_location, start, end):
    df_sentiment_full = pd.read_csv(file_location)
    # convert date to proper datetime string
    df_sentiment_full['date'] = df_sentiment_full['date'].map(lambda a: datetime.datetime.strptime(a, '%Y-%m-%d'))
    # add missing days
    df_sentiment_full = df_sentiment_full.set_index('date')
    new_index = pd.date_range(start.strftime("%Y/%m/%d"), end.strftime("%Y/%m/%d"))
    df_sentiment_full = df_sentiment_full.reindex(new_index)
    # fill now empty rows with column averages
    column_means = df_sentiment_full.mean()
    return df_sentiment_full.fillna(column_means)


def plot_prediction(real_price, predicted_price, image_destination, image_name, title):
    pyplot.plot(real_price, color='red', label='Real Price')
    pyplot.plot(predicted_price, color='blue', label='Predicted Price')
    pyplot.title(title)
    pyplot.xlabel('Days')
    pyplot.ylabel('Price')
    pyplot.legend()
    # create folder if needed
    if not os.path.exists(image_destination):
        os.makedirs(image_destination)
    pyplot.savefig(image_destination + "/" + image_name + ".png")
    pyplot.show()


def calc_mse(array1, array2):
    # size
    mse_sum = 0
    n = array1.shape[0]
    for i in range(0, n):
        mse_sum = mse_sum + (array1[i] - array2[i]) * (array1[i] - array2[i])
    return mse_sum / n


def predict(ticker, model_name, model_abbreviation, title, folder_name, start, end, data_location, nr_observations,
            plot_prediction_bool, sentiment):
    df_ticker = get_ticker_data(ticker, start, end)

    # concatenate ticker and sentiment dataframes
    if sentiment:
        df_features = get_sentiment_data(data_location, start, end)
        # concatenate ticker and sentiment dataframes
        df_feature_vector = df_features.join(df_ticker)
        df_feature_vector = df_feature_vector.dropna()
    else:
        df_feature_vector = df_ticker

    # scale values
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = min_max_scaler.fit_transform(df_feature_vector.values)

    df_time_series_vector = series_to_supervised(scaled_features, 1, 1, True)
    df_time_series_vector.info()

    features = df_time_series_vector.values
    features_X, features_Y = features[:, :-1], features[:, -1]
    features_X = features_X.reshape(features_X.shape[0], 1, features_X.shape[1])

    timesteps_X = []
    timesteps_Y = features_Y[10:]

    for x in range(nr_observations, len(features_X)):
        timesteps_X.append(features_X[x - nr_observations:x, 0])

    timesteps_X = np.array(timesteps_X)

    # load model
    model = keras.models.load_model(f"../Data/Models/{model_name}")

    # make a prediction
    result = model.predict(features_X)
    result = result.reshape((result.shape[0], result.shape[2]))
    mse = calc_mse(result.reshape(result.shape[0]), df_time_series_vector.iloc[:, -1].values)
    # recreate ndarray shape before scaling was applied
    z = np.zeros((result.shape[0], 5))
    prediction = np.append(z, result, axis=1)
    prediction = min_max_scaler.inverse_transform(prediction)
    # only need last column
    prediction = prediction[:, -1]
    # recreate values
    real = df_time_series_vector.values[:, :-1]
    real = min_max_scaler.inverse_transform(real)
    real = real[:, -1]
    pred_to_save = pd.DataFrame(prediction)
    mse_string=format(mse, ".4f")
    mse_string = mse_string.replace('.', ',')
    sent = ' no Sentiment'
    if sentiment:
        sent = 'Sentiment'
    pred_to_save.to_csv(
        f'../Data/Prediction/{ticker} on {model_abbreviation} ({start.strftime("%Y-%m-%d")} - {end.strftime("%Y-%m-%d")}) {mse_string} mse - {sent}.csv',
        sep=',', encoding='utf-8')
    if plot_prediction_bool:
        # plot prediction
        destination = f'../Data/Graphs/{folder_name}/'
        name = title
        sent = 'without sentiment'
        if sentiment:
            sent = 'with sentiment'
        title = title + f' (mse: {format(mse, ".4f")})'
        plot_prediction(real, prediction, destination, name, title)


predict(ticker="TSLA",
        model_name="AAPL--wallstreetbets--epochs100--Vader",
        model_abbreviation="AAPL",
        title="Prediction of TSLA on AAPL trained model with sentiment",
        folder_name='WSB-TSLA-ON-WSB-AAPL',
        start=datetime.datetime(2019, 9, 1),
        end=datetime.datetime(2022, 4, 1) - datetime.timedelta(days=1),
        data_location="../Data/Sentiment/WSB--TSLA--VADER--100--9.2019-4.2022.csv",
        nr_observations=10,
        plot_prediction_bool=True,
        sentiment=True
        )
