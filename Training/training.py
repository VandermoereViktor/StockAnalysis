import pandas_datareader as pdr
import datetime as datetime
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot
# import tensorflow
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


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


def plot_features(dataframe, image_destination, image_name, title):
    features = list(range(0, dataframe.shape[1]))
    i = 1
    pyplot.figure()
    pyplot.suptitle(title)
    for feature in features:
        pyplot.subplot(len(features), 1, i)
        pyplot.plot(dataframe.values[:, feature])
        pyplot.title(dataframe.columns[feature], y=0.5, loc='right')
        i += 1
    # create folder if needed
    if not os.path.exists(image_destination):
        os.makedirs(image_destination)
    pyplot.savefig(image_destination + "/" + image_name + ".png")
    pyplot.show()


def plot_loss(history, image_destination, image_name, title):
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.title(title)
    if not os.path.exists(image_destination):
        os.makedirs(image_destination)
    pyplot.savefig(image_destination + "/" + image_name + ".png")
    pyplot.show()


def plot_test_prediction(real_price, predicted_price, before_price, image_destination, image_name, title):
    before_size = before_price.shape[0]
    after_size = real_price.shape[0]
    before_x = np.arange(0, before_size)
    after_x = np.arange(before_size+1, before_size+after_size+1)
    if before_price is not None:
        pyplot.plot(before_x, before_price, color='green', label='Training')
    pyplot.plot(after_x, real_price, color='red', label='Real Price')
    pyplot.plot(after_x, predicted_price, color='blue', label='Predicted Price')
    pyplot.title(title)
    pyplot.xlabel('Days')
    pyplot.ylabel('Price')
    pyplot.legend()
    # create folder if needed
    if not os.path.exists(image_destination):
        os.makedirs(image_destination)
    pyplot.savefig(image_destination + "/" + image_name + ".png")
    pyplot.show()


def train(ticker, subreddit, folder_name, start, end, data_location, nr_observations, epochs,
          plot_features_bool, plot_loss_bool, plot_prediction_bool, sentiment):
    df_ticker = get_ticker_data(ticker, start, end)
    if sentiment:
        df_features = get_sentiment_data(data_location, start, end)
        # concatenate ticker and sentiment dataframes
        df_feature_vector = df_features.join(df_ticker)
        df_feature_vector = df_feature_vector.dropna()
    else:
        df_feature_vector = df_ticker

    if plot_features_bool:
        # plot features
        destination = f'../Data/Graphs/{folder_name}/'
        name = f'Features {ticker}-{subreddit}({sentiment})'
        title = f'{subreddit} {ticker} {sentiment} {start.strftime("%m/%d/%Y")} - {end.strftime("%m/%d/%Y")}'
        plot_features(df_feature_vector, destination, name, title)

    # scale values
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = min_max_scaler.fit_transform(df_feature_vector.values)

    df_time_series_vector = series_to_supervised(scaled_features, 1, 1, True)

    df_time_series_vector = df_time_series_vector.values
    series_X, series_Y = df_time_series_vector[:, :-1], df_time_series_vector[:, -1]
    series_X = series_X.reshape(series_X.shape[0], 1, series_X.shape[1])

    timesteps_vals_X = []
    timesteps_vals_Y = series_Y[nr_observations-1:]

    for x in range(nr_observations, len(series_X) + 1):
        timesteps_vals_X.append(series_X[x - nr_observations:x, 0])

    vals_X = np.array(timesteps_vals_X)

    # split in training and testing dataframes
    length = vals_X.shape[0]
    train_X = vals_X[0:int(length * 0.8)]
    train_Y = timesteps_vals_Y[0:int(length * 0.8)]
    test_X = vals_X[int(length * 0.8):]
    test_Y = timesteps_vals_Y[int(length * 0.8):]

    # design network
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(1))

    model.compile(loss='mae', optimizer='adam')

    fit = model.fit(train_X, train_Y, epochs=epochs, batch_size=72,
                    validation_data=(test_X, test_Y), verbose=2,
                    shuffle=False)
    if plot_loss_bool:
        # plot loss
        destination = f'../Data/Graphs/{folder_name}/'
        name = f'Loss-Epochs{epochs}-Sentiment({sentiment})-Observations({nr_observations})'
        title = f'Loss {subreddit}: {start.strftime("%m/%d/%Y")} - {end.strftime("%m/%d/%Y")}  obs-{nr_observations}-epochs:{epochs}'
        plot_loss(fit, destination, name, title)

    # save model
    if sentiment:
        model.save(f"../Data/Models/{ticker}--{subreddit}--epochs{epochs}--{sentiment}--observations{nr_observations}")
    else:
        model.save(f"../Data/Models/{ticker}--{subreddit}--epochs{epochs}--Base--observations{nr_observations}")
    # make a prediction
    result = model.predict(test_X)

    if len(result.shape) > 2:
        result = result.reshape((result.shape[0], result.shape[2]))
    # recreate ndarray shape before scaling was applied
    z = np.zeros((result.shape[0], 5))
    prediction = np.append(z, result, axis=1)
    prediction = min_max_scaler.inverse_transform(prediction)
    # only need last column
    prediction = prediction[:, -1]
    # recreate values
    real = np.append(z, test_Y.reshape(test_Y.shape[0], 1), axis=1)
    real = min_max_scaler.inverse_transform(real)
    real = real[:, -1]

    z = np.zeros((train_Y.shape[0], 5))
    before = np.append(z, train_Y.reshape(train_Y.shape[0], 1), axis=1)
    before = min_max_scaler.inverse_transform(before)
    before = before[:, -1]
    if plot_prediction_bool:
        # plot test prediction
        destination = f'../Data/Graphs/{folder_name}/'
        name = f'TestPrediction-Epochs{epochs}-Sentiment({sentiment})-obs{nr_observations}'
        title = ticker+f' testing-data fit {sentiment} ({epochs} Epochs, {nr_observations} timesteps)'
        plot_test_prediction(real, prediction, before, destination, name, title)


train(ticker="^GSPC",
      subreddit="wallstreetbets",
      folder_name='WSB-SPX-VADER-TRAINING',
      start=datetime.datetime(2019, 1, 1),
      end=datetime.datetime(2022, 1, 1) - datetime.timedelta(days=1),
      data_location="../Data/Sentiment/GSPC-1.2019-1.2022(WSB).csv",
      nr_observations=20,
      epochs=100,
      plot_features_bool=False,
      plot_loss_bool=True,
      plot_prediction_bool=True,
      sentiment='Vader',
      )
