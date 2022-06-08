import pandas_datareader as pdr
import datetime as datetime
import pandas as pd
import os
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import tree
from keras.models import Sequential
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from keras.layers import Dense


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


def multi_train(tickers, starts, ends, locations, model):
    i = 0
    for filename in os.listdir('../Data/Sentiment'):
        f = os.path.join('../Data/Sentiment', filename)
        if os.path.isfile(f):
            train(ticker=tickers[i],
                  start=datetime.datetime(starts[i][0], starts[i][1], starts[i][2]),
                  end=datetime.datetime(ends[i][0], ends[i][1], ends[i][2]) - datetime.timedelta(days=1),
                  data_location=locations[i],
                  model_name=model
                  )
            i += 1


def get_model(model_name, X, Y):
    if model_name == "nn":
        # design network
        model = Sequential()
        model.add(Dense(8, input_dim=X.shape[1], activation='relu'))
        model.add(Dense(4, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        model.fit(x=X, y=Y, epochs=200, validation_split=0.2, verbose=1, shuffle=False)
        return model
    elif model_name == "dt":
        model = RandomForestClassifier()
        model.fit(X, Y)
        return model
    elif model_name == "nb":
        model = GaussianNB()
        model.fit(X, Y)
        return model


def train(ticker, start, end, data_location, model_name):
    df_ticker = get_ticker_data(ticker, start, end)

    df_features = get_sentiment_data(data_location, start, end)
    # concatenate ticker and sentiment dataframes
    df_feature_vector = df_features.join(df_ticker)
    df_feature_vector = df_feature_vector.dropna()

    # change close column to binary: rise (1) / lower (0)
    for i in range(1, df_feature_vector.shape[0]):
        if df_feature_vector['Close'][i - 1] < df_feature_vector['Close'][i]:
            df_feature_vector['Close'][i - 1] = 1
        else:
            df_feature_vector['Close'][i - 1] = 0

    df_feature_vector['Close'] = df_feature_vector['Close'].astype(bool)
    df_feature_vector.drop(df_feature_vector.tail(1).index, inplace=True)
    df_feature_vector = df_feature_vector.dropna()

    # scale values
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    data = min_max_scaler.fit_transform(df_feature_vector.values)

    # split in training and testing dataframes
    length = data.shape[0]
    training = data[0:int(length * 0.8)]
    validation = data[int(length * 0.8):]

    X_val = validation[:, :-1]
    Y_val = validation[:, -1]

    X = training[:, :-1]
    Y = training[:, -1]

    model = get_model(model_name, X, Y)

    prediction = model.predict(X_val)

    my_accuracy = accuracy_score(Y_val, prediction.round())
    # print("Prediction:", prediction[:20])
    # print("Real:", Y_val[:20])
    print(ticker + ": ", my_accuracy)
    print("Baseline: ", Y.sum() / len(Y))


ticker_array = ['AAPL', 'AMC', 'AMD', 'AMZN', 'BTC-USD', 'GME', 'TSLA']
start_array = [
    [2016, 1, 1],
    [2020, 6, 1],
    [2020, 1, 1],
    [2018, 1, 1],
    [2017, 1, 1],
    [2019, 6, 1],
    [2019, 9, 1],
]
end_array = [
    [2022, 1, 1],
    [2022, 4, 1],
    [2022, 1, 1],
    [2022, 1, 1],
    [2022, 1, 1],
    [2022, 1, 1],
    [2022, 4, 1],
]
location_array = [
    "../Data/Sentiment/WSB--AAPL--VADER--100--1.2016-1.2022.csv",
    "../Data/Sentiment/WSB--AMC--VADER--100--6.2020-4.2022.csv",
    "../Data/Sentiment/WSB--AMD--VADER--200--1.2020-1.2022.csv",
    "../Data/Sentiment/WSB--AMZN--VADER--100--1.2018-1.2022.csv",
    "../Data/Sentiment/WSB--BTC--VADER--40--1.2017-1.2022.csv",
    "../Data/Sentiment/WSB--GME--VADER--100--6.2019-1.2022.csv",
    "../Data/Sentiment/WSB--TSLA--VADER--100--9.2019-4.2022.csv",
]
multi_train(ticker_array, start_array, end_array, location_array, "dt")
