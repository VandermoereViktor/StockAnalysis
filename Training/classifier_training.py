import pandas_datareader as pdr
import datetime as datetime
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_recall_curve, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from sklearn.naive_bayes import GaussianNB
from keras.layers import Dense
from matplotlib import pyplot

model_dict = {
    "nn": "Neural Network",
    "dt": "Decision Tree",
    "nb": "Naive Bayes"
}


def create_prediction_graph(real, predicted, modeltype, ticker):
    title = modeltype + " correctness for " + ticker

    pyplot.figure(figsize=(15, 2), dpi=200)
    # used to get 'Predict Up' always on top
    initPoints, = pyplot.plot([0, 1], ['Predict Down', 'Predict Up'], marker='o', color='k')

    for i in range(len(real)):
        y = 'Predict Up' if predicted[i] > 0 else 'Predict Down'
        if real[i] == predicted[i]:
            pyplot.plot(i, y, 'go')
        else:
            pyplot.plot(i, y, 'rx')
    pyplot.title(title)
    pyplot.tight_layout()
    # create folder if needed
    if not os.path.exists("../Data/Graphs/Classifier/Accuracy"):
        os.makedirs("../Data/Graphs/Classifier/Accuracy")
    initPoints.remove()
    pyplot.savefig("../Data/Graphs/Classifier/Accuracy" + "/" + title + ".png")
    pyplot.figure().clear()


def create_bar_graph(tickers, accuracy, base, modeltype):

    title = modeltype + " accuracy"

    x = np.arange(len(tickers))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = pyplot.subplots()
    rects1 = ax.bar(x - width / 2, accuracy, width, label='Accuracy')
    rects2 = ax.bar(x + width / 2, base, width, label='BaseLine')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_xticks(x, tickers)
    ax.bar_label(rects1, padding=3, rotation='vertical')
    ax.bar_label(rects2, padding=3, rotation='vertical')
    fig.legend(loc="upper right")
    pyplot.ylim(0, 0.7)
    fig.tight_layout()
    # create folder if needed
    if not os.path.exists("../Data/Graphs/Classifier"):
        os.makedirs("../Data/Graphs/Classifier")
    pyplot.savefig("../Data/Graphs/Classifier" + "/" + title + ".png")


def create_pr_graph(recall, precision, modeltype, ticker):
    # create precision recall curve
    title = modeltype + f" Precision-Recall Curve ({ticker})"
    fig, ax = pyplot.subplots()
    ax.plot(recall, precision, color='red')
    # add axis labels to plot
    ax.set_title(title)
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')

    # create folder if needed
    if not os.path.exists("../Data/Graphs/Classifier/PR"):
        os.makedirs("../Data/Graphs/Classifier/PR")
    pyplot.savefig("../Data/Graphs/Classifier/PR" + "/" + title + ".png")


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


def multi_train(tickers, starts, ends, locations, model, bar_graph=False, prediction_graph=False, precall_graph=False):
    i = 0
    acc_vector = []
    base_vector = []
    fixed_vector = []

    for filename in os.listdir('../Data/Sentiment'):
        if i == len(tickers):
            break
        f = os.path.join('../Data/Sentiment', filename)
        if os.path.isfile(f):
            acc, base, fixed = train(ticker=tickers[i],
                                     start=datetime.datetime(starts[i][0], starts[i][1], starts[i][2]),
                                     end=datetime.datetime(ends[i][0], ends[i][1], ends[i][2]) - datetime.timedelta(
                                         days=1),
                                     data_location=locations[i],
                                     model_name=model,
                                     precall=precall_graph,
                                     prediction_graph=prediction_graph
                                     )
            i += 1
            acc_vector.append((round(acc, 3)))
            base_vector.append((round(base, 3)))
            fixed_vector.append((round(fixed, 3)))

    if bar_graph:
        create_bar_graph(tickers=tickers,
                         accuracy=acc_vector,
                         base=base_vector,
                         modeltype=model_dict[model]
                         )


def get_model(model_name, X, Y):
    if model_name == "nn":
        # design network
        model = Sequential()
        model.add(Dense(8, input_dim=X.shape[1], activation='relu'))
        model.add(Dense(4, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        model.fit(x=X, y=Y, epochs=200, validation_split=0.2, verbose=0, shuffle=False)
        return model
    elif model_name == "dt":
        model = RandomForestClassifier()
        model.fit(X, Y)
        return model
    elif model_name == "nb":
        model = GaussianNB()
        model.fit(X, Y)
        return model


def train(ticker, start, end, data_location, model_name, precall, prediction_graph):
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
    if ticker=="GME":
        print("A")
    if model_name == 'nn':
        prediction = model.predict(X_val)
    else:
        # probabilities for positive (1) used in prec-recall curve
        prediction = model.predict_proba(X_val)
        prediction = prediction[:, [1]].reshape([prediction.shape[0]])

    if prediction_graph:
        create_prediction_graph(real=Y_val, predicted=prediction.round(), modeltype=model_dict[model_name], ticker=ticker)
    my_accuracy = accuracy_score(Y_val, prediction.round())
    print('-' * 50)
    print(ticker + ": ", my_accuracy)
    base = Y.sum() / len(Y)
    base = max(base, 1 - base)
    print("Baseline: ", base)
    fixed = my_accuracy / base
    print("Fixed: ", fixed)
    print('-' * 50)

    if precall:
        Y_val = Y_val.astype(int)
        prediction = prediction.reshape(prediction.shape[0])
        tp, fn, fp, tn = confusion_matrix(Y_val, prediction.round(), labels=(1, 0)).ravel()
        print("True positive: ", tp, "| True negative", tn)
        print("False positive: ", fp, "| False negative", fn)
        prediction = prediction.astype('float64')
        precision, recall, thresholds = precision_recall_curve(y_true=Y_val, probas_pred=prediction)
        create_pr_graph(recall=recall, precision=precision, modeltype=model_dict[model_name], ticker=ticker)

        return my_accuracy, base, fixed


ticker_array = ['AAPL', 'AMC', 'AMD', 'AMZN', 'GME', '^GSPC', 'SPY', 'TSLA']
start_array = [
    [2016, 1, 1],
    [2020, 6, 1],
    [2020, 1, 1],
    [2018, 1, 1],
    [2016, 1, 1],
    [2019, 1, 1],
    [2020, 1, 1],
    [2019, 9, 1],
]
end_array = [
    [2022, 1, 1],
    [2022, 4, 1],
    [2022, 1, 1],
    [2022, 1, 1],
    [2022, 1, 1],
    [2022, 1, 1],
    [2022, 1, 1],
    [2022, 4, 1],
]
location_array = [
    "../Data/Sentiment/AAPL-1.2016-1.2022(WSB).csv",
    "../Data/Sentiment/AMC-6.2020-4.2022(WSB).csv",
    "../Data/Sentiment/AMD-1.2020-1.2022(WSB).csv",
    "../Data/Sentiment/AMZN-1.2018-1.2022(WSB).csv",
    "../Data/Sentiment/GME-1.2016-1.2022(WSB).csv",
    "../Data/Sentiment/GSPC-1.2019-1.2022(WSB).csv",
    "../Data/Sentiment/SPY-1.2020-1.2022(WSB).csv",
    "../Data/Sentiment/TSLA-9.2019-4.2022(WSB).csv"
]

# available models:
# Neural network (code: nn)
# Naive Bayes (code: nb)
# Decision Tree (code: dt)
multi_train(ticker_array, start_array, end_array, location_array, "dt",
            bar_graph=False,
            prediction_graph=False,
            precall_graph=True)
