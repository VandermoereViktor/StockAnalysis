import pandas_datareader as pdr
import datetime as datetime
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot
from tabulate import tabulate


def get_ticker_data(ticker, start, end):
    df_ticker_full = pdr.DataReader(ticker, data_source='yahoo', start=start, end=end)
    return df_ticker_full.drop(['High', 'Low', 'Open', 'Volume', 'Adj Close'], axis=1).iloc[:, -1:]


def get_prediction_data(file_location):
    df_prediction = pd.read_csv(file_location).iloc[:, -1:]
    df_prediction.columns = ['Prediction']
    return df_prediction


def invest_hold_strategy(df, starting_capital, fee_percentage):
    invest_df = pd.DataFrame(columns=['Capital', 'Stock', 'Sum'])
    invest_df.loc[len(invest_df.index)] = [starting_capital, 0, starting_capital]
    stock_value = starting_capital - starting_capital*fee_percentage
    capital = 0
    for i in range(1, df.shape[0]):
        stock_value = stock_value*(df['Real'][i]/df['Real'][i-1])
        invest_df.loc[len(invest_df.index)] = [capital, stock_value, capital+stock_value]
    return invest_df


def invest_minmax_strategy(df, starting_capital, fee_percentage, minimum_fee):
    invest_df = pd.DataFrame(columns=['Capital', 'Stock', 'Sum'])
    invest_df.loc[len(invest_df.index)] = [starting_capital, 0, starting_capital]
    stock_value = 0
    capital = starting_capital
    for i in range(0, df.shape[0]-1):
        # update stock_value
        if i != 0:
            stock_value = stock_value*(df['Real'][i]/df['Real'][i-1])

        # if we predict a rise in price, put all current capital in stock
        if df['Prediction'][i+1] > df['Prediction'][i]:
            # check if making a move is profitable
            if (df['Prediction'][i+1]/df['Prediction'][i]-1) > fee_percentage and capital*(df['Prediction'][i+1]/df['Prediction'][i]-1) > minimum_fee:
                commission_fee = minimum_fee
                if capital*fee_percentage > commission_fee:
                    commission_fee = capital*fee_percentage
                stock_value = stock_value + capital - commission_fee
                capital = 0
        # remove all stock if we predict a loss
        else:
            # check if making a move is profitable
            if (1-df['Prediction'][i + 1] / df['Prediction'][i]) > fee_percentage and stock_value * (
                    1-df['Prediction'][i + 1] / df['Prediction'][i]) > minimum_fee:
                commission_fee = minimum_fee
                if stock_value*fee_percentage > commission_fee:
                    commission_fee = stock_value*fee_percentage
                capital = capital + stock_value - commission_fee
                stock_value = 0

        invest_df.loc[len(invest_df.index)] = [capital, stock_value, capital+stock_value]
    return invest_df


def plot_capitals(hold_data, minmax_data, title):
    pyplot.plot(hold_data['Sum'].values, color='red', label='Hold strategy')
    pyplot.plot(minmax_data['Sum'].values, color='blue', label='Min-Max strategy')
    pyplot.title(title)
    pyplot.xlabel('Days')
    pyplot.ylabel('Capital ($)')
    pyplot.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    pyplot.legend()
    # create folder if needed
    if not os.path.exists('../Data/Graphs/Evaluation'):
        os.makedirs('../Data/Graphs/Evaluation')
    pyplot.savefig('../Data/Graphs/Evaluation/' + title + ".png", bbox_inches='tight')
    pyplot.show()


def plot_capitals_sentiment(hold_data, minmax_sentiment, minmax_base, title):
    pyplot.plot(hold_data['Sum'].values, color='red', label='Hold strategy')
    pyplot.plot(minmax_sentiment['Sum'].values, color='blue', label='Min-Max strategy (Sentiment)')
    pyplot.plot(minmax_base['Sum'].values, color='green', label='Min-Max strategy (Base)')
    pyplot.title(title)
    pyplot.xlabel('Days')
    pyplot.ylabel('Capital ($)')
    pyplot.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    pyplot.legend()
    # create folder if needed
    if not os.path.exists('../Data/Graphs/Evaluation'):
        os.makedirs('../Data/Graphs/Evaluation')
    pyplot.savefig('../Data/Graphs/Evaluation/' + title + ".png", bbox_inches='tight')
    pyplot.show()


def evaluation(prediction_location, ticker, title, start, end):
    price_data = get_prediction_data(prediction_location)
    real = get_ticker_data(
        ticker=ticker,
        start=start,
        end=end)
    price_data['Real'] = real.values.reshape(real.shape[0]).tolist()
    hold_df = invest_hold_strategy(price_data, starting_capital=10000, fee_percentage=0.02)
    minmax_df = invest_minmax_strategy(price_data, starting_capital=10000, fee_percentage=0.02, minimum_fee=50)
    plot_capitals(hold_data=hold_df, minmax_data=minmax_df, title=title)


def evaluation_sentiment(base_pred_loc, sentiment_pred_loc, ticker, title, start, end):
    sentiment_prediction = get_prediction_data(sentiment_pred_loc)
    base_prediction = get_prediction_data(base_pred_loc)
    real = get_ticker_data(
        ticker=ticker,
        start=start,
        end=end)
    sentiment_prediction['Real'] = real.values.reshape(real.shape[0]).tolist()
    base_prediction['Real'] = real.values.reshape(real.shape[0]).tolist()
    sentiment_hold_df = invest_hold_strategy(sentiment_prediction, starting_capital=10000, fee_percentage=0.02)
    sentiment_minmax_df = invest_minmax_strategy(sentiment_prediction, starting_capital=10000, fee_percentage=0.02, minimum_fee=50)
    base_minmax_df = invest_minmax_strategy(base_prediction, starting_capital=10000, fee_percentage=0.02, minimum_fee=50)
    plot_capitals_sentiment(hold_data=sentiment_hold_df, minmax_sentiment=sentiment_minmax_df, minmax_base=base_minmax_df, title=title)
    print(tabulate(sentiment_minmax_df, headers='keys', tablefmt='psql'))
    print(tabulate(base_minmax_df, headers='keys', tablefmt='psql'))


evaluation(
    prediction_location='../Data/Prediction/TSLA on AMD--wallstreetbets--epochs50--Base (2019-09-01 - 2022-03-31).csv',
    ticker='TSLA',
    title='Evaluation TSLA',
    start=datetime.datetime(2019, 9, 1),
    end=datetime.datetime(2022, 3, 30)
)

#
# evaluation_sentiment(
#     base_pred_loc='../Data/Prediction/TSLA on AMD--wallstreetbets--epochs50--Base (2019-09-01 - 2022-03-31).csv',
#     sentiment_pred_loc='../Data/Prediction/TSLA on AMD--wallstreetbets--epochs50--Vader (2019-09-01 - 2022-03-31).csv',
#     ticker='TSLA',
#     title='Evaluation TSLA Sentiment',
#     start=datetime.datetime(2019, 9, 1),
#     end=datetime.datetime(2022, 3, 30)
# )
#
