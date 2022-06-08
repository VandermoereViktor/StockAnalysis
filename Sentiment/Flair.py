from flair.data import Sentence
from flair.models import TextClassifier
import pandas as pd
from tabulate import tabulate
import os
import string


def clear_punctuation(s):
    clear_string = ""
    for symbol in s:
        if symbol not in string.punctuation:
            clear_string += symbol
    return clear_string


def sentiment_analysis(dirty_string):
    if dirty_string == '[removed]' or dirty_string == 'neutral':
        return 0
    clear_string = clear_punctuation(dirty_string)
    sentence = Sentence(clear_string)
    analyzer.predict(sentence)
    score = sentence.labels[0].score
    if sentence.labels[0].value == 'NEGATIVE':
        return score * -1.0
    else:
        return score


def calculate_sentiment(file, filename):
    df = pd.read_csv(file)
    # add new columns for sentiment values
    df['title_sentiment'] = df['title']
    df['selftext_sentiment'] = df['selftext']
    # remove unnamed columns
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    # drop not used rows
    df = df.drop(['author', 'created', 'link_flair_text', 'created_utc', 'title', 'selftext', 'upvote_ratio'], axis=1)
    # fix certain columns
    df['total_awards_received'] = df['total_awards_received'].fillna(0)
    df['selftext_sentiment'] = df['selftext_sentiment'].fillna("neutral")
    # drop rows with missing values
    df = df.dropna()
    # calculate sentiment
    df['title_sentiment'] = df['title_sentiment'].apply(sentiment_analysis)
    df['selftext_sentiment'] = df['selftext_sentiment'].apply(sentiment_analysis)
    # group features
    df = df.groupby(by='date', sort=False, as_index=True).agg(Total_Comments=('num_comments', 'sum'),
                                                              Total_Awards=('total_awards_received', 'sum'),
                                                              Total_Score=('score', 'sum'),
                                                              Title_Sentiment=('title_sentiment', 'mean'),
                                                              Text_Sentiment=('selftext_sentiment', 'mean'))
    # create folder if needed
    if not os.path.exists("../Data/Sentiment/"):
        os.makedirs("../Data/Sentiment/")
    df.to_csv("../Data/Sentiment/" + filename, sep=',', encoding='utf-8')
    print(tabulate(df, headers='keys', tablefmt='psql'))


analyzer = TextClassifier.load('en-sentiment')

calculate_sentiment(file="../Data/Raw/Apple/wallstreetbets_100_1.2016-1.2022.csv",
                    filename="WSB--AAPL--FLAIR--100--1.2016-1.2022.csv")

