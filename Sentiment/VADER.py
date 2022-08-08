from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from tabulate import tabulate
import os


def sentiment_analysis(text_string):
    if text_string == '[removed]' or text_string == 'neutral' or text_string == '[deleted]':
        return 0
    return analyzer.polarity_scores(text_string)['compound']


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
    df.to_csv("../Data/Sentiment/" + filename + ".csv", sep=',', encoding='utf-8')
    print(tabulate(df, headers='keys', tablefmt='psql'))


new_words = {
    'citron': -4.0,
    'coronavirus': -2.0,
    'hidenburg': -4.0,
    'moon': 4.0,
    'highs': 2.0,
    'mooning': 4.0,
    'long': 2.0,
    'short': -2.0,
    'call': 4.0,
    'calls': 4.0,
    'put': -4.0,
    'puts': -4.0,
    'break': 2.0,
    'tendie': 2.0,
    'tendies': 2.0,
    'town': 2.0,
    'overvalued': -3.0,
    'undervalued': 3.0,
    'buy': 4.0,
    'sell': -4.0,
    'bullish': 3.7,
    'bearish': -3.7,
    'bagholder': -1.7,
    'stonk': 1.9,
    'stonks': 2.5,
    'green': 2.5,
    'red': -2.2,
    'print': 2.2,
    'rocket': 2.2,
    'bull': 4.0,
    'bear': -4.0,
    'pumping': -1.0,
    'turmoil': -2.0,
    'volatility': -1.0,
    'volatile': -1.0,
    'sus': -3.0,
    'offering': -2.3,
    'rip': -4.0,
    'downgrade': -3.0,
    'upgrade': 3.0,
    'pump': -1.0,
    'hot': 1.5,
    'drop': -2.5,
    'rebound': 1.5,
    'crack': 2.5,
    'gang': 2.0,
    'scam': -2.0,
    'squeeze': 3.0,
    'bag': -4.0,
    'fly': 2.0,
    'way': 2.0,
    'high': 2.0,
    'volume': 2.5,
    'low': -2.0,
    'trending': 3.0,
    'upwards': 3.0,
    'prediction': 1.0,
    'big': 2.0,
    '🚀': 5.0,
    'bitches': -0.1,
    'fags': -0.1,
    'faggots': -0.1,
    'retards': 1.2,
    'retard': 0.8,
    'cliff': -4.0,
    'millionaire': 1.5,
    'roof': 3.5,
    'loss': -1.5,
    'losses': -2.5,
    'gains': 3.5,
    'gain': 2.5,
    'winning': 3.0,
    'winnings': 2.2,
    'hit': -3.0,
    'potential': 3.0,
    'bubble': -2.5,
    'crash': -4.0,
    'shitty': 0
}

analyzer = SentimentIntensityAnalyzer()
analyzer.lexicon.update(new_words)
calculate_sentiment(file="../Data/Raw/All/wallstreetbets_100_2.2022-7.2022.csv",
                    filename="GSPC-2.2022-7.2022(WSB)",
                    )
