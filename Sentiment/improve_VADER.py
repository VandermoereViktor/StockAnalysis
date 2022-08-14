from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from tabulate import tabulate


def create_subset(name, data, size):
    df = pd.read_csv(data)
    # drop rows with missing data
    df = df.dropna()
    # convert Dtype to string
    df['title'] = df['title'].astype("string")
    df['selftext'] = df['selftext'].astype("string")
    # remove submissions smaller than 100 characters
    df['length'] = df['selftext'].apply(lambda x: len(x))
    cond = df['length'] > 100
    df = df[cond]
    # only used columns remain
    df = df[["title"]]
    # take a random sample of 100 elements
    df = df.sample(size)
    print(tabulate(df, headers='keys', tablefmt='psql'))
    # save to csv
    df.to_csv("../Data/Raw/SubsetForVader/" + name + ".csv", sep=',', encoding='utf-8', index=False)


def read_subset():
    df = pd.read_csv("../Data/Raw/SubsetForVader/subset_valued.csv")
    print(tabulate(df, headers='keys', tablefmt='psql'))


def sentiment_analysis(text_string):
    return analyzer.polarity_scores(text_string)['compound']


def update_vader_lexicon():
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
    analyzer.lexicon.update(new_words)


def apply_vader(name, upperBounds=0.0, update=False, ):
    if update:
        update_vader_lexicon()
    df = pd.read_csv(f"../Data/Raw/SubsetForVader/{name}.csv")
    # add new column for sentiment values
    df['calculated_sentiment'] = df['title']
    # remove unnamed columns

    # calculate sentiment
    df['calculated_sentiment'] = df['calculated_sentiment'].apply(sentiment_analysis)
    # df['selftext_sentiment'] = df['selftext_sentiment'].apply(sentiment_analysis)
    corr = df.corr(method='pearson')
    # shorten string to display nice in table
    df = df.astype(str).apply(lambda x: x.str[:150])
    # only show rows with high disparity between measured sentiment and valued sentiment
    df = df.astype({"title": 'string', "valued_sentiment": 'float64', "calculated_sentiment": 'float64'})
    df = df[abs(df.valued_sentiment - df.calculated_sentiment) >= upperBounds]
    print(tabulate(df, headers='keys', tablefmt='psql'))
    print("CORR: ", name+' ', corr["valued_sentiment"][1])


analyzer = SentimentIntensityAnalyzer()

# create_subset(name="subset_gme",
#               data="../Data/Raw/GME/wallstreetbets_100_6.2019-1.2022.csv",
#               size=50)
print("PRE-UPDATED SENTIMENT:")
print("=="*80)
apply_vader(name="subset_valued", upperBounds=2, update=False)
apply_vader(name="subset_spy_valued", upperBounds=2, update=False)
apply_vader(name="subset_gme_valued", upperBounds=2, update=False)
print("=="*80)
print("POST-UPDATED SENTIMENT:")
apply_vader(name="subset_valued", upperBounds=1, update=True)
apply_vader(name="subset_spy_valued", upperBounds=1, update=True)
apply_vader(name="subset_gme_valued", upperBounds=1, update=True)
print("=="*80)
