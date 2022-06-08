import pandas as pd
import os


def sentiment_statistics(file):
    stat_dict = {}
    df = pd.read_csv(f'../Data/Sentiment/{file}.csv')
    stat_dict['df_size'] = df.shape[0]

    stat_dict['first_date'] = df['date'].iloc[0]
    stat_dict['last_date'] = df['date'].iloc[-1]
    means = df.mean()
    for key, value in means.items():
        stat_dict[key+' (mean)'] = value
    variances = df.std()
    for key, value in variances.items():
        stat_dict[key+' (standard dev)'] = value
    mins = df.min()
    for key, value in mins.items():
        stat_dict[key+' (minimum)'] = value
    maxs = df.max()
    for key, value in maxs.items():
        stat_dict[key+' (maximum)'] = value

    with open(f'../Data/SentimentStatistics/{file}(Statistics).csv', 'w') as f_open:
        for key in stat_dict.keys():
            f_open.write("%s,%s\n" % (key, stat_dict[key]))


for filename in os.listdir('../Data/Sentiment'):
    f = os.path.join('../Data/Sentiment', filename)
    if os.path.isfile(f):
        sentiment_statistics(filename[:len(filename) - 4])
