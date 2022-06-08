import pandas as pd
import datetime
import time
import os
from psaw import PushshiftAPI
from tabulate import tabulate
"""
Options:
  -q, --query TEXT            search term(s)
  -s, --subreddits TEXT       restrict search to subreddit(s)
  -a, --authors TEXT          restrict search to author(s)
  -l, --limit INTEGER         maximum number of items to retrieve
  --before TEXT               restrict to results before date (datetime or int + s,m,h,d; eg, 30d
                              for 30 days)
  --after TEXT                restrict to results after date (datetime or int + s,m,h,d; eg, 30d
                              for 30 days)
  -o, --output PATH           output file for saving all results in a single file
  --output-template TEXT      output file name template for saving each result in a separate file
                              template can include output directory and fields from each result
                              note, if using --filter, any fields in output-template MUST be
                              included in filtered fields

                              example: 'output_path/{author}.{id}.csv'
                              'output_path/{subreddit}_{created_utc}.json'
  --format [json|csv]
  -f, --filter TEXT           filter fields to retrieve (must be in quotes or have no spaces),
                              defaults to all
  --prettify                  make output slightly less ugly (for json only)
  --dry-run                   print potential names of output files, but don't actually write any
                              files
  --no-output-template-check
  --proxy TEXT
  --verbose
  --help                      Show this message and exit.
"""


def convert_date_to_unix(year, month, day):
    dt = datetime.date(year, month, day)
    return int(time.mktime(dt.timetuple()))


def convert_unix_to_date(unix):
    return datetime.datetime.fromtimestamp(unix).date()


def collect_data(subreddit_name, filter_word, filter_arguments, after_unix, before_unix, limit, type):
    api = PushshiftAPI()
    gen = None
    if type == "Submissions":
        gen = api.search_submissions(
            q=filter_word,
            subreddit=subreddit_name,
            filter=filter_arguments,
            after=after_unix,
            before=before_unix,
            limit=limit,
            sort_type='score',
            sort='desc',
        )
    elif type == "Comments":
        gen = api.search_comments(
            q=filter_word,
            subreddit=subreddit_name,
            filter=filter_arguments,
            after=after_unix,
            before=before_unix,
            limit=limit,
            sort_type='score',
            sort='desc',
        )

    df = pd.DataFrame([thing.d_ for thing in gen])
    df['date'] = convert_unix_to_date(after_unix)
    return df


def mass_data_collection(subreddit_name, filter_word, filter_arguments, start_year, start_month, end_year, end_month,
                         location, limit, type):
    # create folder if needed
    if not os.path.exists(location):
        os.makedirs(location)
    # assign length of timespan to look for, we go through this day by day
    after = convert_date_to_unix(start_year, start_month, 1)
    start = after
    until = convert_date_to_unix(end_year, end_month, 1)
    # create empty dataframe with argument columns and date column
    columns = filter_arguments.copy()
    columns.append('date')
    df = pd.DataFrame(columns=filter_arguments)
    # go over every day until end date is reached
    while after <= until:
        before = after + (24 * 60 * 60)  # 1day
        new_df = collect_data(subreddit_name, filter_word, filter_arguments, after, before, limit, type)
        df = df.append(new_df)
        after = before
        print(str((after - start) * 100 / (until - start)) + "%")  # print out percentage done
    # safe data to csv file
    df.to_csv(location + "/" + subreddit_name + "_" + str(limit) + "_" +
              str(start_month) + "." + str(start_year) + "-" +
              str(end_month) + "." + str(end_year)
              + ".csv", sep=',', encoding='utf-8')


arguments = ['author', 'created', 'link_flair_text', 'num_comments',
             'title', 'total_awards_received', 'upvote_ratio', 'score', 'selftext']
mass_data_collection(subreddit_name="wallstreetbets",
                     filter_word="AMZN | Amazon",
                     filter_arguments=arguments,
                     start_year=2018,
                     start_month=1,
                     end_year=2022,
                     end_month=1,
                     location="../Data/Raw/Amazon",
                     limit=100,
                     type="Submissions")

