import numpy as np
import pandas as pd


class TwitterData:
    train_data_path = "../../AWS/aws_local/twitter-2014-dataset/training.dat"

    def __init__(self):
        dataframe = pd.DataFrame()
        pass

    def has_cache(self):
        pass

    def load_from_cache(self):
        pass

    def save_to_cache(self):
        pass

    def load_from_file(self) -> pd.DataFrame:
        # Initializing the dataframe.
        dtypes = np.dtype([
            ('user_id', int),
            ('item_id', int),
            ('id_rating', int),
            ('scraping_time', int),
            ('tweet_in_json', str),
        ])
        data = np.empty(0, dtype=dtypes)
        dataframe = pd.DataFrame(data)

        # Open the training data.
        f = open(TwitterData.train_data_path)

        # Parsing the file.
        lines = f.readlines()
        # Auxiliary structure for building the dataframe
        user_ids = []
        item_ids = []
        id_ratings = []
        scraping_times = []
        tweets_in_json = []

        # Skip the first line.
        for i in range(1, len(lines)):
            line = lines[i]
            # Each row is composed by 5 columns, so they are separated by 4 commas.
            row = line.split(',', 4)
            user_ids.append(int(row[0]))
            item_ids.append(int(row[1]))
            id_ratings.append(int(row[2]))
            scraping_times.append(int(row[3]))
            tweets_in_json.append(row[4])

        # Populate the dataframe
        dataframe['user_id'] = user_ids
        dataframe['item_id'] = item_ids
        dataframe['id_rating'] = id_ratings
        dataframe['scraping_time'] = scraping_times
        dataframe['tweet_in_json'] = tweets_in_json

        return dataframe
