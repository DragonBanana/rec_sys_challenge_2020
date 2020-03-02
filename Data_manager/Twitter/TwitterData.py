import numpy as np
import pandas as pd


class TwitterData:

    train_data_path = "../../AWS/aws_local/dataset/recsys_challenge_2014_dataset/training.dat"

    def __init__(self):

        dataframe = pd.DataFrame()
        pass

    def has_cache(self):
        pass

    def load_from_cache(self):
        pass

    def save_to_cache(self):
        pass

    def load_from_file(self):
        pass

if __name__ == '__main__':

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
    # Skip the first line.
    for line in lines[1:]:
        # Each row is composed by 5 columns, so they are separated by 4 commas.
        row = line.split(',', 4)
        dataframe.append(row)
