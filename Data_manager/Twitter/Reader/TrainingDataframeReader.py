import json
import os
import pickle

import numpy as np
import pandas as pd
from Data_manager.Twitter.Reader.CachedReader import CachedReader
from Utils import PathUtils


class TrainingDataframeReader(CachedReader):
    # Raw data path
    raw_data_path = PathUtils.get_project_root().joinpath(
        'aws_local/twitter-2014-dataset/training.dat')

    # Cached training dataframe path
    cached_data_path = PathUtils.get_project_root().joinpath(
        'aws_local/twitter-2014-cached-dataset/training_dataframe.gz')

    # Other cached resources
    cached_other_data_path = PathUtils.get_project_root().joinpath(
        'aws_local/twitter-2014-cached-dataset')

    def __init__(self, twitter_data):
        self.twitter_data = twitter_data

    def has_cache(self):
        return os.path.isfile(self.cached_data_path) \
               and os.path.isfile(self.cached_other_data_path.joinpath('unique_twitter_user_ids.npz')) \
               and os.path.isfile(self.cached_other_data_path.joinpath('unique_twitter_item_ids.npz')) \
               and os.path.isfile(self.cached_other_data_path.joinpath('unique_internal_user_ids.npz')) \
               and os.path.isfile(self.cached_other_data_path.joinpath('unique_internal_item_ids.npz')) \
               and os.path.isfile(self.cached_other_data_path.joinpath('user_twitter_id_to_internal_id_dict')) \
               and os.path.isfile(self.cached_other_data_path.joinpath('user_internal_id_to_twitter_id_dict')) \
               and os.path.isfile(self.cached_other_data_path.joinpath('item_twitter_id_to_internal_id_dict')) \
               and os.path.isfile(self.cached_other_data_path.joinpath('item_internal_id_to_twitter_id_dict'))


    def load_from_cache(self):
        # Saving the training dataframe
        self.twitter_data.training_dataframe.from_csv(self.cached_data_path)
        self.twitter_data.unique_twitter_user_ids = np.load(
            self.cached_other_data_path.joinpath('unique_twitter_user_ids.npz'))
        self.twitter_data.unique_twitter_item_ids = np.load(
            self.cached_other_data_path.joinpath('unique_twitter_item_ids.npz'))
        self.twitter_data.unique_internal_user_ids = np.load(
            self.cached_other_data_path.joinpath('unique_internal_user_ids.npz'))
        self.twitter_data.unique_internal_item_ids = np.load(
            self.cached_other_data_path.joinpath('unique_internal_item_ids.npz'))
        self.twitter_data.number_of_users = len(self.twitter_data.unique_twitter_user_ids)
        self.twitter_data.number_of_items = len(self.twitter_data.unique_twitter_item_ids)
        self.twitter_data.user_twitter_id_to_internal_id_dict = pickle.load(
            open(self.cached_other_data_path.joinpath('user_twitter_id_to_internal_id_dict'), 'rb'))
        self.twitter_data.user_internal_id_to_twitter_id_dict = pickle.load(
            open(self.cached_other_data_path.joinpath('user_internal_id_to_twitter_id_dict'), 'rb'))
        self.twitter_data.item_twitter_id_to_internal_id_dict = pickle.load(
            open(self.cached_other_data_path.joinpath('item_twitter_id_to_internal_id_dict'), 'rb'))
        self.twitter_data.item_internal_id_to_twitter_id_dict = pickle.load(
            open(self.cached_other_data_path.joinpath('item_internal_id_to_twitter_id_dict'), 'rb'))

    def save_to_cache(self):
        # Saving the training dataframe
        self.twitter_data.training_dataframe.to_csv(self.cached_data_path, compression='gzip')
        np.savez_compressed(self.cached_other_data_path.joinpath('unique_twitter_user_ids'),
                            self.twitter_data.unique_twitter_user_ids)
        np.savez_compressed(self.cached_other_data_path.joinpath('unique_twitter_item_ids'),
                            self.twitter_data.unique_twitter_item_ids)
        np.savez_compressed(self.cached_other_data_path.joinpath('unique_internal_user_ids'),
                            self.twitter_data.unique_internal_user_ids)
        np.savez_compressed(self.cached_other_data_path.joinpath('unique_internal_item_ids'),
                            self.twitter_data.unique_internal_item_ids)
        pickle.dump(self.twitter_data.user_twitter_id_to_internal_id_dict,
                    open(self.cached_other_data_path.joinpath('user_twitter_id_to_internal_id_dict'), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.twitter_data.user_internal_id_to_twitter_id_dict,
                    open(self.cached_other_data_path.joinpath('user_internal_id_to_twitter_id_dict'), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.twitter_data.item_twitter_id_to_internal_id_dict,
                    open(self.cached_other_data_path.joinpath('item_twitter_id_to_internal_id_dict'), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.twitter_data.item_internal_id_to_twitter_id_dict,
                    open(self.cached_other_data_path.joinpath('item_internal_id_to_twitter_id_dict'), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)

    def load_from_raw(self):
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
        f = open(self.raw_data_path)

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

            # Explicitely assigning values, for readability
            user_id = row[0]
            item_id = row[1]
            id_rating = int(row[2])
            scraping_time = int(row[3])
            tweet_in_json = row[4]

            user_ids.append(user_id)
            item_ids.append(item_id)
            id_ratings.append(id_rating)
            scraping_times.append(scraping_time)
            tweets_in_json.append(tweet_in_json)

        # Map each twitter user_id and item_id to an internal id
        # in this way we can reduce the sparsity
        unique_user_ids = np.unique(user_ids)
        unique_item_ids = np.unique(item_ids)
        # User data structures for user id mappings
        user_twitter_id_to_internal_id_dict = {}
        user_internal_id_to_twitter_id_dict = {}
        # Item data structures for user id mappings
        item_twitter_id_to_internal_id_dict = {}
        item_internal_id_to_twitter_id_dict = {}

        # Populating the user dictionary
        for i in range(len(unique_user_ids)):
            user_twitter_id_to_internal_id_dict[unique_user_ids[i]] = i
            user_internal_id_to_twitter_id_dict[i] = unique_user_ids[i]

        # Populating the item dictionary
        for i in range(len(unique_item_ids)):
            item_twitter_id_to_internal_id_dict[unique_item_ids[i]] = i
            item_internal_id_to_twitter_id_dict[i] = unique_item_ids[i]

        # Populating the dataframe
        dataframe['user_id'] = [user_twitter_id_to_internal_id_dict[user_id] for user_id in user_ids]
        dataframe['item_id'] = [item_twitter_id_to_internal_id_dict[item_id] for item_id in item_ids]
        dataframe['id_rating'] = id_ratings
        dataframe['scraping_time'] = scraping_times
        dataframe['tweet_in_json'] = tweets_in_json

        # Storing back the data
        self.twitter_data.training_dataframe = dataframe
        self.twitter_data.unique_twitter_user_ids = unique_user_ids
        self.twitter_data.unique_twitter_item_ids = unique_item_ids
        self.twitter_data.unique_internal_user_ids = np.array(
            [user_twitter_id_to_internal_id_dict[user_id] for user_id in unique_user_ids])
        self.twitter_data.unique_internal_item_ids = np.array(
            [item_twitter_id_to_internal_id_dict[item_id] for item_id in unique_item_ids])
        self.twitter_data.number_of_users = len(self.twitter_data.unique_twitter_user_ids)
        self.twitter_data.number_of_items = len(self.twitter_data.unique_twitter_item_ids)
        self.twitter_data.user_twitter_id_to_internal_id_dict = user_twitter_id_to_internal_id_dict
        self.twitter_data.user_internal_id_to_twitter_id_dict = user_internal_id_to_twitter_id_dict
        self.twitter_data.item_twitter_id_to_internal_id_dict = item_twitter_id_to_internal_id_dict
        self.twitter_data.item_internal_id_to_twitter_id_dict = item_internal_id_to_twitter_id_dict
