import os

import numpy as np
import pandas as pd

from Data_manager.Twitter.Reader.TrainingDataframeReader import TrainingDataframeReader
from Data_manager.Twitter.Reader.TrainingURMReader import TrainingURMReader


class TwitterData:
    # Data structures
    training_dataframe = pd.DataFrame()
    unique_twitter_user_ids = np.array([])
    unique_twitter_item_ids = np.array([])
    unique_internal_user_ids = np.array([])
    unique_internal_item_ids = np.array([])
    number_of_users = 0
    number_of_items = 0
    user_twitter_id_to_internal_id_dict = {}
    user_internal_id_to_twitter_id_dict = {}
    item_twitter_id_to_internal_id_dict = {}
    item_internal_id_to_twitter_id_dict = {}

    def __init__(self):
        # Reading the training dataframe
        self._training_dataframe_reader = TrainingDataframeReader(self)
        if self._training_dataframe_reader.has_cache():
            self._training_dataframe_reader.load_from_cache()
        else:
            self._training_dataframe_reader.load_from_raw()
        # Reading the training urm
        self._training_urm_reader = TrainingURMReader(self.training_dataframe)




if __name__ == '__main__':
    data = TwitterData()
