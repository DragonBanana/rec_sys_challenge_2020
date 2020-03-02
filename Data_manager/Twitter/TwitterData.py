import os

import numpy as np
import pandas as pd

from Data_manager.Twitter.Reader.TrainingDataframeReader import TrainingDataframeReader


class TwitterData:
    # Private reader objects
    _training_dataframe_reader = TrainingDataframeReader()

    # Data structures
    training_dataframe = pd.DataFrame()

    def __init__(self):
        # Reading the training dataframe
        if self._training_dataframe_reader.has_cache():
            self.training_dataframe = self._training_dataframe_reader.load_from_cache()
        else:
            self.training_dataframe = self._training_dataframe_reader.load_from_file()

    def save_to_cache(self):
        # Saving the training dataframe
        self.training_dataframe.to_csv(self._training_dataframe_reader.cached_train_data_path,
                                       compression='gzip')


if __name__ == '__main__':
    data = TwitterData()
