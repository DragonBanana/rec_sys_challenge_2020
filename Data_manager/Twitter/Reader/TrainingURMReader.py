import os
import pandas as pd
import scipy.sparse as sps
from Data_manager.Twitter.Reader.CachedReader import CachedReader
from Utils import PathUtils


class TrainingURMReader(CachedReader):

    # Cached data path
    cached_data_path = PathUtils.get_project_root().joinpath(
        'aws_local/twitter-2014-cached-dataset/cached_training.gz')

    def __init__(self, training_dataframe : pd.DataFrame):
        self._training_dataframe = training_dataframe

    def has_cache(self):
        return os.path.isfile(self.cached_data_path)

    def load_from_cache(self):
        return pd.read_csv(self.cached_data_path)

    def load_from_raw(self) -> sps.csr_matrix:
        user_ids = self._training_dataframe['user_id'].to_list()
        item_ids = self._training_dataframe['item_id'].to_list()
        id_ratings = self._training_dataframe['id_rating'].to_list()

        urm = sps.csr_matrix((id_ratings, (user_ids, item_ids)))

        return urm