import pandas as pd
import pathlib as pl
import numpy as np
import RootPath
from abc import abstractmethod

from Utils.Data.Features.MappedFeatures import *
from Utils.Data.Dictionary.MappingDictionary import *


class TweetTextFeatureDictArrayNumpy(Dictionary):
    """
    It is built only using train and test set.
    Abstract class representing a dictionary array that works with numpy/pickle file.
    """

    def __init__(self, dictionary_name: str, ):
        super().__init__(dictionary_name)
        self.csv_path = pl.Path(f"{Dictionary.ROOT_PATH}/from_text_token/{self.dictionary_name}.csv.gz")
        self.npz_path = pl.Path(f"{Dictionary.ROOT_PATH}/text_features/{self.dictionary_name}.npz")

    def has_dictionary(self):
        return self.npz_path.is_file()

    def load_dictionary(self):
        assert self.has_dictionary(), f"The dictionary {self.dictionary_name} does not exists. Create it first."
        return np.load(self.npz_path, allow_pickle=True)['x']

    @abstractmethod
    def create_dictionary(self):
        pass

    def save_dictionary(self, arr: np.ndarray):
        self.npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.npz_path, x=arr)
        

class TweetTextEmbeddingsFeatureDictArray(TweetTextFeatureDictArrayNumpy):

    def __init__(self):
        super().__init__("tweet_text_feature_dict_array")

    def create_dictionary(self):
        # simply convert the embeddings dataframe to a numpy array (of arrays)
        # get the list of embeddings columns (can vary among different datasets)
        with gzip.open(self.csv_path, "rt") as reader:
            columns = reader.readline().strip().split(',')
        
        # this will be the final dataframe
        embeddings_feature_df = pd.DataFrame()
        # load the tweet id column
        embeddings_feature_df = pd.read_csv(self.csv_path, usecols=[columns[0]])
        for col in columns[1:]:
            # load one embedding column at a time
            embeddings_feature_df[col] = pd.read_csv(self.csv_path, usecols=[col])
            
        # convert to numpy all the columns except the tweet id
        arr = np.array(embeddings_feature_df.sort_values(by='tweet_features_tweet_id')[columns[1:]])
        
        self.save_dictionary(arr)
