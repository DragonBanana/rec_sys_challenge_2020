from Utils.Data.Dictionary.MappingDictionary import *
from Utils.Data.Dictionary.TweetTextFeaturesDictArray import TweetTextEmbeddingsFeatureDictArray
from Utils.Data.Features.Feature import Feature
from Utils.Data.Features.Generated.GeneratedFeature import GeneratedFeaturePickle
import pandas as pd
import numpy as np
import gzip

from Utils.Data.Features.MappedFeatures import MappedFeatureTweetId


class TweetFeatureMappedMentions(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_mentions", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.csv.gz")

    def create_feature(self):

        # Load tweet ids
        tweet_id_feature = MappedFeatureTweetId(self.dataset_id)
        tweet_id_df = tweet_id_feature.load_or_create()

        # Merge train and test mentions
        mentions_array = pd.concat([
            pd.read_csv(f"{RootPath.get_dataset_path()}/Dictionary/from_text_token/test_mentions.csv.gz",
                        compression="gzip", sep="\x01", index_col=0),
            pd.read_csv(f"{RootPath.get_dataset_path()}/Dictionary/from_text_token/train_mentions.csv.gz",
                        compression="gzip", sep="\x01", index_col=0)
        ])['mentions_mapped'].astype(str).map(
            lambda x: np.array(x.split('\t'), dtype=np.str) if x != 'nan' else None
        ).array

        # Compute for each engagement the tweet mentions
        mapped_mentions_df = pd.DataFrame(tweet_id_df[tweet_id_feature.feature_name].map(lambda x: mentions_array[x]))

        # Save the dataframe
        self.save_feature(mapped_mentions_df)

class TweetFeatureNumberOfMentions(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_number_of_mentions", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.csv.gz")

    def create_feature(self):
        # Load the extracted mentions
        mentions_feature = TweetFeatureMappedMentions(self.dataset_id)
        mentions_df = mentions_feature.load_or_create()

        # Compute for each engagement the tweet mentions
        mnumber_of_mentions_df = pd.DataFrame(mentions_df[mentions_feature.feature_name].map(lambda x: len(x) if x is not None else 0))

        # Save the dataframe
        self.save_feature(mnumber_of_mentions_df)
        

class TweetFeatureTextEmbeddings(Feature):

    def __init__(self, feature_name: str, dataset_id: str):
        super().__init__(feature_name, dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.csv.gz")
        
    def has_feature(self):
        return self.csv_path.is_file()

    def load_feature(self):
        assert self.has_feature(), f"The feature {self.feature_name} does not exists. Create it first."
        # get the list of embeddings columns (can vary among different datasets)
        with gzip.open(self.csv_path, "rt") as reader:
            columns = reader.readline().strip().split(',')
            
        dataframe = pd.read_csv(self.csv_path, compression="gzip")
        # load one column at a time
        #for col in columns[1:4]:
            # always load one embedding column at a time
        #    dataframe[col] = pd.read_csv(self.csv_path, usecols=[col], compression="gzip", nrows=5)
            
        return dataframe

    def create_feature(self):
        # Load tweet ids
        tweet_id_feature = MappedFeatureTweetId(self.dataset_id)
        tweet_id_df = tweet_id_feature.load_or_create()
        
        #tweet_id_df = tweet_id_df.head(25)
        #print(tweet_id_df)
        
        tweet_text_embeddings_dict_array = TweetTextEmbeddingsFeatureDictArray(dictionary_name=self.feature_name)
        embeddings_array = tweet_text_embeddings_dict_array.load_or_create()
        
        columns_num = embeddings_array.shape[1]
        
        # this will be the final dataframe
        embeddings_feature_df = pd.DataFrame()
        
        # for each column, map the embeddings dictionary to all the tweets
        for col in range(columns_num):
            embeddings_feature_df[f"embedding_{col}"] = tweet_id_df["mapped_feature_tweet_id"].map(lambda x: embeddings_array[x, col])
            
        #print(embeddings_feature_df)
        
        # Save the dataframe
        self.save_feature(embeddings_feature_df)
        

    def save_feature(self, dataframe: pd.DataFrame):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(self.csv_path, compression='gzip')
