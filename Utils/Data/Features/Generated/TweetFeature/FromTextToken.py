from tqdm import tqdm

from Utils.Data.Data import get_feature_reader
from Utils.Data.Dictionary.MappingDictionary import *
from Utils.Data.Dictionary.TweetTextFeaturesDictArray import *
from Utils.Data.Features.Feature import Feature
from Utils.Data.Features.Generated.GeneratedFeature import GeneratedFeaturePickle
from abc import abstractmethod
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
        

class TweetFeatureTextEmbeddings(GeneratedFeaturePickle):

    def __init__(self, feature_name: str, dataset_id: str):
        super().__init__(feature_name, dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.csv.gz")
        self.embeddings_array = None
        
    @abstractmethod
    def load_embeddings_dictionary(self):
        pass

    def create_feature(self):
        # Load tweet ids
        tweet_id_feature = MappedFeatureTweetId(self.dataset_id)
        tweet_id_df = tweet_id_feature.load_or_create()
        
        #tweet_id_df = tweet_id_df.head(25)
        #print(tweet_id_df)
        
        self.embeddings_array = self.load_embeddings_dictionary()
        
        columns_num = self.embeddings_array.shape[1]
        
        # this will be the final dataframe
        embeddings_feature_df = pd.DataFrame()
        
        # for each column, map the embeddings dictionary to all the tweets
        for col in range(columns_num):
            print("column :", col)
            embeddings_feature_df[f"embedding_{col}"] = tweet_id_df["mapped_feature_tweet_id"].map(lambda x: self.embeddings_array[x, col])
            
        #print(embeddings_feature_df)
        
        # Save the dataframe
        self.save_feature(embeddings_feature_df)
        
        
class TweetFeatureTextEmbeddingsPCA32(TweetFeatureTextEmbeddings):

    def __init__(self, dataset_id: str):
        super().__init__("text_embeddings_clean_PCA_32", dataset_id)
        
    def load_embeddings_dictionary(self):
        self.embeddings_array = TweetTextEmbeddingsPCA32FeatureDictArray().load_or_create()
        

class TweetFeatureTextEmbeddingsPCA10(TweetFeatureTextEmbeddings):

    def __init__(self, dataset_id: str):
        super().__init__("text_embeddings_clean_PCA_10", dataset_id)
        
    def load_embeddings_dictionary(self):
        self.embeddings_array = TweetTextEmbeddingsPCA10FeatureDictArray().load_or_create()
        

class TweetFeatureTextEmbeddingsHashtagsMentionsLDA15(TweetFeatureTextEmbeddings):
        
    def __init__(self, dataset_id: str):
        super().__init__("text_embeddings_hashtags_mentions_LDA_15", dataset_id)
        
    def load_embeddings_dictionary(self):
        self.embeddings_array = TweetTextEmbeddingsHashtagsMentionsLDA15FeatureDictArray().load_or_create()
        

class TweetFeatureTextEmbeddingsHashtagsMentionsLDA20(TweetFeatureTextEmbeddings):
        
    def __init__(self, dataset_id: str):
        super().__init__("text_embeddings_hashtags_mentions_LDA_20", dataset_id)
        
    def load_embeddings_dictionary(self):
        self.embeddings_array = TweetTextEmbeddingsHashtagsMentionsLDA15FeatureDictArray().load_or_create()
        
    
class TweetFeatureDominantTopic(GeneratedFeaturePickle):

    def __init__(self, feature_name : str, dataset_id: str):
        super().__init__(feature_name, dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.csv.gz")
        self.dictionary_array = None
        
    @abstractmethod
    def load_dictionary(self):
        pass

    def create_feature(self):
        # Load the tweet ids
        tweet_id_feature = MappedFeatureTweetId(self.dataset_id)
        tweet_id_df = tweet_id_feature.load_or_create()
        
        self.dictionary_array = self.load_dictionary()
        
        df = pd.DataFrame()
        df["dominant_topic"] = tweet_id_df["mapped_feature_tweet_id"].map(lambda x: np.argmax(self.dictionary_array[x]) if np.max(self.dictionary_array[x]) == np.min(self.dictionary_array[x]) else -1)

        # Save the dataframe
        self.save_feature(df)
        

class TweetFeatureDominantTopicLDA15(TweetFeatureDominantTopic):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_dominant_topic_LDA_15", dataset_id)
        
    def load_dictionary(self):
        self.dictionary_array = TweetTextEmbeddingsHashtagsMentionsLDA15FeatureDictArray().load_or_create()
        
        
class TweetFeatureDominantTopicLDA20(TweetFeatureDominantTopic):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_dominant_topic_LDA_20", dataset_id)
        
    def load_dictionary(self):
        self.dictionary_array = TweetTextEmbeddingsHashtagsMentionsLDA20FeatureDictArray().load_or_create()
        

class TweetFeatureTokenLength(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_token_length", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.csv.gz")

    def create_feature(self):
        # Load the tweet ids
        tweet_id_feature = MappedFeatureTweetId(self.dataset_id)
        tweet_id_df = tweet_id_feature.load_or_create()

        # load the tweet id, token_list dataframe
        tokens_feature_df_reader = get_feature_reader('raw_feature_tweet_text_token', self.dataset_id, chunksize=250000)
        length_arr = None

        for chunk in tqdm(tokens_feature_df_reader):
            curr_arr = chunk['raw_feature_tweet_text_token'] \
                .map(lambda x: x.split('\t')) \
                .map(lambda x: len(x) - 2) \
                .values

            if length_arr is None:
                length_arr = curr_arr
            else:
                length_arr = np.hstack([length_arr, curr_arr])

        length_df = pd.DataFrame({'tweet_feature_token_length': length_arr})
        # Save the dataframe
        self.save_feature(length_df)


class TweetFeatureTokenLengthUnique(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_token_length_unique", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.csv.gz")

    def create_feature(self):
        # Load the tweet ids
        tweet_id_feature = MappedFeatureTweetId(self.dataset_id)
        tweet_id_df = tweet_id_feature.load_or_create()

        # load the tweet id, token_list dataframe
        tokens_feature_df_reader = get_feature_reader('raw_feature_tweet_text_token', self.dataset_id, chunksize=250000)
        length_arr = None

        for chunk in tqdm(tokens_feature_df_reader):
            curr_arr = chunk['raw_feature_tweet_text_token'] \
                .map(lambda x: x.split('\t')) \
                .map(lambda x: set(x))\
                .map(lambda x: len(x) - 2) \
                .values

            if length_arr is None:
                length_arr = curr_arr
            else:
                length_arr = np.hstack([length_arr, curr_arr])

        length_df = pd.DataFrame({'tweet_feature_token_length_unique': length_arr})
        # Save the dataframe
        self.save_feature(length_df)
