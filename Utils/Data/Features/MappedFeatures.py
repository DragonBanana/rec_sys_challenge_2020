import pandas as pd
import pathlib as pl
import numpy as np
import RootPath
from abc import abstractmethod
from Utils.Data.Features.RawFeatures import *
from Utils.Data.Dictionary.MappingDictionary import *


def map_column_single_value(series, dictionary):
    mapped_series = series.map(dictionary).astype(np.int32)
    return pd.DataFrame(mapped_series)


def map_column_array(series, dictionary):
    mapped_series = series.map(
        lambda x: np.array([dictionary[y] for y in x.split('\t')], dtype=np.int32) if x is not pd.NA else None)
    return pd.DataFrame(mapped_series)


class MappedFeaturePickle(Feature):
    """
    Abstract class representing a dictionary that works with pickle file.
    """

    def __init__(self, feature_name: str, dataset_id: str):
        super().__init__(feature_name, dataset_id)
        self.pck_path = pl.Path(f"{Feature.ROOT_PATH}/{self.dataset_id}/mapped/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(f"{Feature.ROOT_PATH}/{self.dataset_id}/mapped/{self.feature_name}.csv.gz")

    def has_feature(self):
        return self.pck_path.is_file()

    def load_feature(self):
        assert self.has_feature(), f"The feature {self.feature_name} does not exists. Create it first."
        df = pd.read_pickle(self.pck_path, compression="gzip")
        # Renaming the column for consistency purpose
        df.columns = [self.feature_name]
        return df

    @abstractmethod
    def create_feature(self):
        pass

    def save_feature(self, dataframe: pd.DataFrame):
        # Changing column name
        dataframe.columns = [self.feature_name]
        self.pck_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_pickle(self.pck_path, compression='gzip')
        # For backup reason
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(self.csv_path, compression='gzip', index=True)


class MappedFeatureTweetLanguage(MappedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("mapped_feature_tweet_language", dataset_id)

    def create_feature(self):
        feature = RawFeatureTweetLanguage(self.dataset_id)
        dataframe = feature.load_or_create()
        dictionary = MappingLanguageDictionary().load_or_create()
        mapped_dataframe = map_column_single_value(dataframe[feature.feature_name], dictionary)

        self.save_feature(mapped_dataframe)


class MappedFeatureTweetId(MappedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("mapped_feature_tweet_id", dataset_id)

    def create_feature(self):
        feature = RawFeatureTweetId(self.dataset_id)
        dataframe = feature.load_or_create()
        dictionary = MappingTweetIdDictionary().load_or_create()
        mapped_dataframe = map_column_single_value(dataframe[feature.feature_name], dictionary)

        self.save_feature(mapped_dataframe)


class MappedFeatureCreatorId(MappedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("mapped_feature_creator_id", dataset_id)

    def create_feature(self):
        feature = RawFeatureCreatorId(self.dataset_id)
        dataframe = feature.load_or_create()
        dictionary = MappingUserIdDictionary().load_or_create()
        mapped_dataframe = map_column_single_value(dataframe[feature.feature_name], dictionary)

        self.save_feature(mapped_dataframe)


class MappedFeatureEngagerId(MappedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("mapped_feature_engager_id", dataset_id)

    def create_feature(self):
        feature = RawFeatureEngagerId(self.dataset_id)
        dataframe = feature.load_or_create()
        dictionary = MappingUserIdDictionary().load_or_create()
        mapped_dataframe = map_column_single_value(dataframe[feature.feature_name], dictionary)

        self.save_feature(mapped_dataframe)


class MappedFeatureTweetHashtags(MappedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("mapped_feature_tweet_hashtags", dataset_id)

    def create_feature(self):
        feature = RawFeatureTweetHashtags(self.dataset_id)
        dataframe = feature.load_or_create()
        dictionary = MappingHashtagDictionary().load_or_create()
        mapped_dataframe = map_column_array(dataframe[feature.feature_name], dictionary)

        self.save_feature(mapped_dataframe)


class MappedFeatureTweetLinks(MappedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("mapped_feature_tweet_links", dataset_id)

    def create_feature(self):
        feature = RawFeatureTweetLinks(self.dataset_id)
        dataframe = feature.load_or_create()
        dictionary = MappingLinkDictionary().load_or_create()
        mapped_dataframe = map_column_array(dataframe[feature.feature_name], dictionary)

        self.save_feature(mapped_dataframe)


class MappedFeatureTweetDomains(MappedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("mapped_feature_tweet_domains", dataset_id)

    def create_feature(self):
        feature = RawFeatureTweetDomains(self.dataset_id)
        dataframe = feature.load_or_create()
        dictionary = MappingDomainDictionary().load_or_create()
        mapped_dataframe = map_column_array(dataframe[feature.feature_name], dictionary)

        self.save_feature(mapped_dataframe)


class MappedFeatureTweetMedia(MappedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("mapped_feature_tweet_media", dataset_id)

    def create_feature(self):
        feature = RawFeatureTweetMedia(self.dataset_id)
        dataframe = feature.load_or_create()
        dictionary = MappingMediaDictionary().load_or_create()
        mapped_dataframe = map_column_array(dataframe[feature.feature_name], dictionary)

        self.save_feature(mapped_dataframe)
