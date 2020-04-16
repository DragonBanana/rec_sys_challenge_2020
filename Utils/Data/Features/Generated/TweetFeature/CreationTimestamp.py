from Utils.Data.Dictionary.MappingDictionary import *
from Utils.Data.Features.Generated.GeneratedFeature import GeneratedFeaturePickle
import pandas as pd
from datetime import datetime as dt


class TweetFeatureCreationTimestampHour(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_creation_timestamp_hour", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/creation_timestamp/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/creation_timestamp/{self.feature_name}.csv.gz")

    def create_feature(self):
        # Load the engagement column
        feature = RawFeatureTweetTimestamp(self.dataset_id)
        feature_df = feature.load_or_create()
        # Count the number of photos
        hour_df = pd.DataFrame(feature_df[feature.feature_name].map(lambda x: dt.fromtimestamp(x).hour))

        self.save_feature(hour_df)

class TweetFeatureCreationTimestampWeekDay(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_creation_timestamp_week_day", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/creation_timestamp/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/creation_timestamp/{self.feature_name}.csv.gz")

    def create_feature(self):
        # Load the engagement column
        feature = RawFeatureTweetTimestamp(self.dataset_id)
        feature_df = feature.load_or_create()
        # Count the number of photos
        week_day_df = pd.DataFrame(feature_df[feature.feature_name].map(lambda x: dt.fromtimestamp(x).weekday()))

        self.save_feature(week_day_df)