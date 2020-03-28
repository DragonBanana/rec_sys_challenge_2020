import pandas as pd
import pathlib as pl
import numpy as np
import RootPath
from abc import abstractmethod

from Utils.Data.Features.MappedFeatures import *
from Utils.Data.Dictionary.MappingDictionary import *


class UserBasicFeatureDictArrayNumpy(Dictionary):
    """
    It is built only using train and test set.
    Abstract class representing a dictionary array that works with numpy/pickle file.
    """

    def __init__(self, dictionary_name: str, ):
        super().__init__(dictionary_name)
        self.npz_path = pl.Path(f"{Dictionary.ROOT_PATH}/basic_features/user/{self.dictionary_name}.npz")

    def has_dictionary(self):
        return self.npz_path.is_file()

    def load_dictionary(self):
        assert self.has_dictionary(), f"The feature {self.dictionary_name} does not exists. Create it first."
        return np.load(self.npz_path, allow_pickle=True)['x']

    @abstractmethod
    def create_dictionary(self):
        pass

    def save_dictionary(self, arr: np.ndarray):
        self.npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.npz_path, x=arr)


class FollowerCountUserBasicFeatureDictArray(UserBasicFeatureDictArrayNumpy):

    def __init__(self):
        super().__init__("follower_count_user_dict_array")

    def create_dictionary(self):
        df = pd.DataFrame()

        # Load the index column
        train_engager_id_feature = MappedFeatureEngagerId("train")
        test_engager_id_feature = MappedFeatureEngagerId("test")
        train_creator_id_feature = MappedFeatureCreatorId("train")
        test_creator_id_feature = MappedFeatureCreatorId("test")

        # Find the mask of uniques one
        engager_train_df = train_engager_id_feature.load_or_create()
        engager_test_df = test_engager_id_feature.load_or_create()
        creator_train_df = train_creator_id_feature.load_or_create()
        creator_test_df = test_creator_id_feature.load_or_create()

        unique = ~engager_train_df[train_engager_id_feature.feature_name].append(
            engager_test_df[test_engager_id_feature.feature_name]).append(
            creator_train_df[train_creator_id_feature.feature_name]).append(
            creator_test_df[test_creator_id_feature.feature_name]).duplicated()

        # Unique tweet ids column
        df['id'] = engager_train_df[train_engager_id_feature.feature_name].append(
            engager_test_df[test_engager_id_feature.feature_name]).append(
            creator_train_df[train_creator_id_feature.feature_name]).append(
            creator_test_df[test_creator_id_feature.feature_name])[unique]

        # Load the target column
        column = "follower_count"
        engager_train_target_feature = RawFeatureEngagerFollowerCount("train")
        engager_test_target_feature = RawFeatureEngagerFollowerCount("test")
        creator_train_target_feature = RawFeatureCreatorFollowerCount("train")
        creator_test_target_feature = RawFeatureCreatorFollowerCount("test")
        engager_train_df = engager_train_target_feature.load_or_create()
        engager_test_df = engager_test_target_feature.load_or_create()
        creator_train_df = creator_train_target_feature.load_or_create()
        creator_test_df = creator_test_target_feature.load_or_create()
        df[column] = engager_train_df[engager_train_target_feature.feature_name].append(
            engager_test_df[engager_test_target_feature.feature_name]).append(
            creator_train_df[creator_train_target_feature.feature_name]).append(
            creator_test_df[creator_test_target_feature.feature_name])[unique]

        # Cast it to a numpy array
        arr = np.array(df.sort_values(by='id')[column].array)

        self.save_dictionary(arr)


class FollowingCountUserBasicFeatureDictArray(UserBasicFeatureDictArrayNumpy):

    def __init__(self):
        super().__init__("following_count_user_dict_array")

    def create_dictionary(self):
        df = pd.DataFrame()

        # Load the index column
        train_engager_id_feature = MappedFeatureEngagerId("train")
        test_engager_id_feature = MappedFeatureEngagerId("test")
        train_creator_id_feature = MappedFeatureCreatorId("train")
        test_creator_id_feature = MappedFeatureCreatorId("test")

        # Find the mask of uniques one
        engager_train_df = train_engager_id_feature.load_or_create()
        engager_test_df = test_engager_id_feature.load_or_create()
        creator_train_df = train_creator_id_feature.load_or_create()
        creator_test_df = test_creator_id_feature.load_or_create()

        unique = ~engager_train_df[train_engager_id_feature.feature_name].append(
            engager_test_df[test_engager_id_feature.feature_name]).append(
            creator_train_df[train_creator_id_feature.feature_name]).append(
            creator_test_df[test_creator_id_feature.feature_name]).duplicated()

        # Unique tweet ids column
        df['id'] = engager_train_df[train_engager_id_feature.feature_name].append(
            engager_test_df[test_engager_id_feature.feature_name]).append(
            creator_train_df[train_creator_id_feature.feature_name]).append(
            creator_test_df[test_creator_id_feature.feature_name])[unique]

        # Load the target column
        column = "following_count"
        engager_train_target_feature = RawFeatureEngagerFollowingCount("train")
        engager_test_target_feature = RawFeatureEngagerFollowingCount("test")
        creator_train_target_feature = RawFeatureCreatorFollowingCount("train")
        creator_test_target_feature = RawFeatureCreatorFollowingCount("test")
        engager_train_df = engager_train_target_feature.load_or_create()
        engager_test_df = engager_test_target_feature.load_or_create()
        creator_train_df = creator_train_target_feature.load_or_create()
        creator_test_df = creator_test_target_feature.load_or_create()
        print(engager_train_df)
        df[column] = engager_train_df[engager_train_target_feature.feature_name].append(
            engager_test_df[engager_test_target_feature.feature_name]).append(
            creator_train_df[creator_train_target_feature.feature_name]).append(
            creator_test_df[creator_test_target_feature.feature_name])[unique]

        # Cast it to a numpy array
        arr = np.array(df.sort_values(by='id')[column].array)

        self.save_dictionary(arr)


class IsVerifiedUserBasicFeatureDictArray(UserBasicFeatureDictArrayNumpy):

    def __init__(self):
        super().__init__("is_verified_user_dict_array")

    def create_dictionary(self):
        df = pd.DataFrame()

        # Load the index column
        train_engager_id_feature = MappedFeatureEngagerId("train")
        test_engager_id_feature = MappedFeatureEngagerId("test")
        train_creator_id_feature = MappedFeatureCreatorId("train")
        test_creator_id_feature = MappedFeatureCreatorId("test")

        # Find the mask of uniques one
        engager_train_df = train_engager_id_feature.load_or_create()
        engager_test_df = test_engager_id_feature.load_or_create()
        creator_train_df = train_creator_id_feature.load_or_create()
        creator_test_df = test_creator_id_feature.load_or_create()

        unique = ~engager_train_df[train_engager_id_feature.feature_name].append(
            engager_test_df[test_engager_id_feature.feature_name]).append(
            creator_train_df[train_creator_id_feature.feature_name]).append(
            creator_test_df[test_creator_id_feature.feature_name]).duplicated()

        # Unique tweet ids column
        df['id'] = engager_train_df[train_engager_id_feature.feature_name].append(
            engager_test_df[test_engager_id_feature.feature_name]).append(
            creator_train_df[train_creator_id_feature.feature_name]).append(
            creator_test_df[test_creator_id_feature.feature_name])[unique]

        # Load the target column
        column = "is_verified"
        engager_train_target_feature = RawFeatureEngagerIsVerified("train")
        engager_test_target_feature = RawFeatureEngagerIsVerified("test")
        creator_train_target_feature = RawFeatureCreatorIsVerified("train")
        creator_test_target_feature = RawFeatureCreatorIsVerified("test")
        engager_train_df = engager_train_target_feature.load_or_create()
        engager_test_df = engager_test_target_feature.load_or_create()
        creator_train_df = creator_train_target_feature.load_or_create()
        creator_test_df = creator_test_target_feature.load_or_create()
        df[column] = engager_train_df[engager_train_target_feature.feature_name].append(
            engager_test_df[engager_test_target_feature.feature_name]).append(
            creator_train_df[creator_train_target_feature.feature_name]).append(
            creator_test_df[creator_test_target_feature.feature_name])[unique]

        # Cast it to a numpy array
        arr = np.array(df.sort_values(by='id')[column].array)

        self.save_dictionary(arr)


class CreationTimestampUserBasicFeatureDictArray(UserBasicFeatureDictArrayNumpy):

    def __init__(self):
        super().__init__("creation_timestamp_user_dict_array")

    def create_dictionary(self):
        df = pd.DataFrame()

        # Load the index column
        train_engager_id_feature = MappedFeatureEngagerId("train")
        test_engager_id_feature = MappedFeatureEngagerId("test")
        train_creator_id_feature = MappedFeatureCreatorId("train")
        test_creator_id_feature = MappedFeatureCreatorId("test")

        # Find the mask of uniques one
        engager_train_df = train_engager_id_feature.load_or_create()
        engager_test_df = test_engager_id_feature.load_or_create()
        creator_train_df = train_creator_id_feature.load_or_create()
        creator_test_df = test_creator_id_feature.load_or_create()

        unique = ~engager_train_df[train_engager_id_feature.feature_name].append(
            engager_test_df[test_engager_id_feature.feature_name]).append(
            creator_train_df[train_creator_id_feature.feature_name]).append(
            creator_test_df[test_creator_id_feature.feature_name]).duplicated()

        # Unique tweet ids column
        df['id'] = engager_train_df[train_engager_id_feature.feature_name].append(
            engager_test_df[test_engager_id_feature.feature_name]).append(
            creator_train_df[train_creator_id_feature.feature_name]).append(
            creator_test_df[test_creator_id_feature.feature_name])[unique]

        # Load the target column
        column = "creation_timestamp"
        engager_train_target_feature = RawFeatureEngagerCreationTimestamp("train")
        engager_test_target_feature = RawFeatureEngagerCreationTimestamp("test")
        creator_train_target_feature = RawFeatureCreatorCreationTimestamp("train")
        creator_test_target_feature = RawFeatureCreatorCreationTimestamp("test")
        engager_train_df = engager_train_target_feature.load_or_create()
        engager_test_df = engager_test_target_feature.load_or_create()
        creator_train_df = creator_train_target_feature.load_or_create()
        creator_test_df = creator_test_target_feature.load_or_create()
        df[column] = engager_train_df[engager_train_target_feature.feature_name].append(
            engager_test_df[engager_test_target_feature.feature_name]).append(
            creator_train_df[creator_train_target_feature.feature_name]).append(
            creator_test_df[creator_test_target_feature.feature_name])[unique]

        # Cast it to a numpy array
        arr = np.array(df.sort_values(by='id')[column].array)

        self.save_dictionary(arr)
