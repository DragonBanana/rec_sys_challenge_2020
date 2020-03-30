from Utils.Data.DataUtils import FEATURES, DICTIONARIES, DICT_ARRAYS
import pandas as pd
import numpy as np


def get_dataset_xgb(dataset_id: str = "train", X_label: list = None, Y_label: list = None):
    """
    :param dataset_id: The dataset id ("train", "test", etc.)
    :param X_label: The X features, the ones the model is trained on.
    :param Y_label:  The Y feature, the one to be predicted.
    :return: 2 dataframes: 1) X_Train, 2) Y_Train
    """
    if X_label is None:
        X_label = [
            "raw_feature_tweet_id",
            "raw_feature_engager_id"
        ]
    if Y_label is None:
        Y_label = [
            "tweet_feature_engagement_is_like"
        ]
    return get_dataset(X_label, dataset_id), get_dataset(Y_label, dataset_id)


def get_dataset_xgb_default_train():
    train_dataset = "train_split_with_timestamp_from_train_random_seed_888_timestamp_threshold_1581465600_holdout_10"
    # Define the X label
    X_label = [
        # "mapped_feature_tweet_id",
        # "mapped_feature_creator_id",
        # "mapped_feature_engager_id",
        "raw_feature_creator_follower_count",
        "raw_feature_creator_following_count",
        "raw_feature_engager_follower_count",
        "raw_feature_engager_following_count"
        "tweet_feature_number_of_photo",
        "tweet_feature_number_of_video",
        "tweet_feature_number_of_gif",
        "tweet_feature_is_reply",
        "tweet_feature_is_retweet",
        "tweet_feature_is_quote",
        "tweet_feature_is_top_level"
    ]
    # Define the Y label
    Y_label = [
        "tweet_feature_engagement_is_like"
    ]
    return get_dataset_xgb(dataset_id=train_dataset, X_label=X_label, Y_label=Y_label)


def get_dataset_xgb_default_test():
    train_dataset = "val_split_with_timestamp_from_train_random_seed_888_timestamp_threshold_1581465600_holdout_10"
    # Define the X label
    X_label = [
        # "mapped_feature_tweet_id",
        # "mapped_feature_creator_id",
        # "mapped_feature_engager_id",
        "raw_feature_creator_follower_count",
        "raw_feature_creator_following_count",
        "raw_feature_engager_follower_count",
        "raw_feature_engager_following_count"
        "tweet_feature_number_of_photo",
        "tweet_feature_number_of_video",
        "tweet_feature_number_of_gif",
        "tweet_feature_is_reply",
        "tweet_feature_is_retweet",
        "tweet_feature_is_quote",
        "tweet_feature_is_top_level"
    ]
    # Define the Y label
    Y_label = [
        "tweet_feature_engagement_is_like"
    ]
    return get_dataset_xgb(dataset_id=train_dataset, X_label=X_label, Y_label=Y_label)


def get_dataset(features: list, dataset_id: str):
    dataframe = pd.concat([get_feature(feature_name, dataset_id) for feature_name in features], axis=1)

    # Some columns are not in the format XGB expects, so the following block of code will cast them to the right format
    for column in dataframe.columns:
        if str(dataframe[column].dtype).lower()[:3] == "int":
            dataframe[column] = dataframe[column].astype(np.int64, copy=False)
    return dataframe


def get_feature(feature_name: str, dataset_id: str):
    if (feature_name, dataset_id) in FEATURES.keys():
        return FEATURES[(feature_name, dataset_id)].load_or_create()


def get_dictionary(dictionary_name: str):
    if dictionary_name in DICTIONARIES.keys():
        return DICTIONARIES[dictionary_name].load_or_create()


def get_dictionary_array(dictionary_name: str):
    if dictionary_name in DICT_ARRAYS.keys():
        return DICT_ARRAYS[dictionary_name].load_or_create()
