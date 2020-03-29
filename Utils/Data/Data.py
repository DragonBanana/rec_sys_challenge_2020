from Utils.Data.DataUtils import FEATURES, DICTIONARIES, DICT_ARRAYS
import pandas as pd


def get_dataset_xgb(dataset_id: str = "train", X_label: list = None, Y_label: str = None):
    """
    :param features: Features in the X_Train matrix, if not specified returns all the known features
    :param dataset_id: The dataset id ("train", "test", etc.)
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

def get_dataset(features: list, dataset_id: str):
    return pd.concat([get_feature(feature_name, dataset_id) for feature_name in features], axis=1)


def get_feature(feature_name: str, dataset_id: str):
    if (feature_name, dataset_id) in FEATURES.keys():
        return FEATURES[(feature_name, dataset_id)].load_or_create()


def get_dictionary(dictionary_name: str):
    if dictionary_name in DICTIONARIES.keys():
        return DICTIONARIES[dictionary_name].load_or_create()


def get_dictionary_array(dictionary_name: str):
    if dictionary_name in DICT_ARRAYS.keys():
        return DICT_ARRAYS[dictionary_name].load_or_create()
