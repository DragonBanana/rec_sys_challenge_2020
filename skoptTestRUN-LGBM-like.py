import numpy as np
import skopt
from skopt import gp_minimize
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import sys
import pandas as pd
import time
import datetime as dt
from ParamTuning.ModelInterface import ModelInterface
from ParamTuning.Optimizer import Optimizer
from Utils.Data import Data

from Utils.Data.Data import oversample


def main():
    # Defining the dataset used
    train_dataset = "holdout/train"
    test_dataset = "holdout/test"

    # Define the X label
    X_label = [
        "raw_feature_creator_follower_count",
        # 0
        "raw_feature_creator_following_count",  # 1
        "raw_feature_engager_follower_count",  # 2
        "raw_feature_engager_following_count",  # 3
        "raw_feature_creator_is_verified",  # 4 CATEGORICAL
        "raw_feature_engager_is_verified",  # 5 CATEGORICAL
        "raw_feature_engagement_creator_follows_engager",  # 6 CATEGORICAL
        "tweet_feature_number_of_photo",  # 7
        "tweet_feature_number_of_video",  # 8
        "tweet_feature_number_of_gif",  # 9
        "tweet_feature_number_of_media",  # 10
        "tweet_feature_is_retweet",  # 11 CATEGORICAL
        "tweet_feature_is_quote",  # 12 CATEGORICAL
        "tweet_feature_is_top_level",  # 13 CATEGORICAL
        "tweet_feature_number_of_hashtags",  # 14
        "tweet_feature_creation_timestamp_hour",  # 15
        "tweet_feature_creation_timestamp_week_day",  # 16
        "tweet_feature_number_of_mentions",  #
        "number_of_engagements_like",
        "number_of_engagements_retweet",
        "number_of_engagements_reply",
        "number_of_engagements_comment",
        "number_of_engagements_negative",
        "number_of_engagements_positive",
        "number_of_engagements_with_language_like",
        "number_of_engagements_with_language_retweet",
        "number_of_engagements_with_language_reply",
        "number_of_engagements_with_language_comment",
        "number_of_engagements_with_language_negative",
        "number_of_engagements_with_language_positive",
        "number_of_engagements_ratio_like",
        "number_of_engagements_ratio_retweet",
        "number_of_engagements_ratio_reply",
        "number_of_engagements_ratio_comment",
        "number_of_engagements_ratio_negative",
        "number_of_engagements_ratio_positive",
        "number_of_engagements_between_creator_and_engager_like",
        "number_of_engagements_between_creator_and_engager_retweet",
        "number_of_engagements_between_creator_and_engager_reply",
        "number_of_engagements_between_creator_and_engager_comment",
        "number_of_engagements_between_creator_and_engager_negative",
        "number_of_engagements_between_creator_and_engager_positive",
        "creator_feature_number_of_like_engagements_received",
        "creator_feature_number_of_retweet_engagements_received",
        "creator_feature_number_of_reply_engagements_received",
        "creator_feature_number_of_comment_engagements_received",
        "creator_feature_number_of_negative_engagements_received",
        "creator_feature_number_of_positive_engagements_received",
        "creator_feature_number_of_like_engagements_given",
        "creator_feature_number_of_retweet_engagements_given",
        "creator_feature_number_of_reply_engagements_given",
        "creator_feature_number_of_comment_engagements_given",
        "creator_feature_number_of_negative_engagements_given",
        "creator_feature_number_of_positive_engagements_given",
        "engager_feature_number_of_like_engagements_received",
        "engager_feature_number_of_retweet_engagements_received",
        "engager_feature_number_of_reply_engagements_received",
        "engager_feature_number_of_comment_engagements_received",
        "engager_feature_number_of_negative_engagements_received",
        "engager_feature_number_of_positive_engagements_received",


        # "engager_feature_number_of_previous_like_engagement",  # 18
        # "engager_feature_number_of_previous_reply_engagement",  # 19
        # "engager_feature_number_of_previous_retweet_engagement",  # 20
        # "engager_feature_number_of_previous_comment_engagement",  # 21
        # "engager_feature_number_of_previous_positive_engagement",  # 22
        # "engager_feature_number_of_previous_negative_engagement",  # 23
        # "engager_feature_number_of_previous_engagement",  # 24
        # "engager_feature_number_of_previous_like_engagement_ratio_1",  # 25
        # "engager_feature_number_of_previous_reply_engagement_ratio_1",  # 26
        # "engager_feature_number_of_previous_retweet_engagement_ratio_1",  # 27
        # "engager_feature_number_of_previous_comment_engagement_ratio_1",  # 28
        # "engager_feature_number_of_previous_positive_engagement_ratio_1",  # 29
        # "engager_feature_number_of_previous_negative_engagement_ratio_1",  # 30
        # "engager_feature_number_of_previous_like_engagement_between_creator_and_engager_by_creator",
        # # 31
        # "engager_feature_number_of_previous_reply_engagement_between_creator_and_engager_by_creator",
        # # 32
        # "engager_feature_number_of_previous_retweet_engagement_between_creator_and_engager_by_creator",
        # # 33
        # "engager_feature_number_of_previous_comment_engagement_between_creator_and_engager_by_creator",
        # # 34
        # "engager_feature_number_of_previous_negative_engagement_between_creator_and_engager_by_creator",
        # # 35
        # "engager_feature_number_of_previous_positive_engagement_between_creator_and_engager_by_creator",
        # # 36
        # "engager_feature_number_of_previous_like_engagement_between_creator_and_engager_by_engager",
        # # 37
        # "engager_feature_number_of_previous_reply_engagement_between_creator_and_engager_by_engager",
        # # 38
        # "engager_feature_number_of_previous_retweet_engagement_between_creator_and_engager_by_engager",
        # # 39
        # "engager_feature_number_of_previous_comment_engagement_between_creator_and_engager_by_engager",
        # # 40
        # "engager_feature_number_of_previous_negative_engagement_between_creator_and_engager_by_engager",
        # # 41
        # "engager_feature_number_of_previous_positive_engagement_between_creator_and_engager_by_engager",
        # # 42
        # "engager_main_language",                      #43 CATEGORICAL
        # "creator_main_language",                      #44 CATEGORICAL
        # "creator_and_engager_have_same_main_language",                      #45 CATEGORICAL       - 43
        # "is_tweet_in_creator_main_language",                        #46 CATEGORICAL                -44
        # "is_tweet_in_engager_main_language",                        #47 CATEGORICAL                45
        # "statistical_probability_main_language_of_engager_engage_tweet_language_1",                     #48     46
        # "statistical_probability_main_language_of_engager_engage_tweet_language_2",                     #49     47
        # "tweet_feature_token_length",                     #50 CATEGORICAL
        # "tweet_feature_token_length_unique",
        # "engager_feature_knows_hashtag_positive",
        # "engager_feature_knows_hashtag_negative",
        # "engager_feature_knows_hashtag_like",
        # "engager_feature_knows_hashtag_reply",
        # "engager_feature_knows_hashtag_rt",
        # "hashtag_similarity_fold_ensembling_positive",  # 48
        # "link_similarity_fold_ensembling_positive",  # 49
        # "domain_similarity_fold_ensembling_positive",  # 50
        "tweet_feature_creation_timestamp_hour_shifted",  # 51
        "tweet_feature_creation_timestamp_day_phase",  # 52
        "tweet_feature_creation_timestamp_day_phase_shifted"  # 53

    ]
    # Define the Y label
    Y_label = [
        "tweet_feature_engagement_is_like"
    ]

    model_name = "lightgbm_classifier"
    kind = "like"

    # Load train data
    loading_data_start_time = time.time()
    X_train, Y_train = Data.get_dataset_xgb_batch(1, 0, train_dataset, X_label, Y_label, 0.30)

    # Load test data
    X_val, Y_val = Data.get_dataset_xgb_batch(2, 0, test_dataset, X_label, Y_label, 1)
    # If oversample is set
    # Oversample the cold users
    use_oversample = True
    os_column_name = "engager_feature_number_of_previous_positive_engagement_ratio_1"
    os_value = -1
    os_percentage = 0.3  # in order to have the 23% of cold users in the validation set
    if use_oversample is True:
        df = pd.concat([X_val, Y_val], axis=1)
        oversampled_df = oversample(df, os_column_name, os_value, os_percentage)
        X_val = oversampled_df[X_label]
        Y_val = oversampled_df[Y_label]
        del df, oversampled_df

    X_test, Y_test = Data.get_dataset_xgb_batch(2, 1, test_dataset, X_label, Y_label, 1)
    # If oversample is set
    # Oversample the cold users
    use_oversample = True
    os_column_name = "engager_feature_number_of_previous_positive_engagement_ratio_1"
    os_value = -1
    os_percentage = 0.3  # in order to have the 23% of cold users in the validation set
    if use_oversample is True:
        df = pd.concat([X_test, Y_test], axis=1)
        oversampled_df = oversample(df, os_column_name, os_value, os_percentage)
        X_test = oversampled_df[X_label]
        Y_test = oversampled_df[Y_label]
        del df, oversampled_df

    print(f"Loading data time: {time.time() - loading_data_start_time} seconds")

    OP = Optimizer(model_name,
                   kind,
                   mode=0,
                   path="like",
                   path_log="LGBM-like-freschi-sovracampione",
                   make_log=True,
                   make_save=False,
                   auto_save=False)

    OP.setParameters(n_calls=40, n_random_starts=15)
    OP.loadTrainData(X_train, Y_train)
    OP.loadTestData(X_test, Y_test)
    OP.loadValData(X_val, Y_val)
    OP.setParamsLGB(objective='binary', early_stopping_rounds=15, eval_metric="binary", is_unbalance=False)
    OP.setCategoricalFeatures(set([4, 5, 6, 11, 12, 13]))
    # OP.loadModelHardCoded()
    res = OP.optimize()

    '''
    #Add this for complete routine check
    print(res.func_vals.shape)
    path = OP.saveModel()
    OP.loadModel(path)
    res = OP.optimize()
    print(res.func_vals.shape)
    print("END")
    '''


if __name__ == "__main__":
    main()