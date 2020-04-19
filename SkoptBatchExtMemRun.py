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
import sklearn.datasets as skd
from tqdm import tqdm


def main():
    #------------------------------------------
    #           BATCH EXAMPLE
    #------------------------------------------
    # Define the X label
    X_label = [
        "raw_feature_creator_follower_count",
        "raw_feature_creator_following_count",
        "raw_feature_engager_follower_count",
        "raw_feature_engager_following_count",
        "tweet_feature_number_of_photo",
        "tweet_feature_number_of_video",
        "tweet_feature_number_of_gif",
        "tweet_feature_is_retweet",
        "tweet_feature_is_quote",
        "tweet_feature_is_top_level",
        "tweet_feature_number_of_hashtags",
        "tweet_feature_creation_timestamp_hour",
        "tweet_feature_creation_timestamp_week_day"
        # "tweet_feature_number_of_mentions",
        # "engager_feature_number_of_previous_like_engagement",
        # "engager_feature_number_of_previous_reply_engagement",
        # "engager_feature_number_of_previous_retweet_engagement",
        # "engager_feature_number_of_previous_comment_engagement",
        # "engager_feature_number_of_previous_positive_engagement",
        # "engager_feature_number_of_previous_negative_engagement",
        # "engager_feature_number_of_previous_engagement",
        # "engager_feature_number_of_previous_like_engagement_ratio",
        # "engager_feature_number_of_previous_reply_engagement_ratio",
        # "engager_feature_number_of_previous_retweet_engagement_ratio",
        # "engager_feature_number_of_previous_comment_engagement_ratio",
        # "engager_feature_number_of_previous_positive_engagement_ratio",
        # "engager_feature_number_of_previous_negative_engagement_ratio"
    ]
    # Define the Y label
    Y_label = [
        "tweet_feature_engagement_is_like"
    ]

    #Name of the model eg. xgboost_classifier
    model_name="xgboost_classifier"
    #Kind of prediction eg. "like"
    kind = "LIKE"

    train_batch_n_split = 1
    val_batch_n_split = 5

    train_dataset_id = "train_days_1"
    val_dataset_id = "val_days_2"

    # for i in tqdm(range(train_batch_n_split)):
    #     X_train, Y_train = Data.get_dataset_xgb_batch(train_batch_n_split, i, train_dataset_id, X_label, Y_label)
    #     skd.dump_svmlight_file(X_train, Y_train[Y_label[0]].array, f"train_batch_{i}.svm")
    #
    # for i in tqdm(range(val_batch_n_split)):
    #     X_train, Y_train = Data.get_dataset_xgb_batch(val_batch_n_split, i, val_dataset_id, X_label, Y_label)
    #     skd.dump_svmlight_file(X_train, Y_train[Y_label[0]].array, f"val_batch_{i}.svm")

    #Declaring optimizer
    OP = Optimizer(model_name,
                   kind,
                   mode=3,
                   make_log=True,
                   make_save=False,
                   auto_save=False)

    OP.setParameters(n_calls=300, n_random_starts=30)
    OP.defineMI()
    OP.MI.setExtMemTrainPaths([
        f"train_batch_{i}.svm" for i in range(train_batch_n_split)
    ])
    OP.MI.setExtMemValPaths([
        f"val_batch_{i}.svm" for i in range(val_batch_n_split)
    ])
    OP.optimize()
    #------------------------------------------



if __name__ == "__main__":
    main()
