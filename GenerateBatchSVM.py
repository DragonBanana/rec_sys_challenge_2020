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
import pathlib as pl


def main():

    labels = [
        "like",
        "reply",
        "retweet",
        "comment"
    ]

    train_batch_n_split = 20
    val_batch_n_split = 5

    train_dataset_id = "train_days_123456"
    val_dataset_id = "val_days_7"

    gen_train = True
    gen_val = True

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
        "tweet_feature_creation_timestamp_week_day",
        "tweet_feature_number_of_mentions",
        "engager_feature_number_of_previous_like_engagement",
        "engager_feature_number_of_previous_reply_engagement",
        "engager_feature_number_of_previous_retweet_engagement",
        "engager_feature_number_of_previous_comment_engagement",
        "engager_feature_number_of_previous_positive_engagement",
        "engager_feature_number_of_previous_negative_engagement",
        "engager_feature_number_of_previous_engagement",
        "engager_feature_number_of_previous_like_engagement_ratio",
        "engager_feature_number_of_previous_reply_engagement_ratio",
        "engager_feature_number_of_previous_retweet_engagement_ratio",
        "engager_feature_number_of_previous_comment_engagement_ratio",
        "engager_feature_number_of_previous_positive_engagement_ratio",
        "engager_feature_number_of_previous_negative_engagement_ratio"
    ]

    def gen(label):
        # Define the Y label
        Y_label = [
            f"tweet_feature_engagement_is_{label}"
        ]

        folder = f"{label}"
        pl.Path(folder).mkdir(parents=True, exist_ok=True)

        if gen_train:
            for i in tqdm(range(train_batch_n_split)):
                X_train, Y_train = Data.get_dataset_xgb_batch(train_batch_n_split, i, train_dataset_id, X_label, Y_label)
                skd.dump_svmlight_file(X_train, Y_train[Y_label[0]].array, f"{folder}/train_batch_{i}.svm")

        if gen_val:
            for i in tqdm(range(val_batch_n_split)):
                X_test, Y_test = Data.get_dataset_xgb_batch(val_batch_n_split, i, val_dataset_id, X_label, Y_label)
                skd.dump_svmlight_file(X_test, Y_test[Y_label[0]].array, f"{folder}/val_batch_{i}.svm")

    [gen(label) for label in labels]



if __name__ == "__main__":
    main()

