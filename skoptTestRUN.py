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


def main():
    #Name of the model eg. xgboost_classifier
    model_name="xgboost_classifier"
    #Kind of prediction eg. "like"
    kind = "LIKE"
    
    OP = Optimizer(model_name, kind, make_log=True, make_save=True)
    OP.setParameters(n_calls=3, n_random_starts=3)
    #OP.loadTrainData(X_train, Y_train)
    #OP.loadTestData(X_test, Y_test)

    # Defining the dataset used
    train_dataset = "train_split_with_timestamp_from_train_random_seed_888_timestamp_threshold_1581465600_holdout_1"
    test_dataset = "val_split_with_timestamp_from_train_random_seed_888_timestamp_threshold_1581465600_holdout_1"

    # Define the X label
    X_label = [
        "raw_feature_creator_follower_count",
        "raw_feature_creator_following_count",
        "raw_feature_creator_is_verified",
        "raw_feature_engager_follower_count",
        "raw_feature_engager_following_count",
        "raw_feature_engager_is_verified",
        "tweet_feature_is_retweet",
        "tweet_feature_is_quote",
        "tweet_feature_is_top_level",
        "raw_feature_engagement_creator_follows_engager"
    ]
    # Define the Y label
    Y_label = [
        "tweet_feature_engagement_is_retweet"
    ]

    # Load train data
    loading_data_start_time = time.time()
    X_train, Y_train = Data.get_dataset_xgb(train_dataset, X_label, Y_label)

    # Load test data
    X_test, Y_test = Data.get_dataset_xgb(test_dataset, X_label, Y_label)
    print(f"Loading data time: {time.time() - loading_data_start_time} seconds")

    OP = Optimizer(model_name, kind, make_log=True, make_save=True)
    OP.setParameters(n_calls=1, n_random_starts=1)
    OP.loadTrainData(X_train, Y_train)
    OP.loadTestData(X_test, Y_test)
    res=OP.optimize()

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
