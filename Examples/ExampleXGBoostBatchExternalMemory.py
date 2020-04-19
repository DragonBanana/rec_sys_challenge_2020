import time
import pandas as pd
import numpy as np
import sklearn.datasets as skd

from Models.GBM.XGBoost import XGBoost
from Utils.Data import Data
from Utils.Submission.Submission import create_submission_file

if __name__ == '__main__':
    train_dataset = "train_days_1"
    test_dataset = "val_days_2"

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
    ]
    # Define the Y label
    Y_label = [
        "tweet_feature_engagement_is_comment"
    ]

    batch_n_split = 10

    batch_paths = [f"batch_{i}" for i in range(batch_n_split)]

    XGB = XGBoost(num_rounds=10, learning_rate=0.01, batch=True, scale_pos_weight=149)

    for split_n in range(batch_n_split):
        # Load train data
        loading_data_start_time = time.time()
        X_train, Y_train = Data.get_dataset_xgb_batch(batch_n_split, split_n, train_dataset, X_label, Y_label)
        print(f"Loading training data time: {time.time() - loading_data_start_time} seconds")

        loading_data_start_time = time.time()
        skd.dump_svmlight_file(X_train, Y_train[Y_label[0]].array, batch_paths[split_n])
        print(f"Saving SVM training data time: {time.time() - loading_data_start_time} seconds")

    for split_n in range(batch_n_split):
        loading_data_start_time = time.time()
        XGB.fit_external_memory(external_memory = batch_paths[split_n])
        print(f"Training data time: {time.time() - loading_data_start_time} seconds")

    # Load test data
    loading_data_start_time = time.time()
    X_test, Y_test = Data.get_dataset_xgb(test_dataset, X_label, Y_label)
    print(f"Loading test data time: {time.time() - loading_data_start_time} seconds")

    # XGB Evaluation
    evaluation_start_time = time.time()
    prauc, rce, confmat, max_tmp, min_tmp, avg_tmp  = XGB.evaluate(X_test, Y_test)
    print(f"Evaluation time: {time.time() - evaluation_start_time} seconds")

    tweets = Data.get_feature("raw_feature_tweet_id", test_dataset)["raw_feature_tweet_id"].array
    users = Data.get_feature("raw_feature_engager_id", test_dataset)["raw_feature_engager_id"].array

    # XGB Prediction
    prediction_start_time = time.time()
    predictions = XGB.get_prediction(X_test)
    print(f"Prediction time: {time.time() - prediction_start_time} seconds")

    create_submission_file(tweets, users, predictions, "test_submission.csv")
