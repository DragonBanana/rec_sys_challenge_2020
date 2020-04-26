import pandas as pd
import numpy as np
from Models.GBM.LightGBM import LightGBM
import time
from Utils.Data import Data
from Utils.Submission.Submission import create_submission_file

if __name__ == '__main__':
    train_dataset = "train_days_123"
    test_dataset = "val_days_4"

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
        #"tweet_feature_number_of_mentions"
    ]
    # Define the Y label
    Y_label = [
        "tweet_feature_engagement_is_like"
    ]

    # Load train data
    loading_data_start_time = time.time()
    X_train, Y_train = Data.get_dataset_xgb(train_dataset, X_label, Y_label)

    # Load test data
    X_test, Y_test = Data.get_dataset_xgb(test_dataset, X_label, Y_label)
    print(f"Loading data time: {time.time() - loading_data_start_time} seconds")

    #Initialize Model
    LGBM = LightGBM(kind="LIKE (TESTING)")
   
    # LGBM Training
    training_start_time = time.time()
    LGBM.fit(X_train, Y_train)
    print(f"Training time: {time.time() - training_start_time} seconds")

    # LGBM Evaluation
    evaluation_start_time = time.time()
    prauc, rce, conf, max_pred, min_pred, avg = LGBM.evaluate(X_test.to_numpy(), Y_test.to_numpy())
    print(f"PRAUC:\t{prauc}")
    print(f"RCE:\t{rce}")
    print(f"TP:\t{conf[0,0]}")
    print(f"FN:\t{conf[0,1]}")
    print(f"FP:\t{conf[1,0]}")
    print(f"TN:\t{conf[1,1]}")
    print(f"MAX_PRED:\t{max_pred}")
    print(f"MIN_PRED:\t{min_pred}")
    print(f"AVG:\t{avg}")
    print(f"Evaluation time: {time.time() - evaluation_start_time} seconds")

    tweets = Data.get_feature("raw_feature_tweet_id", test_dataset)["raw_feature_tweet_id"].array
    users = Data.get_feature("raw_feature_engager_id", test_dataset)["raw_feature_engager_id"].array

    # LGBM Prediction
    prediction_start_time = time.time()
    predictions = LGBM.get_prediction(X_test)
    print(f"Prediction time: {time.time() - prediction_start_time} seconds")

    #Uncomment to plot feature importance at the end of training
    #LGBM.plot_fimportance()

    #create_submission_file(tweets, users, predictions, "test_submission.csv")