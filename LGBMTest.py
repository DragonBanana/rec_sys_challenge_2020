import pandas as pd
import numpy as np
from Models.GBM.LightGBM import LightGBM
import time
from Utils.Data import Data
from Utils.Submission.Submission import create_submission_file

if __name__ == '__main__':
    train_dataset = "train_days_12345"
    test_dataset = "val_days_7"

    # Define the X label
    X_label = [
        "raw_feature_creator_follower_count",                       #(0)
        "raw_feature_creator_following_count",                      #(1)
        "raw_feature_creator_is_verified",                          #(2)categorical
        "raw_feature_engager_follower_count",                       #(3)
        "raw_feature_engager_following_count",                      #(4)
        "raw_feature_engager_is_verified",                          #(5)categorical
        "tweet_feature_is_retweet",                                 #(6)categorical
        "tweet_feature_is_quote",                                   #(7)categorical
        "tweet_feature_is_top_level",                               #(8)categorical
        "raw_feature_engagement_creator_follows_engager",           #(9)categorical
        "tweet_feature_creation_timestamp_hour",                    #(10)
        "tweet_feature_creation_timestamp_week_day",                #(11)
        "engager_feature_number_of_previous_like_engagement_ratio",  #(12)
        "engager_feature_number_of_previous_like_engagement",        #(13)
        "engager_feature_number_of_previous_retweet_engagement_ratio",  #(14)
        "engager_feature_number_of_previous_retweet_engagement",        #(15)
        "engager_feature_number_of_previous_comment_engagement_ratio",  #(16)
        "engager_feature_number_of_previous_comment_engagement",        #(17)
        "engager_feature_number_of_previous_reply_engagement_ratio",  #(18)
        "engager_feature_number_of_previous_reply_engagement",        #(19)
        "mapped_feature_tweet_language",                                 #(20)categorical
    ]
    # Define the Y label
    Y_label = [
        "tweet_feature_engagement_is_like"
    ]
    kind="like"

    # Load train data
    loading_data_start_time = time.time()
    X_train, Y_train = Data.get_dataset_xgb(train_dataset, X_label, Y_label)

    # Load test data
    X_test, Y_test = Data.get_dataset_xgb(test_dataset, X_label, Y_label)
    print(f"Loading data time: {time.time() - loading_data_start_time} seconds")

    #Initialize Model
    LGBM = LightGBM(
        num_iterations    =     400,
        num_leaves        =     43,
        learning_rate     =     0.4082108467287959,
        max_depth         =     91,
        lambda_l1         =     0.20520435539282497,
        lambda_l2         =     1.0,
        colsample_bynode  =     0.4463237463387104,
        colsample_bytree  =     0.772727835899646,
        bagging_fraction  =     0.622227405626987,
        pos_subsample     =     0.7188734449859797,
        neg_subsample     =     0.7341754949841821,
        bagging_freq      =     12,
        max_bin           =     423
        )

    # LGBM Training
    training_start_time = time.time()
    LGBM.fit(X_train, Y_train)
    print(f"Training time: {time.time() - training_start_time} seconds")

    # LGBM Evaluation
    evaluation_start_time = time.time()
    prauc, rce, conf, max_pred, min_pred, avg = LGBM.evaluate(X_test.to_numpy(), Y_test.to_numpy())
    print(f"PRAUC:\t{prauc}")
    print(f"RCE:\t{rce}")
    print(f"TN:\t{conf[0,0]}")
    print(f"FP:\t{conf[0,1]}")
    print(f"FN:\t{conf[1,0]}")
    print(f"TP:\t{conf[1,1]}")
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
    LGBM.plot_fimportance()

    #create_submission_file(tweets, users, predictions, "test_submission.csv")