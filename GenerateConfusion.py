#Importing
import time
import pandas as pd
import numpy as np
from Models.GBM.XGBoost import XGBoost
from Utils.Data import Data
from Utils.Submission.Submission import create_submission_file
from Utils.Eval import ConfMatrix

def main():
    #REPLY MATRIX
    train_dataset = "train"
    test_dataset = "test"

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
        "tweet_feature_number_of_photo",
        "tweet_feature_number_of_video",
        "tweet_feature_number_of_gif",
        "tweet_feature_number_of_media",
        "tweet_feature_number_of_hashtags"
    ]



    #--------------------------------------------------------------------------------
    #REPLY MATRIX
    # Define the Y label
    Y_label = [
        "tweet_feature_engagement_is_reply"
    ]

    # Load train data
    X_train, Y_train = Data.get_dataset_xgb(train_dataset, X_label, Y_label)

    # Load test data
    X_test, Y_test = Data.get_dataset_xgb(test_dataset, X_label, Y_label)

    XGB = XGBoost(
        num_rounds=,
        max_depth=,
        min_child_weight=,
        colsample_bytree=,
        learning_rate=,
        reg_alpha=,
        reg_lambda=,
        scale_pos_weight=,
        gamma=,
        subsample=,
        base_score=,
        batch=True
    )

    X_train_split = np.array_split(X_train, 100)
    Y_train_split = np.array_split(Y_train, 100)

    train_split = zip(X_train_split[:10], Y_train_split[:10])

    for (X,Y) in train_split:
        # XGB Training
        XGB.fit(X, Y)

    # XGB Prediction
    pred = XGB.get_prediction(X_test)

    # Confusion matrix
    tn_reply, fp_reply, fn_reply, tp_reply = confMatrix(Y_test, pred).ravel()



    #--------------------------------------------------------------------------------
    #LIKE MATRIX
    # Define the Y label
    Y_label = [
        "tweet_feature_engagement_is_like"
    ]

    # Load train data
    loading_data_start_time = time.time()
    X_train, Y_train = Data.get_dataset_xgb(train_dataset, X_label, Y_label)

    # Load test data
    X_test, Y_test = Data.get_dataset_xgb(test_dataset, X_label, Y_label)

    XGB = XGBoost(
        num_rounds=,
        max_depth=,
        min_child_weight=,
        colsample_bytree=,
        learning_rate=,
        reg_alpha=,
        reg_lambda=,
        scale_pos_weight=,
        gamma=,
        subsample=,
        base_score=,
        batch=True
    )

    X_train_split = np.array_split(X_train, 100)
    Y_train_split = np.array_split(Y_train, 100)

    train_split = zip(X_train_split[:10], Y_train_split[:10])

    for (X,Y) in train_split:
        # XGB Training
        XGB.fit(X, Y)   

    # XGB Prediction
    pred = XGB.get_prediction(X_test)

    # Confusion matrix
    tn_like, fp_like, fn_like, tp_like = confMatrix(Y_test, pred).ravel()



    #--------------------------------------------------------------------------------
    #RETWEET MATRIX
    # Define the Y label
    Y_label = [
        "tweet_feature_engagement_is_retweet"
    ]

    # Load train data
    loading_data_start_time = time.time()
    X_train, Y_train = Data.get_dataset_xgb(train_dataset, X_label, Y_label)

    # Load test data
    X_test, Y_test = Data.get_dataset_xgb(test_dataset, X_label, Y_label)

    XGB = XGBoost(
        num_rounds=,
        max_depth=,
        min_child_weight=,
        colsample_bytree=,
        learning_rate=,
        reg_alpha=,
        reg_lambda=,
        scale_pos_weight=,
        gamma=,
        subsample=,
        base_score=,
        batch=True
    )

    X_train_split = np.array_split(X_train, 100)
    Y_train_split = np.array_split(Y_train, 100)

    train_split = zip(X_train_split[:10], Y_train_split[:10])

    for (X,Y) in train_split:
        # XGB Training
        XGB.fit(X, Y)   

    # XGB Prediction
    pred = XGB.get_prediction(X_test)

    # Confusion matrix
    tn_retweet, fp_retweet, fn_retweet, tp_retweet = confMatrix(Y_test, pred).ravel()



    #--------------------------------------------------------------------------------
    #COMMENT MATRIX
    # Define the Y label
    Y_label = [
        "tweet_feature_engagement_is_comment"
    ]

    # Load train data
    X_train, Y_train = Data.get_dataset_xgb(train_dataset, X_label, Y_label)

    # Load test data
    X_test, Y_test = Data.get_dataset_xgb(test_dataset, X_label, Y_label)

    XGB = XGBoost(
        num_rounds=,
        max_depth=,
        min_child_weight=,
        colsample_bytree=,
        learning_rate=,
        reg_alpha=,
        reg_lambda=,
        scale_pos_weight=,
        gamma=,
        subsample=,
        base_score=,
        batch=True
    )

    X_train_split = np.array_split(X_train, 100)
    Y_train_split = np.array_split(Y_train, 100)

    train_split = zip(X_train_split[:10], Y_train_split[:10])

    for (X,Y) in train_split:
        # XGB Training
        XGB.fit(X, Y)   

    # XGB Prediction
    pred = XGB.get_prediction(X_test)

    # Confusion matrix
    tn_comment, fp_comment, fn_comment, tp_comment = confMatrix(Y_test, pred).ravel()


if __name__=="__main__":
    main()