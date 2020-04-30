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
    # Defining the dataset used
    train_dataset = "train_days_1234"
    test_dataset = "val_days_5"
    val_dataset = "val_days_5"

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
        "raw_feature_engagement_creator_follows_engager",            #(9)categorical
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
        "engager_feature_know_tweet_language"                         #(21)categorical
    ]
    # Define the Y label
    Y_label = [
        "tweet_feature_engagement_is_like"
    ]

    model_name = "lightgbm_classifier"
    kind = "like"

    # Load train data
    loading_data_start_time = time.time()
    X_train, Y_train = Data.get_dataset_xgb(train_dataset, X_label, Y_label)

    # Load test data
    X_test, Y_test = Data.get_dataset_xgb(test_dataset, X_label, Y_label)

    # Load val data
    X_val, Y_val = Data.get_dataset_xgb(val_dataset, X_label, Y_label)
    print(f"Loading data time: {time.time() - loading_data_start_time} seconds")

    OP = Optimizer(model_name, 
                   kind,
                   mode=0,
                   make_log=True, 
                   make_save=False, 
                   auto_save=False)
    OP.setParameters(n_calls=50, n_random_starts=30)
    OP.loadTrainData(X_train, Y_train)
    OP.loadTestData(X_test, Y_test)
    OP.loadValData(X_val, Y_val)
    OP.setParamsLGB(early_stopping_rounds=5, eval_metric="rmsle")
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