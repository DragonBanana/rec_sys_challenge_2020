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
        "tweet_feature_is_reply",
        "tweet_feature_is_retweet",
        "tweet_feature_is_quote",
        "tweet_feature_is_top_level"
    ]
    # Define the Y label
    Y_label = [
        "tweet_feature_engagement_is_like"
    ]

    #Name of the model eg. xgboost_classifier
    model_name="lightgbm_classifier"
    #Kind of prediction eg. "like"
    kind = "LIKE"
    
    '''
    #Declaring optimizer
    OP = Optimizer(model_name, 
                   kind,
                   mode=1,
                   make_log=True, 
                   make_save=False, 
                   auto_save=False)
    
    OP.setParameters(n_calls=5, n_random_starts=5)
    OP.batchTrain(tot_train_split=5, train_id="train_days_1")
    OP.batchTest(tot_test_split=5, test_id="val_days_2")
    OP.setLabels(X_label, Y_label)
    OP.optimize()
    #------------------------------------------
    '''
    '''
    #------------------------------------------
    #     NESTED CROSS VALIDATION EXAMPLE
    #------------------------------------------
    #Declaring optimizer
    OP = Optimizer(model_name, 
                   kind,
                   mode=2,
                   make_log=True, 
                   make_save=False, 
                   auto_save=False)
    
    OP.setParameters(n_calls=5, n_random_starts=5)
    OP.setLabels(X_label, Y_label)
    OP.optimize()
    #------------------------------------------
    '''


    
    # Defining the dataset used
    train_dataset = "train_days_1"
    test_dataset = "val_days_3"
    val_dataset = "val_days_2"

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
        "tweet_feature_engagement_is_like"
    ]

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
