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
    train_dataset = "cold_train"
    test_dataset = "cold_test"

    # Define the X label
    X_label = [
            "raw_feature_creator_follower_count",                       #0                                                                                   
            "raw_feature_creator_following_count",                      #1                
            "raw_feature_engager_follower_count",                       #2                
            "raw_feature_engager_following_count",                      #3                
            "raw_feature_creator_is_verified",                          #4          
            "raw_feature_engager_is_verified",                          #5            
            "raw_feature_engagement_creator_follows_engager",           #6                            
            "tweet_feature_number_of_photo",                            #7            
            "tweet_feature_number_of_video",                            #8            
            "tweet_feature_number_of_gif",                              #9        
            "tweet_feature_number_of_media",                            #10            
            "tweet_feature_is_retweet",                                 #11  
            "tweet_feature_is_quote",                                   #12    
            "tweet_feature_is_top_level",                               #13        
            "tweet_feature_number_of_hashtags",                         #14            
            "tweet_feature_creation_timestamp_hour",                    #15                    
            "tweet_feature_creation_timestamp_hour_shifted",            #16                    
            "tweet_feature_creation_timestamp_week_day",                #17                       
            "tweet_feature_creation_timestamp_day_phase",               #18                       
            "tweet_feature_creation_timestamp_day_phase_shifted",       #19                       
            "tweet_feature_number_of_mentions",                         #20                                                                       
            "tweet_feature_token_length",                               #21                                                                       
    ]                                                                           
    # Define the Y label
    Y_label = [
        "tweet_feature_engagement_is_like"
    ]

    model_name = "lightgbm_classifier_cold"
    kind = "like"

    # Load train data
    loading_data_start_time = time.time()
    X_train, Y_train = Data.get_dataset_xgb_batch(1, 0, train_dataset, X_label, Y_label, 0.30)

    # Load test data
    X_val, Y_val = Data.get_dataset_xgb_batch(2, 0, test_dataset, X_label, Y_label, 1)

    X_test, Y_test = Data.get_dataset_xgb_batch(2, 1, test_dataset, X_label, Y_label, 1)
    
    print(f"Loading data time: {time.time() - loading_data_start_time} seconds")

    OP = Optimizer(model_name, 
                   kind,
                   mode=0,
                   path="like",
                   path_log="LGBM-like-cold_users",
                   make_log=True, 
                   make_save=False, 
                   auto_save=False)

    OP.setParameters(n_calls=40, n_random_starts=15)
    OP.loadTrainData(X_train, Y_train)
    OP.loadTestData(X_test, Y_test)
    OP.loadValData(X_val, Y_val)
    OP.setParamsLGB(objective='binary',early_stopping_rounds=15, eval_metric="binary",is_unbalance=False)
    OP.setCategoricalFeatures(set([]))
    res=OP.optimize()   


if __name__ == "__main__":
    main()