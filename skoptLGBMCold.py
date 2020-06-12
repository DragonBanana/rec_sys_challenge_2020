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
    X_train, Y_train = Data.get_dataset_xgb(train_dataset, X_label, Y_label)

    x = Data.get_feature("mapped_feature_engager_id", train_dataset)
    y = x.groupby("mapped_feature_engager_id").size()
    a_1 = y[y == 1]

    one_interaction_mask_train = x['mapped_feature_engager_id'].isin(set(a_1.index))

    X_val = X_train[one_interaction_mask_train]
    X_train = X_train[~one_interaction_mask_train]


    Y_val = Y_train[one_interaction_mask_train]
    Y_train = Y_train[~one_interaction_mask_train]

    X_train = X_train.sample(frac=0.3,random_state=0)
    Y_train = Y_train.sample(frac=0.3,random_state=0)

    X_val_temp, Y_val_temp = Data.get_dataset_xgb(test_dataset, X_label, Y_label)    

    x_test = Data.get_feature("engager_feature_number_of_previous_positive_engagements_ratio_1", test_dataset)
    
    cold_mask = x_test["engager_feature_number_of_previous_positive_engagements_ratio_1"] == -1

    X_val = pd.concat([X_val,X_val_temp[cold_mask]],axis=0)
    Y_val = pd.concat([Y_val,Y_val_temp[cold_mask]],axis=0)

    X_val_copy=X_val.copy()
    X_val = X_val_copy.sample(frac=0.5, random_state=0)
    X_test = X_val_copy.drop(X_val.index)

    Y_val_copy=Y_val.copy()
    Y_val = Y_val_copy.sample(frac=0.5, random_state=0)
    Y_test = Y_val_copy.drop(Y_val.index)

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