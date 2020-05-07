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
    train_dataset = "train_days_12345"
    test_dataset = "val_days_6"

    # Define the X label
    X_label = [
        "raw_feature_creator_follower_count",                                       # 0                                                    
        "raw_feature_creator_following_count",                                      # 1               
        "raw_feature_engager_follower_count",                                       # 2               
        "raw_feature_engager_following_count",                                      # 3               
        "tweet_feature_number_of_photo",                                            # 4           
        "tweet_feature_number_of_video",                                            # 5           
        "tweet_feature_number_of_gif",                                              # 6       
        "tweet_feature_is_retweet",                                                 # 7(categorical) 
        "tweet_feature_is_quote",                                                   # 8(categorical   
        "tweet_feature_is_top_level",                                               # 9(categorical      
        "tweet_feature_number_of_hashtags",                                         # 10          
        "tweet_feature_creation_timestamp_hour",                                    # 11                 
        "tweet_feature_creation_timestamp_week_day",                                # 12                       
        "tweet_feature_number_of_mentions",                                         # 13           
        "engager_feature_number_of_previous_like_engagement",                       # 14                               
        "engager_feature_number_of_previous_reply_engagement",                      # 15                               
        "engager_feature_number_of_previous_retweet_engagement",                    # 16                                   
        "engager_feature_number_of_previous_comment_engagement",                    # 17                                  
        "engager_feature_number_of_previous_positive_engagement",                   # 18                                   
        "engager_feature_number_of_previous_negative_engagement",                   # 19                                   
        "engager_feature_number_of_previous_engagement",                            # 20                           
        "engager_feature_number_of_previous_like_engagement_ratio",                 # 21                                   
        "engager_feature_number_of_previous_reply_engagement_ratio",                # 22                                       
        "engager_feature_number_of_previous_retweet_engagement_ratio",              # 23                                       
        "engager_feature_number_of_previous_comment_engagement_ratio",              # 24                                       
        "engager_feature_number_of_previous_positive_engagement_ratio",             # 25                                       
        "engager_feature_number_of_previous_negative_engagement_ratio"              # 26                                       
    ]
    # Define the Y label
    Y_label = [
        "tweet_feature_engagement_is_like"
    ]

    model_name = "lightgbm_classifier"
    kind = "like"

    # Load train data
    loading_data_start_time = time.time()
    X_train, Y_train = Data.get_dataset_xgb_batch(1, 0, train_dataset, X_label, Y_label, 0.25)

    # Load test data
    X_val, Y_val = Data.get_dataset_xgb_batch(2, 0, test_dataset, X_label, Y_label, 1)
    X_test, Y_test = Data.get_dataset_xgb_batch(2, 1, test_dataset, X_label, Y_label, 1)

    print(f"Loading data time: {time.time() - loading_data_start_time} seconds")

    OP = Optimizer(model_name, 
                   kind,
                   mode=0,
                   path="like",
                   path_log="llgbm-like\nCon subsample del dataset per fare più in fretta.\nTrainato su 12345, ES su metà del giorno 6, test sull'altra\
                             metà del giorno 6. Trovato questo minimo lo testeremo sul giorno 7, mai visto durante il training.\nSTAVOLTA GIUSTO.",
                   make_log=True, 
                   make_save=False, 
                   auto_save=False)

    OP.setParameters(n_calls=90, n_random_starts=35)
    OP.loadTrainData(X_train, Y_train)
    OP.loadTestData(X_test, Y_test)
    OP.loadValData(X_val, Y_val)
    OP.setParamsLGB(objective='binary',early_stopping_rounds=15, eval_metric="binary",is_unbalance=False)
    OP.setCategoricalFeatures(set([7,8,9]))
    #OP.loadModelHardCoded()
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