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
    train_dataset = "holdout/train"
    test_dataset = "holdout/test"

    # Define the X label
    X_label = [
        "raw_feature_creator_follower_count",                                                               #0                                                                                   
        "raw_feature_creator_following_count",                                                              #1                
        "raw_feature_engager_follower_count",                                                               #2                
        "raw_feature_engager_following_count",                                                              #3                
        "raw_feature_creator_is_verified",                                                                  #4 CATEGORICAL            
        "raw_feature_engager_is_verified",                                                                  #5 CATEGORICAL            
        "raw_feature_engagement_creator_follows_engager",                                                   #6 CATEGORICAL                            
        "tweet_feature_number_of_photo",                                                                    #7            
        "tweet_feature_number_of_video",                                                                    #8            
        "tweet_feature_number_of_gif",                                                                      #9        
        "tweet_feature_number_of_media",                                                                    #10            
        "tweet_feature_is_retweet",                                                                         #11 CATEGORICAL    
        "tweet_feature_is_quote",                                                                           #12 CATEGORICAL    
        "tweet_feature_is_top_level",                                                                       #13 CATEGORICAL        
        "tweet_feature_number_of_hashtags",                                                                 #14            
        "tweet_feature_creation_timestamp_hour",                                                            #15                    
        "tweet_feature_creation_timestamp_week_day",                                                        #16                       
        "tweet_feature_number_of_mentions",                                                                 #17            
        "engager_feature_number_of_previous_like_engagement",                                               #18                                
        "engager_feature_number_of_previous_reply_engagement",                                              #19                                
        "engager_feature_number_of_previous_retweet_engagement",                                            #20                                    
        "engager_feature_number_of_previous_comment_engagement",                                            #21                                  
        "engager_feature_number_of_previous_positive_engagement",                                           #22                                    
        "engager_feature_number_of_previous_negative_engagement",                                           #23                                    
        "engager_feature_number_of_previous_engagement",                                                    #24                            
        "engager_feature_number_of_previous_like_engagement_ratio",                                         #25                                    
        "engager_feature_number_of_previous_reply_engagement_ratio",                                        #26                                        
        "engager_feature_number_of_previous_retweet_engagement_ratio",                                      #27                                        
        "engager_feature_number_of_previous_comment_engagement_ratio",                                      #28                                        
        "engager_feature_number_of_previous_positive_engagement_ratio",                                     #29                                        
        "engager_feature_number_of_previous_negative_engagement_ratio",                                     #30                                        
        "engager_feature_number_of_previous_like_engagement_between_creator_and_engager_by_creator",        #31                                                                        
        "engager_feature_number_of_previous_reply_engagement_between_creator_and_engager_by_creator",       #32                                                                        
        "engager_feature_number_of_previous_retweet_engagement_between_creator_and_engager_by_creator",     #33                                                                        
        "engager_feature_number_of_previous_comment_engagement_between_creator_and_engager_by_creator",     #34                                                                        
        "engager_feature_number_of_previous_negative_engagement_between_creator_and_engager_by_creator",    #35                                                                            
        "engager_feature_number_of_previous_positive_engagement_between_creator_and_engager_by_creator",    #36                                                                            
        "engager_feature_number_of_previous_like_engagement_between_creator_and_engager_by_engager",        #37                                                                        
        "engager_feature_number_of_previous_reply_engagement_between_creator_and_engager_by_engager",       #38                                                                        
        "engager_feature_number_of_previous_retweet_engagement_between_creator_and_engager_by_engager",     #39                                                                        
        "engager_feature_number_of_previous_comment_engagement_between_creator_and_engager_by_engager",     #40                                                                        
        "engager_feature_number_of_previous_negative_engagement_between_creator_and_engager_by_engager",    #41                                                                            
        "engager_feature_number_of_previous_positive_engagement_between_creator_and_engager_by_engager",    #42                                                                            
        "engager_main_language",                                                                            #43 CATEGORICAL    
        "creator_main_language",                                                                            #44 CATEGORICAL    
        "creator_and_engager_have_same_main_language",                                                      #45 CATEGORICAL                        
        "is_tweet_in_creator_main_language",                                                                #46 CATEGORICAL                
        "is_tweet_in_engager_main_language",                                                                #47 CATEGORICAL                
        "statistical_probability_main_language_of_engager_engage_tweet_language_1",                         #48                                                    
        "statistical_probability_main_language_of_engager_engage_tweet_language_2",                         #49                                                    
        #"tweet_feature_dominant_topic_LDA_15"                                                               #50 CATEGORICAL                             
    ]                                                                           
    # Define the Y label
    Y_label = [
        "tweet_feature_engagement_is_like"
    ]

    model_name = "catboost_classifier"
    kind = "reply"

    print("checkpoint L1")
    # Load train data
    loading_data_start_time = time.time()
    X_train, Y_train = Data.get_dataset_xgb_batch(1, 0, train_dataset, X_label, Y_label, 0.1)
    print("checkpoint L2")
    # Load test data
    X_val, Y_val = Data.get_dataset_xgb_batch(2, 0, test_dataset, X_label, Y_label, 1)
    print("checkpoint L3")
    X_test, Y_test = Data.get_dataset_xgb_batch(2, 1, test_dataset, X_label, Y_label, 1)
    print(f"Loading data time: {time.time() - loading_data_start_time} seconds")



    print("checkpoint 1")
    OP = Optimizer(model_name, 
                   kind,
                   mode=0,
                   path="CatBoostHoldoutReply",
                   path_log="CatBoostHoldoutReply",
                   make_log=True, 
                   make_save=False, 
                   auto_save=True)

    print("checkpoint 2")
    OP.setParameters(n_calls=50, 
                     n_random_starts=20,
                     x0=x,
                     y0=y)
    print("checkpoint 3")
    #Before introducing datasets put the categorical features otherwise they will be ignored.
    OP.setCategoricalFeatures([4,5,6,11,12,13,43,44,45,46,47])
    print("checkpoint 4")
    OP.setParamsCAT(verbosity= 2,
                    boosting_type= "Plain",
                    model_shrink_mode= "Constant",
                    leaf_estimation_method= "Newton",
                    bootstrap_type= "Bernoulli",
                    early_stopping_rounds= 15)
    print("checkpoint 5")
    OP.loadTrainData(X_train, Y_train)
    print("checkpoint 6")
    OP.loadTestData(X_test, Y_test)
    print("checkpoint 7")
    OP.loadValData(X_val, Y_val)
    print("checkpoint 8")
    #OP.setParamsCAT(early_stopping_rounds=15)
    res=OP.optimize()



if __name__ == "__main__":
    main()