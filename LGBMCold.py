import pandas as pd
import numpy as np
from Models.GBM.LightGBM import LightGBM
import time
from Utils.Data import Data
from Utils.Submission.Submission import create_submission_file
from Utils.Data.Data import oversample
from Utils.TelegramBot import telegram_bot_send_update

if __name__ == '__main__':
    train_dataset = "cold_train"
    val_dataset = "cold_test"
    test_dataset="test"

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
            "tweet_feature_token_length",                               #20 
    ]
    # Define the Y label
    Y_label = [
        "tweet_feature_engagement_is_like"
    ]
    kind="like"

    # Load train data
    loading_data_start_time = time.time()
    X_train, Y_train = Data.get_dataset_xgb(train_dataset, X_label, Y_label)

    # Load val data
    X_val, Y_val = Data.get_dataset_xgb_batch(2, 0, val_dataset, X_label, Y_label, 1)

    # Load local_test data
    X_local, Y_local = Data.get_dataset_xgb_batch(2, 1, val_dataset, X_label, Y_label, 1)

    # Load test data
    X_test = Data.get_dataset(X_label, test_dataset)

    print(f"Loading data time: {time.time() - loading_data_start_time} seconds")

    LGBM = LightGBM(
        objective         =     'binary',
        num_threads       =     -1,
        num_iterations    =     1000,
        num_leaves        =     2754, 
        learning_rate     =     0.28615984073261447, 
        max_depth         =     7, 
        lambda_l1         =     27.9468057035752, 
        lambda_l2         =     22.217911321276674, 
        colsample_bynode  =     0.6621896939145201, 
        colsample_bytree  =     0.4497659681733497, 
        bagging_fraction  =     0.25811151715918407, 
        bagging_freq      =     8, 
        max_bin           =     1365, 
        min_data_in_leaf  =     489,
        early_stopping_rounds=15
        )

    # LGBM Training
    training_start_time = time.time()
    #LGBM.fit(X=X_train, Y=Y_train, X_val=X_val, Y_val=Y_val, categorical_feature=set([4,5,6,11,12,13]))
    LGBM.fit(X=X_train, Y=Y_train, X_val=X_val, Y_val=Y_val, categorical_feature=set([]))
    print(f"Training time: {time.time() - training_start_time} seconds")

    # LGBM Evaluation
    evaluation_start_time = time.time()
    prauc, rce, conf, max_pred, min_pred, avg = LGBM.evaluate(X_local.to_numpy(), Y_local.to_numpy())
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
    predictions = LGBM.get_prediction(X_test.to_numpy())
    print(f"Prediction time: {time.time() - prediction_start_time} seconds")

    #Uncomment to plot feature importance at the end of training
    LGBM.plot_fimportance()

    submission_filename = "cold_model_predictions.csv"
    create_submission_file(tweets, users, predictions, submission_filename)
    
    #ip="3.250.69.182"
    #telegram_bot_send_update(f"@lucaconterio la submission è pronta! IP: {ip}, nome del file: {submission_filename}")