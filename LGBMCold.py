import pandas as pd
import numpy as np
from Models.GBM.LightGBM import LightGBM
import time
from Utils.Data import Data
from Utils.Submission.Submission import create_submission_file
from Utils.Data.Data import oversample
from Utils.TelegramBot import telegram_bot_send_update

if __name__ == '__main__':
    train_dataset = "cherry_train"
    val_dataset = "cherry_val"
    test_dataset="new_test"

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

    x = Data.get_feature("mapped_feature_engager_id", train_dataset)
    y = x.groupby("mapped_feature_engager_id").size()
    a_1 = y[y == 1]

    one_interaction_mask_train = x['mapped_feature_engager_id'].isin(set(a_1.index))

    X_val = X_train[one_interaction_mask_train]
    X_train = X_train[~one_interaction_mask_train]


    Y_val = Y_train[one_interaction_mask_train]
    Y_train = Y_train[~one_interaction_mask_train]

    X_val_temp, Y_val_temp = Data.get_dataset_xgb(test_dataset, X_label, Y_label)    

    x_test = Data.get_feature("engager_feature_number_of_previous_positive_engagements_ratio_1", test_dataset)
    
    cold_mask = x_test["engager_feature_number_of_previous_positive_engagements_ratio_1"] == -1

    X_val = pd.concat([X_val,X_val_temp[cold_mask]],axis=0)
    Y_val = pd.concat([Y_val,Y_val_temp[cold_mask]],axis=0)

    # Load test data
    X_test = Data.get_dataset(X_label, test_dataset)

    print(f"Loading data time: {time.time() - loading_data_start_time} seconds")

    param_dict = {
        'objective': "binary",

        'num_threads': -1,

        'num_iterations': 1000,

        'num_leaves': 803, 
        'max_depth': 40, 
        'lambda_l1': 34.569259003648796, 
        'lambda_l2': 16.052913892958095, 
        'colsample_bynode': 0.6211381028707941, 
        'colsample_bytree': 0.7776543289263252, 
        'bagging_fraction': 0.9735959521100843, 
        'bagging_freq': 8, 
        'min_data_in_leaf': 986,

        'early_stopping_rounds': 15
    }

    LGBM = LightGBM(**param_dict)
    # LGBM Training
    training_start_time = time.time()
    #LGBM.fit(X=X_train, Y=Y_train, X_val=X_val, Y_val=Y_val, categorical_feature=set([4,5,6,11,12,13]))
    LGBM.fit(X=X_train, Y=Y_train, X_val=X_val, Y_val=Y_val, categorical_feature=set([]))
    print(f"Training time: {time.time() - training_start_time} seconds")

    # LGBM Evaluation
    evaluation_start_time = time.time()
    prauc, rce, conf, max_pred, min_pred, avg = LGBM.evaluate(X_val.to_numpy(), Y_val.to_numpy())
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