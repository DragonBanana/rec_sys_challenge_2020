import pandas as pd
import numpy as np
from Models.GBM.LightGBM import LightGBM
import time
from Utils.Data import Data
from Utils.Submission.Submission import create_submission_file
from Utils.Data.Data import oversample
from Utils.TelegramBot import telegram_bot_send_update

if __name__ == '__main__':
    train_dataset = "holdout/train"
    val_dataset = "holdout/test"
    test_dataset="test"

    # Define the X label
    X_label = [
            "raw_feature_creator_follower_count",                       #0                                                                                   
            "raw_feature_creator_following_count",                      #1                
            "raw_feature_engager_follower_count",                       #2                
            "raw_feature_engager_following_count",                      #3                
            "raw_feature_creator_is_verified",                      #4 CATEGORICAL            
            "raw_feature_engager_is_verified",                      #5 CATEGORICAL            
            "raw_feature_engagement_creator_follows_engager",                       #6 CATEGORICAL                            
            "tweet_feature_number_of_photo",                        #7            
            "tweet_feature_number_of_video",                        #8            
            "tweet_feature_number_of_gif",                      #9        
            "tweet_feature_number_of_media",                        #10            
            "tweet_feature_is_retweet",                     #11 CATEGORICAL    
            "tweet_feature_is_quote",                       #12 CATEGORICAL    
            "tweet_feature_is_top_level",                       #13 CATEGORICAL        
            "tweet_feature_number_of_hashtags",                     #14            
            "tweet_feature_creation_timestamp_hour",                        #15                    
            "tweet_feature_creation_timestamp_week_day",                        #16                       
            "tweet_feature_number_of_mentions",                     #17            
            "engager_feature_number_of_previous_like_engagement",                       #18                                
            "engager_feature_number_of_previous_reply_engagement",                      #19                                
            "engager_feature_number_of_previous_retweet_engagement",                        #20                                    
            "engager_feature_number_of_previous_comment_engagement",                        #21                                  
            "engager_feature_number_of_previous_positive_engagement",                       #22                                    
            "engager_feature_number_of_previous_negative_engagement",                       #23                                    
            "engager_feature_number_of_previous_engagement",                        #24                            
            "engager_feature_number_of_previous_like_engagement_ratio_1",                       #25                                    
            "engager_feature_number_of_previous_reply_engagement_ratio_1",                      #26                                        
            "engager_feature_number_of_previous_retweet_engagement_ratio_1",                        #27                                        
            "engager_feature_number_of_previous_comment_engagement_ratio_1",                        #28                                        
            "engager_feature_number_of_previous_positive_engagement_ratio_1",                       #29                                        
            "engager_feature_number_of_previous_negative_engagement_ratio_1",                       #30                                        
            "engager_feature_number_of_previous_like_engagement_between_creator_and_engager_by_creator",                        #31                                                                        
            "engager_feature_number_of_previous_reply_engagement_between_creator_and_engager_by_creator",                       #32                                                                        
            "engager_feature_number_of_previous_retweet_engagement_between_creator_and_engager_by_creator",                     #33                                                                        
            "engager_feature_number_of_previous_comment_engagement_between_creator_and_engager_by_creator",                     #34                                                                        
            "engager_feature_number_of_previous_negative_engagement_between_creator_and_engager_by_creator",                        #35                                                                            
            "engager_feature_number_of_previous_positive_engagement_between_creator_and_engager_by_creator",                        #36                                                                            
            "engager_feature_number_of_previous_like_engagement_between_creator_and_engager_by_engager",                        #37                                                                        
            "engager_feature_number_of_previous_reply_engagement_between_creator_and_engager_by_engager",                       #38                                                                        
            "engager_feature_number_of_previous_retweet_engagement_between_creator_and_engager_by_engager",                     #39                                                                        
            "engager_feature_number_of_previous_comment_engagement_between_creator_and_engager_by_engager",                     #40                                                                        
            "engager_feature_number_of_previous_negative_engagement_between_creator_and_engager_by_engager",                        #41                                                                            
            "engager_feature_number_of_previous_positive_engagement_between_creator_and_engager_by_engager",                        #42                                                                            
            # "engager_main_language",                      #43 CATEGORICAL    
            # "creator_main_language",                      #44 CATEGORICAL    
            #"creator_and_engager_have_same_main_language",                      #45 CATEGORICAL       - 43                 
            #"is_tweet_in_creator_main_language",                        #46 CATEGORICAL                -44
            #"is_tweet_in_engager_main_language",                        #47 CATEGORICAL                45
            #"statistical_probability_main_language_of_engager_engage_tweet_language_1",                     #48     46                                               
            #"statistical_probability_main_language_of_engager_engage_tweet_language_2",                     #49     47                                              
            # "tweet_feature_token_length",                     #50 CATEGORICAL     
            # "tweet_feature_token_length_unique",                  
            # "engager_feature_knows_hashtag_positive",                 
            # "engager_feature_knows_hashtag_negative",                 
            # "engager_feature_knows_hashtag_like",                 
            # "engager_feature_knows_hashtag_reply",                    
            # "engager_feature_knows_hashtag_rt",                   
            "hashtag_similarity_fold_ensembling_positive",                  #48
            "link_similarity_fold_ensembling_positive",                 #49
            "domain_similarity_fold_ensembling_positive",               #50
            "tweet_feature_creation_timestamp_hour_shifted",            #51 
            "tweet_feature_creation_timestamp_day_phase",               #52 
            "tweet_feature_creation_timestamp_day_phase_shifted"        #53 
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
    # If oversample is set
    # Oversample the cold users
    use_oversample = True
    os_column_name = "engager_feature_number_of_previous_positive_engagement_ratio_1"
    os_value = -1
    os_percentage = 0.3                      #in order to have the 23% of cold users in the validation set
    if use_oversample is True:
        df = pd.concat([X_val, Y_val], axis=1)
        oversampled_df = oversample(df, os_column_name, os_value, os_percentage)
        X_val = oversampled_df[X_label]
        Y_val = oversampled_df[Y_label]
        del df, oversampled_df

    # Load local_test data
    X_local, Y_local = Data.get_dataset_xgb_batch(2, 1, val_dataset, X_label, Y_label, 1)
    # If oversample is set
    # Oversample the cold users
    use_oversample = True
    os_column_name = "engager_feature_number_of_previous_positive_engagement_ratio_1"
    os_value = -1
    os_percentage = 0.3
    if use_oversample is True:
        df = pd.concat([X_local, Y_local], axis=1)
        oversampled_df = oversample(df, os_column_name, os_value, os_percentage)
        X_local = oversampled_df[X_label]
        Y_local = oversampled_df[Y_label]
        del df, oversampled_df

    # Load test data
    X_test = Data.get_dataset(X_label, test_dataset)

    print(f"Loading data time: {time.time() - loading_data_start_time} seconds")

    LGBM = LightGBM(
        objective         =     'binary',
        num_threads       =     48,
        num_iterations    =     1000,
        num_leaves        =     760,
        learning_rate     =     0.10712396437981637,
        max_depth         =     16,
        lambda_l1         =     32.055790862954424,
        lambda_l2         =     2.9388961426221925,
        colsample_bynode  =     0.5015430512404712,
        colsample_bytree  =     0.400613072648635,
        pos_subsample     =     0.37628392502803776,
        neg_subsample     =     0.3982056057899175,
        bagging_freq      =     2,
        max_bin           =     3512,
        min_data_in_leaf  =     875,
        early_stopping_rounds=15
        )

    # LGBM Training
    training_start_time = time.time()
    LGBM.fit(X=X_train, Y=Y_train, X_val=X_val, Y_val=Y_val, categorical_feature=set([4,5,6,11,12,13]))
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

    submission_filename = "lgbm_like_submission_fresco_sovracampione.csv"
    create_submission_file(tweets, users, predictions, submission_filename)
    
    ip="3.250.69.182"
    telegram_bot_send_update(f"@lucaconterio la submission è pronta! IP: {ip}, nome del file: {submission_filename}")