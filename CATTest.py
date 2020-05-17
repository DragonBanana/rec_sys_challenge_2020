import pandas as pd
import numpy as np
from Models.GBM.CatBoost import CatBoost
from catboost import Pool
import time
from Utils.Data import Data
from Utils.Submission.Submission import create_submission_file

if __name__ == '__main__':
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
    kind="like"
    cat = [4,5,6,11,12,13,43,44,45,46,47,50]

    '''
    # Load train data
    loading_data_start_time = time.time()
    X_train, Y_train = Data.get_dataset_xgb(train_dataset, X_label, Y_label)

    # Load val data
    X_val, Y_val= Data.get_dataset_xgb(val_dataset, X_label, Y_label)

    # Load local_test data
    #X_local, Y_local= Data.get_dataset_xgb(local_test_set, X_label, Y_label)

    # Load test data
    X_test = Data.get_dataset(X_label, test_dataset)
    '''
    # Load train data
    loading_data_start_time = time.time()
    X_train, Y_train = Data.get_dataset_xgb_batch(1, 0, train_dataset, X_label, Y_label, 0.50)

    # Load test data
    X_val, Y_val = Data.get_dataset_xgb_batch(2, 0, test_dataset, X_label, Y_label, 1)
    X_test, Y_test = Data.get_dataset_xgb_batch(2, 1, test_dataset, X_label, Y_label, 1)

    print(f"Loading data time: {time.time() - loading_data_start_time} seconds")

    #Initialize Model
    CAT = CatBoost(iterations=600,
                   depth=16,
                   learning_rate=1.0,
                   l2_leaf_reg=10.0,
                   subsample=0.9,
                   random_strenght=30,
                   colsample_bylevel=1.0,
                   leaf_estimation_iterations=10,
                   scale_pos_weight=1.0,
                   model_shrink_rate=0.03948556779496452,
                   #ES
                   early_stopping_rounds = 15
                )

    # CAT Training
    training_start_time = time.time()
    #Creating Pools to feed the model
    train = Pool(X_train.to_numpy(), label=Y_train.to_numpy(), cat_features=cat)
    val = Pool(X_val.to_numpy(), label=Y_val.to_numpy(), cat_features=cat)
    #Fitting the model
    CAT.fit(pool_train=train, pool_val=val)
    print(f"Training time: {time.time() - training_start_time} seconds")

    '''
    # LGBM Evaluation
    evaluation_start_time = time.time()
    evals = Pool(X_local.to_numpy(), Y_local.to_numpy().astype(np.int32), cat_features=cat)
    prauc, rce, conf, max_pred, min_pred, avg = CAT.evaluate(evals)
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
    '''
    

    tweets = Data.get_feature("raw_feature_tweet_id", test_dataset)["raw_feature_tweet_id"].array
    users = Data.get_feature("raw_feature_engager_id", test_dataset)["raw_feature_engager_id"].array

    # CAT Prediction
    prediction_start_time = time.time()
    predictions = CAT.get_prediction(X_test.to_numpy())
    print(f"Prediction time: {time.time() - prediction_start_time} seconds")

    #Uncomment to print feature importance at the end of training
    #print(CAT.get_feat_importance())

    create_submission_file(tweets, users, predictions, "cat_like_holdout_first.csv")