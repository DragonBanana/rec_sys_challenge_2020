import pandas as pd
import numpy as np
from Models.GBM.CatBoost import CatBoost
from catboost import Pool
import time
from Utils.Data import Data
from Utils.Submission.Submission import create_submission_file

if __name__ == '__main__':
    train_dataset = "train_days_12345"
    val_dataset = "val_days_6"
    local_test_set = "val_days_7"
    test_dataset="test"

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
    kind="like"

    # Load train data
    loading_data_start_time = time.time()
    X_train, Y_train = Data.get_dataset_xgb(train_dataset, X_label, Y_label)

    # Load val data
    X_val, Y_val= Data.get_dataset_xgb(val_dataset, X_label, Y_label)

    # Load local_test data
    X_local, Y_local= Data.get_dataset_xgb(local_test_set, X_label, Y_label)

    # Load test data
    X_test = Data.get_dataset(X_label, test_dataset)

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

    # LGBM Training
    training_start_time = time.time()
    train = Pool(X_train, label=Y_train, cat_features=set([7,8,9]))
    val = Pool(X_val, label=Y_val, cat_features=set([7,8,9]))
    #X=X_train, Y=Y_train, X_val=X_val, Y_val=Y_val
    CAT.fit(pool_train=train, pool_val=val)
    print(f"Training time: {time.time() - training_start_time} seconds")

    
    # LGBM Evaluation
    evaluation_start_time = time.time()
    prauc, rce, conf, max_pred, min_pred, avg = CAT.evaluate(X_local.to_numpy(), Y_local.to_numpy())
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
    predictions = CAT.get_prediction(X_test.to_numpy())
    print(f"Prediction time: {time.time() - prediction_start_time} seconds")

    #Uncomment to plot feature importance at the end of training
    #LGBM.plot_fimportance()

    create_submission_file(tweets, users, predictions, "cat_like_first_submission.csv")




    #Initialize Model
    CAT = CatBoost(iterations=600,
                   depth=12,
                   learning_rate=0.2032390715790451,
                   l2_leaf_reg=10.0,
                   subsample=0.9,
                   random_strenght=20.816707472109698,
                   colsample_bylevel=0.8645906447696082,
                   leaf_estimation_iterations=52,
                   scale_pos_weight=1.0,
                   model_shrink_rate=0.002086451762532185,
                   #ES
                   early_stopping_rounds = 15
                )

    # CAT Training
    training_start_time = time.time()
    train = Pool(X_train, Y_train)
    val = Pool(X_val, Y_val)
    #X=X_train, Y=Y_train, X_val=X_val, Y_val=Y_val
    CAT.fit(pool_train=train, pool_val=val, cat_feat=set([7,8,9]))
    print(f"Training time: {time.time() - training_start_time} seconds")

    
    # CAT Evaluation
    evaluation_start_time = time.time()
    prauc, rce, conf, max_pred, min_pred, avg = CAT.evaluate(X_local.to_numpy(), Y_local.to_numpy())
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

    # CAT Prediction
    prediction_start_time = time.time()
    predictions = CAT.get_prediction(X_test.to_numpy())
    print(f"Prediction time: {time.time() - prediction_start_time} seconds")

    #Uncomment to plot feature importance at the end of training
    #LGBM.plot_fimportance()

    create_submission_file(tweets, users, predictions, "cat_like_second_submission.csv")




'''

CatBoostSmallerSet
ITERATION NUMBER 61

iterations= 600

depth= 16

learning_rate= 1.0

l2_leaf_reg= 10.0

subsample= 0.9

random_strenght= 30.0

colsample_bylevel= 1.0

leaf_estimation_iterations= 10

scale_pos_weight= 1.0

model_shrink_rate= 0.03948556779496452
CatBoostSmallerSet
-------
best_es_iteration: 42
-------
PRAUC = 0.7356456969450872
RCE   = 17.08616492123297
-------
TN %  = 43.271%
FP %  = 12.332%
FN %  = 17.514%
TP %  = 26.883%
-------
TN    = 4642925
FP    = 1323239
FN    = 1879247
TP    = 2884511
-------
PREC    = 0.68552%
RECALL    = 0.60551%
F1    = 0.64304%
-------
MAX   =0.9991703592829889
MIN   =0.0002690306245719121
AVG   =0.4679609110623672
-------
OBJECTIVE: -12.56936370159913
'''

'''
CatBoostSmallerSet
ITERATION NUMBER 60

iterations= 601

depth= 12

learning_rate= 0.2032390715790451

l2_leaf_reg= 10.0

subsample= 0.9

random_strenght= 20.816707472109698

colsample_bylevel= 0.8645906447696082

leaf_estimation_iterations= 52

scale_pos_weight= 1.0

model_shrink_rate= 0.002086451762532185
CatBoostSmallerSet
-------
best_es_iteration: 453
-------
PRAUC = 0.7442111087000572
RCE   = 18.229543990666976
-------
TN %  = 43.312%
FP %  = 12.291%
FN %  = 17.006%
TP %  = 27.391%
-------
TN    = 4647325
FP    = 1318839
FN    = 1824774
TP    = 2938984
-------
PREC    = 0.69026%
RECALL    = 0.61695%
F1    = 0.65155%
-------
MAX   =0.9997311662166893
MIN   =0.0002822997182683521
AVG   =0.4692099661828366
-------
OBJECTIVE: -13.566629144390737
'''