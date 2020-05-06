import pandas as pd
import numpy as np
from Models.GBM.LightGBM import LightGBM
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

    '''
    PARAMETERS USED FOR LIKE SUBMISSION
    ITERATION NUMBER 30
    num_iterations=             669
    num_leaves=                 56
    learning rate=              0.24415290666671663
    max_depth=                  5
    lambda_l1=                  0.10764946399034334
    lambda_l2=                  1.0
    colsample_bynode=           0.31156106645302606
    colsample_bytree=           0.1
    bagging_fraction=           0.8438671225197703
    pos_subsample=              0.17208466069341272
    neg_subsample=              0.37003597658609644
    bagging_freq=               47
    max_bin=                    3319
    -------
    best_es_iteration: 94
    -------
    PRAUC = 0.7136959333851709
    RCE   = 18.937288652839634
    -------
    TN %  = 50.529%
    FP %  = 10.173%
    FN %  = 17.270%
    TP %  = 22.027%
    -------
    TN    = 9907440
    FP    = 1994715
    FN    = 3386232
    TP    = 4318864
    -------
    PREC    = 0.68406%
    RECALL    = 0.56052%
    F1    = 0.61616%
    -------
    MAX   =0.9918682556033757
    MIN   =0.010975114181572168
    AVG   =0.40627307013611136
    -------
    OBJECTIVE: -13.515465900872789
    '''

    #Initialize Model
    LGBM = LightGBM(
        objective         =     'binary',
        num_threads       =     32,
        num_iterations    =     669,
        num_leaves        =     56,
        learning_rate     =     0.24415290666671663,
        max_depth         =     5,
        lambda_l1         =     0.10764946399034334,
        lambda_l2         =     1.0,
        colsample_bynode  =     0.31156106645302606,
        colsample_bytree  =     0.1,
        bagging_fraction  =     0.8438671225197703,
        pos_subsample     =     0.17208466069341272,
        neg_subsample     =     0.37003597658609644,
        bagging_freq      =     47,
        max_bin           =     3319
        )

    # LGBM Training
    training_start_time = time.time()
    LGBM.fit(X=X_train, Y=Y_train, X_val=X_val, Y_val=Y_val, categorical_feature=set([7,8,9]))
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
    #LGBM.plot_fimportance()

    create_submission_file(tweets, users, predictions, "lgbm_like_submission.csv")