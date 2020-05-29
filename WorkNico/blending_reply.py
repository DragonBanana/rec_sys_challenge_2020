import time
from ParamTuning.Optimizer import Optimizer
from Utils.Data import Data
import pandas as pd
from Utils.Data.Features.Generated.EnsemblingFeature.LGBMEnsemblingFeature import LGBMEnsemblingFeature
from sklearn.model_selection import train_test_split
import time


def main():
    features = [
        "raw_feature_creator_follower_count",  # 0
        "raw_feature_creator_following_count",  # 1
        "raw_feature_engager_follower_count",  # 2
        "raw_feature_engager_following_count",  # 3
        "raw_feature_creator_is_verified",  # 4 CATEGORICAL
        "raw_feature_engager_is_verified",  # 5 CATEGORICAL
        "raw_feature_engagement_creator_follows_engager",  # 6 CATEGORICAL
        "tweet_feature_number_of_photo",  # 7
        "tweet_feature_number_of_video",  # 8
        "tweet_feature_number_of_gif",  # 9
        "tweet_feature_number_of_media",  # 10
        "tweet_feature_is_retweet",  # 11 CATEGORICAL
        "tweet_feature_is_quote",  # 12 CATEGORICAL
        "tweet_feature_is_top_level",  # 13 CATEGORICAL
        "tweet_feature_number_of_hashtags",  # 14
        "tweet_feature_creation_timestamp_hour",  # 15
        "tweet_feature_creation_timestamp_week_day",  # 16
        "tweet_feature_number_of_mentions",  # 17
        "engager_feature_number_of_previous_like_engagement",  # 18
        "engager_feature_number_of_previous_reply_engagement",  # 19
        "engager_feature_number_of_previous_retweet_engagement",  # 20
        "engager_feature_number_of_previous_comment_engagement",  # 21
        "engager_feature_number_of_previous_positive_engagement",  # 22
        "engager_feature_number_of_previous_negative_engagement",  # 23
        "engager_feature_number_of_previous_engagement",  # 24
        "engager_feature_number_of_previous_like_engagement_ratio_1",  # 25
        "engager_feature_number_of_previous_reply_engagement_ratio_1",  # 26
        "engager_feature_number_of_previous_retweet_engagement_ratio_1",  # 27
        "engager_feature_number_of_previous_comment_engagement_ratio_1",  # 28
        "engager_feature_number_of_previous_positive_engagement_ratio_1",  # 29
        "engager_feature_number_of_previous_negative_engagement_ratio_1",  # 30
        "hashtag_similarity_fold_ensembling_positive",  # 31
        "link_similarity_fold_ensembling_positive",  # 32
        "domain_similarity_fold_ensembling_positive",  # 33
        "tweet_feature_creation_timestamp_hour_shifted",  # 34 CATEGORICAL
        "tweet_feature_creation_timestamp_day_phase",  # 35 CATEGORICAL
        "tweet_feature_creation_timestamp_day_phase_shifted"  # 36 CATEGORICAL
    ]

    label = [
        "tweet_feature_engagement_is_reply"
    ]

    train_dataset = "holdout/train"
    val_dataset = "holdout/test"
    test_dataset = "test"

    # PARAM DICTS
    # 16
    '''
    LGBM-reply-opt-latest
    ITERATION NUMBER 16
    
    num_leaves= 2881
    
    learning rate= 0.08758677650601243
    
    max_depth= 16
    
    lambda_l1= 4.046966260480756
    
    lambda_l2= 18.215469571026347
    
    colsample_bynode= 0.40991913646014816
    
    colsample_bytree= 0.8760714468777756
    
    pos_subsample= 0.8420326910275198
    
    neg_subsample= 0.6682094774884858
    
    bagging_freq= 4
    
    max_bin= 2959
    
    min_data_in_leaf= 782
    
    best_es_iteration: 100
    '''
    param_dict_1 = {"num_iterations": 100,
                    "num_leaves": 2881,
                    "learning_rate": 0.08758677650601243,
                    "max_depth": 16,
                    "lambda_l1": 4.046966260480756,
                    "lambda_l2": 18.215469571026347,
                    "colsample_bynode": 0.40991913646014816,
                    "colsample_bytree": 0.8760714468777756,
                    "pos_subsample": 0.8420326910275198,
                    "neg_subsample": 0.6682094774884858,
                    "bagging_freq": 4,
                    "max_bin": 2959,
                    "min_data_in_leaf": 782}
    '''
    LGBM-reply-opt-latest
    ITERATION NUMBER 87
    
    num_leaves= 1365
    
    learning rate= 0.09873269569395177
    
    max_depth= 35
    
    lambda_l1= 28.9780315731426
    
    lambda_l2= 43.29727494688447
    
    colsample_bynode= 0.5355879177924052
    
    colsample_bytree= 0.6872796168633171
    
    pos_subsample= 0.6453155116796576
    
    neg_subsample= 0.6593317435851668
    
    bagging_freq= 1
    
    max_bin= 4954
    
    min_data_in_leaf= 1478
    
    LGBM-reply-opt-latest
    -------
    EXECUTION TIME: 751.4716079235077
    -------
    best_es_iteration: 100
    -------
    PRAUC = 0.15548183660740444
    RCE   = 16.449036928227756
    '''
    param_dict_2 = {"num_iterations": 100,
                    "num_leaves": 35,
                    "learning_rate": 0.09873269569395177,
                    "max_depth": 16,
                    "lambda_l1": 28.9780315731426,
                    "lambda_l2": 43.29727494688447,
                    "colsample_bynode": 0.5355879177924052,
                    "colsample_bytree": 0.6872796168633171,
                    "pos_subsample": 0.6453155116796576,
                    "neg_subsample": 0.6593317435851668,
                    "bagging_freq": 1,
                    "max_bin": 4954,
                    "min_data_in_leaf": 1478}

    '''
    LGBM-reply-opt-latest
    ITERATION NUMBER 76
    
    num_leaves= 2136
    
    learning rate= 0.033200228961358734
    
    max_depth= 31
    
    lambda_l1= 21.189321612695124
    
    lambda_l2= 2.134035022515942
    
    colsample_bynode= 0.9364940987306792
    
    colsample_bytree= 0.7433032763839553
    
    pos_subsample= 0.5933369324600773
    
    neg_subsample= 0.6040860424150478
    
    bagging_freq= 7
    
    max_bin= 2740
    
    min_data_in_leaf= 817
    
    LGBM-reply-opt-latest
    -------
    EXECUTION TIME: 711.1479604244232
    -------
    best_es_iteration: 100
    -------
    PRAUC = 0.1533623977112988
    RCE   = 16.106309000275587

    '''
    param_dict_3 = {"num_iterations": 100,
                    "num_leaves": 2136,
                    "learning_rate": 0.033200228961358734,
                    "max_depth": 31,
                    "lambda_l1": 21.189321612695124,
                    "lambda_l2": 2.134035022515942,
                    "colsample_bynode": 0.9364940987306792,
                    "colsample_bytree": 0.7433032763839553,
                    "pos_subsample": 0.8420326910275198,
                    "neg_subsample": 0.6682094774884858,
                    "bagging_freq": 7,
                    "max_bin": 2740,
                    "min_data_in_leaf": 817}

    # categorical_features_set = {4, 5, 6, 11, 12, 13, 43, 44, 45, 46, 47}
    # categorical_features_set = set([])
    categorical_features_set = {4, 5, 6, 11, 12, 13, 34, 35, 36}

    # Load train data
    loading_data_start_time = time.time()
    df_train, df_train_label = Data.get_dataset_xgb(train_dataset, features, label)
    print(f"Loading train data time: {loading_data_start_time - time.time()} seconds")

    # Load val data
    df_val, df_val_label = Data.get_dataset_xgb(val_dataset, features, label)

    # Load test data
    df_test = Data.get_dataset(features, test_dataset)

    new_index = pd.Series(df_test.index).map(lambda x: x + len(df_val))
    df_test.set_index(new_index, inplace=True)

    # df to be predicted by the lgbm blending feature
    df_to_predict = pd.concat([df_val, df_test])

    # BLENDING FEATURE DECLARATION

    # Load blending feature data
    first_feature_start_time = time.time()
    feat_1 = LGBMEnsemblingFeature(dataset_id=train_dataset,
                                  df_train=df_train,
                                  df_train_label=df_train_label,
                                  df_to_predict=df_to_predict,
                                  param_dict=param_dict_1,
                                  categorical_features_set=categorical_features_set)

    df_feat_1 = feat_1.load_or_create()

    first_feature_end_time = time.time()
    print(f"Loading first feature time: {first_feature_end_time - first_feature_start_time} seconds")

    feat_2 = LGBMEnsemblingFeature(dataset_id=train_dataset,
                                   df_train=df_train,
                                   df_train_label=df_train_label,
                                   df_to_predict=df_to_predict,
                                   param_dict=param_dict_2,
                                   categorical_features_set=categorical_features_set)

    df_feat_2 = feat_2.load_or_create()

    second_feature_end_time = time.time()
    print(f"Loading second feature time: {second_feature_end_time -first_feature_end_time} seconds")

    feat_3 = LGBMEnsemblingFeature(dataset_id=train_dataset,
                                   df_train=df_train,
                                   df_train_label=df_train_label,
                                   df_to_predict=df_to_predict,
                                   param_dict=param_dict_3,
                                   categorical_features_set=categorical_features_set)

    df_feat_3 = feat_3.load_or_create()

    third_feature_end_time = time.time()
    print(f"Loading third feature time: {third_feature_end_time - second_feature_end_time} seconds")

    # check dimensions
    len_val = len(df_val)
    assert len(df_feat_1) == len(df_feat_2) == len(df_feat_3) == (len_val + len(df_test)), \
        f"Blending features are not of dimension expected, len val: {len_val} len test: {len(df_test)}\n " \
        f"obtained len1: {len(df_feat_1)}\n" \
        f"len2: {len(df_feat_2)}\n" \
        f"len3: {len(df_feat_3)}\n"

    # split feature dataframe in validation and testing
    df_feat_1_val = df_feat_1.iloc[:len_val]
    df_feat_1_test = df_feat_1.iloc[len_val:]

    df_feat_2_val = df_feat_2.iloc[:len_val]
    df_feat_2_test = df_feat_2.iloc[len_val:]

    df_feat_3_val = df_feat_3.iloc[:len_val]
    df_feat_3_test = df_feat_3.iloc[len_val:]

    # creating the new validation set on which we will do meta optimization
    df_val = pd.concat([df_val, df_feat_1_val, df_feat_2_val, df_feat_3_val, df_val_label], axis=1)

    # now we are in full meta-model mode
    # watchout! they are unsorted now, you got to re-sort the dfs
    df_metatrain, df_metaval = train_test_split(df_val, test_size=0.3)
    df_metatrain.sort_index(inplace=True)
    df_metaval.sort_index(inplace=True)

    # split dataframe columns in train and label
    extended_features = features + [df_feat_1.columns[0], df_feat_2.columns[0], df_feat_3.columns[0]]

    df_metatrain_label = df_metatrain[label]
    df_metatrain = df_metatrain[extended_features]

    df_metaval_label = df_metaval[label]
    df_metaval = df_metaval[extended_features]

    model_name = "lightgbm_classifier"
    kind = "reply"

    OP = Optimizer(model_name,
                   kind,
                   mode=0,
                   path="reply",
                   path_log="#blending-lgbm-reply",
                   make_log=True,
                   make_save=False,
                   auto_save=False
                   )

    OP.setParameters(n_calls=100, n_random_starts=20)
    OP.loadTrainData(df_metatrain, df_metatrain_label)

    OP.loadValData(df_metaval, df_metaval_label)  # early stopping

    OP.loadTestData(df_metaval, df_metaval_label)  # evaluate objective

    OP.setParamsLGB(objective='binary', early_stopping_rounds=10, eval_metric="binary", is_unbalance=True)
    OP.setCategoricalFeatures(categorical_features_set)
    # OP.loadModelHardCoded()
    res = OP.optimize()


if __name__ == '__main__':
    main()
