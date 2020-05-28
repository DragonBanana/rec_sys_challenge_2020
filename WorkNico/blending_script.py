import time
from ParamTuning.Optimizer import Optimizer
from Utils.Data import Data
import pandas as pd
from Utils.Data.Features.Generated.EnsemblingFeature.LGBMEnsemblingFeature import LGBMEnsemblingFeature
from sklearn.model_selection import train_test_split
import time


def main():
    features = [
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
        "statistical_probability_main_language_of_engager_engage_tweet_language_2"                          #49
    ]

    label = [
        "tweet_feature_engagement_is_reply"
    ]

    train_dataset = "holdout/train"
    val_dataset = "holdout/test"
    test_dataset = "test"

    # PARAM DICTS
    # 91
    param_dict_1 = {"num_iterations": 385,
                  "num_leaves": 49,
                  "learning_rate": 0.019538269336244873,
                  "max_depth": 74, "lambda_l1": 0.1,
                  "lambda_l2": 0.19052504682453342,
                  "colsample_bynode": 0.7231061657217172,
                  "colsample_bytree": 0.4793056076569261,
                  "bagging_fraction": 0.6568652636652283,
                  "pos_subsample": 0.027676620202986415,
                  "neg_subsample": 0.9300456178804056,
                  "bagging_freq": 36,
                  "max_bin": 2327}
    # 92
    param_dict_2 ={"num_iterations": 268,
                   "num_leaves": 48,
                   "learning_rate": 0.01925807472939145,
                   "max_depth": 54,
                   "lambda_l1": 0.15294583081165222,
                   "lambda_l2": 0.19607203711205476,
                   "colsample_bynode": 0.6432327418688643,
                   "colsample_bytree": 0.9057232769698761,
                   "bagging_fraction": 0.7356974582117795,
                   "pos_subsample": 0.03144099582708874,
                   "neg_subsample": 0.9478232498465374,
                   "bagging_freq": 35,
                   "max_bin": 2030}
    # 85
    param_dict_3 = {"num_iterations": 363,
                    "num_leaves": 48,
                    "learning_rate": 0.018928042409241633,
                    "max_depth": 63,
                    "lambda_l1": 0.18997647965029454,
                    "lambda_l2": 0.20810484898050355,
                    "colsample_bynode": 0.21219927478050338,
                    "colsample_bytree": 0.7321086416923552,
                    "bagging_fraction": 0.7488543272279146,
                    "pos_subsample": 0.035496933452999944,
                    "neg_subsample": 0.9770616468788216,
                    "bagging_freq": 33,
                    "max_bin": 1949}

    # categorical_features_set = {4, 5, 6, 11, 12, 13, 43, 44, 45, 46, 47}
    categorical_features_set = set([])

    # Load train data
    loading_data_start_time = time.time()
    df_train, df_train_label = Data.get_dataset_xgb(train_dataset, features, label)
    print(f"Loading train data time: {loading_data_start_time - time.time()} seconds")

    # Load val data
    df_val, df_val_label = Data.get_dataset_xgb(val_dataset, features, label)

    # Load test data
    df_test = Data.get_dataset(features, test_dataset)

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
                   path_log="lgbm-reply",
                   make_log=True,
                   make_save=False,
                   auto_save=False
                   )

    OP.setParameters(n_calls=100, n_random_starts=20)
    OP.loadTrainData(df_metatrain, df_metatrain_label)
    # OP.loadTestData(X_test, Y_test)
    OP.loadValData(df_metaval, df_metaval_label)
    OP.setParamsLGB(objective='binary', early_stopping_rounds=10, eval_metric="binary", is_unbalance=True)
    OP.setCategoricalFeatures(categorical_features_set)
    # OP.loadModelHardCoded()
    res = OP.optimize()


if __name__ == '__main__':
    main()
