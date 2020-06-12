import time
from ParamTuning.Optimizer import Optimizer
from Utils.Data import Data
import pandas as pd
from Utils.Data.Features.Generated.EnsemblingFeature.LGBMEnsemblingFeature import LGBMEnsemblingFeature
from sklearn.model_selection import train_test_split
import time


def main():
    features = [
        "raw_feature_creator_follower_count",   # 0
        "raw_feature_creator_following_count",  # 1
        "raw_feature_engager_follower_count",   # 2
        "raw_feature_engager_following_count",  # 3
        "raw_feature_creator_is_verified",      # 4
        "raw_feature_engager_is_verified",      # 5
        "raw_feature_engagement_creator_follows_engager",  # 6
        "tweet_feature_number_of_photo",   # 7
        "tweet_feature_number_of_video",   # 8
        "tweet_feature_number_of_gif",     # 9
        "tweet_feature_number_of_media",   # 10
        "tweet_feature_is_retweet",        # 11
        "tweet_feature_is_quote",          # 12
        "tweet_feature_is_top_level",      # 13
        "tweet_feature_number_of_hashtags",  # 14
        "tweet_feature_creation_timestamp_hour",  # 15
        "tweet_feature_creation_timestamp_week_day",
        "tweet_feature_number_of_mentions",
        "engager_feature_number_of_previous_like_engagement",
        "engager_feature_number_of_previous_reply_engagement",
        "engager_feature_number_of_previous_retweet_engagement",
        "engager_feature_number_of_previous_comment_engagement",
        "engager_feature_number_of_previous_positive_engagement",
        "engager_feature_number_of_previous_negative_engagement",
        "engager_feature_number_of_previous_engagement",
        "engager_feature_number_of_previous_like_engagement_ratio_1",
        "engager_feature_number_of_previous_reply_engagement_ratio_1",
        "engager_feature_number_of_previous_retweet_engagement_ratio_1",
        "engager_feature_number_of_previous_comment_engagement_ratio_1",
        "engager_feature_number_of_previous_positive_engagement_ratio_1",
        "engager_feature_number_of_previous_negative_engagement_ratio_1",
        "engager_feature_number_of_previous_like_engagement_ratio",
        "engager_feature_number_of_previous_reply_engagement_ratio",
        "engager_feature_number_of_previous_retweet_engagement_ratio",
        "engager_feature_number_of_previous_comment_engagement_ratio",
        "engager_feature_number_of_previous_positive_engagement_ratio",
        "engager_feature_number_of_previous_negative_engagement_ratio",
        "engager_feature_number_of_previous_like_engagement_between_creator_and_engager_by_creator",
        "engager_feature_number_of_previous_reply_engagement_between_creator_and_engager_by_creator",
        "engager_feature_number_of_previous_retweet_engagement_between_creator_and_engager_by_creator",
        "engager_feature_number_of_previous_comment_engagement_between_creator_and_engager_by_creator",
        "engager_feature_number_of_previous_negative_engagement_between_creator_and_engager_by_creator",
        "engager_feature_number_of_previous_positive_engagement_between_creator_and_engager_by_creator",
        "engager_feature_number_of_previous_like_engagement_between_creator_and_engager_by_engager",
        "engager_feature_number_of_previous_reply_engagement_between_creator_and_engager_by_engager",
        "engager_feature_number_of_previous_retweet_engagement_between_creator_and_engager_by_engager",
        "engager_feature_number_of_previous_comment_engagement_between_creator_and_engager_by_engager",
        "engager_feature_number_of_previous_negative_engagement_between_creator_and_engager_by_engager",
        "engager_feature_number_of_previous_positive_engagement_between_creator_and_engager_by_engager",
        "tweet_feature_number_of_previous_like_engagements",
        "tweet_feature_number_of_previous_reply_engagements",
        "tweet_feature_number_of_previous_retweet_engagements",
        "tweet_feature_number_of_previous_comment_engagements",
        "tweet_feature_number_of_previous_positive_engagements",
        "tweet_feature_number_of_previous_negative_engagements",
        "creator_feature_number_of_previous_like_engagements_given",
        "creator_feature_number_of_previous_reply_engagements_given",
        "creator_feature_number_of_previous_retweet_engagements_given",
        "creator_feature_number_of_previous_comment_engagements_given",
        "creator_feature_number_of_previous_positive_engagements_given",
        "creator_feature_number_of_previous_negative_engagements_given",
        "creator_feature_number_of_previous_like_engagements_received",
        "creator_feature_number_of_previous_reply_engagements_received",
        "creator_feature_number_of_previous_retweet_engagements_received",
        "creator_feature_number_of_previous_comment_engagements_received",
        "creator_feature_number_of_previous_positive_engagements_received",
        "creator_feature_number_of_previous_negative_engagements_received",
        # "engager_feature_number_of_previous_like_engagement_with_language",
        # "engager_feature_number_of_previous_reply_engagement_with_language",
        # "engager_feature_number_of_previous_retweet_engagement_with_language",
        # "engager_feature_number_of_previous_comment_engagement_with_language",
        # "engager_feature_number_of_previous_positive_engagement_with_language",
        # "engager_feature_number_of_previous_negative_engagement_with_language",
        # "engager_feature_knows_hashtag_positive",
        # "engager_feature_knows_hashtag_negative",
        # "engager_feature_knows_hashtag_like",
        # "engager_feature_knows_hashtag_reply",
        # "engager_feature_knows_hashtag_rt",
        # "engager_feature_knows_hashtag_comment",
        # "creator_and_engager_have_same_main_language",
        # "is_tweet_in_creator_main_language",
        # "is_tweet_in_engager_main_language",
        # "statistical_probability_main_language_of_engager_engage_tweet_language_1",
        # "statistical_probability_main_language_of_engager_engage_tweet_language_2",
        # "hashtag_similarity_fold_ensembling_positive",
        # "link_similarity_fold_ensembling_positive",
        # "domain_similarity_fold_ensembling_positive"
        "tweet_feature_creation_timestamp_hour_shifted",
        "tweet_feature_creation_timestamp_day_phase",
        "tweet_feature_creation_timestamp_day_phase_shifted"
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
    LITERATION NUMBER 24
    num_leaves= 2491
    learning rate= 0.07711745164455673
    max_depth= 19
    lambda_l1= 13.659668991262592
    lambda_l2= 48.58804711556127
    colsample_bynode= 0.620699712270728
    colsample_bytree= 0.8498078262545382
    pos_subsample= 0.2970870375654674
    neg_subsample= 0.2808850800907874
    bagging_freq= 0
    max_bin= 4405
    min_data_in_leaf= 1781
    -------
    EXECUTION TIME: 2966.6233174800873
    -------
    best_es_iteration: 633
    -------
    PRAUC = 0.1659458700519899
    RCE   = 17.3149847659256
    -------
    '''
    param_dict_1 = {"num_iterations": 633,
                    "num_leaves": 2491,
                    "learning_rate": 0.07711745164455673,
                    "max_depth": 16,
                    "lambda_l1": 13.659668991262592,
                    "lambda_l2": 48.58804711556127,
                    "colsample_bynode": 0.620699712270728,
                    "colsample_bytree": 0.8498078262545382,
                    "pos_subsample": 0.2970870375654674,
                    "neg_subsample": 0.2808850800907874,
                    "bagging_freq": 0,
                    "max_bin": 4405,
                    "min_data_in_leaf": 1781}

    '''
    ITERATION NUMBER 40
    num_leaves= 2805
    learning rate= 0.015432063600124588
    max_depth= 29
    lambda_l1= 14.551578079511918
    lambda_l2= 2.4182464162196693
    colsample_bynode= 1.0
    colsample_bytree= 0.9188456319593815
    pos_subsample= 1.0
    neg_subsample= 1.0
    bagging_freq= 0
    max_bin= 4644
    min_data_in_leaf= 400
    -------
    EXECUTION TIME: 6636.464026689529
    -------
    best_es_iteration: 1000
    -------
    PRAUC = 0.16694768428903073
    RCE   = 17.390420458550203
    '''
    param_dict_2 = {"num_iterations": 1000,
                    "num_leaves": 2805,
                    "learning_rate": 0.015432063600124588,
                    "max_depth": 29,
                    "lambda_l1": 14.551578079511918,
                    "lambda_l2": 2.4182464162196693,
                    "colsample_bynode": 1.0,
                    "colsample_bytree": 0.9188456319593815,
                    "pos_subsample": 1.0,
                    "neg_subsample": 1.0,
                    "bagging_freq": 0,
                    "max_bin": 4644,
                    "min_data_in_leaf": 400}

    '''
    ITERATION NUMBER 37
    num_leaves= 4095
    learning rate= 0.015044832319931386
    max_depth= 22
    lambda_l1= 1.0
    lambda_l2= 50.0
    colsample_bynode= 0.5350241548737203
    colsample_bytree= 1.0
    pos_subsample= 0.9770849887273056
    neg_subsample= 0.1
    bagging_freq= 0
    max_bin= 3305
    min_data_in_leaf= 1970
    -------
    EXECUTION TIME: 5724.907956123352
    -------
    best_es_iteration: 1000
    -------
    PRAUC = 0.16375069740899847
    RCE   = 17.1015505074945
    '''
    param_dict_3 = {"num_iterations": 1000,
                    "num_leaves": 4095,
                    "learning_rate": 0.015044832319931386,
                    "max_depth": 22,
                    "lambda_l1": 1.0,
                    "lambda_l2": 50.0,
                    "colsample_bynode": 0.5350241548737203,
                    "colsample_bytree": 1.0,
                    "pos_subsample": 0.9770849887273056,
                    "neg_subsample": 0.1,
                    "bagging_freq": 0,
                    "max_bin": 3305,
                    "min_data_in_leaf": 1970}

    # categorical_features_set = {4, 5, 6, 11, 12, 13, 43, 44, 45, 46, 47}
    categorical_features_set = set([])
    # categorical_features_set = {4, 5, 6, 11, 12, 13, 34, 35, 36}

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
    print(f"Loading second feature time: {second_feature_end_time - first_feature_end_time} seconds")

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
                   path_log="blending-lgbm-reply-Better",
                   make_log=True,
                   make_save=False,
                   auto_save=False
                   )

    OP.setParameters(n_calls=100, n_random_starts=20)
    OP.loadTrainData(df_metatrain, df_metatrain_label)

    OP.loadValData(df_metaval, df_metaval_label)  # early stopping

    OP.loadTestData(df_metaval, df_metaval_label)  # evaluate objective

    OP.setParamsLGB(objective='binary', early_stopping_rounds=10, eval_metric="binary", is_unbalance=False)
    OP.setCategoricalFeatures(categorical_features_set)
    # OP.loadModelHardCoded()
    res = OP.optimize()


if __name__ == '__main__':
    main()
