import time
from ParamTuning.Optimizer import Optimizer
from Utils.Data import Data
import pandas as pd

from Utils.Data.Data import get_dataset_xgb_batch
from Utils.Data.Features.Generated.EnsemblingFeature.LGBMEnsemblingFeature import LGBMEnsemblingFeature
from sklearn.model_selection import train_test_split
import time
import Blending.like_params as like_params
import Blending.reply_params as reply_params
import Blending.retweet_params as retweet_params
import Blending.comment_params as comment_params
from Utils.Data.Features.Generated.EnsemblingFeature.XGBEnsembling import XGBEnsembling
import argparse


def main():
    # Instantiate the parser
    parser = argparse.ArgumentParser()

    parser.add_argument('label', type=str,
                        help='required argument: label')

    args = parser.parse_args()

    LABEL = args.label

    assert LABEL in ["like", "reply", "retweet", "comment"], "LABEL not valid."

    features = ["raw_feature_creator_follower_count",
               "raw_feature_creator_following_count",
               "raw_feature_engager_follower_count",
               "raw_feature_engager_following_count",
               "raw_feature_creator_is_verified",
               "raw_feature_engager_is_verified",
               "raw_feature_engagement_creator_follows_engager",
               "tweet_feature_number_of_photo",
               "tweet_feature_number_of_video",
               "tweet_feature_number_of_gif",
               "tweet_feature_number_of_media",
               "tweet_feature_is_retweet",
               "tweet_feature_is_quote",
               "tweet_feature_is_top_level",
               "tweet_feature_number_of_hashtags",
               "tweet_feature_creation_timestamp_hour",
               "tweet_feature_creation_timestamp_week_day",
               # "tweet_feature_number_of_mentions",
               "tweet_feature_token_length",
               "tweet_feature_token_length_unique",
               "tweet_feature_text_topic_word_count_adult_content",
               "tweet_feature_text_topic_word_count_kpop",
               "tweet_feature_text_topic_word_count_covid",
               "tweet_feature_text_topic_word_count_sport",
               "number_of_engagements_with_language_like",
               "number_of_engagements_with_language_retweet",
               "number_of_engagements_with_language_reply",
               "number_of_engagements_with_language_comment",
               "number_of_engagements_with_language_negative",
               "number_of_engagements_with_language_positive",
               "number_of_engagements_ratio_like",
               "number_of_engagements_ratio_retweet",
               "number_of_engagements_ratio_reply",
               "number_of_engagements_ratio_comment",
               "number_of_engagements_ratio_negative",
               "number_of_engagements_ratio_positive",
               "number_of_engagements_between_creator_and_engager_like",
               "number_of_engagements_between_creator_and_engager_retweet",
               "number_of_engagements_between_creator_and_engager_reply",
               "number_of_engagements_between_creator_and_engager_comment",
               "number_of_engagements_between_creator_and_engager_negative",
               "number_of_engagements_between_creator_and_engager_positive",
               "creator_feature_number_of_like_engagements_received",
               "creator_feature_number_of_retweet_engagements_received",
               "creator_feature_number_of_reply_engagements_received",
               "creator_feature_number_of_comment_engagements_received",
               "creator_feature_number_of_negative_engagements_received",
               "creator_feature_number_of_positive_engagements_received",
               "creator_feature_number_of_like_engagements_given",
               "creator_feature_number_of_retweet_engagements_given",
               "creator_feature_number_of_reply_engagements_given",
               "creator_feature_number_of_comment_engagements_given",
               "creator_feature_number_of_negative_engagements_given",
               "creator_feature_number_of_positive_engagements_given",
               "engager_feature_number_of_like_engagements_received",
               "engager_feature_number_of_retweet_engagements_received",
               "engager_feature_number_of_reply_engagements_received",
               "engager_feature_number_of_comment_engagements_received",
               "engager_feature_number_of_negative_engagements_received",
               "engager_feature_number_of_positive_engagements_received",
               "number_of_engagements_like",
               "number_of_engagements_retweet",
               "number_of_engagements_reply",
               "number_of_engagements_comment",
               "number_of_engagements_negative",
               "number_of_engagements_positive",
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
               # "tweet_feature_number_of_previous_like_engagements",
               # "tweet_feature_number_of_previous_reply_engagements",
               # "tweet_feature_number_of_previous_retweet_engagements",
               # "tweet_feature_number_of_previous_comment_engagements",
               # "tweet_feature_number_of_previous_positive_engagements",
               # "tweet_feature_number_of_previous_negative_engagements",
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
               "engager_feature_number_of_previous_like_engagement_with_language",
               "engager_feature_number_of_previous_reply_engagement_with_language",
               "engager_feature_number_of_previous_retweet_engagement_with_language",
               "engager_feature_number_of_previous_comment_engagement_with_language",
               "engager_feature_number_of_previous_positive_engagement_with_language",
               "engager_feature_number_of_previous_negative_engagement_with_language",
               "engager_feature_knows_hashtag_positive",
               "engager_feature_knows_hashtag_negative",
               "engager_feature_knows_hashtag_like",
               "engager_feature_knows_hashtag_reply",
               "engager_feature_knows_hashtag_rt",
               "engager_feature_knows_hashtag_comment",
               "creator_and_engager_have_same_main_language",
               "is_tweet_in_creator_main_language",
               "is_tweet_in_engager_main_language",
               # "statistical_probability_main_language_of_engager_engage_tweet_language_1",
               # "statistical_probability_main_language_of_engager_engage_tweet_language_2",
               "creator_and_engager_have_same_main_grouped_language",
               "is_tweet_in_creator_main_grouped_language",
               "is_tweet_in_engager_main_grouped_language",
               # # "hashtag_similarity_fold_ensembling_positive",
               # # "link_similarity_fold_ensembling_positive",
               # # "domain_similarity_fold_ensembling_positive"
               "tweet_feature_creation_timestamp_hour_shifted",
               "tweet_feature_creation_timestamp_day_phase",
               "tweet_feature_creation_timestamp_day_phase_shifted"
               ]

    label = [
        f"tweet_feature_engagement_is_{LABEL}"
    ]

    train_dataset = "cherry_train"
    val_dataset = "cherry_val"
    test_dataset = "new_test"

    if LABEL is "like":
        lgbm_params = like_params.lgbm_get_params()
        xgb_params = like_params.xgb_get_params()
    elif LABEL is "reply":
        lgbm_params = reply_params.lgbm_get_params()
        xgb_params = reply_params.xgb_get_params()
    elif LABEL is "retweet":
        lgbm_params = retweet_params.lgbm_get_params()
        xgb_params = retweet_params.xgb_get_params()
    elif LABEL is "comment":
        lgbm_params = comment_params.lgbm_get_params()
        xgb_params = comment_params.xgb_get_params()

    categorical_features_set = set([])


    # Load train data
    # loading_data_start_time = time.time()
    # df_train, df_train_label = Data.get_dataset_xgb(train_dataset, features, label)
    # print(f"Loading train data time: {loading_data_start_time - time.time()} seconds")

    # Load val data
    df_val, df_val_label = Data.get_dataset_xgb(val_dataset, features, label)

    # Load test data
    df_test = Data.get_dataset(features, test_dataset)

    new_index = pd.Series(df_test.index).map(lambda x: x + len(df_val))
    df_test.set_index(new_index, inplace=True)

    # df to be predicted by the lgbm blending feature
    df_to_predict = pd.concat([df_val, df_test])

    # BLENDING FEATURE DECLARATION

    feature_list = []

    for lgbm_param_dict in lgbm_params:
        start_time = time.time()

        df_train, df_train_label = get_dataset_xgb_batch(total_n_split=1, split_n=0, dataset_id=train_dataset,
                                                             X_label=features, Y_label=label, sample=0.3)

        feature_list.append(LGBMEnsemblingFeature(dataset_id=train_dataset,
                                   df_train=df_train,
                                   df_train_label=df_train_label,
                                   df_to_predict=df_to_predict,
                                   param_dict=lgbm_param_dict,
                                   categorical_features_set=categorical_features_set))

    for xgb_param_dict in xgb_params:
        start_time = time.time()
        df_train, df_train_label = get_dataset_xgb_batch(total_n_split=1, split_n=0, dataset_id=train_dataset,
                                                         X_label=features, Y_label=label, sample=0.1)
        feature_list.append(XGBEnsembling(dataset_id=train_dataset,
                                   df_train=df_train,
                                   df_train_label=df_train_label,
                                   df_to_predict=df_to_predict,
                                   param_dict=xgb_param_dict, ))

    df_feature_list = [x.load_or_create() for x in feature_list]

    # check dimensions
    len_val = len(df_val)

    for df_feat in df_feature_list:
        assert len(df_feat) == (len_val + len(df_test)), \
            f"Blending features are not of dimension expected, len val: {len_val} len test: {len(df_test)}\n " \
            f"obtained len: {len(df_feat)} of {df_feat.columns[0]}\n"

    # split feature dataframe in validation and testing
    df_feat_val_list = [df_feat.iloc[:len_val] for df_feat in df_feature_list]
    df_feat_test_list = [df_feat.iloc[len_val:] for df_feat in df_feature_list]

    df_to_be_concatenated_list = [df_val] + df_feat_val_list + [df_val_label]

    # creating the new validation set on which we will do meta optimization
    df_val = pd.concat(df_to_be_concatenated_list, axis=1)

    # now we are in full meta-model mode
    # watchout! they are unsorted now, you got to re-sort the dfs
    df_metatrain, df_metaval = train_test_split(df_val, test_size=0.3)
    df_metatrain.sort_index(inplace=True)
    df_metaval.sort_index(inplace=True)

    # split dataframe columns in train and label
    col_names_list = [df_feat.columns[0] for df_feat in df_feature_list]

    extended_features = features + col_names_list
    df_metatrain_label = df_metatrain[label]
    df_metatrain = df_metatrain[extended_features]

    df_metaval_label = df_metaval[label]
    df_metaval = df_metaval[extended_features]

    model_name = "lightgbm_classifier"
    kind = LABEL

    OP = Optimizer(model_name,
                   kind,
                   mode=0,
                   path=LABEL,
                   path_log=f"blending-lgbm-{LABEL}",
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
