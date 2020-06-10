import numpy as np
import skopt
from skopt import gp_minimize
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import sys
import pandas as pd
import time
import datetime as dt
from ParamTuning.ModelInterface import ModelInterface
from ParamTuning.Optimizer import Optimizer
from Utils.Data import Data

from Utils.Data.Data import oversample


def main():  
    # Defining the dataset used
    train_dataset = "cherry_train"
    test_dataset = "cherry_val"

    # Define the X label

    X_label = ["raw_feature_creator_follower_count",
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
    #"tweet_feature_number_of_mentions",
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
    #"creator_and_engager_have_same_main_language",
    #"is_tweet_in_creator_main_language",
    #"is_tweet_in_engager_main_language",
    #"statistical_probability_main_language_of_engager_engage_tweet_language_1",
    #"statistical_probability_main_language_of_engager_engage_tweet_language_2",
    #"creator_and_engager_have_same_main_grouped_language",
    #"is_tweet_in_creator_main_grouped_language",
    #"is_tweet_in_engager_main_grouped_language",
    # # "hashtag_similarity_fold_ensembling_positive",
    # # "link_similarity_fold_ensembling_positive",
    # # "domain_similarity_fold_ensembling_positive"
    "tweet_feature_creation_timestamp_hour_shifted",
    "tweet_feature_creation_timestamp_day_phase",
    "tweet_feature_creation_timestamp_day_phase_shifted",
    "tweet_feature_has_discriminative_hashtag_like",
    "tweet_feature_has_discriminative_hashtag_reply",
    "tweet_feature_has_discriminative_hashtag_retweet",
    "tweet_feature_has_discriminative_hashtag_comment",
    "tweet_feature_number_of_discriminative_hashtag_like",
    "tweet_feature_number_of_discriminative_hashtag_reply",
    "tweet_feature_number_of_discriminative_hashtag_retweet",
    "tweet_feature_number_of_discriminative_hashtag_comment"
    ]
    
                                                    
    # Define the Y label
    Y_label = [
        "tweet_feature_engagement_is_like"
    ]

    model_name = "lightgbm_classifier"
    kind = "like"

    # Load train data
    loading_data_start_time = time.time()
    X_train, Y_train = Data.get_dataset_xgb_batch(1, 0, train_dataset, X_label, Y_label, 0.3)

    #additional_features=[
    #"tweet_feature_has_discriminative_hashtag_like",
    #"tweet_feature_has_discriminative_hashtag_reply",
    #"tweet_feature_has_discriminative_hashtag_retweet",
    #"tweet_feature_has_discriminative_hashtag_comment",
    #"tweet_feature_number_of_discriminative_hashtag_like",
    #"tweet_feature_number_of_discriminative_hashtag_reply",
    #"tweet_feature_number_of_discriminative_hashtag_retweet",
    #"tweet_feature_number_of_discriminative_hashtag_comment",
    #]
    #
    #add_feat = []
    #
    #for feature in additional_features:
    #    add_feat.append(Data.get_feature_batch(feature, train_dataset, 1, 0, 0.3))
    #
    #all_feat = pd.concat(add_feat, axis=1)
    #X_train = pd.concat([X_train,add_feat], axis=1)

    # Load test data
    X_val, Y_val = Data.get_dataset_xgb_batch(2, 0, test_dataset, X_label, Y_label, 1)    

    #add_feat = []
    #
    #for feature in additional_features:
    #    add_feat.append(Data.get_feature_batch(feature, test_dataset, 1, 0, 0.3))
    #
    #all_feat = pd.concat(add_feat, axis=1)
    #X_val = pd.concat([X_test,add_feat], axis=1)

    X_test, Y_test = Data.get_dataset_xgb_batch(2, 1, test_dataset, X_label, Y_label, 1)
    
    #add_feat = []
    #
    #for feature in additional_features:
    #    add_feat.append(Data.get_feature_batch(feature, test_dataset, 1, 0, 0.3))
    #
    #all_feat = pd.concat(add_feat, axis=1)
    #X_test = pd.concat([X_test,add_feat], axis=1)

    print(f"Loading data time: {time.time() - loading_data_start_time} seconds")

    OP = Optimizer(model_name, 
                   kind,
                   mode=0,
                   path="like",
                   path_log="LGBM-like-HASHTAGS-FEATURES",
                   make_log=True, 
                   make_save=False, 
                   auto_save=False)

    OP.setParameters(n_calls=40, n_random_starts=15)
    OP.loadTrainData(X_train, Y_train)
    OP.loadTestData(X_test, Y_test)
    OP.loadValData(X_val, Y_val)
    OP.setParamsLGB(objective='binary',early_stopping_rounds=15, eval_metric="binary",is_unbalance=False)
    #OP.setCategoricalFeatures(set([4,5,6,11,12,13]))
    OP.setCategoricalFeatures(set([]))
    #OP.loadModelHardCoded()
    res=OP.optimize()

    '''
    #Add this for complete routine check
    print(res.func_vals.shape)
    path = OP.saveModel()
    OP.loadModel(path)
    res = OP.optimize()
    print(res.func_vals.shape)
    print("END")
    '''
    


if __name__ == "__main__":
    main()


