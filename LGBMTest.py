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
    test_dataset="test"

    # Define the X label
    X_label = [
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
    "creator_and_engager_have_same_main_language",
    "is_tweet_in_creator_main_language",
    "is_tweet_in_engager_main_language",
    "statistical_probability_main_language_of_engager_engage_tweet_language_1",
    "statistical_probability_main_language_of_engager_engage_tweet_language_2",
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

    # Load local_test data
    X_local, Y_local = Data.get_dataset_xgb_batch(2, 1, val_dataset, X_label, Y_label, 1)

    # Load test data
    X_test = Data.get_dataset(X_label, test_dataset)

    print(f"Loading data time: {time.time() - loading_data_start_time} seconds")

num_leaves=         2730
learning rate=      0.055466274331429706
max_depth=          37
lambda_l1=          24.7293614309037
lambda_l2=          43.708181220600984
colsample_bynode=   0.8634350494410573
colsample_bytree=   0.92785113065472
bagging_fraction=   0.5322034449307513
bagging_freq=       7
max_bin=            4755
min_data_in_leaf=   1628

    LGBM = LightGBM(
        objective         =     'binary',
        num_threads       =     -1,
        num_iterations    =     1000,
        num_leaves        =     2730,
        learning_rate     =     0.055466274331429706,
        max_depth         =     37,
        lambda_l1         =     24.7293614309037,
        lambda_l2         =     43.708181220600984,
        colsample_bynode  =     0.8634350494410573,
        colsample_bytree  =     0.92785113065472,
        pos_subsample     =     0.5322034449307513,
        neg_subsample     =     7,
        bagging_freq      =     4755,
        max_bin           =     1628,
        min_data_in_leaf  =     875,
        early_stopping_rounds=15
        )

    # LGBM Training
    training_start_time = time.time()
    #LGBM.fit(X=X_train, Y=Y_train, X_val=X_val, Y_val=Y_val, categorical_feature=set([4,5,6,11,12,13]))
    LGBM.fit(X=X_train, Y=Y_train, X_val=X_val, Y_val=Y_val, categorical_feature=set([]))
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