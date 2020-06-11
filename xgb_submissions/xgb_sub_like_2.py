import xgboost as xgb
import pandas as pd
import numpy as np
from Models.GBM.XGBoost import XGBoost
from Utils.Data.Data import get_dataset, get_dataset_batch, get_feature, oversample
from Utils.Data.DataUtils import cache_dataset_as_svm

# Parameters to be configured for the run

# The run will load:
# 1) a training set
# 2) a local validation set for early stopping
# 3) a remote valiation set for local evaluation
# 4) a test set

# The label to be predicted
from Utils.Submission.Submission import create_submission_file

label = "like"

# The name of the submission file
submission_filename = f"xgb_submission_{label}_2.csv"

# Train dataset id
# train_dataset_id = "holdout/train"
train_dataset_id = "cherry_train"

# Validation dataset id
# val_dataset_id = "holdout/test"
val_dataset_id = "cherry_val"

# Test dataset id
test_dataset_id = "new_test"

# Last test dataset id
last_test_dataset_id = "last_test"

# Cached svm filename
svm_filename = "cached_svm_submission"

# Model filename
xgb_model_filename = f"xgb_model_{label}.model"

# XGB parameters
xgb_parameters = {
    'max_depth': 2,
    'min_child_weight': 1,
    'colsample_bytree': 0.8483601464893439,
    'learning_rate': 0.025000000000000005,
    'reg_alpha': 1.0,
    'reg_lambda': 0.030912186986807588,
    'scale_pos_weight': 1,
    'gamma': 1.7184464463142841,
    'subsample': 1.0,
    'base_score': 0.4392,
    'max_delta_step': 0.0,
    'num_parallel_tree': 8
}

# XGBoost model
xgboost = XGBoost(
    eval_metric="logloss",
    tree_method='gpu_hist',
    early_stopping_rounds=50,
    num_rounds=2500,
    **xgb_parameters
)


# The features
X_label = [
    "raw_feature_creator_follower_count",
    "raw_feature_creator_following_count",
    "raw_feature_engager_follower_count",
    "raw_feature_engager_following_count",
    "raw_feature_creator_is_verified",
    "raw_feature_engager_is_verified",
    "raw_feature_engagement_creator_follows_engager",
    "raw_feature_creator_creation_timestamp",
    "raw_feature_engager_creation_timestamp",
    "raw_feature_tweet_timestamp",
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
    "adjacency_between_creator_and_engager_retweet",
    "adjacency_between_creator_and_engager_reply",
    "adjacency_between_creator_and_engager_comment",
    "adjacency_between_creator_and_engager_like",
    "adjacency_between_creator_and_engager_positive",
    "adjacency_between_creator_and_engager_negative",
    "graph_two_steps_adjacency_positive",
    "graph_two_steps_adjacency_negative",
    "graph_two_steps_adjacency_like",
    "graph_two_steps_adjacency_reply",
    "graph_two_steps_adjacency_retweet",
    "graph_two_steps_adjacency_comment",
    "graph_two_steps_positive",
    "graph_two_steps_negative",
    "graph_two_steps_like",
    "graph_two_steps_reply",
    "graph_two_steps_retweet",
    "graph_two_steps_comment",
    "xgb_fold_ensembling_like_2",
    "xgb_fold_ensembling_retweet_2",
    "xgb_fold_ensembling_reply_2",
    "xgb_fold_ensembling_comment_2",
    "tweet_feature_creation_timestamp_hour_shifted",
    "tweet_feature_creation_timestamp_day_phase",
    "tweet_feature_creation_timestamp_day_phase_shifted"
]

# The labels
Y_label = [
    f"tweet_feature_engagement_is_{label}"
]

def evaluation(model, test):
    prauc, rce, conf, max_v, min_v, avg_v = model.evaluate(test)
    print(f"local eval - prauc: {prauc}")
    print(f"local eval - rce: {rce}")
    print(f"local eval - max: {max_v}")
    print(f"local eval - min: {min_v}")
    print(f"local eval - avg: {avg_v}")
    print(f"-------------------------")


def run_xgb():
    # Load the training dataset
    X_train = get_dataset_batch(X_label, train_dataset_id, 1, 0, 0.05)
    Y_train = get_dataset_batch(Y_label, train_dataset_id, 1, 0, 0.05)
    # Cache the training dataset
    cache_dataset_as_svm(svm_filename, X_train, Y_train)
    train = xgb.DMatrix(f"{svm_filename}.svm")
    train.feature_names = X_label
    # Delete the data structure that are not useful anymore
    del X_train, Y_train

    # Load the local validation dataset for early stopping
    X_local_val = get_dataset_batch(X_label, val_dataset_id, 2, 0, 0.99)
    Y_local_val = get_dataset_batch(Y_label, val_dataset_id, 2, 0, 0.99)
    cache_dataset_as_svm(f"{svm_filename}_local_val", X_local_val, Y_local_val)
    local_val = xgb.DMatrix(f"{svm_filename}_local_val.svm")
    local_val.feature_names = X_label
    del X_local_val, Y_local_val

    # Fit the model
    xgboost.fit(train, local_val)

    # Save the model
    xgboost.save_model(xgb_model_filename)
    del train, local_val

    # Load the remote validation dataset for testing
    X_remote_val = get_dataset_batch(X_label, val_dataset_id, 2, 1, 0.99)
    Y_remote_val = get_dataset_batch(Y_label, val_dataset_id, 2, 1, 0.99)
    cache_dataset_as_svm(f"{svm_filename}_remote_val", X_remote_val, Y_remote_val, no_fuck_my_self=True)
    remote_val = xgb.DMatrix(f"{svm_filename}_remote_val.svm")
    remote_val.feature_names = X_label
    cold_mask = np.where(X_remote_val["engager_feature_number_of_previous_positive_engagement_ratio_1"] == -1)[0]
    hot_mask = np.where(X_remote_val["engager_feature_number_of_previous_positive_engagement_ratio_1"] != -1)[0]
    del X_remote_val, Y_remote_val

    # Evaluate the model
    evaluation(xgboost, remote_val)
    print(f"cold_users are {len(cold_mask)}")
    evaluation(xgboost, remote_val.slice(cold_mask))
    print(f"hot_users are {len(hot_mask)}")
    evaluation(xgboost, remote_val.slice(hot_mask))
    del remote_val

    # Load the remote validation dataset for testing
    X_test = get_dataset(X_label, test_dataset_id)
    print(X_test)
    cache_dataset_as_svm(f"{svm_filename}_test", X_test, no_fuck_my_self=True)
    test = xgb.DMatrix(f"{svm_filename}_test.svm")
    test.feature_names = X_label
    del X_test

    # Retrieve the predictions
    predictions = xgboost.get_prediction(test)
    print(f"remote submission - max: {predictions.max()}")
    print(f"remote submission - min: {predictions.min()}")
    print(f"remote submission - avg: {predictions.mean()}")

    # Retrieve users and tweets
    tweets = get_feature("raw_feature_tweet_id", test_dataset_id)["raw_feature_tweet_id"].array
    users = get_feature("raw_feature_engager_id", test_dataset_id)["raw_feature_engager_id"].array

    # Write the submission file
    create_submission_file(tweets, users, predictions, submission_filename)

    # Load the last test dataset for testing
    X_test = get_dataset(X_label, last_test_dataset_id)
    print(X_test)
    cache_dataset_as_svm(f"{svm_filename}_last_test", X_test, no_fuck_my_self=True)
    test = xgb.DMatrix(f"{svm_filename}_last_test.svm")
    test.feature_names = X_label
    del X_test

    # Retrieve the predictions
    predictions = xgboost.get_prediction(test)
    print(f"remote submission - max: {predictions.max()}")
    print(f"remote submission - min: {predictions.min()}")
    print(f"remote submission - avg: {predictions.mean()}")

    # Retrieve users and tweets
    tweets = get_feature("raw_feature_tweet_id", last_test_dataset_id)["raw_feature_tweet_id"].array
    users = get_feature("raw_feature_engager_id", last_test_dataset_id)["raw_feature_engager_id"].array

    # Write the submission file
    create_submission_file(tweets, users, predictions, f"last_{submission_filename}")

if __name__ == '__main__':
    run_xgb()