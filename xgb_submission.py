import xgboost as xgb
import pandas as pd

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
submission_filename = f"xgb_submission_{label}.py"

# Train dataset id
train_dataset_id = "holdout/train"

# Validation dataset id
val_dataset_id = "holdout/test"

# Test dataset id
test_dataset_id = "test"

# Oversample the cold users
use_oversample = True
os_column_name = "engager_feature_number_of_previous_positive_engagement_ratio_1"
os_value = -1
os_percentage = 0.15

# Cached svm filename
svm_filename = "cached_svm_submission"

# Model filename
xgb_model_filename = f"xgb_model_{label}.model"

# XGBoost model
xgboost = XGBoost(
    early_stopping_rounds=20,
    num_rounds=1000,
    max_depth=14,
    min_child_weight=2,
    colsample_bytree=0.986681081839595,
    learning_rate=0.084682683195233,
    reg_alpha=0.000709687642837,
    reg_lambda=0.004524218043733,
    scale_pos_weight=0.701767017039216,
    gamma=8.5138404902064,
    subsample=0.726578069020986,
    base_score=0.5,
    max_delta_step=0.018196005449355,
    num_parallel_tree=13
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
    "creator_and_engager_have_same_main_language",
    "is_tweet_in_creator_main_language",
    "is_tweet_in_engager_main_language",
    "statistical_probability_main_language_of_engager_engage_tweet_language_1",
    "statistical_probability_main_language_of_engager_engage_tweet_language_2"
]

# The labels
Y_label = [
    f"tweet_feature_engagement_is_{label}"
]


def run_xgb():
    # Load the training dataset
    X_train = get_dataset(X_label, train_dataset_id)
    Y_train = get_dataset(Y_label, train_dataset_id)
    # Cache the training dataset
    cache_dataset_as_svm(svm_filename, X_train, Y_train)
    train = xgb.DMatrix(f"{svm_filename}.svm#{svm_filename}.cache")
    train.feature_names = X_train.columns
    # Delete the data structure that are not useful anymore
    del X_train, Y_train

    # Load the local validation dataset for early stopping
    X_local_val = get_dataset_batch(X_label, val_dataset_id, 2, 0, 1)
    Y_local_val = get_dataset_batch(Y_label, val_dataset_id, 2, 0, 1)
    # If oversample is set
    if use_oversample is True:
        df = pd.concat([X_local_val, Y_local_val], axis=1)
        oversampled_df = oversample(df, os_column_name, os_value, os_percentage)
        X_local_val = oversampled_df[X_label]
        Y_local_val = oversampled_df[Y_label]
        del df, oversampled_df
    local_val = xgb.DMatrix(X_local_val, Y_local_val)
    del X_local_val, Y_local_val

    # Fit the model
    xgboost.fit(train, local_val)

    # Save the model
    xgboost.save_model(xgb_model_filename)
    del train, local_val

    # Load the remote validation dataset for testing
    X_remote_val = get_dataset_batch(X_label, val_dataset_id, 2, 1, 1)
    Y_remote_val = get_dataset_batch(Y_label, val_dataset_id, 2, 1, 1)
    # If oversample is set
    if use_oversample is True:
        df = pd.concat([X_remote_val, Y_remote_val], axis=1)
        oversampled_df = oversample(df, os_column_name, os_value, os_percentage)
        X_remote_val = oversampled_df[X_label]
        Y_remote_val = oversampled_df[Y_label]
        del df, oversampled_df
    remote_val = xgb.DMatrix(X_remote_val, Y_remote_val)
    del X_remote_val, Y_remote_val

    # Evaluate the model
    prauc, rce, conf, max_v, min_v, avg_v = xgboost.evaluate(remote_val)
    print(f"prauc: {prauc}")
    print(f"rce: {rce}")
    print(f"max: {max_v}")
    print(f"min: {min_v}")
    print(f"avg: {avg_v}")
    del remote_val

    # Load the remote validation dataset for testing
    X_test = get_dataset(X_label, test_dataset_id)
    test = xgb.DMatrix(X_test)
    del X_test

    # Retrieve the predictions
    predictions = xgboost.get_prediction(test)

    # Retrieve users and tweets
    tweets = get_feature("raw_feature_tweet_id", test_dataset_id)["raw_feature_tweet_id"].array
    users = get_feature("raw_feature_engager_id", test_dataset_id)["raw_feature_engager_id"].array

    # Write the submission file
    create_submission_file(tweets, users, predictions, submission_filename)


if __name__ == '__main__':
    run_xgb()
