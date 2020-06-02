import eli5
import sklearn
import xgboost as xgb
import pandas as pd
import numpy as np
from eli5.sklearn import PermutationImportance
import sklearn.inspection as sk_ins
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
from Utils.Importance.XGBImportance import XGBImportance
from Utils.Submission.Submission import create_submission_file

label = "reply"

# The name of the submission file
submission_filename = f"xgb_submission_{label}.csv"

# Train dataset id
# train_dataset_id = "holdout/train"
train_dataset_id = "train_days_123456"

# Validation dataset id
# val_dataset_id = "holdout/test"
val_dataset_id = "val_days_7"

# Test dataset id
test_dataset_id = "new_test"

# Oversample the cold users
use_oversample = False
os_column_name = "engager_feature_number_of_previous_positive_engagement_ratio_1"
os_value = -1
os_percentage = 0.25

# Cached svm filename
svm_filename = "cached_svm_submission"

# Model filename
xgb_model_filename = f"xgb_model_{label}.model"

# XGBoost model
xgboost = XGBoost(
    eval_metric="logloss",
    tree_method='gpu_hist',
    early_stopping_rounds=50,
    num_rounds=2500,
    max_depth=14,
    min_child_weight=27,
    colsample_bytree=0.2907551623457,
    learning_rate=0.13789861457,
    reg_alpha=0.033194608,
    reg_lambda=0.002818679,
    scale_pos_weight=1,
    gamma=0.2657285621,
    subsample=0.9297063,
    base_score=0.0274,
    max_delta_step=69.6521399,
    num_parallel_tree=5
)

# The features
X_label = [
    "raw_feature_creator_follower_count",
    "raw_feature_creator_following_count",
    "raw_feature_engager_follower_count",
    "raw_feature_engager_following_count",
    # "raw_feature_creator_is_verified",
    # "raw_feature_engager_is_verified",
    # "raw_feature_engagement_creator_follows_engager",
    # "tweet_feature_number_of_photo",
    # "tweet_feature_number_of_video",
    # "tweet_feature_number_of_gif",
    # "tweet_feature_number_of_media",
    # "tweet_feature_is_retweet",
    # "tweet_feature_is_quote",
    # "tweet_feature_is_top_level",
    # "tweet_feature_number_of_hashtags",
    # "tweet_feature_creation_timestamp_hour",
    # "tweet_feature_creation_timestamp_week_day",
    # "tweet_feature_number_of_mentions",
    # "tweet_feature_token_length",
    # "tweet_feature_token_length_unique",
    # "tweet_feature_text_topic_word_count_adult_content",
    # "tweet_feature_text_topic_word_count_kpop",
    # "tweet_feature_text_topic_word_count_covid",
    # "tweet_feature_text_topic_word_count_sport",
    # "number_of_engagements_with_language_like",
    # "number_of_engagements_with_language_retweet",
    # "number_of_engagements_with_language_reply",
    # "number_of_engagements_with_language_comment",
    # "number_of_engagements_with_language_negative",
    # "number_of_engagements_with_language_positive",
    # "number_of_engagements_ratio_like",
    # "number_of_engagements_ratio_retweet",
    # "number_of_engagements_ratio_reply",
    # "number_of_engagements_ratio_comment",
    # "number_of_engagements_ratio_negative",
    # "number_of_engagements_ratio_positive",
    # "number_of_engagements_between_creator_and_engager_like",
    # "number_of_engagements_between_creator_and_engager_retweet",
    # "number_of_engagements_between_creator_and_engager_reply",
    # "number_of_engagements_between_creator_and_engager_comment",
    # "number_of_engagements_between_creator_and_engager_negative",
    # "number_of_engagements_between_creator_and_engager_positive",
    # "creator_feature_number_of_like_engagements_received",
    # "creator_feature_number_of_retweet_engagements_received",
    # "creator_feature_number_of_reply_engagements_received",
    # "creator_feature_number_of_comment_engagements_received",
    # "creator_feature_number_of_negative_engagements_received",
    # "creator_feature_number_of_positive_engagements_received",
    # "creator_feature_number_of_like_engagements_given",
    # "creator_feature_number_of_retweet_engagements_given",
    # "creator_feature_number_of_reply_engagements_given",
    # "creator_feature_number_of_comment_engagements_given",
    # "creator_feature_number_of_negative_engagements_given",
    # "creator_feature_number_of_positive_engagements_given",
    # "engager_feature_number_of_like_engagements_received",
    # "engager_feature_number_of_retweet_engagements_received",
    # "engager_feature_number_of_reply_engagements_received",
    # "engager_feature_number_of_comment_engagements_received",
    # "engager_feature_number_of_negative_engagements_received",
    # "engager_feature_number_of_positive_engagements_received",
    # "number_of_engagements_like",
    # "number_of_engagements_retweet",
    # "number_of_engagements_reply",
    # "number_of_engagements_comment",
    # "number_of_engagements_negative",
    # "number_of_engagements_positive",
    # "engager_feature_number_of_previous_like_engagement",
    # "engager_feature_number_of_previous_reply_engagement",
    # "engager_feature_number_of_previous_retweet_engagement",
    # "engager_feature_number_of_previous_comment_engagement",
    # "engager_feature_number_of_previous_positive_engagement",
    # "engager_feature_number_of_previous_negative_engagement",
    # "engager_feature_number_of_previous_engagement",
    # "engager_feature_number_of_previous_like_engagement_ratio_1",
    # "engager_feature_number_of_previous_reply_engagement_ratio_1",
    # "engager_feature_number_of_previous_retweet_engagement_ratio_1",
    # "engager_feature_number_of_previous_comment_engagement_ratio_1",
    # "engager_feature_number_of_previous_positive_engagement_ratio_1",
    # "engager_feature_number_of_previous_negative_engagement_ratio_1",
    # "engager_feature_number_of_previous_like_engagement_ratio",
    # "engager_feature_number_of_previous_reply_engagement_ratio",
    # "engager_feature_number_of_previous_retweet_engagement_ratio",
    # "engager_feature_number_of_previous_comment_engagement_ratio",
    # "engager_feature_number_of_previous_positive_engagement_ratio",
    # "engager_feature_number_of_previous_negative_engagement_ratio",
    # "engager_feature_number_of_previous_like_engagement_between_creator_and_engager_by_creator",
    # "engager_feature_number_of_previous_reply_engagement_between_creator_and_engager_by_creator",
    # "engager_feature_number_of_previous_retweet_engagement_between_creator_and_engager_by_creator",
    # "engager_feature_number_of_previous_comment_engagement_between_creator_and_engager_by_creator",
    # "engager_feature_number_of_previous_negative_engagement_between_creator_and_engager_by_creator",
    # "engager_feature_number_of_previous_positive_engagement_between_creator_and_engager_by_creator",
    # "engager_feature_number_of_previous_like_engagement_between_creator_and_engager_by_engager",
    # "engager_feature_number_of_previous_reply_engagement_between_creator_and_engager_by_engager",
    # "engager_feature_number_of_previous_retweet_engagement_between_creator_and_engager_by_engager",
    # "engager_feature_number_of_previous_comment_engagement_between_creator_and_engager_by_engager",
    # "engager_feature_number_of_previous_negative_engagement_between_creator_and_engager_by_engager",
    # "engager_feature_number_of_previous_positive_engagement_between_creator_and_engager_by_engager",
    # # "tweet_feature_number_of_previous_like_engagements",
    # # "tweet_feature_number_of_previous_reply_engagements",
    # # "tweet_feature_number_of_previous_retweet_engagements",
    # # "tweet_feature_number_of_previous_comment_engagements",
    # # "tweet_feature_number_of_previous_positive_engagements",
    # # "tweet_feature_number_of_previous_negative_engagements",
    # "creator_feature_number_of_previous_like_engagements_given",
    # "creator_feature_number_of_previous_reply_engagements_given",
    # "creator_feature_number_of_previous_retweet_engagements_given",
    # "creator_feature_number_of_previous_comment_engagements_given",
    # "creator_feature_number_of_previous_positive_engagements_given",
    # "creator_feature_number_of_previous_negative_engagements_given",
    # "creator_feature_number_of_previous_like_engagements_received",
    # "creator_feature_number_of_previous_reply_engagements_received",
    # "creator_feature_number_of_previous_retweet_engagements_received",
    # "creator_feature_number_of_previous_comment_engagements_received",
    # "creator_feature_number_of_previous_positive_engagements_received",
    # "creator_feature_number_of_previous_negative_engagements_received",
    # "engager_feature_number_of_previous_like_engagement_with_language",
    # "engager_feature_number_of_previous_reply_engagement_with_language",
    # "engager_feature_number_of_previous_retweet_engagement_with_language",
    # "engager_feature_number_of_previous_comment_engagement_with_language",
    # "engager_feature_number_of_previous_positive_engagement_with_language",
    # "engager_feature_number_of_previous_negative_engagement_with_language",
    # # "engager_feature_knows_hashtag_positive",
    # # "engager_feature_knows_hashtag_negative",
    # # "engager_feature_knows_hashtag_like",
    # # "engager_feature_knows_hashtag_reply",
    # # "engager_feature_knows_hashtag_rt",
    # # "engager_feature_knows_hashtag_comment",
    # # # "creator_and_engager_have_same_main_language",
    # # # "is_tweet_in_creator_main_language",
    # # # "is_tweet_in_engager_main_language",
    # # # "statistical_probability_main_language_of_engager_engage_tweet_language_1",
    # # # "statistical_probability_main_language_of_engager_engage_tweet_language_2",
    # # # "hashtag_similarity_fold_ensembling_positive",
    # # # "link_similarity_fold_ensembling_positive",
    # # # "domain_similarity_fold_ensembling_positive"
    # "xgb_fold_ensembling_like_1",
    # "xgb_fold_ensembling_retweet_1",
    # "xgb_fold_ensembling_reply_1",
    # "xgb_fold_ensembling_comment_1",
    # "tweet_feature_creation_timestamp_hour_shifted",
    # "tweet_feature_creation_timestamp_day_phase",
    # "tweet_feature_creation_timestamp_day_phase_shifted"
]

# The labels
Y_label = [
    f"tweet_feature_engagement_is_{label}"
]

if __name__ == '__main__':

    # Loading XGB model
    xgb_importance = XGBImportance(f"xgb_model_reply.model")

    # XGB Loading test
    X_remote_val = get_dataset_batch(X_label, val_dataset_id, 1, 0, 0.05)
    Y_remote_val = get_dataset_batch(Y_label, val_dataset_id, 1, 0, 0.05)

    r = sk_ins.permutation_importance(xgb_importance, X_remote_val, Y_remote_val, n_jobs=20)
    for i in r.importances_mean.argsort()[::-1]:
        print(f"{X_label[i]},"
              f"{r.importances_mean[i]:.3f},"
              f"{r.importances_std[i]:.3f}")

    # perm = PermutationImportance(xgb_importance, random_state=1).fit(X_remote_val, Y_remote_val)
    # result = eli5.show_weights(perm, top=None, feature_names=X_label)
    # with open(f"permutation_importance_{label}.html", "w") as file:
    #     file.write(result.data)

# class XGBWrapper():
#
#     def __init__(self, model):
#         self.model = model
#
#     def fit(self, X_test, Y_test):
