from ParamTuning.Optimizer import Optimizer
import pathlib as pl
import xgboost as xgb

from Utils.Data.Data import get_dataset_xgb_batch, get_dataset, get_dataset_batch
from Utils.Data.DataUtils import cache_dataset_as_svm

labels = [
    "like",
    "retweet",
    "reply",
    "comment"
]

folder = f"skopt_result"

# Cached svm filename
svm_filename = "skopt_svm_file"

train_dataset_id = "train_days_123456"
val_dataset_id = "val_days_7"

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
    "engager_main_language",
    "creator_main_language",
    "creator_and_engager_have_same_main_language",
    "is_tweet_in_creator_main_language",
    "is_tweet_in_engager_main_language",
    "statistical_probability_main_language_of_engager_engage_tweet_language_1",
    "statistical_probability_main_language_of_engager_engage_tweet_language_2",
    "tweet_feature_token_length",
    "tweet_feature_token_length_unique",
    "engager_feature_knows_hashtag_positive",
    "engager_feature_knows_hashtag_negative",
    "engager_feature_knows_hashtag_like",
    "engager_feature_knows_hashtag_reply",
    "engager_feature_knows_hashtag_rt",
    "hashtag_similarity_fold_ensembling_positive",
    "link_similarity_fold_ensembling_positive",
    "domain_similarity_fold_ensembling_positive"
]


def main():
    for label in labels:
        run(label)

def run(label: str):
    model_name = "xgboost_classifier"
    kind = label

    # Define the Y label
    Y_label = [
        f"tweet_feature_engagement_is_{label}"
    ]

    OP = Optimizer(model_name,
                   kind,
                   mode=0,
                   make_log=True,
                   make_save=True,
                   auto_save=True,
                   path=f"{folder}/{label}",
                   path_log=f"{folder}/{label}")

    # if pl.Path(f"{label}.save.npz").is_file():
    #     OP.loadModel(f"{label}.save.npz")

    # Load the training dataset
    X_train = get_dataset_batch(X_label, train_dataset_id, 1, 0, 0.5)
    Y_train = get_dataset_batch(Y_label, train_dataset_id, 1, 0, 0.5)
    # Cache the training dataset
    cache_dataset_as_svm(svm_filename, X_train, Y_train)
    train = xgb.DMatrix(f"{svm_filename}.svm#~/{svm_filename}.cache")
    train.feature_names = X_train.columns
    # Delete the data structure that are not useful anymore
    del X_train, Y_train

    X_val, Y_val = get_dataset_xgb_batch(2, 0, val_dataset_id, X_label, Y_label, 1)
    val = xgb.DMatrix(X_val, Y_val)
    X_test, Y_test = get_dataset_xgb_batch(2, 1, val_dataset_id, X_label, Y_label, 1)
    test = xgb.DMatrix(X_test, Y_test)




    del X_train, Y_train, X_val, Y_val, X_test, Y_test

    OP.setParameters(n_calls=50, n_random_starts=20)
    OP.defineMI()
    # Use GenerateBatchSVM in order to generate the batches
    OP.loadTrainData(holder_train=train)
    OP.loadValData(holder_val=val)
    OP.loadTestData(holder_test=test)
    OP.setParamsXGB(early_stopping_rounds=15, tree_method='hist')

    OP.optimize()


if __name__ == "__main__":
    main()
