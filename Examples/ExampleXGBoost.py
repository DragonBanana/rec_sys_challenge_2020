from Models.GBM.XGBoost import XGBoost
from Utils.Data import Data
from Utils.Submission.Submission import create_submission_file

if __name__ == '__main__':
    train_dataset = "train_split_with_timestamp_from_train_random_seed_888_timestamp_threshold_1581465600_holdout_75"
    test_dataset = "val_split_with_timestamp_from_train_random_seed_888_timestamp_threshold_1581465600_holdout_75"

    # Define the X label
    X_label = [
        # "mapped_feature_tweet_id",
        # "mapped_feature_creator_id",
        # "mapped_feature_engager_id",
        "raw_feature_creator_follower_count",
        "raw_feature_creator_following_count",
        "raw_feature_engager_follower_count",
        "raw_feature_engager_following_count"
        "tweet_feature_number_of_photo",
        "tweet_feature_number_of_video",
        "tweet_feature_number_of_gif",
        "tweet_feature_is_reply",
        "tweet_feature_is_retweet",
        "tweet_feature_is_quote",
        "tweet_feature_is_top_level"
    ]
    # Define the Y label
    Y_label = [
        "tweet_feature_engagement_is_like"
    ]

    # Load train data
    X_train, Y_train = Data.get_dataset_xgb(train_dataset, X_label, Y_label)

    # Load test data
    X_test, Y_test = Data.get_dataset_xgb(test_dataset, X_label, Y_label)

    XGB = XGBoost()

    XGB.fit(X_train, Y_train)
    predictions = XGB.evaluate(X_test, Y_test)

    tweets = Data.get_feature("raw_feature_tweet_id", test_dataset)["raw_feature_tweet_id"].array
    users = Data.get_feature("raw_feature_engager_id", test_dataset)["raw_feature_engager_id"].array

    create_submission_file(tweets, users, predictions, "test_submission.csv")