from Utils.Data import Data
from Utils.Data.Data import get_dataset_xgb, get_feature
from Utils.Data.DataUtils import create_all
from Utils.Data.Split import TimestampBasedSplit
import pandas as pd

if __name__ == '__main__':

    # x = get_dataset_xgb()
    #
    # pd.set_option('display.max_rows', 500)
    # pd.set_option('display.max_columns', 500)
    # pd.set_option('display.width', 1000)
    #
    # x = Data.get_dataset([
    #     "raw_feature_tweet_id",
    #     # "tweet_feature_number_of_photo",
    #     # "tweet_feature_number_of_video",
    #     # "tweet_feature_number_of_gif",
    #     # "tweet_feature_is_reply",
    #     # "tweet_feature_is_retweet",
    #     # "tweet_feature_is_quote",
    #     # "tweet_feature_is_top_level",
    #     "tweet_feature_engagement_is_like",
    #     "tweet_feature_engagement_is_retweet",
    #     "tweet_feature_engagement_is_comment",
    #     "tweet_feature_engagement_is_reply",
    #     "tweet_feature_engagement_is_positive"
    # ], "train_split_with_timestamp_from_train_random_seed_888_timestamp_threshold_1581465600_holdout_75")
    #
    # for column in x.columns:
    #     print(x[column])

    X_label = [
        "mapped_feature_tweet_id",
        "mapped_feature_creator_id",
        "mapped_feature_engager_id",
        "tweet_feature_number_of_photo",
        "tweet_feature_number_of_video",
        "tweet_feature_number_of_gif",
        "tweet_feature_is_reply",
        "tweet_feature_is_retweet",
        "tweet_feature_is_quote",
        "tweet_feature_is_top_level"
    ]
    # Define the Y label
    Y_label = "tweet_feature_engagement_is_like"

    for x in X_label:
        print(get_feature(x, "train_split_with_timestamp_from_train_random_seed_888_timestamp_threshold_1581465600_holdout_75"))