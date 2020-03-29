from Utils.Data import Data
from Utils.Data.Data import get_dataset_xgb
from Utils.Data.DataUtils import create_all
from Utils.Data.Split import TimestampBasedSplit
import pandas as pd

if __name__ == '__main__':

    x = get_dataset_xgb()

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
    # ], "train")
    #
    # for column in x.columns:
    #     print(x[column])