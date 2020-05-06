import time
from Models.GBM.XGBoost import XGBoost
from Utils.Data.Data import get_dataset
from Utils.Importance.XGBImportance import XGBImportance
import xgboost as xgb
import eli5
from eli5.sklearn import PermutationImportance

run_name = "submission_test"

train_dataset = "train_days_12345"
val_dataset = "val_days_6"
test_dataset = "val_days_7"

label = "like"

# Define the X label
X_label = [
    "raw_feature_creator_follower_count",
    "raw_feature_creator_following_count",
    "raw_feature_engager_follower_count",
    "raw_feature_engager_following_count",
    "tweet_feature_number_of_photo",
    "tweet_feature_number_of_video",
    "tweet_feature_number_of_gif",
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
    "engager_feature_number_of_previous_like_engagement_ratio",
    "engager_feature_number_of_previous_reply_engagement_ratio",
    "engager_feature_number_of_previous_retweet_engagement_ratio",
    "engager_feature_number_of_previous_comment_engagement_ratio",
    "engager_feature_number_of_previous_positive_engagement_ratio",
    "engager_feature_number_of_previous_negative_engagement_ratio",
    "engager_feature_number_of_previous_like_engagement_betweet_creator_and_engager_by_creator",
    "engager_feature_number_of_previous_reply_engagement_betweet_creator_and_engager_by_creator",
    "engager_feature_number_of_previous_retweet_engagement_betweet_creator_and_engager_by_creator",
    "engager_feature_number_of_previous_comment_engagement_betweet_creator_and_engager_by_creator",
    "engager_feature_number_of_previous_negative_engagement_betweet_creator_and_engager_by_creator",
    "engager_feature_number_of_previous_positive_engagement_betweet_creator_and_engager_by_creator",
    "engager_feature_number_of_previous_like_engagement_betweet_creator_and_engager_by_engager",
    "engager_feature_number_of_previous_reply_engagement_betweet_creator_and_engager_by_engager",
    "engager_feature_number_of_previous_retweet_engagement_betweet_creator_and_engager_by_engager",
    "engager_feature_number_of_previous_comment_engagement_betweet_creator_and_engager_by_engager",
    "engager_feature_number_of_previous_negative_engagement_betweet_creator_and_engager_by_engager",
    "engager_feature_number_of_previous_positive_engagement_betweet_creator_and_engager_by_engager",
    # "creator_and_engager_have_same_main_language",
    # "is_tweet_in_creator_main_language",
    # "is_tweet_in_engager_main_language",
    # "statistical_probability_main_language_of_engager_engage_tweet_language_1",
    # "statistical_probability_main_language_of_engager_engage_tweet_language_2"
]
# Define the Y label
Y_label = [
    f"tweet_feature_engagement_is_{label}"
]

if __name__ == '__main__':

    XGB = XGBoost(
        early_stopping_rounds=10,
        num_rounds=600,
        max_depth=5,
        min_child_weight=10,
        colsample_bytree=0.3,
        learning_rate = 0.005006767331003627,
        reg_alpha = 0.9555422741002999,
        reg_lambda = 0.03478629336559846,
        scale_pos_weight = 0.8,
        gamma = 0.3161358940346798,
        subsample = 0.3,
        base_score = 0.4,
        max_delta_step = 0.0
    )

    # To generate model

    X_train = get_dataset(X_label, train_dataset)
    Y_train = get_dataset(Y_label, train_dataset)

    X_val = get_dataset(X_label, val_dataset)
    Y_val = get_dataset(Y_label, val_dataset)

    # XGB Training
    training_start_time = time.time()
    XGB.fit(
        xgb.DMatrix(X_train, Y_train),
        xgb.DMatrix(X_val, Y_val)
            )
    print(f"Training time: {time.time() - training_start_time} seconds")

    # Saving XGB model
    XGB.save_model(f"{label}_permutation_model")

    # To load model

    # Loading XGB model
    xgb_importance = XGBImportance(f"{label}_permutation__model.model")


    # XGB Loading test
    X_test = get_dataset(X_label, test_dataset)
    Y_test = get_dataset(Y_label, test_dataset)

    perm = PermutationImportance(xgb_importance, random_state=1).fit(X_test, Y_test)
    result = eli5.show_weights(perm, top=None, feature_names=X_test.columns.tolist())
    with open(f"permutation_importance_{label}.html", "w") as file:
        file.write(result.data)

# class XGBWrapper():
#
#     def __init__(self, model):
#         self.model = model
#
#     def fit(self, X_test, Y_test):

