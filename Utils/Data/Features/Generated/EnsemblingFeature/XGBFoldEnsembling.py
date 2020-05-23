from Utils.Data.Features.Generated.EnsemblingFeature.XGBFoldEnsemblingAbstract import XGBFoldEnsemblingAbstract


class XGBFoldEnsemblingLike1(XGBFoldEnsemblingAbstract):

    def __init__(self, dataset_id: str):
        label = "like"
        dataset_id = dataset_id
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
        Y_label = [f"tweet_feature_engagement_is_{label}"]
        xgb_parameters = {
            'num_rounds': 125,
            'max_depth': 14,
            'min_child_weight': 2,
            'colsample_bytree': 0.986681081839595,
            'learning_rate': 0.084682683195233,
            'reg_alpha': 0.000709687642837,
            'reg_lambda': 0.004524218043733,
            'scale_pos_weight': 0.701767017039216,
            'gamma': 8.5138404902064,
            'subsample': 0.726578069020986,
            'base_score': 0.5,
            'max_delta_step': 0.018196005449355,
            'num_parallel_tree': 13
        }
        super().__init__(dataset_id, X_label, Y_label, xgb_parameters)

class XGBFoldEnsemblingRetweet1(XGBFoldEnsemblingAbstract):

    def __init__(self, dataset_id: str):
        label = "retweet"
        dataset_id = dataset_id
        X_label = ["raw_feature_creator_follower_count"]
        Y_label = [f"tweet_feature_engagement_is_{label}"]
        xgb_parameters = {
            'num_rounds': 1000,
            'max_depth': 15,
            'min_child_weight': 6,
            'colsample_bytree': 0.33818954844496046,
            'learning_rate': 0.130817833734442,
            'reg_alpha': 0.0005311830218970207,
            'reg_lambda': 0.00018776522886741493,
            'scale_pos_weight': 0.7170586642475405,
            'gamma': 0.38859834472037047,
            'subsample': 0.3071905565109999,
            'base_score': 0.40486498623622924,
            'max_delta_step': 0.0653504311420456,
            'num_parallel_tree': 4
        }
        super().__init__(dataset_id, X_label, Y_label, xgb_parameters)

class XGBFoldEnsemblingReply1(XGBFoldEnsemblingAbstract):

    def __init__(self, dataset_id: str):
        label = "reply"
        dataset_id = dataset_id
        X_label = ["raw_feature_creator_follower_count"]
        Y_label = [f"tweet_feature_engagement_is_{label}"]
        xgb_parameters = {
            'num_rounds': 1000,
            'max_depth': 15,
            'min_child_weight': 6,
            'colsample_bytree': 0.33818954844496046,
            'learning_rate': 0.130817833734442,
            'reg_alpha': 0.0005311830218970207,
            'reg_lambda': 0.00018776522886741493,
            'scale_pos_weight': 0.7170586642475405,
            'gamma': 0.38859834472037047,
            'subsample': 0.3071905565109999,
            'base_score': 0.40486498623622924,
            'max_delta_step': 0.0653504311420456,
            'num_parallel_tree': 4
        }
        super().__init__(dataset_id, X_label, Y_label, xgb_parameters)


class XGBFoldEnsemblingComment1(XGBFoldEnsemblingAbstract):

    def __init__(self, dataset_id: str):
        label = "comment"
        dataset_id = dataset_id
        X_label = ["raw_feature_creator_follower_count"]
        Y_label = [f"tweet_feature_engagement_is_{label}"]
        xgb_parameters = {
            'num_rounds': 1000,
            'max_depth': 15,
            'min_child_weight': 6,
            'colsample_bytree': 0.33818954844496046,
            'learning_rate': 0.130817833734442,
            'reg_alpha': 0.0005311830218970207,
            'reg_lambda': 0.00018776522886741493,
            'scale_pos_weight': 0.7170586642475405,
            'gamma': 0.38859834472037047,
            'subsample': 0.3071905565109999,
            'base_score': 0.40486498623622924,
            'max_delta_step': 0.0653504311420456,
            'num_parallel_tree': 4
        }
        super().__init__(dataset_id, X_label, Y_label, xgb_parameters)