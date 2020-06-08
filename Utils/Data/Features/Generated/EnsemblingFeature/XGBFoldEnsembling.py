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
            "tweet_feature_number_of_previous_like_engagements",
            "tweet_feature_number_of_previous_reply_engagements",
            "tweet_feature_number_of_previous_retweet_engagements",
            "tweet_feature_number_of_previous_comment_engagements",
            "tweet_feature_number_of_previous_positive_engagements",
            "tweet_feature_number_of_previous_negative_engagements",
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
            "tweet_feature_creation_timestamp_hour_shifted",
            "tweet_feature_creation_timestamp_day_phase",
            "tweet_feature_creation_timestamp_day_phase_shifted"
        ]
        Y_label = [f"tweet_feature_engagement_is_{label}"]
        xgb_parameters = {
            'num_rounds': 1000,
            'max_depth': 15,
            'min_child_weight': 31,
            'colsample_bytree': 0.527867627326924,
            'learning_rate': 0.005512832288229653,
            'reg_alpha': 0.009776474546945663,
            'reg_lambda': 0.00019426092791670796,
            'scale_pos_weight': 1,
            'gamma': 3.634726434679246,
            'subsample': 0.1,
            'base_score': 0.4392,
            'max_delta_step': 14.898340668257987,
            'num_parallel_tree': 4
        }
        super().__init__(dataset_id, X_label, Y_label, xgb_parameters)

class XGBFoldEnsemblingLike2(XGBFoldEnsemblingAbstract):

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
            "tweet_feature_creation_timestamp_hour_shifted",
            "tweet_feature_creation_timestamp_day_phase",
            "tweet_feature_creation_timestamp_day_phase_shifted"
        ]
        Y_label = [f"tweet_feature_engagement_is_{label}"]
        xgb_parameters = {
            'num_rounds': 1000,
            'max_depth': 15,
            'min_child_weight': 48,
            'colsample_bytree': 0.6686372658935652,
            'learning_rate': 0.08643458398278793,
            'reg_alpha': 0.0037793354464627812,
            'reg_lambda': 0.0007101634591379154,
            'scale_pos_weight': 1,
            'gamma': 1.4499129276201115,
            'subsample': 0.9254329432712624,
            'base_score': 0.4392,
            'max_delta_step': 32.47945890085462,
            'num_parallel_tree': 7
        }
        super().__init__(dataset_id, X_label, Y_label, xgb_parameters)


class XGBFoldEnsemblingRetweet1(XGBFoldEnsemblingAbstract):

    def __init__(self, dataset_id: str):
        label = "retweet"
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
            "tweet_feature_number_of_previous_like_engagements",
            "tweet_feature_number_of_previous_reply_engagements",
            "tweet_feature_number_of_previous_retweet_engagements",
            "tweet_feature_number_of_previous_comment_engagements",
            "tweet_feature_number_of_previous_positive_engagements",
            "tweet_feature_number_of_previous_negative_engagements",
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
            "tweet_feature_creation_timestamp_hour_shifted",
            "tweet_feature_creation_timestamp_day_phase",
            "tweet_feature_creation_timestamp_day_phase_shifted"
        ]
        Y_label = [f"tweet_feature_engagement_is_{label}"]
        xgb_parameters = {
            'num_rounds': 1000,
            'max_depth': 10,
            'min_child_weight': 27,
            'colsample_bytree': 0.5986429226741212,
            'learning_rate': 0.011074446706129549,
            'reg_alpha': 0.2107137541643742,
            'reg_lambda': 0.001054755517667771,
            'scale_pos_weight': 1,
            'gamma': 0.5475527101663964,
            'subsample': 0.1999280583101749,
            'base_score': 0.1131,
            'max_delta_step': 5.451325914597373,
            'num_parallel_tree': 3
        }
        super().__init__(dataset_id, X_label, Y_label, xgb_parameters)

class XGBFoldEnsemblingReply1(XGBFoldEnsemblingAbstract):

    def __init__(self, dataset_id: str):
        label = "reply"
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
            "tweet_feature_number_of_previous_like_engagements",
            "tweet_feature_number_of_previous_reply_engagements",
            "tweet_feature_number_of_previous_retweet_engagements",
            "tweet_feature_number_of_previous_comment_engagements",
            "tweet_feature_number_of_previous_positive_engagements",
            "tweet_feature_number_of_previous_negative_engagements",
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
            "tweet_feature_creation_timestamp_hour_shifted",
            "tweet_feature_creation_timestamp_day_phase",
            "tweet_feature_creation_timestamp_day_phase_shifted"
        ]
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
            "tweet_feature_number_of_previous_like_engagements",
            "tweet_feature_number_of_previous_reply_engagements",
            "tweet_feature_number_of_previous_retweet_engagements",
            "tweet_feature_number_of_previous_comment_engagements",
            "tweet_feature_number_of_previous_positive_engagements",
            "tweet_feature_number_of_previous_negative_engagements",
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
            "tweet_feature_creation_timestamp_hour_shifted",
            "tweet_feature_creation_timestamp_day_phase",
            "tweet_feature_creation_timestamp_day_phase_shifted"
        ]
        Y_label = [f"tweet_feature_engagement_is_{label}"]
        xgb_parameters = {
            'num_rounds': 1000,
            'max_depth': 9,
            'min_child_weight': 2,
            'colsample_bytree': 0.5494029357749396,
            'learning_rate': 0.02453873911722272,
            'reg_alpha': 0.12197464179130557,
            'reg_lambda': 0.040075335957880695,
            'scale_pos_weight': 1,
            'gamma': 1.16012924256167954,
            'subsample': 0.9056298933716602,
            'base_score': 0.0274,
            'max_delta_step': 69.98549377129551,
            'num_parallel_tree': 7
        }
        super().__init__(dataset_id, X_label, Y_label, xgb_parameters)