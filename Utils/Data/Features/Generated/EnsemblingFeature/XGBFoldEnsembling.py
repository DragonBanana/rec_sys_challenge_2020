from Utils.Data.Features.Generated.EnsemblingFeature.XGBFoldEnsemblingAbstract import XGBFoldEnsemblingAbstract


class XGBFoldEnsemblingLike1(XGBFoldEnsemblingAbstract):

    def __init__(self, dataset_id: str):
        label = "like"
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