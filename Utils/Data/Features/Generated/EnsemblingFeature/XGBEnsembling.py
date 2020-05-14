import pathlib

from Models.GBM.XGBoost import XGBoost
from Utils.Data.Data import get_dataset_xgb_batch, get_dataset
from Utils.Data.DataUtils import cache_dataset_as_svm
from Utils.Data.Features.Generated.EnsemblingFeature.EnsemblingFeatureAbstract import EnsemblingFeatureAbstract
import xgboost as xgb
import random
import pandas as pd


class XGBEnsembling(EnsemblingFeatureAbstract):
    path = "xgb_ensembling"
    feature_name = "xgb_ensembling"

    def __init__(self,
                 dataset_id: str,
                 df_train: pd.DataFrame,
                 df_train_label: pd.DataFrame,
                 df_to_predict: pd.DataFrame,
                 param_dict: dict
                 ):
        self.dataset_id = dataset_id
        super().__init__(df_train, df_train_label, df_to_predict, param_dict)

    def _get_dataset_id(self):
        return self.dataset_id

    def _get_path(self):
        return self.dataset_id

    def _get_feature_name(self):
        return self.path

    def _load_model(self):
        model = XGBoost()
        model.load_model(self.model_path)
        return model

    def _train_and_save(self):
        # Generate a random number
        random_n = random.random()
        # Initiate XGBoost wrapper
        xgb_wrapper = XGBoost(
            num_rounds=self.param_dict['num_rounds'],
            max_depth=self.param_dict['max_depth'],
            min_child_weight=self.param_dict['min_child_weight'],
            colsample_bytree=self.param_dict['colsample_bytree'],
            learning_rate=self.param_dict['learning_rate'],
            reg_alpha=self.param_dict['reg_alpha'],
            reg_lambda=self.param_dict['reg_lambda'],
            scale_pos_weight=self.param_dict['scale_pos_weight'],
            gamma=self.param_dict['gamma'],
            subsample=self.param_dict['subsample'],
            base_score=self.param_dict['base_score'],
            max_delta_step=self.param_dict['max_delta_step'],
            num_parallel_tree=self.param_dict['num_parallel_tree']
        )
        # Cache the train matrix as libsvm
        cache_dataset_as_svm(f"temp_ensembling_{random_n}", self.df_train, self.df_train_label)
        # Load the train matrix + external memory
        train = xgb.DMatrix(f"temp_ensembling_{random_n}.svm#temp_ensembling_{random_n}.cache")
        # Overwrite the feature names for consistency
        train.feature_names = self.df_train.columns
        # Fit the model
        xgb_wrapper.fit(dmat_train=train)
        # Create the directory (where the model is saved) if it does not exist
        pathlib.Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        # Save the model
        xgb_wrapper.save_model(filename = self.model_path)

    def create_feature(self):
        # Load the model
        model = self._get_model()
        # Create the test DMatrix for xgboost
        test = xgb.DMatrix(self.df_to_predict)
        # Predict the labels
        predictions = model.get_prediction(test)
        # Encapsulate the labels
        result = pd.DataFrame(predictions, index=self.df_to_predict.index)
        # Sort the labels
        result.sort_index(inplace=True)
        # Save the result
        self.save_feature(result)


# Example
if __name__ == '__main__':
    label = "like"
    dataset_id = "test_xgb_ensemble_train_dataset"
    train_dataset_id = "train"
    test_dataset_id = "test"
    X_label = ["raw_feature_creator_follower_count"]
    Y_label = [f"tweet_feature_engagement_is_{label}"]
    X_train = get_dataset(X_label, train_dataset_id)
    Y_train = get_dataset(Y_label, train_dataset_id)
    X_test = get_dataset(X_label, test_dataset_id)
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
    feature = XGBEnsembling(dataset_id, X_train, Y_train, X_test, xgb_parameters)
    feature.create_feature()