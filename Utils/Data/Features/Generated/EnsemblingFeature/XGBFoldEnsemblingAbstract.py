from Utils.Data.DatasetUtils import is_test_or_val_set, get_train_set_id_from_test_or_val_set, \
    get_test_or_val_set_id_from_train
from Utils.Data.Features.Feature import Feature
from Utils.Data.Features.Generated.EnsemblingFeature.XGBEnsembling import XGBEnsembling
from Utils.Data.Features.Generated.GeneratedFeature import GeneratedFeaturePickle
import pathlib as pl
import numpy as np
import pandas as pd
import hashlib

class XGBFoldEnsemblingAbstract(GeneratedFeaturePickle):

    def __init__(self,
                 dataset_id: str,
                 features: list,
                 label: list,
                 param_dict: dict,
                 number_of_folds: int = 5,
                 ):
        hash_features = hashlib.md5(repr(features).encode('utf-8')).hexdigest()
        hash_label = hashlib.md5(repr(label).encode('utf-8')).hexdigest()
        hash_param_dict = hashlib.md5(repr(param_dict.items()).encode('utf-8')).hexdigest()
        hashcode = f"{hash_features}_{hash_label}_{hash_param_dict}"
        feature_name = f"xgb_fold_ensembling_{hashcode}"
        super().__init__(feature_name, dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/fold_ensembling/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/fold_ensembling/{self.feature_name}.csv.gz")
        self.features = features
        self.label = label
        self.param_dict = param_dict
        self.number_of_folds = number_of_folds

    def create_feature(self):

        # Check if the dataset id is train or test
        if not is_test_or_val_set(self.dataset_id):
            # Compute train and test dataset ids
            train_dataset_id = self.dataset_id

            # Load the dataset and shuffle it
            import Utils.Data.Data as data
            X_train = data.get_dataset(features=self.features, dataset_id=train_dataset_id).sample(frac=1)
            Y_train = data.get_dataset(features=self.label, dataset_id=train_dataset_id).sample(frac=1)

            # Compute the folds
            X_train_folds = np.array_split(X_train, self.number_of_folds)
            Y_train_folds = np.array_split(Y_train, self.number_of_folds)

            # Declare list of scores (of each folds)
            # used for aggregating results
            scores = []

            # Train multiple models with 1-fold out strategy
            for i in range(self.number_of_folds):
                # Compute the train set
                X_train = pd.concat([X_train_folds[x] for x in range(self.number_of_folds) if x is not i]).sample(frac=0.05)
                Y_train = pd.concat([Y_train_folds[x] for x in range(self.number_of_folds) if x is not i]).sample(frac=0.05)

                # Compute the test set
                X_test = X_train_folds[i]

                # Generate the dataset id for this fold
                fold_dataset_id = f"{self.feature_name}_{self.dataset_id}_fold_{i}"

                # Create the sub-feature
                feature = XGBEnsembling(fold_dataset_id, X_train, Y_train, X_test, self.param_dict)

                # Retrieve the scores
                scores.append(pd.DataFrame(feature.load_or_create(), index=X_test.index))
                print(scores)

            # Compute the resulting dataframe and sort the results
            result = pd.concat(scores).sort_index()

            # Save it as a feature
            self.save_feature(result)

        else:
            test_dataset_id = self.dataset_id
            train_dataset_id = get_train_set_id_from_test_or_val_set(test_dataset_id)

            # Load the train dataset
            import Utils.Data.Data as data
            X_train = data.get_dataset(features=self.features, dataset_id=train_dataset_id).sample(frac=0.05)
            Y_train = data.get_dataset(features=self.label, dataset_id=train_dataset_id).sample(frac=0.05)

            # Load the test dataset
            X_test = data.get_dataset(features=self.features, dataset_id=test_dataset_id)

            fold_dataset_id = f"{self.feature_name}_{self.dataset_id}"

            # Create the sub-feature
            feature = XGBEnsembling(fold_dataset_id, X_train, Y_train, X_test, self.param_dict)

            # Retrieve the scores
            result = pd.DataFrame(feature.load_or_create(), index=X_test.index)

            # Save it as a feature
            self.save_feature(result)

# Example
if __name__ == '__main__':
    label = "like"
    dataset_id = "train"
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
    feature = XGBFoldEnsemblingAbstract(dataset_id, X_label, Y_label, xgb_parameters)
    x = feature.load_or_create()