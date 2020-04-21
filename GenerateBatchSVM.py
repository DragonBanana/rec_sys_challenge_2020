import numpy as np
from Utils.Data import Data
import sklearn.datasets as skd
import pathlib as pl
import multiprocessing as mp
import functools
import pandas as pd

def load_and_dump(
        i,
        batch_n_split,
        dataset_id,
        X_label,
        folder,
        sample=1
):
    if dataset_id != "test":
        X, _ = Data.get_dataset_xgb_batch(
            batch_n_split,
            i,
            dataset_id,
            X_label,
            sample=sample)
        for label in labels:
            Y, _ = Data.get_dataset_xgb_batch(
                batch_n_split,
                i,
                dataset_id,
                [f"tweet_feature_engagement_is_{label}"],
                sample=sample
            )
            assert len(X) == len(Y)
            skd.dump_svmlight_file(
                X=X,
                y=Y[f"tweet_feature_engagement_is_{label}"].array,
                f=f"{folder}/{dataset_id}_{label}_batch_{i}.svm"
            )
    else:
        X_train, _ = Data.get_dataset_xgb_batch(batch_n_split, i, dataset_id, X_label)
        skd.dump_svmlight_file(X_train, np.zeros(len(X_train)), f"{folder}/{dataset_id}_batch_{i}.svm")


if __name__ == "__main__":
    labels = [
        # "like",
        "reply",
        "retweet",
        "comment"
    ]

    train_batch_n_split = 5
    val_batch_n_split = 1

    train_dataset_id = "train_days_123456"
    val_dataset_id = "val_days_7"

    main_dir = "svm_files"

    gen_train = True
    gen_val = True

    sample = 0.25

    # ------------------------------------------
    #           BATCH EXAMPLE
    # ------------------------------------------
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
        "engager_feature_number_of_previous_negative_engagement_ratio"
    ]

    Y_label = [
        f"tweet_feature_engagement_is_{label}" for label in labels
    ]

    folder = f"{main_dir}"

    pl.Path(folder).mkdir(parents=True, exist_ok=True)

    if gen_train:
        with mp.Pool(6) as pool:
            pool.map(functools.partial(load_and_dump,
                                       batch_n_split=train_batch_n_split,
                                       dataset_id=train_dataset_id,
                                       X_label=X_label,
                                       folder=folder,
                                       sample=sample
                                       ), range(train_batch_n_split))

    if gen_val:
        with mp.Pool(6) as pool:
            pool.map(functools.partial(load_and_dump,
                                       batch_n_split=val_batch_n_split,
                                       dataset_id=val_dataset_id,
                                       X_label=X_label,
                                       folder=folder,
                                       sample=1
                                       ), range(val_batch_n_split))


