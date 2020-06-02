from Models.NN.NNRec import DistilBertRec
from Utils.NN.TorchModels import FFNN2, FFNN1
from Utils.Data.Data import get_dataset, get_feature, get_feature_reader
from Utils.Submission.Submission import create_submission_file
# from sklearn.model_selection import train_test_split
import numpy as np
import time

def main():
    '''
    feature_list = [
                "raw_feature_creator_follower_count",  # 0
                "raw_feature_creator_following_count",  # 1
                "raw_feature_engager_follower_count",  # 2
                "raw_feature_engager_following_count",  # 3
                "tweet_feature_number_of_photo",  # 4
                "tweet_feature_number_of_video",  # 5
                "tweet_feature_number_of_gif",  # 6
                "tweet_feature_number_of_hashtags",  # 7
                "tweet_feature_creation_timestamp_hour",  # 8
                "tweet_feature_creation_timestamp_week_day",  # 9
                "tweet_feature_number_of_mentions",  # 10
                "engager_feature_number_of_previous_like_engagement",  # 11
                "engager_feature_number_of_previous_reply_engagement",  # 12
                "engager_feature_number_of_previous_retweet_engagement",  # 13
                "engager_feature_number_of_previous_comment_engagement",  # 14
                "engager_feature_number_of_previous_positive_engagement",  # 15
                "engager_feature_number_of_previous_negative_engagement",  # 16
                "engager_feature_number_of_previous_engagement",  # 17 ciao nico :-)
                "engager_feature_number_of_previous_like_engagement_ratio",  # 18
                "engager_feature_number_of_previous_reply_engagement_ratio",  # 19
                "engager_feature_number_of_previous_retweet_engagement_ratio",  # 20
                "engager_feature_number_of_previous_comment_engagement_ratio",  # 21
                "engager_feature_number_of_previous_positive_engagement_ratio",  # 22
                "engager_feature_number_of_previous_negative_engagement_ratio"  # 23
    ]

    '''
    feature_list = [
        "raw_feature_creator_follower_count",  # 0
        "raw_feature_creator_following_count",  # 1
    ]

    chunksize = 2000
    n_data_train = 2000 #* 10000
    n_data_val = 2000 #* 10000

    train_dataset = "holdout/train"
    val_dataset = "holdout/test"
    test_dataset = "test"

    class_label = "retweet"   # retweet, reply, like, comment

    ffnn_params = {'hidden_size_1': 128, 'hidden_size_2': 64, 'hidden_dropout_prob_1': 0.5, 'hidden_dropout_prob_2': 0.5}
    rec_params = {'epochs': 5, 'weight_decay': 1e-5, 'lr': 2e-5, 'cap_length': 128, 'ffnn_params': ffnn_params, 'class_label': class_label}

    rec = DistilBertRec(**rec_params)

    print(f"n_data_train: {n_data_train}")
    print(f"n_data_val: {n_data_val}")

    print(f"ffnn_params: {ffnn_params}")
    print(f"bert_params: {rec_params}")

    print(f"train_dataset: {train_dataset}")
    print(f"val_dataset: {val_dataset}")

    feature_train_df = get_dataset(features=feature_list, dataset_id=train_dataset)
    #   feature_train_df, _ = train_test_split(feature_train_df, train_size=0.2)
    feature_train_df = feature_train_df.head(n_data_train)

    label_train_df = get_feature(feature_name=f"tweet_feature_engagement_is_{class_label}", dataset_id=train_dataset)
    label_train_df = label_train_df.head(n_data_train)

    text_train_reader_df = get_feature_reader(feature_name="raw_feature_tweet_text_token", dataset_id=train_dataset,
                                              chunksize=chunksize)

    #    label_train_df, _ = train_test_split(label_train_df, train_size=0.2)

    feature_val_df = get_dataset(features=feature_list, dataset_id=val_dataset)
    feature_val_df = feature_val_df.head(n_data_val)

    label_val_df = get_feature(feature_name=f"tweet_feature_engagement_is_{class_label}", dataset_id=val_dataset)
    label_val_df = label_val_df.head(n_data_val)

    text_val_reader_df = get_feature_reader(feature_name="raw_feature_tweet_text_token", dataset_id=val_dataset,
                                            chunksize=chunksize)

    ###   TRAINING
    stats = rec.fit(df_train_features=feature_train_df,
                    df_train_tokens_reader=text_train_reader_df,
                    df_train_label=label_train_df,
                    df_val_features=feature_val_df,
                    df_val_tokens_reader=text_val_reader_df,
                    df_val_label=label_val_df,
                    cat_feature_set=set([]),
                    subsample=0.1) # subsample percentage of each batch

    print("STATS: \n")
    print(stats)
    with open('stats.txt', 'w+') as f:
        for s in stats:
            f.write(str(s) + '\n')


if __name__ == '__main__':
    main()
