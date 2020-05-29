# from Models.NN.NNRecNew import NNRecNew
from Models.NN.NNRec import DistilBertRec
from Utils.Data.Data import get_dataset, get_feature, get_feature_reader
# from sklearn.model_selection import train_test_split
import numpy as np


def run(params):

    rec = DistilBertRec(**params)

    feature_list = [
        #        "raw_feature_creator_follower_count",  # 0
        #        "raw_feature_creator_following_count",  # 1
        #        "raw_feature_engager_follower_count",  # 2
        #        "raw_feature_engager_following_count",  # 3
        #        "tweet_feature_number_of_photo",  # 4
        #        "tweet_feature_number_of_video",  # 5
        #        "tweet_feature_number_of_gif",  # 6
        #        "tweet_feature_number_of_hashtags",  # 7
        #        "tweet_feature_creation_timestamp_hour",  # 8
        #        "tweet_feature_creation_timestamp_week_day",  # 9
        #        "tweet_feature_number_of_mentions",  # 10
        #        "engager_feature_number_of_previous_like_engagement",  # 11
        #        "engager_feature_number_of_previous_reply_engagement",  # 12
        #        "engager_feature_number_of_previous_retweet_engagement",  # 13
        #        "engager_feature_number_of_previous_comment_engagement",  # 14
        #        "engager_feature_number_of_previous_positive_engagement",  # 15
        #        "engager_feature_number_of_previous_negative_engagement",  # 16
        #"engager_feature_number_of_previous_engagement",  # 17 ciao nico :-)
        "engager_feature_number_of_previous_like_engagement_ratio",  # 18
        "engager_feature_number_of_previous_reply_engagement_ratio",  # 19
        "engager_feature_number_of_previous_retweet_engagement_ratio",  # 20
        "engager_feature_number_of_previous_comment_engagement_ratio",  # 21
        "engager_feature_number_of_previous_positive_engagement_ratio",  # 22
        "engager_feature_number_of_previous_negative_engagement_ratio"  # 23
    ]

    chunksize = 100
    n_data_train = 1000
    n_data_val = 1000

    train_dataset = "holdout/train"
    val_dataset = "holdout/test"
    test_dataset = "test"

    print(f"n_data_train: {n_data_train}")
    print(f"n_data_val: {n_data_val}")

    print("params:")
    print(params)

    print(f"train_dataset: {train_dataset}")
    print(f"val_dataset: {val_dataset}")

    feature_train_df = get_dataset(features=feature_list, dataset_id=train_dataset)
    #   feature_train_df, _ = train_test_split(feature_train_df, train_size=0.2)
    feature_train_df = feature_train_df.head(n_data_train)

    label_train_df = get_feature(feature_name="tweet_feature_engagement_is_like", dataset_id=train_dataset)
    label_train_df = label_train_df.head(n_data_train)

    text_train_reader_df = get_feature_reader(feature_name="raw_feature_tweet_text_token", dataset_id=train_dataset,
                                              chunksize=chunksize)

    #    label_train_df, _ = train_test_split(label_train_df, train_size=0.2)

    feature_val_df = get_dataset(features=feature_list, dataset_id=val_dataset)
    feature_val_df = feature_val_df.head(n_data_val)

    label_val_df = get_feature(feature_name="tweet_feature_engagement_is_like", dataset_id=val_dataset)
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

    ###   PREDICTION
    test_df = get_dataset(features=feature_list, dataset_id=test_dataset)

    prediction_start_time = time.time()
    predictions = rec.get_prediction(test_df.to_numpy())
    print(f"Prediction time: {time.time() - prediction_start_time} seconds")

    tweets = get_feature("raw_feature_tweet_id", test_dataset)["raw_feature_tweet_id"].array
    users = get_feature("raw_feature_engager_id", test_dataset)["raw_feature_engager_id"].array

    create_submission_file(tweets, users, predictions, "nn_submission_like.csv")


def main():
    #for dropout in [0.3, 0.5]:
    #    for hidden_size_2 in [32, 64]:
    params = {'hidden_dropout_prob': 0.5, 'weight_decay': 1e-5, 'hidden_size_2': 256, 'hidden_size_3': 64}
    run(params)


if __name__ == '__main__':
    main()

