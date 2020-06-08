# from Models.NN.NNRecNew import NNRecNew
from Models.NN.NNRec import DistilBertRec
from Utils.Data.Data import get_dataset, get_feature, get_feature_reader
from Utils.Submission.Submission import create_submission_file
# from sklearn.model_selection import train_test_split
import numpy as np
import time
from Utils.TelegramBot import telegram_bot_send_update

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
                "number_of_engagements_like", # 11
                "number_of_engagements_retweet", #  12
                "number_of_engagements_reply", # 13
                "number_of_engagements_comment", #  14
                "number_of_engagements_positive", #  15
                "number_of_engagements_negative", # 16
                "engager_feature_number_of_previous_like_engagement_ratio",  # 17
                "engager_feature_number_of_previous_reply_engagement_ratio",  # 18
                "engager_feature_number_of_previous_retweet_engagement_ratio",  # 19
                "engager_feature_number_of_previous_comment_engagement_ratio",  # 20
                "engager_feature_number_of_previous_positive_engagement_ratio",  # 21
                "engager_feature_number_of_previous_negative_engagement_ratio"  # 22
    ]
    '''
    '''
    feature_list = [
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
        #"tweet_feature_number_of_mentions",
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
        "number_of_engagements_like",
        "number_of_engagements_retweet",
        "number_of_engagements_reply",
        "number_of_engagements_comment",
        "number_of_engagements_negative",
        "number_of_engagements_positive",
        "tweet_feature_creation_timestamp_hour_shifted",
        "tweet_feature_creation_timestamp_day_phase",
        "tweet_feature_creation_timestamp_day_phase_shifted"
    ]
    '''
    
    feature_list = [
        "raw_feature_creator_follower_count",  # 0
        "raw_feature_creator_following_count",  # 1
    ]

        class_label = "like"

    ip = '34.240.194.211'
    submission_filename = f"nn_yj_only_ratios_{class_label}.csv"

    chunksize = 2048

    train_dataset = "cherry_train"
    test_dataset = "new_test"

    ffnn_params = {'hidden_size_1': 128, 'hidden_size_2': 64, 'hidden_dropout_prob_1': 0.1, 'hidden_dropout_prob_2': 0.1}
    rec_params = {'epochs': 5, 'weight_decay': 1e-5, 'lr': 1e-4, 'cap_length': 128, 'ffnn_params': ffnn_params, 'class_label': class_label}

    saved_model_path = "saved_models/saved_model_yj_like_0.0001_774_128_64_0.1_0.1_epoch_5"

    rec = DistilBertRec(**rec_params)

    train_df = get_dataset(features=feature_list, dataset_id=train_dataset)
    train_df = train_df.head(1920000)
    train_df = rec._normalize_features(train_df, is_train=True)

    ###   PREDICTION
    test_df = get_dataset(features=feature_list, dataset_id=test_dataset)
    #test_df = test_df.head(2500)

    prediction_start_time = time.time()

    text_test_reader_df = get_feature_reader(feature_name="raw_feature_tweet_text_token",
                                            dataset_id=test_dataset,
                                            chunksize=chunksize)
    predictions = rec.get_prediction(df_test_features=test_df,
                                     df_test_tokens_reader=text_test_reader_df,
                                     pretrained_model_dict_path=saved_model_path)
    print(f"Prediction time: {time.time() - prediction_start_time} seconds")

    print(predictions)
    print(predictions.shape)

    tweets = get_feature("raw_feature_tweet_id", test_dataset)["raw_feature_tweet_id"].array
    users = get_feature("raw_feature_engager_id", test_dataset)["raw_feature_engager_id"].array

    #tweets = tweets.head(2500).array
    #users = users.head(2500).array

    create_submission_file(tweets, users, predictions, submission_filename)

    bot_string = f"DistilBertDoubleInput NN - {class_label} \n ---------------- \n"
    bot_string = bot_string + f"@lucaconterio la submission pronta! \nIP: {ip} \nFile: {submission_filename}"
    telegram_bot_send_update(bot_string)


if __name__ == '__main__':
    main()
