# from Models.NN.NNRecNew import NNRecNew
from Models.NN.NNRec import DistilBertRec
from Utils.Data.Data import get_dataset, get_feature, get_feature_reader
from Utils.Submission.Submission import create_submission_file
# from sklearn.model_selection import train_test_split
import numpy as np
import time
from Utils.TelegramBot import 

def main():

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
    '''

    class_label = "retweet"

    ip = '34.240.194.211'
    submission_filename = f"nn_submission_{class_label}.csv"

    chunksize = 2048

    test_dataset = "test"

    ffnn_params = {'hidden_size_1': 128, 'hidden_size_2': 64, 'hidden_dropout_prob_1': 0.5, 'hidden_dropout_prob_2': 0.5}
    rec_params = {'epochs': 5, 'weight_decay': 1e-5, 'lr': 2e-5, 'cap_length': 128, 'ffnn_params': ffnn_params, 'class_label': class_label}

    saved_model_path = "saved_models/saved_model_0.5_128_64_epoch_4"

    rec = DistilBertRec(**rec_params)

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
    bot_string = bot_string + f"@lucaconterio la submission è pronta! \nIP: {ip} \nFile: {submission_filename}"
    telegram_bot_send_update(bot_string)


if __name__ == '__main__':
    main()
