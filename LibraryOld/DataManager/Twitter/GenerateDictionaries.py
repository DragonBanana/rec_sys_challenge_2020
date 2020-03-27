import TwitterData
import pandas as pd
import gzip
import json


def save_dictionary(dictionary, path):
    with gzip.GzipFile(path, 'w') as outfile:
        outfile.write(json.dumps(dictionary).encode('utf-8'))


def generate_tweet_id_dictionary():
    data = pd.concat([
        TwitterData.get_resource("training_raw_tweet_features_tweet_id")["training_raw_tweet_features_tweet_id"],
        TwitterData.get_resource("validation_raw_tweet_features_tweet_id")["validation_raw_tweet_features_tweet_id"]
    ])

    dictionary = pd.DataFrame(data.unique()).to_dict()[0]

    save_dictionary(dictionary, "tweet_id_inverse_mapping.json.gz")

    inverse_dictionary = {v: k for k, v in dictionary.items()}

    save_dictionary(inverse_dictionary, "tweet_id_direct_mapping.json.gz")


def generate_user_id_dictionary():
    data = pd.concat([
        TwitterData.get_resource("training_raw_creator_features_user_id")["training_raw_creator_features_user_id"],
        TwitterData.get_resource("validation_raw_creator_features_user_id")["validation_raw_creator_features_user_id"],
        TwitterData.get_resource("training_raw_engager_features_user_id")["training_raw_engager_features_user_id"],
        TwitterData.get_resource("validation_raw_engager_features_user_id")["validation_raw_engager_features_user_id"],
    ])

    dictionary = pd.DataFrame(data.unique()).to_dict()[0]

    save_dictionary(dictionary, "user_id_inverse_mapping.json.gz")

    inverse_dictionary = {v: k for k, v in dictionary.items()}

    save_dictionary(inverse_dictionary, "user_id_direct_mapping.json.gz")


def generate_language_id_dictionary():
    data = pd.concat([
        TwitterData.get_resource("training_raw_tweet_features_language")["training_raw_tweet_features_language"],
        TwitterData.get_resource("validation_raw_tweet_features_language")["validation_raw_tweet_features_language"],
    ])

    dictionary = pd.DataFrame(data.unique()).to_dict()[0]

    save_dictionary(dictionary, "language_id_inverse_mapping.json.gz")

    inverse_dictionary = {v: k for k, v in dictionary.items()}

    save_dictionary(inverse_dictionary, "language_id_direct_mapping.json.gz")


def generate_domain_id_dictionary():
    data = pd.concat([
        TwitterData.get_resource("training_raw_tweet_features_domains")["training_raw_tweet_features_domains"],
        TwitterData.get_resource("validation_raw_tweet_features_domains")["validation_raw_tweet_features_domains"],
    ])

    data = pd.DataFrame([y for x in data.dropna() for y in x.split('\t')])
    data = data[data.columns[0]]

    dictionary = pd.DataFrame(data.unique()).to_dict()[0]

    save_dictionary(dictionary, "domain_id_inverse_mapping.json.gz")

    inverse_dictionary = {v: k for k, v in dictionary.items()}

    save_dictionary(inverse_dictionary, "domain_id_direct_mapping.json.gz")


def generate_link_id_dictionary():
    data = pd.concat([
        TwitterData.get_resource("training_raw_tweet_features_links")["training_raw_tweet_features_links"],
        TwitterData.get_resource("validation_raw_tweet_features_links")["validation_raw_tweet_features_links"],
    ])

    data = pd.DataFrame([y for x in data.dropna() for y in x.split('\t')])
    data = data[data.columns[0]]

    dictionary = pd.DataFrame(data.unique()).to_dict()[0]

    save_dictionary(dictionary, "link_id_inverse_mapping.json.gz")

    inverse_dictionary = {v: k for k, v in dictionary.items()}

    save_dictionary(inverse_dictionary, "link_id_direct_mapping.json.gz")


def generate_hashtag_id_dictionary():
    data = pd.concat([
        TwitterData.get_resource("training_raw_tweet_features_hashtags")["training_raw_tweet_features_hashtags"],
        TwitterData.get_resource("validation_raw_tweet_features_hashtags")["validation_raw_tweet_features_hashtags"],
    ])

    data = pd.DataFrame([y for x in data.dropna() for y in x.split('\t')])
    data = data[data.columns[0]]

    dictionary = pd.DataFrame(data.unique()).to_dict()[0]

    save_dictionary(dictionary, "hashtag_id_inverse_mapping.json.gz")

    inverse_dictionary = {v: k for k, v in dictionary.items()}

    save_dictionary(inverse_dictionary, "hashtag_id_direct_mapping.json.gz")


def generate():
    generate_tweet_id_dictionary()
    generate_user_id_dictionary()
    generate_language_id_dictionary()
    generate_domain_id_dictionary()
    generate_link_id_dictionary()
    generate_hashtag_id_dictionary()
