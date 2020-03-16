import pandas as pd
import numpy as np
import TwitterData


def map_column_single_value(series, dictionary):
    mapped_series = series.map(dictionary).astype(np.int32)
    return pd.DataFrame(mapped_series)

def map_column_array(series, dictionary):
    mapped_series = series.map(lambda x: np.array([dictionary[y] for y in x.split('\t')], dtype=np.int32) if x is not pd.NA else None)
    return pd.DataFrame(mapped_series)

def generate():

    # ----------------------------------------------------------------

    # LANGUAGE

    #   TRAINING
    column_name = "training_raw_tweet_features_language"
    dictionary_name = "dictionary_language_id_direct"
    dictionary = TwitterData.get_resource(dictionary_name)
    df = TwitterData.get_resource(column_name)
    mapped_df = map_column_single_value(df[column_name], dictionary)
    mapped_df.to_csv(column_name + ".csv.gz", compression="gzip")
    mapped_df.to_pickle(column_name + ".pck.gz", compression="gzip")

    #   VALIDATION
    column_name = "validation_raw_tweet_features_language"
    dictionary_name = "dictionary_language_id_direct"
    dictionary = TwitterData.get_resource(dictionary_name)
    df = TwitterData.get_resource(column_name)
    mapped_df = map_column_single_value(df[column_name], dictionary)
    mapped_df.to_csv(column_name + ".csv.gz", compression="gzip")
    mapped_df.to_pickle(column_name + ".pck.gz", compression="gzip")

    # ----------------------------------------------------------------

    # TWEET ID

    #   TRAINING
    column_name = "training_raw_tweet_features_tweet_id"
    dictionary_name = "dictionary_tweet_id_direct"
    dictionary = TwitterData.get_resource(dictionary_name)
    df = TwitterData.get_resource(column_name)
    mapped_df = map_column_single_value(df[column_name], dictionary)
    mapped_df.to_csv(column_name + ".csv.gz", compression="gzip")
    mapped_df.to_pickle(column_name + ".pck.gz", compression="gzip")

    #   VALIDATION
    column_name = "validation_raw_tweet_features_tweet_id"
    dictionary_name = "dictionary_tweet_id_direct"
    dictionary = TwitterData.get_resource(dictionary_name)
    df = TwitterData.get_resource(column_name)
    mapped_df = map_column_single_value(df[column_name], dictionary)
    mapped_df.to_csv(column_name + ".csv.gz", compression="gzip")
    mapped_df.to_pickle(column_name + ".pck.gz", compression="gzip")

    # ----------------------------------------------------------------

    # USER ID

    #   TRAINING
    column_name = "training_raw_creator_features_user_id"
    dictionary_name = "dictionary_user_id_direct"
    dictionary = TwitterData.get_resource(dictionary_name)
    df = TwitterData.get_resource(column_name)
    mapped_df = map_column_single_value(df[column_name], dictionary)
    mapped_df.to_csv(column_name + ".csv.gz", compression="gzip")
    mapped_df.to_pickle(column_name + ".pck.gz", compression="gzip")

    column_name = "training_raw_engager_features_user_id"
    df = TwitterData.get_resource(column_name)
    mapped_df = map_column_single_value(df[column_name], dictionary)
    mapped_df.to_csv(column_name + ".csv.gz", compression="gzip")
    mapped_df.to_pickle(column_name + ".pck.gz", compression="gzip")

    #   VALIDATION
    column_name = "validation_raw_creator_features_user_id"
    dictionary_name = "dictionary_user_id_direct"
    dictionary = TwitterData.get_resource(dictionary_name)
    df = TwitterData.get_resource(column_name)
    mapped_df = map_column_single_value(df[column_name], dictionary)
    mapped_df.to_csv(column_name + ".csv.gz", compression="gzip")
    mapped_df.to_pickle(column_name + ".pck.gz", compression="gzip")

    column_name = "validation_raw_engager_features_user_id"
    df = TwitterData.get_resource(column_name)
    mapped_df = map_column_single_value(df[column_name], dictionary)
    mapped_df.to_csv(column_name + ".csv.gz", compression="gzip")
    mapped_df.to_pickle(column_name + ".pck.gz", compression="gzip")

    # ----------------------------------------------------------------

    # HASHTAGS

    #   TRAINING
    column_name = "training_raw_tweet_features_hashtags"
    dictionary_name = "dictionary_hashtag_id_direct"
    dictionary = TwitterData.get_resource(dictionary_name)
    df = TwitterData.get_resource(column_name)
    mapped_df = map_column_array(df[column_name], dictionary)
    mapped_df.to_csv(column_name + ".csv.gz", compression="gzip")
    mapped_df.to_pickle(column_name + ".pck.gz", compression="gzip")

    #   VALIDATION
    column_name = "validation_raw_tweet_features_hashtags"
    dictionary_name = "dictionary_hashtag_id_direct"
    dictionary = TwitterData.get_resource(dictionary_name)
    df = TwitterData.get_resource(column_name)
    mapped_df = map_column_array(df[column_name], dictionary)
    mapped_df.to_csv(column_name + ".csv.gz", compression="gzip")
    mapped_df.to_pickle(column_name + ".pck.gz", compression="gzip")

    # ----------------------------------------------------------------

    # DOMAIN

    #   TRAINING
    column_name = "training_raw_tweet_features_domains"
    dictionary_name = "dictionary_domain_id_direct"
    dictionary = TwitterData.get_resource(dictionary_name)
    df = TwitterData.get_resource(column_name)
    mapped_df = map_column_array(df[column_name], dictionary)
    mapped_df.to_csv(column_name + ".csv.gz", compression="gzip")
    mapped_df.to_pickle(column_name + ".pck.gz", compression="gzip")

    #   VALIDATION
    column_name = "validation_raw_tweet_features_domains"
    dictionary_name = "dictionary_domain_id_direct"
    dictionary = TwitterData.get_resource(dictionary_name)
    df = TwitterData.get_resource(column_name)
    mapped_df = map_column_array(df[column_name], dictionary)
    mapped_df.to_csv(column_name + ".csv.gz", compression="gzip")
    mapped_df.to_pickle(column_name + ".pck.gz", compression="gzip")

    # ----------------------------------------------------------------

    # LINK

    #   TRAINING
    column_name = "training_raw_tweet_features_links"
    dictionary_name = "dictionary_link_id_direct"
    dictionary = TwitterData.get_resource(dictionary_name)
    df = TwitterData.get_resource(column_name)
    mapped_df = map_column_array(df[column_name], dictionary)
    mapped_df.to_csv(column_name + ".csv.gz", compression="gzip")
    mapped_df.to_pickle(column_name + ".pck.gz", compression="gzip")

    #   VALIDATION
    column_name = "validation_raw_tweet_features_links"
    dictionary_name = "dictionary_link_id_direct"
    dictionary = TwitterData.get_resource(dictionary_name)
    df = TwitterData.get_resource(column_name)
    mapped_df = map_column_array(df[column_name], dictionary)
    mapped_df.to_csv(column_name + ".csv.gz", compression="gzip")
    mapped_df.to_pickle(column_name + ".pck.gz", compression="gzip")