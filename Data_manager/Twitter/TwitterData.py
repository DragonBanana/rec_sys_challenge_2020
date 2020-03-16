import pandas as pd
import gzip as gz
import json
import numpy as np

root = "/home/jovyan/work/"

switcher = {
    # TRAINING RAW
    #   TWEET FEATURES
    "training_raw_tweet_features_text_token":
        lambda: pd.read_csv(root+"data/training/raw_columns/tweet_features/text_tokens.csv.gz", header=0, index_col=0, names=["training_raw_tweet_features_text_token"], dtype={"training_raw_tweet_features_text_token": pd.StringDtype()}),
    "training_raw_tweet_features_hashtags":
        lambda: pd.read_csv(root+"data/training/raw_columns/tweet_features/hashtags.csv.gz", header=0, index_col=0, names=["training_raw_tweet_features_hashtags"], dtype={"training_raw_tweet_features_hashtags": pd.StringDtype()}),
    "training_raw_tweet_features_tweet_id":
        lambda: pd.read_pickle(root+"data/training/raw_columns/tweet_features/tweet_id.pck.gz"),
    "training_raw_tweet_features_media":
        lambda: pd.read_csv(root+"data/training/raw_columns/tweet_features/media.csv.gz", header=0, index_col=0, names=["training_raw_tweet_features_media"], dtype={"training_raw_tweet_features_media": pd.StringDtype()}),
    "training_raw_tweet_features_links":
        lambda: pd.read_csv(root+"data/training/raw_columns/tweet_features/links.csv.gz", header=0, index_col=0, names=["training_raw_tweet_features_links"], dtype={"training_raw_tweet_features_links": pd.StringDtype()}),
    "training_raw_tweet_features_domains":
        lambda: pd.read_csv(root+"data/training/raw_columns/tweet_features/domains.csv.gz", header=0, index_col=0, names=["training_raw_tweet_features_domains"], dtype={"training_raw_tweet_features_domains": pd.StringDtype()}),
    "training_raw_tweet_features_type":
        lambda: pd.read_pickle(root+"data/training/raw_columns/tweet_features/type.pck.gz"),
    "training_raw_tweet_features_language":
        lambda: pd.read_pickle(root+"data/training/raw_columns/tweet_features/language.pck.gz"),
    #   CREATOR FEATURES
    "training_raw_tweet_features_timestamp":
        lambda: pd.read_pickle(root+"data/training/raw_columns/tweet_features/timestamp.pck.gz"),
    "training_raw_creator_features_user_id":
        lambda: pd.read_pickle(root+"data/training/raw_columns/creator_features/user_id.pck.gz"),
    "training_raw_creator_features_follower_count":
        lambda: pd.read_pickle(root+"data/training/raw_columns/creator_features/follower_count.pck.gz"),
    "training_raw_creator_features_following_count":
        lambda: pd.read_pickle(root+"data/training/raw_columns/creator_features/following_count.pck.gz"),
    "training_raw_creator_features_is_verified":
        lambda: pd.read_pickle(root+"data/training/raw_columns/creator_features/is_verified.pck.gz"),
    "training_raw_creator_features_creation_timestamp":
        lambda: pd.read_pickle(root+"data/training/raw_columns/creator_features/creation_timestamp.pck.gz"),
    #   ENGAGER FEATURES
    "training_raw_engager_features_user_id":
        lambda: pd.read_pickle(root+"data/training/raw_columns/engager_features/user_id.pck.gz"),
    "training_raw_engager_features_follower_count":
        lambda: pd.read_pickle(root+"data/training/raw_columns/engager_features/follower_count.pck.gz"),
    "training_raw_engager_features_following_count":
        lambda: pd.read_pickle(root+"data/training/raw_columns/engager_features/following_count.pck.gz"),
    "training_raw_engager_features_is_verified":
        lambda: pd.read_pickle(root+"data/training/raw_columns/engager_features/is_verified.pck.gz"),
    "training_raw_engager_features_creation_timestamp":
        lambda: pd.read_pickle(root+"data/training/raw_columns/engager_features/creation_timestamp.pck.gz"),
    #   ENGAGEMENT FEATURES
    "training_raw_engagement_features_creator_follow_engager":
        lambda: pd.read_pickle(root+"data/training/raw_columns/engagement_features/creator_follow_engager.pck.gz"),
    "training_raw_engagement_features_reply_timestamp":
        lambda: pd.read_pickle(root+"data/training/raw_columns/engagement_features/reply_timestamp.pck.gz"),
    "training_raw_engagement_features_retweet_timestamp":
        lambda: pd.read_pickle(root+"data/training/raw_columns/engagement_features/retweet_timestamp.pck.gz"),
    "training_raw_engagement_features_retweet_comment_timestamp":
        lambda: pd.read_pickle(root+"data/training/raw_columns/engagement_features/retweet_comment_timestamp.pck.gz"),
    "training_raw_engagement_features_like_timestamp":
        lambda: pd.read_pickle(root+"data/training/raw_columns/engagement_features/like_timestamp.pck.gz"),

    # VALIDATION RAW
    #   TWEET FEATURES
    "validation_raw_tweet_features_text_token":
        lambda: pd.read_csv(root+"data/validation/raw_columns/tweet_features/text_tokens.csv.gz", header=0, index_col=0, names=["validation_raw_tweet_features_text_token"], dtype={"validation_raw_tweet_features_text_token": pd.StringDtype()}),
    "validation_raw_tweet_features_hashtags":
        lambda: pd.read_csv(root+"data/validation/raw_columns/tweet_features/hashtags.csv.gz", header=0, index_col=0, names=["validation_raw_tweet_features_hashtags"], dtype={"validation_raw_tweet_features_hashtags": pd.StringDtype()}),
    "validation_raw_tweet_features_tweet_id":
        lambda: pd.read_pickle(root+"data/validation/raw_columns/tweet_features/tweet_id.pck.gz"),
    "validation_raw_tweet_features_media":
        lambda: pd.read_csv(root+"data/validation/raw_columns/tweet_features/media.csv.gz", header=0, index_col=0, names=["validation_raw_tweet_features_media"], dtype={"validation_raw_tweet_features_media": pd.StringDtype()}),
    "validation_raw_tweet_features_links":
        lambda: pd.read_csv(root+"data/validation/raw_columns/tweet_features/links.csv.gz", header=0, index_col=0, names=["validation_raw_tweet_features_links"], dtype={"validation_raw_tweet_features_links": pd.StringDtype()}),
    "validation_raw_tweet_features_domains":
        lambda: pd.read_csv(root+"data/validation/raw_columns/tweet_features/domains.csv.gz", header=0, index_col=0, names=["validation_raw_tweet_features_domains"], dtype={"validation_raw_tweet_features_domains": pd.StringDtype()}),
    "validation_raw_tweet_features_type":
        lambda: pd.read_pickle(root+"data/validation/raw_columns/tweet_features/type.pck.gz"),
    "validation_raw_tweet_features_language":
        lambda: pd.read_pickle(root+"data/validation/raw_columns/tweet_features/language.pck.gz"),
    "validation_raw_tweet_features_timestamp":
        lambda: pd.read_pickle(root+"data/validation/raw_columns/tweet_features/timestamp.pck.gz"),
    #   CREATOR FEATURES
    "validation_raw_creator_features_user_id":
        lambda: pd.read_pickle(root+"data/validation/raw_columns/creator_features/user_id.pck.gz"),
    "validation_raw_creator_features_follower_count":
        lambda: pd.read_pickle(root+"data/validation/raw_columns/creator_features/follower_count.pck.gz"),
    "validation_raw_creator_features_following_count":
        lambda: pd.read_pickle(root+"data/validation/raw_columns/creator_features/following_count.pck.gz"),
    "validation_raw_creator_features_is_verified":
        lambda: pd.read_pickle(root+"data/validation/raw_columns/creator_features/is_verified.pck.gz"),
    "validation_raw_creator_features_creation_timestamp":
        lambda: pd.read_pickle(root+"data/validation/raw_columns/creator_features/creation_timestamp.pck.gz"),
    #   ENGAGER FEATURES
    "validation_raw_engager_features_user_id":
        lambda: pd.read_pickle(root+"data/validation/raw_columns/engager_features/user_id.pck.gz"),
    "validation_raw_engager_features_follower_count":
        lambda: pd.read_pickle(root+"data/validation/raw_columns/engager_features/follower_count.pck.gz"),
    "validation_raw_engager_features_following_count":
        lambda: pd.read_pickle(root+"data/validation/raw_columns/engager_features/following_count.pck.gz"),
    "validation_raw_engager_features_is_verified":
        lambda: pd.read_pickle(root+"data/validation/raw_columns/engager_features/is_verified.pck.gz"),
    "validation_raw_engager_features_creation_timestamp":
        lambda: pd.read_pickle(root+"data/validation/raw_columns/engager_features/creation_timestamp.pck.gz"),
    #   ENGAGEMENT FEATURES
    "validation_raw_engagement_features_creator_follow_engager":
        lambda: pd.read_pickle(root+"data/validation/raw_columns/engagement_features/creator_follow_engager.pck.gz"),

    # TRAINING MAPPED
    "training_mapped_tweet_features_hashtags":
        lambda: pd.read_pickle(root + "data/training/mapped_columns/hashtags.pck.gz"),
    "training_mapped_tweet_features_tweet_id":
        lambda: pd.read_pickle(root + "data/training/mapped_columns/tweet_id.pck.gz"),
    "training_mapped_tweet_features_links":
        lambda: pd.read_pickle(root + "data/training/mapped_columns/links.pck.gz"),
    "training_mapped_tweet_features_domains":
        lambda: pd.read_pickle(root + "data/training/mapped_columns/domains.pck.gz"),
    "training_mapped_tweet_features_language":
        lambda: pd.read_pickle(root + "data/training/mapped_columns/language.pck.gz"),
    "training_mapped_creator_features_user_id":
        lambda: pd.read_pickle(root + "data/training/mapped_columns/creator_user_id.pck.gz"),
    "training_mapped_engager_features_user_id":
        lambda: pd.read_pickle(root + "data/training/mapped_columns/engager_user_id.pck.gz"),

    # VALIDATION MAPPED
    "validation_mapped_tweet_features_hashtags":
        lambda: pd.read_pickle(root + "data/validation/mapped_columns/hashtags.pck.gz"),
    "validation_mapped_tweet_features_tweet_id":
        lambda: pd.read_pickle(root + "data/validation/mapped_columns/tweet_id.pck.gz"),
    "validation_mapped_tweet_features_links":
        lambda: pd.read_pickle(root + "data/validation/mapped_columns/links.pck.gz"),
    "validation_mapped_tweet_features_domains":
        lambda: pd.read_pickle(root + "data/validation/mapped_columns/domains.pck.gz"),
    "validation_mapped_tweet_features_language":
        lambda: pd.read_pickle(root + "data/validation/mapped_columns/language.pck.gz"),
    "validation_mapped_creator_features_user_id":
        lambda: pd.read_pickle(root + "data/validation/mapped_columns/creator_user_id.pck.gz"),
    "validation_mapped_engager_features_user_id":
        lambda: pd.read_pickle(root + "data/validation/mapped_columns/engager_user_id.pck.gz"),

    # DICTIONARIES
    "dictionary_domain_id_direct":
        lambda: json.loads(gz.GzipFile(root+"data/dictionary/domain_id/direct_mapping.json.gz", 'r').read().decode('utf-8')),
    "dictionary_domain_id_inverse":
        lambda: json.loads(gz.GzipFile(root+"data/dictionary/domain_id/inverse_mapping.json.gz", 'r').read().decode('utf-8')),
    "dictionary_link_id_direct":
        lambda: json.loads(gz.GzipFile(root+"data/dictionary/link_id/direct_mapping.json.gz", 'r').read().decode('utf-8')),
    "dictionary_link_id_inverse":
        lambda: json.loads(gz.GzipFile(root+"data/dictionary/link_id/inverse_mapping.json.gz", 'r').read().decode('utf-8')),
    "dictionary_hashtag_id_direct":
        lambda: json.loads(gz.GzipFile(root+"data/dictionary/hashtag_id/direct_mapping.json.gz", 'r').read().decode('utf-8')),
    "dictionary_hashtag_id_inverse":
        lambda: json.loads(gz.GzipFile(root+"data/dictionary/hashtag_id/inverse_mapping.json.gz", 'r').read().decode('utf-8')),
    "dictionary_language_id_direct":
        lambda: json.loads(gz.GzipFile(root+"data/dictionary/language_id/direct_mapping.json.gz", 'r').read().decode('utf-8')),
    "dictionary_language_id_inverse":
        lambda: json.loads(gz.GzipFile(root+"data/dictionary/language_id/inverse_mapping.json.gz", 'r').read().decode('utf-8')),
    "dictionary_tweet_id_direct":
        lambda: json.loads(gz.GzipFile(root+"data/dictionary/tweet_id/direct_mapping.json.gz", 'r').read().decode('utf-8')),
    "dictionary_tweet_id_inverse":
        lambda: json.loads(gz.GzipFile(root+"data/dictionary/tweet_id/inverse_mapping.json.gz", 'r').read().decode('utf-8')),
    "dictionary_user_id_direct":
        lambda: json.loads(gz.GzipFile(root+"data/dictionary/user_id/direct_mapping.json.gz", 'r').read().decode('utf-8')),
    "dictionary_user_id_inverse":
        lambda: json.loads(gz.GzipFile(root+"data/dictionary/user_id/inverse_mapping.json.gz", 'r').read().decode('utf-8')),

}


def get_resource(resource_id: str):
    result = switcher[resource_id]()

    if isinstance(result, pd.DataFrame):
        result.columns = [resource_id]

    return result
