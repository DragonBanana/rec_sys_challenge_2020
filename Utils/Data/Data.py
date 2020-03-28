import pandas as pd
import numpy as np
import multiprocessing as mp

from Utils.Data.Dictionary.TweetBasicFeaturesDictArray import *
from Utils.Data.Dictionary.UserBasicFeaturesDictArray import *
from Utils.Data.Features.MappedFeatures import *
from Utils.Data.Dictionary.MappingDictionary import *
from Utils.Data.Features.RawFeatures import *

FEATURES = {
    # RAW TRAIN
    ("raw_feature_tweet_text_token", "train"): RawFeatureTweetTextToken("train"),
    ("raw_feature_tweet_hashtags", "train"): RawFeatureTweetHashtags("train"),
    ("raw_feature_tweet_id", "train"): RawFeatureTweetId("train"),
    ("raw_feature_tweet_media", "train"): RawFeatureTweetMedia("train"),
    ("raw_feature_tweet_links", "train"): RawFeatureTweetLinks("train"),
    ("raw_feature_tweet_domains", "train"): RawFeatureTweetDomains("train"),
    ("raw_feature_tweet_type", "train"): RawFeatureTweetType("train"),
    ("raw_feature_tweet_language", "train"): RawFeatureTweetLanguage("train"),
    ("raw_feature_tweet_timestamp", "train"): RawFeatureTweetTimestamp("train"),
    ("raw_feature_creator_id", "train"): RawFeatureCreatorId("train"),
    ("raw_feature_creator_follower_count", "train"): RawFeatureCreatorFollowerCount("train"),
    ("raw_feature_creator_following_count", "train"): RawFeatureCreatorFollowingCount("train"),
    ("raw_feature_creator_is_verified", "train"): RawFeatureCreatorIsVerified("train"),
    ("raw_feature_creator_creation_timestamp", "train"): RawFeatureCreatorCreationTimestamp("train"),
    ("raw_feature_engager_id", "train"): RawFeatureEngagerId("train"),
    ("raw_feature_engager_follower_count", "train"): RawFeatureEngagerFollowerCount("train"),
    ("raw_feature_engager_following_count", "train"): RawFeatureEngagerFollowingCount("train"),
    ("raw_feature_engager_is_verified", "train"): RawFeatureEngagerIsVerified("train"),
    ("raw_feature_engager_creation_timestamp", "train"): RawFeatureEngagerCreationTimestamp("train"),
    ("raw_feature_engagement_creator_follows_engager", "train"): RawFeatureEngagementCreatorFollowsEngager("train"),
    ("raw_feature_engagement_reply_timestamp", "train"): RawFeatureEngagementReplyTimestamp("train"),
    ("raw_feature_engagement_retweet_timestamp", "train"): RawFeatureEngagementRetweetTimestamp("train"),
    ("raw_feature_engagement_comment_timestamp", "train"): RawFeatureEngagementCommentTimestamp("train"),
    ("raw_feature_engagement_like_timestamp", "train"): RawFeatureEngagementLikeTimestamp("train"),
    # RAW TEST
    ("raw_feature_tweet_text_token", "test"): RawFeatureTweetTextToken("test"),
    ("raw_feature_tweet_hashtags", "test"): RawFeatureTweetHashtags("test"),
    ("raw_feature_tweet_id", "test"): RawFeatureTweetId("test"),
    ("raw_feature_tweet_media", "test"): RawFeatureTweetMedia("test"),
    ("raw_feature_tweet_links", "test"): RawFeatureTweetLinks("test"),
    ("raw_feature_tweet_domains", "test"): RawFeatureTweetDomains("test"),
    ("raw_feature_tweet_type", "test"): RawFeatureTweetType("test"),
    ("raw_feature_tweet_language", "test"): RawFeatureTweetLanguage("test"),
    ("raw_feature_tweet_timestamp", "test"): RawFeatureTweetTimestamp("test"),
    ("raw_feature_creator_id", "test"): RawFeatureCreatorId("test"),
    ("raw_feature_creator_follower_count", "test"): RawFeatureCreatorFollowerCount("test"),
    ("raw_feature_creator_following_count", "test"): RawFeatureCreatorFollowingCount("test"),
    ("raw_feature_creator_is_verified", "test"): RawFeatureCreatorIsVerified("test"),
    ("raw_feature_creator_creation_timestamp", "test"): RawFeatureCreatorCreationTimestamp("test"),
    ("raw_feature_engager_id", "test"): RawFeatureEngagerId("test"),
    ("raw_feature_engager_follower_count", "test"): RawFeatureEngagerFollowerCount("test"),
    ("raw_feature_engager_following_count", "test"): RawFeatureEngagerFollowingCount("test"),
    ("raw_feature_engager_is_verified", "test"): RawFeatureEngagerIsVerified("test"),
    ("raw_feature_engager_creation_timestamp", "test"): RawFeatureEngagerCreationTimestamp("test"),
    ("raw_feature_engagement_creator_follows_engager", "test"): RawFeatureEngagementCreatorFollowsEngager("test"),
    # MAPPED TRAIN
    ("mapped_feature_tweet_hashtags", "train"): MappedFeatureTweetHashtags("train"),
    ("mapped_feature_tweet_id", "train"): MappedFeatureTweetId("train"),
    ("mapped_feature_tweet_media", "train"): MappedFeatureTweetMedia("train"),
    ("mapped_feature_tweet_links", "train"): MappedFeatureTweetLinks("train"),
    ("mapped_feature_tweet_domains", "train"): MappedFeatureTweetDomains("train"),
    ("mapped_feature_tweet_language", "train"): MappedFeatureTweetLanguage("train"),
    ("mapped_feature_creator_id", "train"): MappedFeatureCreatorId("train"),
    ("mapped_feature_engager_id", "train"): MappedFeatureEngagerId("train"),
    # MAPPED TEST
    ("mapped_feature_tweet_hashtags", "test"): MappedFeatureTweetHashtags("test"),
    ("mapped_feature_tweet_id", "test"): MappedFeatureTweetId("test"),
    ("mapped_feature_tweet_media", "test"): MappedFeatureTweetMedia("test"),
    ("mapped_feature_tweet_links", "test"): MappedFeatureTweetLinks("test"),
    ("mapped_feature_tweet_domains", "test"): MappedFeatureTweetDomains("test"),
    ("mapped_feature_tweet_language", "test"): MappedFeatureTweetLanguage("test"),
    ("mapped_feature_creator_id", "test"): MappedFeatureCreatorId("test"),
    ("mapped_feature_engager_id", "test"): MappedFeatureEngagerId("test")
}

DICTIONARIES = {
    "mapping_tweet_id_direct": MappingTweetIdDictionary(inverse=False),
    "mapping_tweet_id_inverse": MappingTweetIdDictionary(inverse=True),
    "mapping_user_id_direct": MappingUserIdDictionary(inverse=False),
    "mapping_user_id_inverse": MappingUserIdDictionary(inverse=True),
    "mapping_language_id_direct": MappingLanguageDictionary(inverse=False),
    "mapping_language_id_inverse": MappingLanguageDictionary(inverse=True),
    "mapping_domain_id_direct": MappingDomainDictionary(inverse=False),
    "mapping_domain_id_inverse": MappingDomainDictionary(inverse=True),
    "mapping_link_id_direct": MappingLinkDictionary(inverse=False),
    "mapping_link_id_inverse": MappingLinkDictionary(inverse=True),
    "mapping_media_id_direct": MappingMediaDictionary(inverse=False),
    "mapping_media_id_inverse": MappingMediaDictionary(inverse=True),
    "mapping_hashtag_id_direct": MappingHashtagDictionary(inverse=False),
    "mapping_hashtag_id_inverse": MappingHashtagDictionary(inverse=True)
}

DICT_ARRAYS = {
    # TWEET BASIC FEATURES
    "hashtags_tweet_dict_array": HashtagsTweetBasicFeatureDictArray(),
    "media_tweet_dict_array": MediaTweetBasicFeatureDictArray(),
    "links_tweet_dict_array": MediaTweetBasicFeatureDictArray(),
    "domains_tweet_dict_array": DomainsTweetBasicFeatureDictArray(),
    "type_tweet_dict_array": TypeTweetBasicFeatureDictArray(),
    "timestamp_tweet_dict_array": TimestampTweetBasicFeatureDictArray(),
    "creator_id_tweet_dict_array": CreatorIdTweetBasicFeatureDictArray(),
    # USER BASIC FEATURES
    "follower_count_user_dict_array": FollowerCountUserBasicFeatureDictArray(),
    "following_count_user_dict_array": FollowingCountUserBasicFeatureDictArray(),
    "is_verified_user_dict_array": IsVerifiedUserBasicFeatureDictArray(),
    "creation_timestamp_user_dict_array": CreationTimestampUserBasicFeatureDictArray(),

}


def get_xgb_train():
    pass


def get_feature(feature_name: str, dataset_id: str):
    if (feature_name, dataset_id) in FEATURES.keys():
        return FEATURES[(feature_name, dataset_id)]


def _create_all():
    with mp.Pool(3) as p:
        p.map(_create_feature, FEATURES.values())
        p.map(_create_dictionary, DICTIONARIES.values())
        p.map(_create_dictionary, DICT_ARRAYS.values())


def _create_feature(feature: Feature):
    if not feature.has_feature():
        feature.create_feature()


def _create_dictionary(dictionary: Dictionary):
    if not dictionary.has_dictionary():
        dictionary.create_dictionary()


if __name__ == '__main__':
    _create_all()
