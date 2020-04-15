from Utils.Data.Dictionary.TweetBasicFeaturesDictArray import *
from Utils.Data.Dictionary.UserBasicFeaturesDictArray import *
from Utils.Data.Features.Generated.EngagerFeature.EngagerKnowTweetLanguage import *
from Utils.Data.Features.Generated.EngagerFeature.KnownEngagementCount import *
from Utils.Data.Features.Generated.TweetFeature.IsEngagementType import *
from Utils.Data.Features.Generated.TweetFeature.IsLanguage import *
from Utils.Data.Features.Generated.TweetFeature.IsTweetType import *
from Utils.Data.Features.Generated.TweetFeature.NumberOfHashtags import TweetFeatureNumberOfHashtags
from Utils.Data.Features.Generated.TweetFeature.NumberOfMedia import *
from Utils.Data.Features.MappedFeatures import *
from Utils.Data.Dictionary.MappingDictionary import *
from Utils.Data.Features.RawFeatures import *
from Utils.Data.Sparse.CSR.HashtagMatrix import *
from Utils.Data.Sparse.CSR.DomainMatrix import *
from Utils.Data.Sparse.CSR.Language.LanguageMatrixOnlyPositive import LanguageMatrixOnlyPositive
from Utils.Data.Sparse.CSR.LinkMatrix import *

import multiprocessing as mp


DATASET_IDS = [
    "train",
    "test",
    "train_days_1",
    "train_days_12",
    "train_days_123",
    "train_days_1234",
    "train_days_12345",
    "train_days_123456",
    "val_days_2",
    "val_days_3",
    "val_days_4",
    "val_days_5",
    "val_days_6",
    "val_days_7",
]


def populate_features():
    result = {}
    for dataset_id in DATASET_IDS:
        # RAW
        result[("raw_feature_tweet_text_token", dataset_id)] = RawFeatureTweetTextToken(dataset_id)
        result[("raw_feature_tweet_hashtags", dataset_id)] = RawFeatureTweetHashtags(dataset_id)
        result[("raw_feature_tweet_id", dataset_id)] = RawFeatureTweetId(dataset_id)
        result[("raw_feature_tweet_media", dataset_id)] = RawFeatureTweetMedia(dataset_id)
        result[("raw_feature_tweet_links", dataset_id)] = RawFeatureTweetLinks(dataset_id)
        result[("raw_feature_tweet_domains", dataset_id)] = RawFeatureTweetDomains(dataset_id)
        result[("raw_feature_tweet_type", dataset_id)] = RawFeatureTweetType(dataset_id)
        result[("raw_feature_tweet_language", dataset_id)] = RawFeatureTweetLanguage(dataset_id)
        result[("raw_feature_tweet_timestamp", dataset_id)] = RawFeatureTweetTimestamp(dataset_id)
        result[("raw_feature_creator_id", dataset_id)] = RawFeatureCreatorId(dataset_id)
        result[("raw_feature_creator_follower_count", dataset_id)] = RawFeatureCreatorFollowerCount(dataset_id)
        result[("raw_feature_creator_following_count", dataset_id)] = RawFeatureCreatorFollowingCount(dataset_id)
        result[("raw_feature_creator_is_verified", dataset_id)] = RawFeatureCreatorIsVerified(dataset_id)
        result[("raw_feature_creator_creation_timestamp", dataset_id)] = RawFeatureCreatorCreationTimestamp(dataset_id)
        result[("raw_feature_engager_id", dataset_id)] = RawFeatureEngagerId(dataset_id)
        result[("raw_feature_engager_follower_count", dataset_id)] = RawFeatureEngagerFollowerCount(dataset_id)
        result[("raw_feature_engager_following_count", dataset_id)] = RawFeatureEngagerFollowingCount(dataset_id)
        result[("raw_feature_engager_is_verified", dataset_id)] = RawFeatureEngagerIsVerified(dataset_id)
        result[("raw_feature_engager_creation_timestamp", dataset_id)] = RawFeatureEngagerCreationTimestamp(dataset_id)
        result[
            ("raw_feature_engagement_creator_follows_engager", dataset_id)] = RawFeatureEngagementCreatorFollowsEngager(
            dataset_id)
        if dataset_id != "test":
            result[("raw_feature_engagement_reply_timestamp", dataset_id)] = RawFeatureEngagementReplyTimestamp(
                dataset_id)
            result[("raw_feature_engagement_retweet_timestamp", dataset_id)] = RawFeatureEngagementRetweetTimestamp(
                dataset_id)
            result[("raw_feature_engagement_comment_timestamp", dataset_id)] = RawFeatureEngagementCommentTimestamp(
                dataset_id)
            result[("raw_feature_engagement_like_timestamp", dataset_id)] = RawFeatureEngagementLikeTimestamp(
                dataset_id)
        # MAPPED
        result[("mapped_feature_tweet_hashtags", dataset_id)] = MappedFeatureTweetHashtags(dataset_id)
        result[("mapped_feature_tweet_id", dataset_id)] = MappedFeatureTweetId(dataset_id)
        result[("mapped_feature_tweet_media", dataset_id)] = MappedFeatureTweetMedia(dataset_id)
        result[("mapped_feature_tweet_links", dataset_id)] = MappedFeatureTweetLinks(dataset_id)
        result[("mapped_feature_tweet_domains", dataset_id)] = MappedFeatureTweetDomains(dataset_id)
        result[("mapped_feature_tweet_language", dataset_id)] = MappedFeatureTweetLanguage(dataset_id)
        result[("mapped_feature_creator_id", dataset_id)] = MappedFeatureCreatorId(dataset_id)
        result[("mapped_feature_engager_id", dataset_id)] = MappedFeatureEngagerId(dataset_id)
        # GENERATED
        # TWEET FEATURE
        # NUMBER OF MEDIA
        result[("tweet_feature_number_of_photo", dataset_id)] = TweetFeatureNumberOfPhoto(dataset_id)
        result[("tweet_feature_number_of_video", dataset_id)] = TweetFeatureNumberOfVideo(dataset_id)
        result[("tweet_feature_number_of_gif", dataset_id)] = TweetFeatureNumberOfGif(dataset_id)
        result[("tweet_feature_number_of_media", dataset_id)] = TweetFeatureNumberOfMedia(dataset_id)
        # NUMBER OF HASHTAGS
        result[("tweet_feature_number_of_hashtags", dataset_id)] = TweetFeatureNumberOfHashtags(dataset_id)
        # IS TWEET TYPE
        result[("tweet_feature_is_reply", dataset_id)] = TweetFeatureIsReply(dataset_id)
        result[("tweet_feature_is_retweet", dataset_id)] = TweetFeatureIsRetweet(dataset_id)
        result[("tweet_feature_is_quote", dataset_id)] = TweetFeatureIsQuote(dataset_id)
        result[("tweet_feature_is_top_level", dataset_id)] = TweetFeatureIsTopLevel(dataset_id)
        # IS IN LANGUAGE
        # result[("tweet_is_language_x", dataset_id)] = TweetFeatureIsLanguage(dataset_id, top_popular_language(dataset_id, top_n=10))
        # IS ENGAGEMENT TYPE
        if dataset_id != "test":
            result[("tweet_feature_engagement_is_like", dataset_id)] = TweetFeatureEngagementIsLike(dataset_id)
            result[("tweet_feature_engagement_is_retweet", dataset_id)] = TweetFeatureEngagementIsRetweet(dataset_id)
            result[("tweet_feature_engagement_is_comment", dataset_id)] = TweetFeatureEngagementIsComment(dataset_id)
            result[("tweet_feature_engagement_is_reply", dataset_id)] = TweetFeatureEngagementIsReply(dataset_id)
            result[("tweet_feature_engagement_is_positive", dataset_id)] = TweetFeatureEngagementIsPositive(dataset_id)
            result[("tweet_feature_engagement_is_negative", dataset_id)] = TweetFeatureEngagementIsNegative(dataset_id)
        # CREATOR FEATURE
        # KNOWN COUNT OF ENGAGEMENT
        # BAD IMPLEMENTATION - DOES NOT RESPECT TIME
        # result[("engager_feature_known_number_of_like_engagement", dataset_id)] = EngagerFeatureKnowNumberOfLikeEngagement(dataset_id)
        # result[("engager_feature_known_number_of_reply_engagement", dataset_id)] = EngagerFeatureKnowNumberOfReplyEngagement(dataset_id)
        # result[("engager_feature_known_number_of_retweet_engagement", dataset_id)] = EngagerFeatureKnowNumberOfRetweetEngagement(dataset_id)
        # result[("engager_feature_known_number_of_comment_engagement", dataset_id)] = EngagerFeatureKnowNumberOfCommentEngagement(dataset_id)
        # result[("engager_feature_known_number_of_positive_engagement", dataset_id)] = EngagerFeatureKnowNumberOfPositiveEngagement(dataset_id)
        # result[("engager_feature_known_number_of_negative_engagement", dataset_id)] = EngagerFeatureKnowNumberOfNegativeEngagement(dataset_id)
        # KNOW TWEET LANGUAGE
        # BAD IMPLEMENTATION - DOES NOT RESPECT TIME
        # result[("engager_feature_know_tweet_language", dataset_id)] = EngagerFeatureKnowTweetLanguage(dataset_id)


    return result


FEATURES = populate_features()

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
    "links_tweet_dict_array": LinksTweetBasicFeatureDictArray(),
    "domains_tweet_dict_array": DomainsTweetBasicFeatureDictArray(),
    "type_tweet_dict_array": TypeTweetBasicFeatureDictArray(),
    "timestamp_tweet_dict_array": TimestampTweetBasicFeatureDictArray(),
    "creator_id_tweet_dict_array": CreatorIdTweetBasicFeatureDictArray(),
    # USER BASIC FEATURES
    "follower_count_user_dict_array": FollowerCountUserBasicFeatureDictArray(),
    "following_count_user_dict_array": FollowingCountUserBasicFeatureDictArray(),
    "is_verified_user_dict_array": IsVerifiedUserBasicFeatureDictArray(),
    "creation_timestamp_user_dict_array": CreationTimestampUserBasicFeatureDictArray(),
    "language_user_dict_array": LanguageUserBasicFeatureDictArray(),

}

SPARSE_MATRIXES = {
    # ICM
    "tweet_hashtags_csr_matrix": HashtagMatrix(),
    "tweet_links_csr_matrix": LinkMatrix(),
    "tweet_domains_csr_matrix": DomainMatrix(),
    "tweet_language_csr_matrix": LanguageMatrixOnlyPositive()
}

def create_all():
    # For more parallelism
    features_grouped = [[v for k, v in FEATURES.items() if k[1] == dataset_id] for dataset_id in DATASET_IDS]
    for features in features_grouped:
        for feature in features:
            print(feature.dataset_id)
    with mp.Pool(4) as pool:
        pool.map(create_features, features_grouped)
    # list(map(create_feature, FEATURES.values()))
    list(map(create_dictionary, DICTIONARIES.values()))
    list(map(create_dictionary, DICT_ARRAYS.values()))
    list(map(create_matrix, SPARSE_MATRIXES.values()))

def create_features(feature_list: list):
    list(map(create_feature, feature_list))

def create_feature(feature: Feature):
    if not feature.has_feature():
        print(f"creating: {feature.dataset_id}_{feature.feature_name}")
        feature.create_feature()
    else:
        print(f"already created: {feature.dataset_id}_{feature.feature_name}")


def create_dictionary(dictionary: Dictionary):
    if not dictionary.has_dictionary():
        print(f"creating: {dictionary.dictionary_name}")
        dictionary.create_dictionary()
    else:
        print(f"already created: {dictionary.dictionary_name}")

def create_matrix(matrix: CSR_SparseMatrix):
    if not matrix.has_matrix():
        print(f"creating: {matrix.matrix_name}")
        matrix.create_matrix()
    else:
        print(f"already created: {matrix.matrix_name}")


def consistency_check(dataset_id: str):
    features = np.array(FEATURES.items())
    lenghts = np.array([len(v.load_or_create()) for k, v in FEATURES.items() if k[1] == dataset_id])
    if all(lenghts == lenghts[0]):
        print(f"{dataset_id} is consistent")
    else:
        not_consistent_features_mask = lenghts != lenghts[0]
        for feature in features[lenghts][not_consistent_features_mask]:
            print(feature)


def consistency_check_all():
    for dataset_id in DATASET_IDS:
        consistency_check(dataset_id)
