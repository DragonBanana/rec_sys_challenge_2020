# Example
### Get feature
```python
from Utils.Data.Data import get_feature
feature_df = get_feature(feature_name="mapped_feature_tweet_id", dataset_id="train")
```

### Get multiple feature
```python
from Utils.Data.Data import get_dataset
features = [
    "tweet_feature_number_of_photo",
    "tweet_feature_number_of_media",
    "tweet_feature_number_of_mentions"
]
feature_df = get_dataset(features=features, dataset_id="train")
```

# Features

For each dataset we have the following features:

### Raw Features

- **raw_feature_tweet_text_token**: str:
 <br>Ordered list of Bert ids corresponding to Bert tokenization of Tweet text.
- **raw_feature_tweet_hashtags**: str:
 <br>Tab separated list of hastags (hashed identifiers) present in the tweet.
- **raw_feature_tweet_text_token**: str:
 <br>Tweet identifier (hashed).
- **raw_feature_tweet_media**: str:
 <br>Tab separated list of media types. Media type can be in (Photo, Video, Gif)
- **raw_feature_tweet_links**: str:
 <br>Tab separeted list of links (hashed identifiers) included in the Tweet.
- **raw_feature_tweet_domains**: str:
 <br>Tab separated list of hashed domains included in the Tweet (twitter.com, dogs.com).
- **raw_feature_tweet_type**: str:
 <br>Tweet type, can be either Retweet, Quote, Reply, or Toplevel.
- **raw_feature_tweet_language**: str:
 <br>Identifier corresponding to the inferred language of the Tweet.
- **raw_feature_tweet_timestamp**: int:
 <br>Unix timestamp, in sec of the creation time of the Tweet.
 

- **raw_feature_creator_id**: str:
 <br>User identifier.
- **raw_feature_creator_follower_count**: int:
 <br>Number of followers of the user.
- **raw_feature_creator_following_count**: int:
 <br>Number of accounts the user is following.
- **raw_feature_creator_is_verified**: bool:
 <br>Is the account verified?
- **raw_feature_creator_creation_timestamp**: int:
 <br>Unix timestamp, in seconds, of the creation time of the account.
 
 
 - **raw_feature_creator_id**: str:
 <br>User identifier.
- **raw_feature_creator_follower_count**: int:
 <br>Number of followers of the user.
- **raw_feature_creator_following_count**: int:
 <br>Number of accounts the user is following.
- **raw_feature_creator_is_verified**: bool:
 <br>Is the account verified?
- **raw_feature_creator_creation_timestamp**: int:
 <br>Unix timestamp, in seconds, of the creation time of the account.
 
 
- **raw_feature_engagement_creator_follows_engager**: bool:
 <br>Does the account of the engaged tweet author follow the account that has made the engagement?
- **raw_feature_engagement_reply_timestamp**: int: **[only train/val set]**
 <br>If there is at least one, unix timestamp, in s, of one of the replies.
- **raw_feature_engagement_retweet_timestamp**: int: **[only train/val set]**
 <br>If there is one, unix timestamp, in s, of the retweet of the tweet by the engaging user.
- **raw_feature_engagement_comment_timestamp**: int: **[only train/val set]**
 <br>If there is at least one, unix timestamp, in s, of one of the retweet with comment of the tweet by the engaging user.
- **raw_feature_engagement_like_timestamp**: int: **[only train/val set]**
 <br>If there is one, Unix timestamp, in s, of the like.
 
 ### Mapped Features
 
 Those features are just the same as raw features but each identifier has been mapped to a positive integer:
 
 - **mapped_feature_tweet_hashtags**: list of int:
 <br>List of hashtags present in the tweet. *None* otherwise.
  - **mapped_feature_tweet_id**: int:
 <br>Tweet identifier.
  - **mapped_feature_tweet_media**: list of int:
 <br>List of media present in the tweet. *None* otherwise.
  - **mapped_feature_tweet_links**: list of int:
 <br>List of links present in the tweet. *None* otherwise.
  - **mapped_feature_tweet_domains**: list of int:
 <br>List of domains present in the tweet. *None* otherwise.
  - **mapped_feature_tweet_language**: int:
 <br>Tweet language.
  - **mapped_feature_creator_id**: int:
 <br>User identifier of the creator.
  - **mapped_feature_engager_id**: int:
 <br>User identifier of the engager.
 
 ### Generated Features
 
 Those features has been extracted from the previous one. All the identifiers used in these features are mapped using the internal dictionary.

 ### Generated Tweet Features
 
 #### Number of media

  - **tweet_feature_number_of_photo**: int:
 <br>Number of photo in the tweet.
  - **tweet_feature_number_of_video**: int:
 <br>Number of video in the tweet.
  - **tweet_feature_number_of_gif**: int:
 <br>Number of gif in the tweet.
   - **tweet_feature_number_of_media**: int:
 <br>Number of media (photo, video and gif) in the tweet.
 
 #### Number of hashtags
 
   - **tweet_feature_number_of_hashtags**: int:
 <br>Number of hashtags in the tweet.
 
 #### Is tweet type

  - **tweet_feature_is_reply**: bool:
 <br>True if the tweet is a reply.
  - **tweet_feature_is_retweet**: bool:
 <br>True if the tweet is a retweet.
  - **tweet_feature_is_quote**: bool:
 <br>True if the tweet is a quote.
   - **tweet_feature_is_top_level**: bool:
 <br>True if the tweet is a top_level.
 
 #### Extracted from text token
 
   - **tweet_feature_mentions**: list of ints (or None):
 <br>Mentions extracted from the tweet. 
 
   - **tweet_feature_number_of_mentions**: int:
 <br>Number of mentions in the tweet.
 
 #### Creation timestamp
 
   - **tweet_feature_creation_timestamp_hour**: int:
 <br>The hour when the tweet has been created. (0-23 UTC hour)
 
   - **tweet_feature_creation_timestamp_week_day**: int:
 <br>The week day when the tweet has been created (0-6 UTC date)

 #### Is engagement type
 **Only for train and local validation test**
 
   - **tweet_feature_engagement_is_like**: bool:
 <br>True if the tweet has been liked by the engager.
   - **tweet_feature_engagement_is_retweet**: bool:
 <br>True if the tweet has been retweeted by the engager.
   - **tweet_feature_engagement_is_comment**: bool:
 <br>True if the tweet has been commented by the engager.
   - **tweet_feature_engagement_is_reply**: bool:
 <br>True if the tweet has been replied by the engager.
   - **tweet_feature_engagement_is_positive**: bool:
 <br>True if the tweet has been involved in a positive engagement by the engager.
   - **tweet_feature_engagement_is_negative**: bool:
 <br>True if the tweet has been involved in a pseudo negative engagement by the engager.
 
 #### Main Language

  - **engager_main_language**: int:
 <br>The main language of the engager.
  - **creator_main_language**: int:
 <br>The main language of the creator.
  - **creator_and_engager_have_same_main_language**: int:
 <br>True if the creator and the engager have the same main language.
   - **is_tweet_in_creator_main_language**: int:
 <br>True if the tweet is in the creator main language.
   - **is_tweet_in_engager_main_language**: int:
 <br>True if the tweet is in the engager main language.
   - **is_tweet_in_engager_main_language**: int:
 <br>True if the tweet is in the engager main language.
   - **statistical_probability_main_language_of_engager_engage_tweet_language_1**: int:
 <br>Statical data explaining how probable a user that have a certain language know also the tweet language. (Excluding the relation language_X - language_X)
   - **statistical_probability_main_language_of_engager_engage_tweet_language_2**: int:
 <br>Statical data explaining how probable a user that have a certain language know also the tweet language. (Including the relation language_X - language_X)
