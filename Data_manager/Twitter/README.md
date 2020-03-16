# TwitterData

The TwitterData class is a wrapper that allow users to access the twitter dataset.
The data that can be accessed can be of multiple type.
The following sections the data and shows how to access them.


# Training Raw columns

They are the columns without filtering and processing. The value in those columns can be Integers, Booleans or String. All the data is wrapped in Pandas Dataframe. <br>
The training set is composed by ~ 150M rows. Those columns are saved both as csv and pck files. <br>
**missing data**: missing data has value pandas.NA

### Training Raw columns - Tweet Features

| Resource uid | Description | Type | 
|---|---|---|
| **training_raw_tweet_features_text_token** | Ordered list of Bert ids corresponding to Bert tokenization of Tweet text   | String |
| **training_raw_tweet_features_hashtags**   | Tab separated list of hastags (identifiers) present in the tweet            | String |
| **training_raw_tweet_features_tweet_id**   | Tweet identifier                                                            | String |
| **training_raw_tweet_features_media**      | Tab separated list of media types. Media type can be in (Photo, Video, Gif) | String |
| **training_raw_tweet_features_links**      | Tab separeted list of links (identifiers) included in the Tweet             | String |
| **training_raw_tweet_features_domains**    | Tab separated list of domains included in the Tweet (twitter.com, dogs.com) | String |
| **training_raw_tweet_features_type**       | Tweet type, can be either Retweet, Quote, Reply, or Toplevel                | String |
| **training_raw_tweet_features_language**   | Identifier corresponding to the inferred language of the Tweet              | String |
| **training_raw_tweet_features_timestamp**  | Unix timestamp, in sec of the creation time of the Tweet                    | Int32  |

### Raw columns - Creator features
| Resource uid | Description | Type | 
|---|---|---|
| **training_raw_creator_user_id**             | User identifier                                                             | String  |
| **training_raw_creator_follower_count**      | Number of followers of the user                                             | Int32   |
| **training_raw_creator_following_count**     | Number of accounts the user is following                                    | Int32   |
| **training_raw_creator_is_verified**         | Is the account verified?                                                    | Boolean |
| **training_raw_creator_creation_timestamp**  | Unix timestamp, in seconds, of the creation time of the account             | Int32   |

### Raw columns - Engager Features
| Resource uid | Description | Type | 
|---|---|---|
| **training_raw_engager_user_id**             | User identifier                                                             | String  |
| **training_raw_engager_follower_count**      | Number of followers of the user                                             | Int32   |
| **training_raw_engager_following_count**     | Number of accounts the user is following                                    | Int32   |
| **training_raw_engager_is_verified**         | Is the account verified?                                                    | Boolean |
| **training_raw_engager_creation_timestamp**  | Unix timestamp, in seconds, of the creation time of the account             | Int32   |

### Raw columns - Engagement Features
| Resource uid | Description | Type | 
|---|---|---|
| **training_raw_engagegement_creator_follows_engager** | Does the account of the engaged tweet author follow the account that has made the engagement?                       | Boolean |
| **training_raw_engager_reply_timestamp**              | If there is at least one, unix timestamp, in s, of one of the replies                                               | Int32   |
| **training_raw_engager_retweet_timestamp**            | If there is one, unix timestamp, in s, of the retweet of the tweet by the engaging user                             | Int32   |
| **training_raw_engager_retweet_comment_timestamp**    | If there is at least one, unix timestamp, in s, of one of the retweet with comment of the tweet by the engaging user| Int32   |
| **training_raw_engager_like_timestamp**               | If there is one, Unix timestamp, in s, of the like                                                                  | Int32   |

# Validation Raw columns

They are the columns without filtering and processing. The value in those columns can be Integers, Booleans or String. All the data is wrapped in Pandas Dataframe. <br>
The validation set is composed by ~15M rows. Those columns are saved both as csv and pck files.<br>
**missing data**: missing data has value pandas.NA

### Validation Raw columns - Tweet Features

| Resource uid | Description | Type | 
|---|---|---|
| **training_raw_tweet_features_text_token** | Ordered list of Bert ids corresponding to Bert tokenization of Tweet text   | String |
| **training_raw_tweet_features_hashtags**   | Tab separated list of hastags (identifiers) present in the tweet            | String |
| **training_raw_tweet_features_tweet_id**   | Tweet identifier                                                            | String |
| **training_raw_tweet_features_media**      | Tab separated list of media types. Media type can be in (Photo, Video, Gif) | String |
| **training_raw_tweet_features_links**      | Tab separeted list of links (identifiers) included in the Tweet             | String |
| **training_raw_tweet_features_domains**    | Tab separated list of domains included in the Tweet (twitter.com, dogs.com) | String |
| **training_raw_tweet_features_type**       | Tweet type, can be either Retweet, Quote, Reply, or Toplevel                | String |
| **training_raw_tweet_features_language**   | Identifier corresponding to the inferred language of the Tweet              | String |
| **training_raw_tweet_features_timestamp**  | Unix timestamp, in sec of the creation time of the Tweet                    | Int32  |

### Raw columns - Creator features
| Resource uid | Description | Type | 
|---|---|---|
| **training_raw_creator_user_id**             | User identifier                                                             | String  |
| **training_raw_creator_follower_count**      | Number of followers of the user                                             | Int32   |
| **training_raw_creator_following_count**     | Number of accounts the user is following                                    | Int32   |
| **training_raw_creator_is_verified**         | Is the account verified?                                                    | Boolean |
| **training_raw_creator_creation_timestamp**  | Unix timestamp, in seconds, of the creation time of the account             | Int32   |

### Raw columns - Engager Features
| Resource uid | Description | Type | 
|---|---|---|
| **training_raw_engager_user_id**             | User identifier                                                             | String  |
| **training_raw_engager_follower_count**      | Number of followers of the user                                             | Int32   |
| **training_raw_engager_following_count**     | Number of accounts the user is following                                    | Int32   |
| **training_raw_engager_is_verified**         | Is the account verified?                                                    | Boolean |
| **training_raw_engager_creation_timestamp**  | Unix timestamp, in seconds, of the creation time of the account             | Int32   |

### Raw columns - Engagement Features
| Resource uid | Description | Type | 
|---|---|---|
| **training_raw_engagegement_creator_follows_engager** | Does the account of the engaged tweet author follow the account that has made the engagement?                       | Boolean |
| **training_raw_engager_reply_timestamp**              | If there is at least one, unix timestamp, in s, of one of the replies                                               | Int32   |
| **training_raw_engager_retweet_timestamp**            | If there is one, unix timestamp, in s, of the retweet of the tweet by the engaging user                             | Int32   |
| **training_raw_engager_retweet_comment_timestamp**    | If there is at least one, unix timestamp, in s, of one of the retweet with comment of the tweet by the engaging user| Int32   |
| **training_raw_engager_like_timestamp**               | If there is one, Unix timestamp, in s, of the like                                                                  | Int32   |


# Training Mapped columns

They are the columns after applying a dictionary. The value in those columns can be Integers, Booleans or Numpy Arrays. All the data is wrapped in Pandas Dataframe. <br>
The training set is composed by ~ 150M rows. Those columns are saved both as csv and pck files. <br>
**missing data**: missing data has value None

### Training Raw columns - Details

| Resource uid | Description | Type | 
|---|---|---|
| **training_mapped_tweet_features_hashtags**     | Tab separated list of hastags (identifiers) present in the tweet            | Arrary of Int32 |
| **training_mapped_tweet_features_tweet_id**     | Tweet identifier                                                            | Int32           |
| **training_mapped_tweet_features_links**        | Tab separeted list of links (identifiers) included in the Tweet             | Arrary of Int32 |
| **training_mapped_tweet_features_domains**      | Tab separated list of domains included in the Tweet (twitter.com, dogs.com) | Arrary of Int32 |
| **training_mapped_tweet_features_language**     | Identifier corresponding to the inferred language of the Tweet              | Int32           |
| **training_mapped_creator_user_id**             | User identifier                                                             | Int32           |
| **training_mapped_engager_user_id**             | User identifier                                                             | Int32           |

# Training Mapped columns

They are the columns after applying a dictionary. The value in those columns can be Integers, Booleans or Numpy Arrays. All the data is wrapped in Pandas Dataframe. <br>
The validation set is composed by ~ 15M rows. Those columns are saved both as csv and pck files. <br>
**missing data**: missing data has value None

### Validation Raw columns - Details

| Resource uid | Description | Type | 
|---|---|---|
| **validation_mapped_tweet_features_hashtags**     | Tab separated list of hastags (identifiers) present in the tweet            | Arrary of Int32 |
| **validation_mapped_tweet_features_tweet_id**     | Tweet identifier                                                            | Int32           |
| **validation_mapped_tweet_features_links**        | Tab separeted list of links (identifiers) included in the Tweet             | Arrary of Int32 |
| **validation_mapped_tweet_features_domains**      | Tab separated list of domains included in the Tweet (twitter.com, dogs.com) | Arrary of Int32 |
| **validation_mapped_tweet_features_language**     | Identifier corresponding to the inferred language of the Tweet              | Int32           |
| **validation_mapped_creator_user_id**             | User identifier                                                             | Int32           |
| **validation_mapped_engager_user_id**             | User identifier                                                             | Int32           |      


# Dictionaries

They are python dictionaries that contains mapping from raw string enconding to integers. <br>
The dictionaries are saved in both directions, both from (raw -> int) and (int -> raw)
The data are saved in json format.

### Dictionaries
| Resource uid | Description |
|---|---|
| **dictionary_domain_id_direct**     | Mapping between raw domain id (String) and custom domain id (Integer)     |
| **dictionary_domain_id_inverse**    | Mapping between custom domain id (Integer) and raw domain id (String)     |
| **dictionary_link_id_direct**       | Mapping between raw link id (String) and custom link id (Integer)         |
| **dictionary_link_id_inverse**      | Mapping between custom link id (Integer) and  raw link id (String)        |
| **dictionary_hashtag_id_direct**    | Mapping between raw hashtag id (String) and custom hashtag id (Integer)   |
| **dictionary_hashtag_id_inverse**   | Mapping between custom hashtag id (Integer) and raw hashtag id (String)   |
| **dictionary_language_id_direct**   | Mapping between raw language id (String) and custom language id (Integer) |
| **dictionary_language_id_inverse**  | Mapping between custom language id (Integer) and raw language id (String) |
| **dictionary_tweet_id_direct**      | Mapping between raw tweet id (String) and custom tweet id (Integer)       |
| **dictionary_tweet_id_inverse**     | Mapping between custom tweet id (Integer) and raw tweet id (String)       |
| **dictionary_user_id_direct**       | Mapping between raw user id (String) and custom user id (Integer)         |
| **dictionary_user_id_inverse**      | Mapping between custom user id (Integer) and raw user id (String)         |