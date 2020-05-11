import pathlib as pl
import RootPath
import json
from Utils.Data.Dictionary.TweetBasicFeaturesDictArray import CreatorIdTweetBasicFeatureDictArray
from Utils.Data.Dictionary.UserBasicFeaturesDictArray import IsVerifiedUserBasicFeatureDictArray


def get_max_user_id():
    info_path = RootPath.get_dataset_path().joinpath("info.json")
    info_file = open(info_path)
    if info_path.exists():
        with open(info_path, "r") as info_file:
            info = json.load(info_file)
            if "max_user_id" in info.keys():
                max_user_id = info['max_user_id']
            else:
                max_user_id = len(IsVerifiedUserBasicFeatureDictArray().load_or_create())
                info['max_user_id'] = max_user_id
                with open(info_path, "w") as info_file:
                    json.dump(info, info_file)
    else:
        info_file = open(info_path, "w")
        info = {}
        max_user_id = len(IsVerifiedUserBasicFeatureDictArray().load_or_create())
        info['max_user_id'] = max_user_id
        with open(info_path, "w") as info_file:
            json.dump(info, info_file)
    return max_user_id

def get_max_tweet_id():
    info_path = RootPath.get_dataset_path().joinpath("info.json")
    if info_path.exists():
        with open(info_path, "r") as info_file:
            info = json.load(info_file)
            if "max_tweet_id" in info.keys():
                max_tweet_id = info['max_tweet_id']
            else:
                max_tweet_id = len(CreatorIdTweetBasicFeatureDictArray().load_or_create())
                info['max_tweet_id'] = max_tweet_id
                with open(info_path, "w") as info_file:
                    json.dump(info, info_file)
    else:
        info = {}
        max_tweet_id = len(CreatorIdTweetBasicFeatureDictArray().load_or_create())
        info['max_tweet_id'] = max_tweet_id
        with open(info_path, "w") as info_file:
            json.dump(info, info_file)
    return max_tweet_id