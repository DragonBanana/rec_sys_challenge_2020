from Utils.Data.Dictionary.MappingDictionary import *
from Utils.Data.Features.Generated.GeneratedFeature import GeneratedFeaturePickle
from Utils.Data.Features.MappedFeatures import MappedFeatureTweetHashtags
import RootPath as rp
import json, gzip


class HasDiscriminativeHashtag_Like(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_has_discriminative_hashtag_like", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/discriminative_hashtags/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/discriminative_hashtags/{self.feature_name}.csv.gz")

    def create_feature(self):
        # Load the hashtags column
        feature = MappedFeatureTweetHashtags(self.dataset_id)
        feature_df = feature.load_or_create()
        print(feature_df)
        # Load the list of discriminative for the like class
        kind_pos, kind_neg = loadDiscriminative("like")
        # Create the feature
        kind_disc_df = pd.DataFrame()
        kind_disc_df[self.feature_name+"pos"] =  feature_df[feature.feature_name].map(lambda x: containsHashtag(x,kind_pos) if x is not None else False)
        kind_disc_df[self.feature_name+"neg"] =  feature_df[feature.feature_name].map(lambda x: containsHashtag(x,kind_neg) if x is not None else False)
        kind_disc_df = kind_disc_df.astype(int)

        self.save_feature(kind_disc_df)

class HasDiscriminativeHashtag_Reply(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_has_discriminative_hashtag_reply", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/discriminative_hashtags/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/discriminative_hashtags/{self.feature_name}.csv.gz")

    def create_feature(self):
        # Load the hashtags column
        feature = MappedFeatureTweetHashtags(self.dataset_id)
        feature_df = feature.load_or_create()
        # Load the list of discriminative for the reply class
        kind_pos, kind_neg = loadDiscriminative("reply")
        # Create the feature
        kind_disc_df = pd.DataFrame()
        kind_disc_df[self.feature_name+"pos"] =  feature_df[feature.feature_name].map(lambda x: containsHashtag(x,kind_pos) if x is not None else False)
        kind_disc_df[self.feature_name+"neg"] =  feature_df[feature.feature_name].map(lambda x: containsHashtag(x,kind_neg) if x is not None else False)
        kind_disc_df = kind_disc_df.astype(int)

        self.save_feature(kind_disc_df)

class HasDiscriminativeHashtag_Retweet(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_has_discriminative_hashtag_retweet", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/discriminative_hashtags/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/discriminative_hashtags/{self.feature_name}.csv.gz")

    def create_feature(self):
        # Load the hashtags column
        feature = MappedFeatureTweetHashtags(self.dataset_id)
        feature_df = feature.load_or_create()
        # Load the list of discriminative for the retweet class
        kind_pos, kind_neg = loadDiscriminative("retweet")
        # Create the feature
        kind_disc_df = pd.DataFrame()
        kind_disc_df[self.feature_name+"pos"] =  feature_df[feature.feature_name].map(lambda x: containsHashtag(x,kind_pos) if x is not None else False)
        kind_disc_df[self.feature_name+"neg"] =  feature_df[feature.feature_name].map(lambda x: containsHashtag(x,kind_neg) if x is not None else False)
        kind_disc_df = kind_disc_df.astype(int)

        self.save_feature(kind_disc_df)

class HasDiscriminativeHashtag_Comment(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_has_discriminative_hashtag_comment", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/discriminative_hashtags/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/discriminative_hashtags/{self.feature_name}.csv.gz")

    def create_feature(self):
        # Load the hashtags column
        feature = MappedFeatureTweetHashtags(self.dataset_id)
        feature_df = feature.load_or_create()

        # Load the list of discriminative for the comment class
        kind_pos, kind_neg = loadDiscriminative("comment")
        # Create the feature
        kind_disc_df = pd.DataFrame()
        kind_disc_df[self.feature_name+"pos"] =  feature_df[feature.feature_name].map(lambda x: containsHashtag(x,kind_pos) if x is not None else False)
        kind_disc_df[self.feature_name+"neg"] =  feature_df[feature.feature_name].map(lambda x: containsHashtag(x,kind_neg) if x is not None else False)
        kind_disc_df = kind_disc_df.astype(int)
        print(kind_disc_df)
        self.save_feature(kind_disc_df)

###########################################                 
#               IMPORTANT                 #      
########################################### 
#The loaded lists have been computed independently by analyzing the whole provided dataset

def loadDiscriminative(kind):
    #list of positive discriminative hashtags for the kind class
    jsonfilename = rp.get_root().joinpath(f"Dataset/Dictionary/discriminative_hashtags/{kind}_pos.gz")
    with gzip.GzipFile(jsonfilename, 'r') as fin:
        json_bytes = fin.read()                      
    json_str = json_bytes.decode('utf-8')            
    ret_pos = json.loads(json_str)
    #list of negative discriminative hashtags for the kind class
    jsonfilename = rp.get_root().joinpath(f"Dataset/Dictionary/discriminative_hashtags/{kind}_neg.gz")
    with gzip.GzipFile(jsonfilename, 'r') as fin:
        json_bytes = fin.read()                      
    json_str = json_bytes.decode('utf-8')            
    ret_neg = json.loads(json_str)
    
    return ret_pos, ret_neg

def containsHashtag(lst, disc_pos):
    for hashtag in lst:
        if hashtag in disc_pos:
            return True
    return False