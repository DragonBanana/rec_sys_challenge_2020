from Utils.Data.Dictionary.MappingDictionary import *
from Utils.Data.Features.Generated.TweetFeature.IsEngagementType import *
from Utils.Data.Features.Generated.GeneratedFeature import GeneratedFeaturePickle
from Utils.Data.Features.MappedFeatures import MappedFeatureTweetHashtags
import RootPath as rp
import numpy as np

import json, gzip

class UtilityMethodsClass():
    ###########################################                 
    #               IMPORTANT                 #      
    ########################################### 
    #The loaded lists have been computed independently by analyzing the whole provided dataset

    def loadDiscriminative(self,kind, dataset_id, hashtag_df, hashtag_col):
        #load the engagement type column to use as mask
        if kind == "like":
            feature = TweetFeatureEngagementIsLike(dataset_id)
        if kind == "reply":
            feature = TweetFeatureEngagementIsReply(dataset_id)
        if kind == "retweet":
            feature = TweetFeatureEngagementIsRetweet(dataset_id)
        if kind == "comment":
            feature = TweetFeatureEngagementIsComment(dataset_id)
        
        feature_df = feature.load_or_create()

        eng_mask = ~feature_df[feature.feature_name]
        neg_mask = ~eng_mask

        # COMPUTE THE MAX HASHATAG ID FOR THE POSITIVE INTERACTIONS
        pos_max_id = 0

        disc_max = pd.DataFrame()
        disc_max["max"] = hashtag_df[eng_mask][hashtag_col].map(lambda x: getMax(x))
        pos_max_id = disc_max["max"].max() +1
        
        # COUNT THE HASHTAGS OCCreplyURRENCIES FOR THE POSITIVE INTERACTIONS

        self.pos_counts = [0 for i in range(0,int(pos_max_id))]

        tags_to_be_processed = hashtag_df[eng_mask][hashtag_col].dropna()
        tags_to_be_processed.map(self.countPos)

        del disc_max

        # COMPUTE THE MAX HASHATAG ID FOR THE NEGATIVE INTERACTIONS
        neg_max_id = 0

        disc_max = pd.DataFrame()
        disc_max["max"] = hashtag_df[neg_mask][hashtag_col].map(lambda x: getMax(x))

        neg_max_id = disc_max["max"].max() +1
        
        # COUNT THE HASHTAGS OCCreplyURRENCIES FOR THE POSITIVE INTERACTIONS

        self.neg_counts = [0 for i in range(0,int(neg_max_id))]

        tags_to_be_processed = hashtag_df[neg_mask][hashtag_col].dropna()
        tags_to_be_processed.map(self.countNeg)
        
        #---------------------

        c_df_p = pd.DataFrame()
        c_df_p["h"] = self.pos_counts

        c_df_n = pd.DataFrame()
        c_df_n["h"] = self.neg_counts

        pos_and_neg_counts = eng_mask.value_counts()
        unbalancement_ratio = pos_and_neg_counts[1]/pos_and_neg_counts[0]

        # ONLY USE THE HASHTAGS WITH MORE THAN THRESHOLD INTERACTIONS
        threshold = 1000

        pos_search = c_df_p[c_df_p["h"] > threshold]
        neg_search = c_df_n[c_df_n["h"] > threshold/unbalancement_ratio]

        # Populate the lists of "unbalanced" hashtags for both the positive and the negative class

        ret_pos = []
        for i in list(pos_search.index):
            i = int(i)
            if pos_search["h"][i]/c_df_n["h"][i] > 6*unbalancement_ratio:
                ret_pos.append(i)

        ret_neg = []
        for i in list(neg_search.index):
            i = int(i)
            if c_df_p["h"][i]/neg_search["h"][i] < unbalancement_ratio/60:
                ret_neg.append(i)
        
        return ret_pos, ret_neg

    def countPos(self,array):
        for hashtag_id in array:
            self.pos_counts[hashtag_id] = self.pos_counts[hashtag_id] + 1
            # For strange reason, but this return is needed
        return 1
    
    def countNeg(self,array):
        for hashtag_id in array:
            self.neg_counts[hashtag_id] = self.neg_counts[hashtag_id] + 1
            # For strange reason, but this return is needed
        return 1

class HasDiscriminativeHashtag_Like(GeneratedFeaturePickle, UtilityMethodsClass):

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
        # Load the list of discriminative for the like class
        kind_pos, kind_neg = self.loadDiscriminative("like", self.dataset_id, feature_df, feature.feature_name)
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
        kind_pos, kind_neg = self.loadDiscriminative("reply", self.dataset_id, feature_df, feature.feature_name)
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
        kind_pos, kind_neg = self.loadDiscriminative("retweet", self.dataset_id, feature_df, feature.feature_name)
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
        kind_pos, kind_neg = self.loadDiscriminative("comment", self.dataset_id, feature_df, feature.feature_name)
        # Create the feature
        kind_disc_df = pd.DataFrame()
        kind_disc_df[self.feature_name+"pos"] =  feature_df[feature.feature_name].map(lambda x: containsHashtag(x,kind_pos) if x is not None else False)
        kind_disc_df[self.feature_name+"neg"] =  feature_df[feature.feature_name].map(lambda x: containsHashtag(x,kind_neg) if x is not None else False)
        kind_disc_df = kind_disc_df.astype(int)
        
        self.save_feature(kind_disc_df)

def containsHashtag(lst, disc_pos):
    for hashtag in lst:
        if hashtag in disc_pos:
            return True
    return False

def getMax(x):
    if x is not None:
        return max(np.array(x))