from Utils.Data.Dictionary.MappingDictionary import *
from Utils.Data.Dictionary.TweetTextFeaturesDictArray import *
from Utils.Data.Features.Feature import Feature
from Utils.Data.Features.Generated.GeneratedFeature import GeneratedFeaturePickle
from abc import abstractmethod
import pandas as pd
import numpy as np
import gzip

from Utils.Data.Features.MappedFeatures import MappedFeatureTweetId
from Utils.Data.Features.RawFeatures import RawFeatureTweetTextToken

from BERT.TokenizerWrapper import TokenizerWrapper


class TweetFeatureMappedMentions(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_mentions", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.csv.gz")

    def create_feature(self):

        # Load tweet ids
        tweet_id_feature = MappedFeatureTweetId(self.dataset_id)
        tweet_id_df = tweet_id_feature.load_or_create()

        # Merge train and test mentions
        mentions_array = pd.concat([
            pd.read_csv(f"{RootPath.get_dataset_path()}/Dictionary/from_text_token/test_mentions.csv.gz",
                        compression="gzip", sep="\x01", index_col=0),
            pd.read_csv(f"{RootPath.get_dataset_path()}/Dictionary/from_text_token/train_mentions.csv.gz",
                        compression="gzip", sep="\x01", index_col=0)
        ])['mentions_mapped'].astype(str).map(
            lambda x: np.array(x.split('\t'), dtype=np.str) if x != 'nan' else None
        ).array

        # Compute for each engagement the tweet mentions
        mapped_mentions_df = pd.DataFrame(tweet_id_df[tweet_id_feature.feature_name].map(lambda x: mentions_array[x]))

        # Save the dataframe
        self.save_feature(mapped_mentions_df)

class TweetFeatureNumberOfMentions(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_number_of_mentions", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.csv.gz")

    def create_feature(self):
        # Load the extracted mentions
        mentions_feature = TweetFeatureMappedMentions(self.dataset_id)
        mentions_df = mentions_feature.load_or_create()

        # Compute for each engagement the tweet mentions
        mnumber_of_mentions_df = pd.DataFrame(mentions_df[mentions_feature.feature_name].map(lambda x: len(x) if x is not None else 0))

        # Save the dataframe
        self.save_feature(mnumber_of_mentions_df)
        

class TweetFeatureTextEmbeddings(GeneratedFeaturePickle):

    def __init__(self, feature_name: str, dataset_id: str):
        super().__init__(feature_name, dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.csv.gz")
        self.embeddings_array = None
        
    @abstractmethod
    def load_embeddings_dictionary(self):
        pass

    def create_feature(self):
        # Load tweet ids
        tweet_id_feature = MappedFeatureTweetId(self.dataset_id)
        tweet_id_df = tweet_id_feature.load_or_create()
        
        #tweet_id_df = tweet_id_df.head(25)
        #print(tweet_id_df)
        
        self.embeddings_array = self.load_embeddings_dictionary()
        
        columns_num = self.embeddings_array.shape[1]
        
        # this will be the final dataframe
        embeddings_feature_df = pd.DataFrame()
        
        # for each column, map the embeddings dictionary to all the tweets
        for col in range(columns_num):
            print("column :", col)
            embeddings_feature_df[f"embedding_{col}"] = tweet_id_df["mapped_feature_tweet_id"].map(lambda x: self.embeddings_array[x, col])
            
        #print(embeddings_feature_df)
        
        # Save the dataframe
        self.save_feature(embeddings_feature_df)
        
        
class TweetFeatureTextEmbeddingsPCA32(TweetFeatureTextEmbeddings):

    def __init__(self, dataset_id: str):
        super().__init__("text_embeddings_clean_PCA_32", dataset_id)
        
    def load_embeddings_dictionary(self):
        self.embeddings_array = TweetTextEmbeddingsPCA32FeatureDictArray().load_or_create()
        

class TweetFeatureTextEmbeddingsPCA10(TweetFeatureTextEmbeddings):

    def __init__(self, dataset_id: str):
        super().__init__("text_embeddings_clean_PCA_10", dataset_id)
        
    def load_embeddings_dictionary(self):
        self.embeddings_array = TweetTextEmbeddingsPCA10FeatureDictArray().load_or_create()
        

class TweetFeatureTextEmbeddingsHashtagsMentionsLDA15(TweetFeatureTextEmbeddings):
        
    def __init__(self, dataset_id: str):
        super().__init__("text_embeddings_hashtags_mentions_LDA_15", dataset_id)
        
    def load_embeddings_dictionary(self):
        self.embeddings_array = TweetTextEmbeddingsHashtagsMentionsLDA15FeatureDictArray().load_or_create()
        

class TweetFeatureTextEmbeddingsHashtagsMentionsLDA20(TweetFeatureTextEmbeddings):
        
    def __init__(self, dataset_id: str):
        super().__init__("text_embeddings_hashtags_mentions_LDA_20", dataset_id)
        
    def load_embeddings_dictionary(self):
        self.embeddings_array = TweetTextEmbeddingsHashtagsMentionsLDA15FeatureDictArray().load_or_create()
        
    
class TweetFeatureDominantTopic(GeneratedFeaturePickle):

    def __init__(self, feature_name : str, dataset_id: str):
        super().__init__(feature_name, dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.csv.gz")
        self.dictionary_array = None
        
    @abstractmethod
    def load_dictionary(self):
        pass

    def create_feature(self):
        # Load the tweet ids
        tweet_id_feature = MappedFeatureTweetId(self.dataset_id)
        tweet_id_df = tweet_id_feature.load_or_create()
        
        self.dictionary_array = self.load_dictionary()
        
        df = pd.DataFrame()
        df["dominant_topic"] = tweet_id_df["mapped_feature_tweet_id"].map(lambda x: np.argmax(self.dictionary_array[x]) if np.max(self.dictionary_array[x]) == np.min(self.dictionary_array[x]) else -1)

        # Save the dataframe
        self.save_feature(df)
        

class TweetFeatureDominantTopicLDA15(TweetFeatureDominantTopic):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_dominant_topic_LDA_15", dataset_id)
        
    def load_dictionary(self):
        self.dictionary_array = TweetTextEmbeddingsHashtagsMentionsLDA15FeatureDictArray().load_or_create()
        
        
class TweetFeatureDominantTopicLDA20(TweetFeatureDominantTopic):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_dominant_topic_LDA_20", dataset_id)
        
    def load_dictionary(self):
        self.dictionary_array = TweetTextEmbeddingsHashtagsMentionsLDA20FeatureDictArray().load_or_create()
        

class TweetFeatureTokenLength(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_token_length", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.csv.gz")

    def create_feature(self):
        # Load the tweet ids
        tweet_id_feature = MappedFeatureTweetId(self.dataset_id)
        tweet_id_df = tweet_id_feature.load_or_create()

        # load the length dictionary
        tweet_length_dict = TweetTokenLengthFeatureDictArray().load_or_create()


        # Compute for each engagement the tweet mentions
        length_df = pd.DataFrame(tweet_id_df['mapped_feature_tweet_id'].map(lambda t_id: tweet_length_dict[t_id]))


        # Save the dataframe
        self.save_feature(length_df)


class TweetFeatureTokenLengthUnique(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_token_length_unique", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.csv.gz")

    def create_feature(self):
        # Load the tweet ids
        tweet_id_feature = MappedFeatureTweetId(self.dataset_id)
        tweet_id_df = tweet_id_feature.load_or_create()

        # load the length dictionary
        tweet_length_unique_dict = TweetTokenLengthUniqueFeatureDictArray().load_or_create()

        # Compute for each engagement the tweet mentions
        length_df = pd.DataFrame(tweet_id_df['mapped_feature_tweet_id'].map(lambda t_id: tweet_length_unique_dict[t_id]))

        # Save the dataframe
        self.save_feature(length_df)


class TweetFeatureTextContainsAdultContent(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_number_of_adult_content_words", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.csv.gz")

        self.tok = TokenizerWrapper("bert-base-multilingual-cased")
        
        self.adult_content_words = ['adult content', 'adult film', 'adult movie', 'adult video', 'anal', 'ass', 'bara', 'barely legal', 'bdsm', 'bestiality', 'bisexual', 'bitch', 'blowjob', 'bondage', 'boob', 'boobs', 'boobies', 'boobys', 'booty', 'bound & gagged', 'bound and gagged', 'breast', 'breasts', 'bukkake', 'butt', 'cameltoe', 'creampie', 'cock', 'condom', 'cuck-old', 'cuckold', 'cum', 'cumshot', 'cunt', 'deep thraot', 'deap throat', 'deep thraoting', 'deap throating', 'deep-thraot', 'deap-throat', 'deep-thraoting', 'deap-throating', 'deepthraot', 'deapthroat', 'deepthraoting', 'deapthroating', 'dick', 'dildo', 'emetophilia', 'erotic', 'erotica', 'erection', 'erections', 'escort', 'facesitting', 'facial', 'felching', 'femdon', 'fetish', 'fisting', 'futanari', 'fuck', 'fucking', 'fucked', 'fucks', 'fucker', 'gangbang', 'gapping', 'gay', 'gentlemens club', 'gloryhole', 'glory hole', 'gonzo', 'gore', 'guro', 'handjob', 'hardon', 'hard-on', 'hentai', 'hermaphrodite', 'hidden camera', 'hump', 'humped', 'humping', 'hustler', 'incest', 'jerk off', 'jerking off', 'kinky', 'lesbian', 'lolicon ', 'masturbate', 'masturbating', 'masturbation', 'mature', 'mens club', 'menstrual', 'menstral', 'menstraul', 'milf', 'milking', 'naked', 'naughty', 'nude', 'orgasm', 'orgy', 'orgie', 'pearl necklace', 'pegging', 'penis', 'penetration', 'playboy', 'playguy', 'playgirl', 'porn', 'pornography', 'pornstar', 'pov', 'pregnant', 'preggo', 'pubic', 'pussy', 'rape', 'rimjob', 'scat', 'semen', 'sex', 'sexual', 'sexy', 'sexting', 'shemale', 'skank', 'slut', 'snuff', 'snuf', 'sperm', 'squirt', 'suck', 'swapping', 'tit', 'trans', 'transman', 'transsexual', 'transgender', 'threesome', 'tube8', 'twink', 'upskirt', 'vagina', 'virgin', 'whore', 'wore', 'xxx', 'yaoi', 'yif', 'yiff', 'yiffy', 'yuri', 'youporn']

    def check_if_contains_adult_words(self, row):
        tokens = row.replace('\n','').split('\t')
        sentence = self.tok.decode(tokens).lower()
        count = 0
        for w in self.adult_content_words:
            if w in sentence:
                count += 1
        return count
    
    def create_feature(self):
        # Load the tweet ids and tokens
        tweet_tokens_feature = RawFeatureTweetTextToken(self.dataset_id)
        tweet_tokens_df = tweet_tokens_feature.load_or_create()

        #print(tweet_tokens_df)

        number_of_adult_words_df = pd.DataFrame(tweet_tokens_df['raw_feature_tweet_text_token'].apply(self.check_if_contains_adult_words))
        
        #print(number_of_adult_words_df)
        
        print("Number of rows with feature == 0 :", (number_of_adult_words_df['raw_feature_tweet_text_token'] == 0).sum())

        # Save the dataframe
        self.save_feature(number_of_adult_words_df)