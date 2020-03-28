from abc import abstractmethod
from Utils.Data.Dictionary.Dictionary import Dictionary
import pathlib as pl
import pickle
import gzip
import json

from Utils.Data.Features.RawFeatures import *


class MappingDictionary(Dictionary):
    """
    Mapping dictionaries are built with the data of train and test set.
    """

    def __init__(self, dictionary_name: str, inverse: bool = False):
        super().__init__(dictionary_name)
        self.inverse = inverse
        self.direct_path_pck_path = pl.Path(f"{Dictionary.ROOT_PATH}/mapping/{self.dictionary_name}/direct.pck.gz")
        self.inverse_path_pck_path = pl.Path(f"{Dictionary.ROOT_PATH}/mapping/{self.dictionary_name}/inverse.pck.gz")
        self.direct_path_json_path = pl.Path(f"{Dictionary.ROOT_PATH}/mapping/{self.dictionary_name}/direct.json.gz")
        self.inverse_path_json_path = pl.Path(f"{Dictionary.ROOT_PATH}/mapping/{self.dictionary_name}/inverse.json.gz")

    def has_dictionary(self):
        if self.inverse:
            return self.inverse_path_pck_path.is_file()
        else:
            return self.direct_path_pck_path.is_file()

    def load_dictionary(self):
        if self.inverse:
            with gzip.GzipFile(self.inverse_path_pck_path, 'rb') as file:
                return pickle.load(file)
        else:
            with gzip.GzipFile(self.direct_path_pck_path, 'rb') as file:
                return pickle.load(file)

    @abstractmethod
    def create_dictionary(self):
        pass

    def save_dictionary(self, inverse_dictionary):
        dictionary = {v: k for k, v in inverse_dictionary.items()}
        self.direct_path_pck_path.parent.mkdir(parents=True, exist_ok=True)
        self.inverse_path_pck_path.parent.mkdir(parents=True, exist_ok=True)
        self.direct_path_json_path.parent.mkdir(parents=True, exist_ok=True)
        self.inverse_path_json_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.GzipFile(self.direct_path_pck_path, 'wb') as file:
            pickle.dump(dictionary, file, protocol=pickle.HIGHEST_PROTOCOL)
        with gzip.GzipFile(self.inverse_path_pck_path, 'wb') as file:
            pickle.dump(inverse_dictionary, file, protocol=pickle.HIGHEST_PROTOCOL)
        with gzip.GzipFile(self.direct_path_json_path, 'wb') as file:
            file.write(json.dumps(dictionary).encode('utf-8'))
        with gzip.GzipFile(self.inverse_path_json_path, 'wb') as file:
            file.write(json.dumps(inverse_dictionary).encode('utf-8'))


class MappingTweetIdDictionary(MappingDictionary):

    def __init__(self, inverse: bool = False):
        super().__init__("mapping_tweet_id_dictionary", inverse)

    def create_dictionary(self):
        train_feature = RawFeatureTweetId("train")
        test_feature = RawFeatureTweetId("test")
        data = pd.concat([
            train_feature.load_or_create()[train_feature.feature_name],
            test_feature.load_or_create()[test_feature.feature_name]
        ])
        dictionary = pd.DataFrame(data.unique()).to_dict()[0]

        self.save_dictionary(dictionary)


class MappingUserIdDictionary(MappingDictionary):

    def __init__(self, inverse: bool = False):
        super().__init__("mapping_user_id_dictionary", inverse)

    def create_dictionary(self):
        train_feature_creator = RawFeatureCreatorId("train")
        test_feature_creator = RawFeatureCreatorId("test")
        train_feature_engager = RawFeatureEngagerId("train")
        test_feature_engager = RawFeatureEngagerId("test")
        data = pd.concat([
            train_feature_creator.load_or_create()[train_feature_creator.feature_name],
            test_feature_creator.load_or_create()[test_feature_creator.feature_name],
            train_feature_engager.load_or_create()[train_feature_engager.feature_name],
            test_feature_engager.load_or_create()[test_feature_engager.feature_name]
        ])
        dictionary = pd.DataFrame(data.unique()).to_dict()[0]

        self.save_dictionary(dictionary)


class MappingLanguageDictionary(MappingDictionary):

    def __init__(self, inverse: bool = False):
        super().__init__("mapping_language_id_dictionary", inverse)

    def create_dictionary(self):
        train_feature = RawFeatureTweetLanguage("train")
        test_feature = RawFeatureTweetLanguage("test")
        data = pd.concat([
            train_feature.load_or_create()[train_feature.feature_name],
            test_feature.load_or_create()[test_feature.feature_name]
        ])
        dictionary = pd.DataFrame(data.unique()).to_dict()[0]

        self.save_dictionary(dictionary)


class MappingDomainDictionary(MappingDictionary):

    def __init__(self, inverse: bool = False):
        super().__init__("mapping_domain_id_dictionary", inverse)

    def create_dictionary(self):
        train_feature = RawFeatureTweetDomains("train")
        test_feature = RawFeatureTweetDomains("test")
        data = pd.concat([
            train_feature.load_or_create()[train_feature.feature_name],
            test_feature.load_or_create()[test_feature.feature_name]
        ])
        data = pd.DataFrame([y for x in data.dropna() for y in x.split('\t')])
        data = data[data.columns[0]]
        dictionary = pd.DataFrame(data.unique()).to_dict()[0]

        self.save_dictionary(dictionary)


class MappingLinkDictionary(MappingDictionary):

    def __init__(self, inverse: bool = False):
        super().__init__("mapping_link_id_dictionary", inverse)

    def create_dictionary(self):
        train_feature = RawFeatureTweetLinks("train")
        test_feature = RawFeatureTweetLinks("test")
        data = pd.concat([
            train_feature.load_or_create()[train_feature.feature_name],
            test_feature.load_or_create()[test_feature.feature_name]
        ])
        data = pd.DataFrame([y for x in data.dropna() for y in x.split('\t')])
        data = data[data.columns[0]]
        dictionary = pd.DataFrame(data.unique()).to_dict()[0]

        self.save_dictionary(dictionary)


class MappingHashtagDictionary(MappingDictionary):

    def __init__(self, inverse: bool = False):
        super().__init__("mapping_hashtag_id_dictionary", inverse)

    def create_dictionary(self):
        train_feature = RawFeatureTweetHashtags("train")
        test_feature = RawFeatureTweetHashtags("test")
        data = pd.concat([
            train_feature.load_or_create()[train_feature.feature_name],
            test_feature.load_or_create()[test_feature.feature_name]
        ])
        data = pd.DataFrame([y for x in data.dropna() for y in x.split('\t')])
        data = data[data.columns[0]]
        dictionary = pd.DataFrame(data.unique()).to_dict()[0]

        self.save_dictionary(dictionary)


class MappingMediaDictionary(MappingDictionary):

    def __init__(self, inverse: bool = False):
        super().__init__("mapping_media_id_dictionary", inverse)

    def create_dictionary(self):
        train_feature = RawFeatureTweetMedia("train")
        test_feature = RawFeatureTweetMedia("test")
        data = pd.concat([
            train_feature.load_or_create()[train_feature.feature_name],
            test_feature.load_or_create()[test_feature.feature_name]
        ])
        data = pd.DataFrame([y for x in data.dropna() for y in x.split('\t')])
        data = data[data.columns[0]]
        dictionary = pd.DataFrame(data.unique()).to_dict()[0]

        self.save_dictionary(dictionary)
