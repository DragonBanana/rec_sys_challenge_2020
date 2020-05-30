import sys
sys.path.append('/home/jovyan/work/github_repo')
from RootPath import *
from Utils.Data.Features.Generated.TweetFeature.FromTextToken import *
from Utils.Data.Features.MappedFeatures import MappedFeatureGroupedTweetLanguage

feature = TweetFeatureTextTokenDecoded("train_days_1")
feature.create_feature()

feature = TweetFeatureTextTopicWordCountAdultContent("train_days_1")
feature.create_feature()
feature = TweetFeatureTextTopicWordCountKpop("train_days_1")
feature.create_feature()
feature = TweetFeatureTextTopicWordCountCovid("train_days_1")
feature.create_feature()
feature = TweetFeatureTextTopicWordCountSport("train_days_1")
feature.create_feature()