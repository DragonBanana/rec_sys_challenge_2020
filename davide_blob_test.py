from Utils.Data import Data
from Utils.Data.Data import get_dataset_xgb, get_feature
from Utils.Data.DataUtils import create_all, consistency_check, consistency_check_all
from Utils.Data.Features.Generated.EngagerFeature.KnownEngagementCount import *
from Utils.Data.Features.MappedFeatures import MappedFeatureCreatorId
from Utils.Data.Split import TimestampBasedSplit
import pandas as pd
import numpy as np

from Utils.Data.Statistic.ColdUsers import analyze_cold_user

if __name__ == '__main__':
    TimestampBasedSplit.split_with_timestamp("train", pc_hold_out=0.01)
    #
    create_all(nthread=6)
    #
    # analyze_cold_user("train")

    consistency_check_all()