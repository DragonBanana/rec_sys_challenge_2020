import time

import numpy as np
import pandas as pd
import sys
from tqdm.contrib.concurrent import process_map

from Utils.Data.Data import get_feature
from Utils.Data.Features.Generated.EngagerFeature.KnownEngagementCount import *
from Utils.Data.Features.MappedFeatures import MappedFeatureCreatorId, MappedFeatureTweetId
from Utils.Data.Sparse.CSR.CreatorTweetMatrix import CreatorTweetMatrix
from Utils.Data.Sparse.CSR.DomainMatrix import DomainMatrix
from Utils.Data.Sparse.CSR.HashtagMatrix import HashtagMatrix
from Utils.Data.Sparse.CSR.LinkMatrix import LinkMatrix
from Utils.Data.Sparse.CSR.URM import URM

# PART THAT GENERATES THE TRAIN DATASET

dataset_id = sys.argv[1]
sim_matrix_name = "link"
sim = LinkMatrix().load_similarity().tocsc()
n_folds = 5
labels = ["like", "retweet", "reply", "comment", "positive", "negative"]

# dataset_id = "train"
# sim_matrix_name = "hashtag"
# sim = HashtagMatrix().load_similarity().tocsc()
# n_folds = 5
# label = "like"

creator_feature = MappedFeatureCreatorId(dataset_id)
engager_feature = MappedFeatureEngagerId(dataset_id)
tweet_feature = MappedFeatureTweetId(dataset_id)

cold_item_dict = np.full(sim.shape[0], True)
cold_item_dict[pd.unique(sim.nonzero()[0])] = False

for label in labels:
    prime_dataframe = pd.concat(
        [
            creator_feature.load_or_create(),
            engager_feature.load_or_create(),
            tweet_feature.load_or_create(),
            get_feature(f"tweet_feature_engagement_is_{label}", dataset_id=dataset_id)
        ],
        axis=1
    ).sample(frac=1)

    prime_dataframe.columns = [
        creator_feature.feature_name,
        engager_feature.feature_name,
        tweet_feature.feature_name,
        "engagement"
    ]

    engager_urm_columns = [
        engager_feature.feature_name,
        tweet_feature.feature_name,
        "engagement"
    ]

    creator_urm_columns = [
        creator_feature.feature_name,
        tweet_feature.feature_name
    ]
    print("data loaded")

    ctm = CreatorTweetMatrix(prime_dataframe[creator_urm_columns]).get_as_urm().astype(np.uint8)

    for actual_fold in range(n_folds):

        start_time = time.time()
        folds = np.array_split(prime_dataframe, n_folds)
        train_folds = pd.concat([folds[i][engager_urm_columns] for i in range(n_folds) if i is not actual_fold])
        print(f"time taken to generate folds: {time.time() - start_time}")

        start_time = time.time()
        urm = URM(train_folds).get_as_urm().astype(np.uint8) + ctm
        urm = urm.tocsr()
        print(f"time taken to create urm: {time.time() - start_time}")

        dataframe = folds[actual_fold][[
            engager_feature.feature_name,
            tweet_feature.feature_name
        ]].copy()

        index = dataframe.index

        start_time = time.time()
        cold_user_dict = np.full(urm.shape[0], True)
        cold_user_dict[pd.unique(urm.nonzero()[0])] = False

        dataframe['cold_user'] = dataframe[engager_feature.feature_name].map(lambda x: cold_user_dict[x])
        dataframe['cold_item'] = dataframe[tweet_feature.feature_name].map(lambda x: cold_item_dict[x])
        dataframe['cold'] = dataframe['cold_user'] | dataframe['cold_item']
        dataframe = dataframe[dataframe['cold'] == False]
        print(f"time taken to filter dataframe: {time.time() - start_time}")

        dataframe = dataframe[[
            engager_feature.feature_name,
            tweet_feature.feature_name
        ]]

        # PART OF CODE THE GENERATE user_array_dict

        idx = list(range(0, urm.shape[0], 100000))
        idx.extend([urm.shape[0]])

        urm_chunks = []
        for i in range(len(idx) - 1):
            urm_chunks.append(urm[idx[i]:idx[i + 1]])


        def p_urm_chunks(urm):
            return [urm[i].indices if not cold_user_dict[i] else None for i in range(urm.shape[0])]

        result = process_map(p_urm_chunks, urm_chunks, max_workers=48)
        # user_array_dict = np.hstack(result)
        user_array_dict = []
        for x in result:
            user_array_dict += x

        def select_value_from_csr(matrix, row, cols):
            i_start, i_stop = matrix.indptr[row], matrix.indptr[row + 1]
            result = 0
            j_start = 0
            j_stop = len(cols) - 1
            i = i_start
            j = j_start
            while (True):
                if matrix.indices[i] < cols[j]:
                    i += 1
                elif matrix.indices[i] > cols[j]:
                    j += 1
                else:
                    result += matrix.data[i]
                    i += 1
                    j += 1
                if i > i_stop or j > j_stop:
                    break
            return result


        def compute_score(dataframe):
            result = pd.DataFrame(
                [
                    ((select_value_from_csr(sim, item, user_array_dict[user])) if user_array_dict[user] is not None else 0)
                    for user, item
                    in zip(dataframe[engager_feature.feature_name],
                           dataframe[tweet_feature.feature_name])
                ],
                index=dataframe.index
            )
            return result

        chunks = np.array_split(dataframe, 10000)
        result = process_map(compute_score, chunks, max_workers=48)
        r = pd.concat(result)
        r = pd.concat([pd.DataFrame(index=index), r], axis=1)
        r.fillna(0, inplace=True)
        r.to_pickle(path=f"{sim_matrix_name}_{dataset_id.replace('/', '_')}_fold_n_{actual_fold}_{label}", compression='gzip')