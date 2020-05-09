from Utils.Data.Data import get_dataset
from Utils.Data.Sparse.CSR_SparseMatrix import CSR_SparseMatrix
import scipy.sparse as sps
import numpy as np


class URM(CSR_SparseMatrix):

    def __init__(self, dataset_id: str):
        super().__init__("urm_csr_matrix")
        self.dataset_id = dataset_id

    def create_matrix(self):
        # creation of the urm

        features_urm = [
            "mapped_feature_tweet_id",
            "mapped_feature_engager_id",
            "tweet_feature_engagement_is_positive"
        ]

        df = get_dataset(features=features_urm, dataset_id=self.dataset_id)

        # taking only the positive interactions
        # it could be interesting to take also the negative with a -1 value
        df = df[df['tweet_feature_engagement_is_positive'] == True]

        # the matrix will be User X Tweet
        engager_ids_arr = df['mapped_feature_engager_id'].values
        tweet_ids_arr = df['mapped_feature_tweet_id'].values
        interactions_arr = np.array([1] * len(df))

        urm = sps.coo_matrix((interactions_arr, (engager_ids_arr, tweet_ids_arr))).tocsr()
        sps.save_npz('urm.npz', urm)
