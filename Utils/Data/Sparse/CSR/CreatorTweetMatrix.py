from Utils.Data.Data import get_dataset
from Utils.Data.Sparse.CSR_SparseMatrix import CSR_SparseMatrix
import numpy as np
import scipy.sparse as sps


class CreatorTweetMatrix(CSR_SparseMatrix):

    def __init__(self, dataset_id: str):
        super().__init__("creator_tweet_csr_matrix")
        self.dataset_id = dataset_id


    def create_matrix(self):

        # creation of the creator - tweet matrix
        features_creator = [
            "mapped_feature_tweet_id",
            "mapped_feature_creator_id",
            "tweet_feature_engagement_is_positive"
        ]

        df = get_dataset(features=features_creator, dataset_id=self.dataset_id)
        df = df.drop_duplicates()

        # the matrix will be User X Tweet
        creator_ids_arr = df['mapped_feature_creator_id'].values
        tweet_ids_arr = df['mapped_feature_tweet_id'].values
        creations_arr = np.array([1] * len(df))

        ctm = sps.coo_matrix((creations_arr, (creator_ids_arr, tweet_ids_arr))).tocsr()
        sps.save_npz('ctm.npz', ctm)
