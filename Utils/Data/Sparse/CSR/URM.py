from Utils.Data.Data import get_dataset
from Utils.Data.Sparse.CSR_SparseMatrix import CSR_SparseMatrix
import scipy.sparse as sps
import numpy as np
import pandas as pd


class URM(CSR_SparseMatrix):

    def __init__(self, df: pd.DataFrame, max_user_id, max_tweet_id):
        super().__init__("urm_csr_matrix")

        assert df.columns.shape[0] == 3, "The dataframe must have exactly three columns"
        assert 'mapped_feature_engager_id' in df.columns, "The dataframe must have mapped_feature_engager_id column"
        assert 'mapped_feature_tweet_id' in df.columns, "The dataframe must have mapped_feature_tweet_id column"
        assert 'engagement' in df.columns, "The dataframe must have engagement column"

        self.df = df

        self.max_user_id = max_user_id
        self.max_tweet_id = max_tweet_id

    def create_matrix(self):
        # creation of the urm
        # taking only the positive interactions
        # it could be interesting to take also the negative with a -1 value
        self.df = self.df[self.df['tweet_feature_engagement_is_positive']==True]


        # the matrix will be User X Tweet
        engager_ids_arr = self.df['mapped_feature_engager_id'].values
        tweet_ids_arr = self.df['mapped_feature_tweet_id'].values
        interactions_arr = np.array([1] * len(self.df))

        urm = sps.coo_matrix((interactions_arr, (engager_ids_arr, tweet_ids_arr)),
                             shape=(self.max_user_id+1, self.max_tweet_id+1)).tocsr()

        sps.save_npz('urm.npz', urm)
