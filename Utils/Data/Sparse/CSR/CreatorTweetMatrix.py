from Utils.Data.Data import get_dataset
from Utils.Data.Sparse.CSR_SparseMatrix import CSR_SparseMatrix
import numpy as np
import scipy.sparse as sps
import pandas as pd

class CreatorTweetMatrix(CSR_SparseMatrix):

    def __init__(self, df: pd.DataFrame, max_user_id, max_tweet_id):
        super().__init__("creator_tweet_csr_matrix")

        assert df.columns.shape[0] == 2, "The dataframe must have exactly two columns"
        assert 'mapped_feature_creator_id' in df.columns, "The dataframe must have mapped_feature_creator_id column"
        assert 'mapped_feature_tweet_id' in df.columns, "The dataframe must have mapped_feature_tweet_id column"

        self.df = df
        self.max_user_id = max_user_id
        self.max_tweet_id = max_tweet_id


    def create_matrix(self):

        # creation of the creator - tweet matrix
        self.df = self.df.drop_duplicates()
        # the matrix will be User X Tweet
        creator_ids_arr = self.df['mapped_feature_creator_id'].values
        tweet_ids_arr = self.df['mapped_feature_tweet_id'].values
        creations_arr = np.array([1] * len(self.df))

        ctm = sps.coo_matrix((creations_arr, (creator_ids_arr, tweet_ids_arr)),
                             shape=(self.max_user_id+1, self.max_tweet_id+1)).tocsr()
        
        sps.save_npz('ctm.npz', ctm)
