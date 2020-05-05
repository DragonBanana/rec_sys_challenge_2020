from Utils.Data.Dictionary.UserBasicFeaturesDictArray import UserBasicFeatureDictArrayNumpy
from Utils.Data.Sparse.CSR.Language.LanguageMatrixOnlyPositive import LanguageMatrixOnlyPositive
import pandas as pd

class MainLanguageUserBasicFeatureDictArray(UserBasicFeatureDictArrayNumpy):

    def __init__(self):
        super().__init__("main_language_user_dict_array")

    def create_dictionary(self):
        # Load Language matrix
        csr_matrix = LanguageMatrixOnlyPositive().load_or_create()

        # Cast it to dataframe
        df = pd.DataFrame.sparse.from_spmatrix(csr_matrix)

        # Save the base columns
        base_columns = df.columns

        # Find the main language
        df["main_language"] = df.idxmax(axis=1)

        # Find the amount of time a user has spoken a language
        df['total_count_language_occurence'] = df[base_columns].sum(axis=1)

        # Override the value for users that have never spoken a language
        df["main_language"] = [x if y > 0 else -1 for x, y in
                               zip(df["main_language"], df['total_count_language_occurence'])]

        # To create the some statistics
        # df.drop(columns=['total_count_language_occurence'])
        #
        # # To numpy matrix
        # matrix = df.groupby("main_language").sum()[1:].to_numpy(dtype=int)

        # Save the numpy matrix
        self.save_dictionary(df["main_language"].to_numpy())