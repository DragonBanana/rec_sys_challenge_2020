import pandas as pd

from Utils.Data.Features.Generated.EngagerFeature.NumberOfPreviousEngagementRatio import \
    EngagerFeatureNumberOfPreviousPositiveEngagementRatio1

predictions_file = "test_submission.csv"
dataset_id = "test"

# The dataframes are structured as follow
def compute(cold, hot):

    print(cold)
    print(hot)


if __name__ == '__main__':

    predictions = pd.read_csv(predictions_file)
    predictions.columns = ["tweet", "user", "prediction"]

    n_engagements = EngagerFeatureNumberOfPreviousPositiveEngagementRatio1(dataset_id).load_or_create()
    n_engagements.columns = [0]

    cold_mask = n_engagements[0] > 0
    hot_mask = ~cold_mask

    compute(predictions[cold_mask], predictions[hot_mask])