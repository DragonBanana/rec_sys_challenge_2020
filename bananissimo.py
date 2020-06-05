
import pandas as pd

from Utils.Data.Features.Generated.EngagerFeature.NumberOfPreviousEngagementRatio import \
    EngagerFeatureNumberOfPreviousPositiveEngagementRatio1

predictions_file = "xgb_submission_like.csv"
predictions_file_2 = "nn_submission_like.csv"
dataset_id = "test"

if __name__ == '__main__':

    predictions = pd.read_csv(predictions_file, header=None, names=["tweet", "user", "prediction"])

    n_engagements = EngagerFeatureNumberOfPreviousPositiveEngagementRatio1(dataset_id).load_or_create()
    n_engagements.columns = [0]

    cold_mask = n_engagements[0] == -1
    hot_mask = ~cold_mask

    hot = predictions[hot_mask]
    cold = predictions[cold_mask]

    #print(hot)
    #print(cold)

    predictions_2 = pd.read_csv(predictions_file_2, header=None, names=["tweet", "user", "prediction"])

    hot_2 = predictions_2[hot_mask]
    cold_2 = predictions_2[cold_mask]

    #print(hot_2)
    #print(cold_2)

    #print("hot xgb :", hot.max(), hot.min(), hot.mean())
    #print("hot nn :", hot_2.max(), hot_2.min(), hot_2.mean())
    #print("cold xgb :", cold.max(), cold.min(), cold.mean())
    #print("cold nn :", cold_2.max(), cold_2.min(), cold_2.mean())

    hot['prediction'] = (hot['prediction']+hot_2['prediction'])/2.0

    predictions = pd.concat([hot, cold_2]).sort_index()

    #print(predictions)

    predictions.to_csv("xgb_nn_predictions_3.csv", header=False, index=False)