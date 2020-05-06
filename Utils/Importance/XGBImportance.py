import xgboost as xgb

from Models.GBM.XGBoost import XGBoost
from Utils.Eval.Metrics import ComputeMetrics


class XGBImportance:

    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def load_model(self, path):
        model = XGBoost()
        model.load_model(path)
        return model

    def fit(self, *params):
        print("FIT PARAMS")
        print(params)

    def score(self, X_test, Y_test):
        predictions = self.model.get_prediction(dmat_test=xgb.DMatrix(X_test))
        cm = ComputeMetrics(predictions, Y_test.to_numpy())

        # Evaluating
        prauc = cm.compute_prauc()
        rce = cm.compute_rce()

        print(rce)
        return rce
