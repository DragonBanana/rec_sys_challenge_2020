import sys
import os.path
from Models.GBM.XGBoost import XGBoost
from Utils.Base.ParamRangeDict import DictDispenser


class ModelInterface(object):
    def __init__(self, model_name, kind):
        self.model_name = model_name
        self.kind = kind

    def blackBoxXGB(self, **param):
        #Initializing the model it it wasn't already
        model = XGBoost(**param, kind=self.kind)    
        
        #Without passing data it should fetch it automatically
        model.fit()

        prauc, rce = model.evaluate()
        return prauc


    def getParams(self):
        DD = DictDispenser()
        #Initializing the model if it wasn't already
        if self.model_name in "xgboost_classifier":
            param_dict = DD.xgb()

        if self.model_name in "lightgbm_classifier":
            param_dict = DD.lgb()

        if self.model_name in "catboost_classifier":
            param_dict = DD.cat()

        return param_dict

