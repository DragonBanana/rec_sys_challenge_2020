import sys
from Models import *


class ModelInterface(object):
    def __init__(self, model_name):
        self.model_name = model_name

        #CLASS VARIABLES
        #Is the class initialized
        self.is_init = False
        #Picked model
        self.model = None
    
    #This method declare and initialize the model
    def initModel(self):
        #Surely not elegant, but functional
        if self.model_name in "xgboost_classifier":
            model = XGBoost()

        if self.model_name in "lightgbm_classifier":
            model = LightGBM()

        if self.model_name in "catboost_classifier":
            model = CatBoost()

        #Choosing the declared model as active model
        self.model=model

        self.is_init = True

    def blackBox(self, param):
        #Initializing the model it it wasn't already
        if (not self.is_init):
            self.initModel()
        
        self.model.set_parameters(param) #Not working for catboost
        self.model.get_data()
        self.model.fit()
        score = self.model.evaluate()
        
        #Should figure out how PRAUC and RCE are combined
        #in order to optimize only a single metric       
        
        #evaluate should return the score
        #get_prediction the predictions

        
        return score


    def getParams(self):
        #Initializing the model if it wasn't already
        if (not self.is_init):
            self.initModel()

        return self.model.get_param_dict()
