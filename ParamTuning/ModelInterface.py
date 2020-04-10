import sys
import os.path
from Models.GBM.XGBoost import XGBoost
from Utils.Base.ParamRangeDict import xgbRange
from Utils.Base.ParamRangeDict import xgbName
from Utils.Data.Data import get_dataset_xgb
import pandas as pd


class ModelInterface(object):
    def __init__(self, model_name, kind):
        self.model_name = model_name
        self.kind = kind
        #Datasets
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        
        
#------------------------------------------------------
#                   SINGLE TRAIN
#------------------------------------------------------
    #Score function for the XGBoost model
    def blackBoxXGB(self, param):
        #print(param)
        #Initializing the model it it wasn't already
        model = XGBoost(kind=self.kind,
                        batch = True,
                        #Not in tuning dict
                        objective="binary:logistic",
                        num_parallel_tree= 4,
                        eval_metric= "auc",
                        #In tuning dict
                        num_rounds = param[0],
                        max_depth = param[1],
                        min_child_weight = param[2],
                        colsample_bytree= param[3],
                        learning_rate= param[4],
                        reg_alpha= param[5],
                        reg_lambda= param[6],
                        #max_delta_step= param[7],
                        scale_pos_weight= param[7],
                        gamma= param[8],                        
                        subsample= param[9],
                        base_score= param[10])       
        
        
        #Training on custom set
        if (self.X_train is None) or (self.Y_train is None):
            #Without passing data it should fetch it automatically
            model.fit()
        else:
            model.fit(self.X_train, self.Y_train)


        #Evaluating on custom set
        if (self.X_test is None) or (self.Y_test is None):
            prauc, rce = model.evaluate()
        else:
            prauc, rce = model.evaluate(self.X_test, self.Y_test)
        
        #Returning the dumbly combined scores
        return self.metriComb(prauc, rce)


#------------------------------------------------------
#                   BATCH TRAIN
#------------------------------------------------------
#       Use the self.autoLoadTrain method
#         In order to load the dataset
#------------------------------------------------------
    #Score function for the XGBoost model
    def blackBoxXgbBatch(self, param):
        #print(param)
        #Initializing the model it it wasn't already
        model = XGBoost(kind=self.kind,
                        batch = True,
                        #Not in tuning dict
                        objective="binary:logistic",
                        num_parallel_tree= 4,
                        eval_metric= "auc",
                        #In tuning dict
                        num_rounds = param[0],
                        max_depth = param[1],
                        min_child_weight = param[2],
                        colsample_bytree= param[3],
                        learning_rate= param[4],
                        reg_alpha= param[5],
                        reg_lambda= param[6],
                        max_delta_step= param[7],
                        gamma= param[8],
                        #scale_pos_weight= param[7],                        
                        subsample= param[9],
                        base_score= param[10])
        
        for dataset in self.data_id:
            #Iteratively fetching the dataset
            X, Y = get_dataset_xgb(self.data_id, self.x_label_train, self.y_label_train)
            #Multistage model fitting
            model.fit(X, Y)

        #Evaluating on custom set
        if (self.X_test is None) or (self.Y_test is None):
            prauc, rce = model.evaluate()
        else:
            prauc, rce = model.evaluate(self.X_test, self.Y_test)
        
        #Returning the dumbly combined scores
        return self.metriComb(prauc, rce)


#-----------------------------------------
#       Future implementation
#-----------------------------------------
    #Score function for the lightGBM model
    def blackBoxLGB(self, param):
        #TODO: implement this
        return None

    
    #Score function for the CatBoost model
    def blackBoxCAT(self, param):
        #TODO: implement this
        return None
#------------------------------------------


#-----------------------------------------------------
#           Parameters informations
#-----------------------------------------------------
    #Returns the ordered parameter dictionary
    def getParams(self):
        #Returning an array containing the hyperparameters
        if self.model_name in "xgboost_classifier":
            param_dict = xgbRange()

        if self.model_name in "lightgbm_classifier":
            param_dict =  []

        if self.model_name in "catboost_classifier":
            param_dict = []

        return param_dict


    #Returns the ordered names parameter dictionary
    def getParamNames(self):
        #Returning the names of the hyperparameters
        if self.model_name in "xgboost_classifier":
            names_dict = xgbName()

        if self.model_name in "lightgbm_classifier":
            names_dict =  []

        if self.model_name in "catboost_classifier":
            names_dict = []

        return names_dict
#-----------------------------------------------------


#--------------------------------------------------------------
#               Return the method to optimize
#--------------------------------------------------------------
    #This method returns the score function based on model name
    def getScoreFunc(self):
        if self.model_name in "xgboost_classifier":
            score_func = self.blackBoxXGB
        
        if self.model_name in "lightgbm_classifier":
            score_func = self.blackBoxLGB

        if self.model_name in "catboost_classifier":
            score_func = self.blackBoxCAT

        return score_func
#---------------------------------------------------------------


#-------------------------------------------------   
#           Combine the two metrics
#-------------------------------------------------
    #Returns a combination of the two metrics
    def metriComb(self, prauc, rce):
        #Superdumb combination
        return -rce
#-------------------------------------------------


#-------------------------------------------------
#           Load dataset methods
#-------------------------------------------------
    #Loads a custom train set
    def loadTrainData(self, X_train, Y_train):
        self.X_train=X_train
        self.Y_train=Y_train

    
    #Loads a custom data set
    def loadTestData(self, X_test, Y_test):
        self.X_test=X_test
        self.Y_test=Y_test
#--------------------------------------------------


#--------------------------------------------------
#               Autofetch methods
#--------------------------------------------------
    #Train
    def autoLoadTrain(self, ids, x_label, y_label):
        #Defining the datasets' ids to work with
        self.train_id = ids
        #Define the labels to import
        self.x_label_train = x_label
        self.y_label_train = y_label


    #Test
    def autoLoadTest(self):
        None
#--------------------------------------------------
