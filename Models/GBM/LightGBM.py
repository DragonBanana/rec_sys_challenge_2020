import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, log_loss
import time
import pickle
import os.path
import datetime as dt
import sys
from Utils.Base.RecommenderGBM import RecommenderGBM
from Utils.Eval.Metrics import ComputeMetrics as CoMe

#TODO: The categorical features must be imported from load_data() method

class LightGBM(RecommenderGBM):
    #---------------------------------------------------------------------------------------------------
    #n_rounds:      Number of rounds for boosting
    #param:         Parameters of the XGB model
    #kind:          Name of the kind of prediction to print [LIKE, REPLY, REWTEET, RETWEET WITH COMMENT]
    #---------------------------------------------------------------------------------------------------
    #Not all the parameters are explicitated
    #PARAMETERS DOCUMENTATION:  https://lightgbm.readthedocs.io/en/latest/Parameters.html
    #---------------------------------------------------------------------------------------------------
    def __init__(self,
                 kind="NO_NAME_GIVEN",
                 batch=False,
                 #Not in tuning dict
                 objective= 'binary',
                 num_threads= 4,
                 metric= ('auc', 'binary_logloss'),
                 #In tuning dict
                 num_rounds=15,
                 num_leaves= 31,
                 learning_rate= 0.2,
                 max_depth= 150, 
                 lambda_l1= 0.01,
                 lambda_l2= 0.01,
                 colsample_bynode= 0.5,
                 colsample_bytree= 0.5,
                 subsample= 0.7,
                 pos_subsample= 0.5, #In classification positive and negative-
                 neg_subsample= 0.5, #Subsample ratio.
                 bagging_freq= 5):   #Default 0 perform bagging every k iterations
                 

        super(LightGBM, self).__init__(
                name="lightgbm_classifier", #name of the recommender
                kind=kind,                  #what does it recommends
                batch=batch)                #if it's a batch type

        
        #PARAMETERS' RANGE DICTIONARY
        self.param_range_dict={'num_rounds': (5,1000),
                               'num_leaves': (15,500),
                               'learning_rate': (0.00001,0.1),
                               'max_depth': (15,500),
                               'lambda_l1': (0.00001,0.1),
                               'lambda_l2': (0.00001,0.1),
                               'colsample_bynode': (0,1),
                               'colsample_bytree': (0,1),
                               'subsample': (0,1),
                               'pos_subsample': (0,1),
                               'neg_subsample': (0,1), 
                               'bagging_freq': (0,1)}

        #INPUTS
        self.kind=kind
        self.batch=batch
        #Parameters
        self.objective= objective
        self.num_rounds=num_rounds
        self.num_leaves= num_leaves
        self.learning_rate= learning_rate
        self.num_threads= num_threads
        self.max_depth= max_depth 
        self.lambda_l1= lambda_l1
        self.lambda_l2= lambda_l2
        self.colsample_bynode= colsample_bynode
        self.colsample_bytree= colsample_bytree
        self.subsample= subsample
        self.pos_subsample= pos_subsample       #In classification positive and negative-
        self.neg_subsample= neg_subsample       #-subsample ratio.
        self.bagging_freq= bagging_freq         #Default 0 perform bagging every k iterations
        self.metric=metric

        #CLASS VARIABLES
        #Models
        self.sround_model = None    #No need to differentiate, but it's
        self.batch_model = None     #way more readable.
        # List of categorical features
        self.cat_feat = "auto" 
        #Extension of saving file
        self.ext=".txt"

    

    def fit(self, X=None, Y=None, cat_feat=None):
        #Tries to load X and Y if not directly passed        
        if (X is None) or (Y is None):
            X, Y = self.load_data(self.test_dataset)
            print("Train set loaded from file.")

        #If categorical features are null taking default value
        if (cat_feat is None):
            cat_feat = self.cat_feat            

        #Learning in a single round
        if self.batch is False:
            #Declaring LightGBM Dataset
            train = lgb.Dataset(data=X,
                                label=Y,
                                categorical_feature=self.cat_feat) 

            #Defining and fitting the model
            self.sround_model = lgb.train(self.param,  
                                          train_set=train,       
                                          num_boost_rounds=self.num_rounds)

            
        #Learning by consecutive batches
        else:
            #Declaring LightGBM Dataset
            train = lgb.Dataset(data=X,
                                label=Y,
                                categorical_feature=self.cat_feat)

            #Defining and training the models
            self.batch_model = lgb.train(self.param,  
                                         train_set=train,      
                                         num_boost_rounds=self.num_rounds,
                                         init_model=self.batch_model)


    # Returns the predictions and evaluates them
    #---------------------------------------------------------------------------
    #                           evaluate(...)
    #---------------------------------------------------------------------------
    #X_tst:     Features of the test set
    #Y_tst      Ground truth, target of the test set
    #---------------------------------------------------------------------------
    #           Works for both for batch and single training
    #---------------------------------------------------------------------------
    def evaluate(self, X_tst=None, Y_tst=None):
        Y_pred = None
        #Tries to load X and Y if not directly passed        
        if (X_tst is None) or (Y_tst is None):
            X_tst, Y_tst = self.load_data(self.test_dataset)
            print("Test set loaded from file.")
        if (self.sround_model is None) and (self.batch_model is None):
            print("No model trained yet.")
        else:
            #Selecting the coherent model for the evaluation
            if self.batch is False:
                model = self.sround_model
            else:
                model = self.batch_model            

            #Making predictions
            #Error when it gets LightGBM's Dataset
            Y_pred = self.get_prediction(X_tst)

            # Declaring the class containing the
            # metrics.
            cm = CoMe(Y_pred, Y_tst)

            #Evaluating
            prauc = cm.compute_prauc()
            rce = cm.compute_rce()
            print("PRAUC "+self.kind+": {0}".format(prauc))
            print("RCE "+self.kind+": {0}".format(rce))
            print("MAX: {0}".format(max(Y_pred)))
            print("MIN: {0}\n".format(min(Y_pred)))
        return prauc, rce


    # This method returns only the predictions
    #-------------------------------------------
    #           get_predictions(...)
    #-------------------------------------------
    #X_tst:     Features of the test set
    #-------------------------------------------
    # As above, but without computing the scores
    #-------------------------------------------
    def get_prediction(self, X_tst=None):
        Y_pred = None
        #Tries to load X and Y if not directly passed        
        if (X_tst is None) or (Y_tst is None):
            X_tst, Y_tst = self.load_data(self.test_dataset)
            print("Test set loaded from file.")
        if (self.sround_model is None) and (self.batch_model is None):
            print("No model trained yet.")
        else:
            #Selecting the coherent model for the evaluation
            #According to the initial declaration (batch/single round)
            if self.batch is False:
                model = self.sround_model
            else:
                model = self.batch_model

            #Making predictions
            #Wants row data for prediction
            Y_pred = model.predict(X_tst)
            return Y_pred


    #This method loads a model
    #-------------------------
    #path: path to the model
    #-------------------------
    def load_model(self, path):
        if (self.batch is False):
            #Reinitializing model
            self.sround_model = lgb.Booster(model_file=path)
            print("Model correctly loaded.\n")    

        else:
            #By loading in this way it is possible to keep on learning            
            self.batch_model = lgb.Booster(model_file=path)
            print("Batch model correctly loaded.\n")


    #Returns/prints the importance of the features
    #-------------------------------------------------
    #verbose:   it also prints the features importance
    #-------------------------------------------------
    def get_feat_importance(self, verbose = False):
        
        if (self.batch is False):
            model = self.sround_model
        else:
            model = self.batch_model
        
        #Getting the importance
        importance = model.feature_importance(importance_type="gain")

        if verbose is True:
            print("F_pos\tF_importance")
            for k in range(len(importance)):
                print("{0}:\t{1}".format(k,importance[k]))            
            
        return importance

    
    #Returns the parameters in dictionary form
    def get_param_dict(self):
        
        param_dict={'objective': self.objective,
                    'num_leaves': self.num_leaves,
                    'learning_rate': self.learning_rate,
                    'num_threads': self.num_threads,
                    'num_iterations': self.num_iterations,
                    'max_depth': self.max_depth, 
                    'lambda_l1': self.lambda_l1,
                    'lambda_l2': self.lambda_l2,
                    'colsample_bynode': self.colsample_bynode,
                    'colsample_bytree': self.colsample_bytree,
                    'subsample': self.subsample,
                    'pos_subsample': self.pos_subsample,
                    'neg_subsample': self.neg_subsample,
                    'bagging_freq': self.bagging_freq,
                    'metric': self.metric}

        return param_dict


    #This method loads the dataset from file
    #-----------------------------------------------
    #dataset:   defines if will be load the train or
    #           test set, should be equal to either:
    #
    #           self.train_dataset
    #           self.test_dataset
    #-----------------------------------------------
    # TODO: has to be extended if we want to use it
    # too for LightGBM and CatBoost
    #-----------------------------------------------
    def load_data(self, dataset):
        #X, Y = Data.get_dataset_xgb(train_dataset, X_label, Y_label)
        #X, Y, self.cat_feat = Data.get_dataset_xgb(...)
        X = None
        Y = None
        return X, Y
