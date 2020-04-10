import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, log_loss
import time
import pickle
import os.path
import datetime as dt
import sys
import math
from Utils.Base.RecommenderGBM import RecommenderGBM
from Utils.Eval.Metrics import ComputeMetrics as CoMe
from Utils.Data import Data
from Utils.Submission.Submission import create_submission_file


class XGBoost(RecommenderGBM):
    #---------------------------------------------------------------------------------------------------
    #n_rounds:      Number of rounds for boosting
    #param:         Parameters of the XGB model
    #kind:          Name of the kind of prediction to print [LIKE, REPLY, REWTEET, RETWEET WITH COMMENT]
    #---------------------------------------------------------------------------------------------------
    #Not all the parameters are explicitated
    #PARAMETERS DOCUMENTATION:https://xgboost.readthedocs.io/en/latest/parameter.html
    #---------------------------------------------------------------------------------------------------

    def __init__(self, 
                 kind = "NO_KIND_GIVEN",
                 batch = False,
                 #Not in tuning dict
                 objective="binary:logistic", #outputs the binary classification probability
                 num_parallel_tree= 4, #Number of parallel trees
                 eval_metric= "auc",    #WORKS ONLY IF A VALIDATION SET IS PASSED IN TRAINING PHASE
                 early_stopping_rounds= None,
                 #In tuning dict
                 num_rounds = 10,
                 colsample_bytree= 1,
                 learning_rate= 0.3,
                 max_depth= 6, #Max depth per tree
                 reg_alpha= 0, #L1 regularization
                 reg_lambda= 1, #L2 regularization
                 min_child_weight= 1,#Minimum sum of instance weight (hessian) needed in a child.
                 scale_pos_weight= 1,
                 gamma= 0,
                 #max_delta_step= 0,
                 base_score= 0.5,
                 subsample= 1):

        super(XGBoost, self).__init__(
            name="xgboost_classifier", #name of the recommender
            kind=kind,                 #what does it recommends
            batch=batch)               #if it's a batch type
        
        #INPUTS
        self.kind = kind
        self.batch = batch  #False: single round| True: batch learning
        #Parameters
        self.num_rounds=num_rounds
        self.objective=objective
        self.early_stopping_rounds=early_stopping_rounds
        self.eval_metric=eval_metric
        self.colsample_bytree=colsample_bytree
        self.learning_rate=learning_rate
        self.max_depth=max_depth
        self.reg_alpha=reg_alpha
        self.reg_lambda=reg_lambda
        self.num_parallel_tree=num_parallel_tree
        self.min_child_weight=min_child_weight
        self.scale_pos_weight=scale_pos_weight
        self.subsample=subsample
        self.gamma=gamma
        #self.max_delta_step=max_delta_step
        self.base_score=base_score

        #CLASS VARIABLES
        #Model
        self.sround_model = None
        self.batch_model = None
        #Prediction
        self.Y_pred = None
        #Extension of saved file
        self.ext=".model"

    
    #-----------------------------------------------------
    #                    fit(...)
    #-----------------------------------------------------
    #X:         Learning features of the dataset
    #Y:         Target feature of the dataset
    #batch:     Enable learning by batch
    #-----------------------------------------------------
    # sround_model and batch_model are differentiated
    # in order to avoid overwriting. (Maybe not necessary)
    #-----------------------------------------------------
    def fit(self, X=None, Y=None, X_valid=None, Y_valid=None):
        
        #Tries to load X and Y if not directly passed        
        if (X is None) or (Y is None):
            X, Y = Data.get_dataset_xgb_default_train()
            print("Train set loaded from file.")
        
        #In case validation set is not provided set early stopping rounds to default
        if (X_valid is None) or (Y_valid is None):
            self.early_stopping_rounds = None
            valid = []
        else:
            valid = xgb.DMatrix(X_valid,
                                label=Y_valid)

        #Learning in a single round
        if self.batch is False:
            #Transforming matrices in DMatrix type
            train = xgb.DMatrix(X, 
                                label=Y)     	    
            
            #Defining and fitting the models
            self.sround_model = xgb.train(self.get_param_dict(),
                                          early_stopping_rounds=self.early_stopping_rounds,
                                          evals=valid,
                                          dtrain=train,
                                          num_boost_round=math.ceil(self.num_rounds))
            
        #Learning by consecutive batches
        else:
            #Transforming matrices in DMatrix type
            train = xgb.DMatrix(X, 
                                label=Y)
            
            #Defining and training the model
            self.batch_model = xgb.train(self.get_param_dict(),
                                         early_stopping_rounds=self.early_stopping_rounds,
                                         evals=valid,
                                         dtrain=train,   
                                         xgb_model=self.batch_model)


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
            X_tst, Y_tst = Data.get_dataset_xgb_default_test()
            print("Test set loaded from file.")
        #Y_tst = np.array(Y_tst[Y_tst.columns[0]].astype(float))
        if (self.sround_model is None) and (self.batch_model is None):
            print("No model trained yet.")
        else:
            #Selecting the coherent model for the evaluation
            #According to the initial declaration (batch/single round)
            if self.batch is False:
                model = self.sround_model
            else:
                model = self.batch_model
            
            #Preparing DMatrix
            #d_test = xgb.DMatrix(X_tst)
            #Making predictions
            #Y_pred = model.predict(d_test)
            Y_pred = self.get_prediction(X_tst)

            # Declaring the class containing the
            # metrics.
            cm = CoMe(Y_pred, Y_tst)

            #Evaluating
            prauc = cm.compute_prauc()
            rce = cm.compute_rce()
            '''
            print("PRAUC "+self.kind+": {0}".format(prauc))
            print("RCE "+self.kind+": {0}".format(rce))
            print("MAX: {0}".format(max(Y_pred)))
            print("MIN: {0}".format(min(Y_pred)))
            '''
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
        if (X_tst is None):
            X_tst, _ = Data.get_dataset_xgb_default_test()
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
            
            #Preparing DMatrix
            d_test = xgb.DMatrix(X_tst)

            #Making predictions
            Y_pred = model.predict(d_test)
            return Y_pred


    #This method loads a model
    #-------------------------
    #path: path to the model
    #-------------------------
    def load_model(self, path):
        if (self.batch is False):
            #Reinitializing model
            self.sround_model = xgb.Booster()
            self.sround_model.load_model(path)
            print("Model correctly loaded.\n")    

        else:
            #By loading in this way it is possible to keep on learning            
            self.batch_model = xgb.Booster()
            self.batch_model.load_model(path)
            print("Batch model correctly loaded.\n")

    
    #Returns/prints the importance of the features
    #-------------------------------------------------
    #verbose:   it also prints the features importance
    #-------------------------------------------------
    def get_feat_importance(self, verbose = False):
        
        if (self.batch is False):
            importance = self.sround_model.get_score(importance_type='gain')
        else:
            importance = self.batch_model.get_score(importance_type='gain')

        if verbose is True:
            for k, v in importance.items():
                print("{0}:\t{1}".format(k,v))
            
        return importance


    #Returns parameters in dicrionary form
    def get_param_dict(self):
        param_dict = {'objective':self.objective,
                      'eval_metric':self.eval_metric,
                      'colsample_bytree':self.colsample_bytree,
                      'learning_rate':self.learning_rate,
                      'max_depth':math.ceil(self.max_depth),
                      'reg_alpha':self.reg_alpha,
                      'reg_lambda':self.reg_lambda,
                      'num_parallel_tree':self.num_parallel_tree,
                      'min_child_weight':self.min_child_weight,
                      'scale_pos_weight':self.scale_pos_weight,
                      'subsample':self.subsample,
                      'gamma':self.gamma,
                      #'max_delta_step':self.max_delta_step,
                      'base_score':self.base_score}
        
        return param_dict
