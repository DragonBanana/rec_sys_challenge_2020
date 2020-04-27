from catboost import CatBoostClassifier, Pool
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
import catboost

#TODO: The categorical features must be imported from load_data() method

class CatBoost(RecommenderGBM):
    def __init__(self,
                 kind="NO_NAME_GIVEN",
                 batch=False,
                 #Not in tuning dict
                 verbose=True,
                 loss_function="Logloss",
                 eval_metric="AUC",
                 #In tuning dict
                 iterations=20,
                 depth=16,
                 learning_rate = 0.1,
                 l2_leaf_reg = 0.01,
                 bootstrap_type = "Bernoulli",
                 subsample = 0.8,
                 max_leaves = 31,
                 min_data_in_leaf = 1,
                 leaf_estimation_method = "Newton",
                 leaf_estimation_iterations= 10,
                 scale_pos_weight = 1,
                 model_shrink_mode = "Constant",
                 model_shrink_rate = 0.5,
                 random_strenght = 0.5,
                 colsample_bylevel = 0.5,
                 early_stopping_rounds = 10,
                 od_type = "Iter"):

        super(CatBoost, self).__init__(
              name="catboost_classifier",
              kind=kind,
              batch=batch)

        #Inputs
        #self.param=param
        self.kind=kind
        self.batch=batch

        #TODO: Dictionary containing pamarameters' range
        self.param_dict = None

        #CLASS VARIABLES
        #Model
        self.model = None
        #Prediction
        self.Y_pred = None
        #Categorical features
        self.cat_feat = None #Default value --> No categorical features
        #Save extension
        self.ext=".cbm"
        #Cannot pass parameters as a dict
        #Explicitating parameters (set to default)
        self.loss_function=loss_function
        self.eval_metric=eval_metric
        self.verbose=verbose
        self.iterations=iterations
        self.depth=depth
        self.learning_rate=learning_rate
        self.l2_leaf_reg=l2_leaf_reg
        self.bootstrap_type = bootstrap_type
        self.subsample = subsample
        self.max_leaves = max_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.leaf_estimation_method = leaf_estimation_method
        self.leaf_estimation_iterations = leaf_estimation_iterations
        self.scale_pos_weight = scale_pos_weight
        self.model_shrink_mode = model_shrink_mode
        self.model_shrink_rate = model_shrink_rate
        self.random_strenght = random_strenght
        self.colsample_bylevel = colsample_bylevel
        # ES parameters
        self.early_stopping_rounds = early_stopping_rounds
        self.od_type = od_type


    def init_model(self):
        return CatBoostClassifier(loss_function= self.loss_function,
                                  eval_metric= self.eval_metric,
                                  verbose= self.verbose,
                                  iterations= self.iterations,
                                  depth= self.depth,
                                  learning_rate= self.learning_rate,
                                  l2_leaf_reg= self.l2_leaf_reg,
                                  bootstrap_type=self.bootstrap_type,
                                  subsample=self.subsample,
                                  max_leaves=self.max_leaves,
                                  min_data_in_leaf=self.min_data_in_leaf,
                                  leaf_estimation_method=self.leaf_estimation_method,
                                  leaf_estimation_iterations=self.leaf_estimation_iterations,
                                  scale_pos_weight=self.scale_pos_weight,
                                  model_shrink_mode=self.model_shrink_mode,
                                  model_shrink_rate=self.model_shrink_rate,
                                  random_strength=self.random_strenght,
                                  colsample_bylevel=self.colsample_bylevel,
                                  od_wait=self.early_stopping_rounds,       # ES set here
                                  od_type=self.od_type)                     # ES set here
        

    #-----------------------------------------------------
    #                    fit(...)
    #-----------------------------------------------------
    #X:         Learning features of the dataset
    #Y:         Target feature of the dataset
    #-----------------------------------------------------
    # sround_model and batch_model are differentiated
    # in order to avoid overwriting. (Maybe not necessary)
    #-----------------------------------------------------
    def fit(self, pool_train = None, pool_val = None):

        # In case validation set is not provided set early stopping rounds to default
        if (pool_val is None):
            self.early_stopping_rounds = None
            self.od_type = None


        if self.model is not None:
            self.model.fit(pool_train,
                           eval_set=pool_val, 
                           init_model=self.model)

        else:
            #Defining and fitting the models
            self.model = self.init_model()
            self.model.fit(pool_train,
                           eval_set=pool_val)


       


    # Returns the predictions and evaluates them
    #---------------------------------------------------------------------------
    #                           evaluate(...)
    #---------------------------------------------------------------------------
    #X_tst:     Features of the test set
    #Y_tst      Ground truth, target of the test set
    #---------------------------------------------------------------------------
    #           Works for both for batch and single training
    #---------------------------------------------------------------------------
    def evaluate(self, pool_tst=None):
        if (pool_tst is None):
            print("No dataset provided.")    
        if (self.model is None):
            print("No model trained yet.")
        else:            
            #Preparing DMatrix
            #p_test = Pool(X_tst, label=Y_tst)
            #Making predictions
            #Y_pred = model.predict_proba(p_test)
            Y_pred = self.get_prediction(pool_tst)

            # Declaring the class containing the
            # metrics.
            cm = CoMe(Y_pred, pool_tst.get_label())

            # Evaluating
            prauc = cm.compute_prauc()
            rce = cm.compute_rce()
            # Confusion matrix
            conf = cm.confMatrix()
            # Prediction stats
            max_pred, min_pred, avg = cm.computeStatistics()

            return prauc, rce, conf, max_pred, min_pred, avg

    
    # This method returns only the predictions
    #-------------------------------------------
    #           get_predictions(...)
    #-------------------------------------------
    # pool_tst:     Features of the test set
    #-------------------------------------------
    # As above, but without computing the scores
    #-------------------------------------------
    def get_prediction(self, pool_tst=None):
        Y_pred = None
        #Tries to load X and Y if not directly passed        
        if (pool_tst is None):
            print("No dataset provided.")
        if (self.model is None):
            print("No model trained yet.")
        else:
            #Preparing DMatrix
            #p_test = Pool(X_tst)

            #Making predictions            
            #Commented part gives probability but 2 columns
            # First column probability to be 0
            # Second column probability to be 1
            Y_pred = self.model.predict_proba(pool_tst)
            Y_pred = Y_pred[:,1]
            return Y_pred


    #This method loads a model
    #-------------------------
    # path: path to the model
    #-------------------------
    def load_model(self, path):      
        self.model = CatBoostClassifier()
        self.model.load_model(path)
        print("Model correctly loaded.\n")
            
    
    # Returns/prints the importance of the features
    #-------------------------------------------------
    # verbose:   it also prints the features importance
    #-------------------------------------------------
    def get_feat_importance(self, verbose = False):
        
        #Getting feature importance
        importance = self.model.get_feature_importance(verbose=verbose)
        #for fstr_type parameter assign it something like = catboost.EFStrType.SharpValues
            
        return importance


    #-----------------------------------------------------
    #        Get the best iteration with ES
    #-----------------------------------------------------
    def getBestIter(self):
        return self.model.best_iteration_






















































#WHAT DOES THE CAT SAY?[semicit.]
#Miao
#Meaw
#Nyan
#Muwaa'
#Meo
#Meong
#Meu
#Miaou
#Miau
#Miauw
#Miaow
#Miyav
#Miav
#Mjau
#Miyau
#Mao
#Meogre
#Ngiiyaw