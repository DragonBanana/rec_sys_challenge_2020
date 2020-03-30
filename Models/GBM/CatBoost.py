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

#TODO: The categorical features must be imported from load_data() method

class CatBoost(RecommenderGBM):
    def __init__(self,
                 kind="NO_NAME_GIVEN",
                 batch=False,
                 #Not in tuning dict
                 verbose=False,
                 loss_function="Logloss",
                 eval_metric="AUC",
                 #In tuning dict
                 iterations=10,
                 depth=10,
                 learning_rate = 0.1,
                 l2_leaf_reg = 0.01):

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
        self.sround_model = None
        self.batch_model = None
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

        

    #-----------------------------------------------------
    #                    fit(...)
    #-----------------------------------------------------
    #X:         Learning features of the dataset
    #Y:         Target feature of the dataset
    #-----------------------------------------------------
    # sround_model and batch_model are differentiated
    # in order to avoid overwriting. (Maybe not necessary)
    #-----------------------------------------------------
    def fit(self, X=None, Y=None):
        if (X is None) or (Y is None):
            X, Y = self.load_data(self.test_dataset)
            print("Test set loaded from file.")

        #Learning in a single round
        if self.batch is False:
            #Getting the CatBoost Pool matrix format
            train = Pool(data=X, 
                         label=Y,
                         cat_features=self.cat_feat)
     	
            #Defining and fitting the models
            self.sround_model = self.init_model()
            self.sround_model.fit(train)

        #Learning by consecutive batches
        else:
            #Getting the CatBoost Pool matrix format
            train = Pool(data=X, 
                         label=Y,
                         cat_features=self.cat_feat)

            #Declaring and training the model
            #Initializing the model only if it wasn't already
            if (self.batch_model is None):
                self.batch_model = self.init_model() 
                self.batch_model.fit(train)

            else:
                self.batch_model.fit(train, init_model=self.batch_model)


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
        if (X_tst is None) or (Y_tst is None):
            X_tst, Y_tst = self.load_data(self.test_dataset)
            print("Train set loaded from file.")
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
            #p_test = Pool(X_tst)
            #Making predictions
            #Y_pred = model.predict(p_test)
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
        if (X_tst is None):
            X_tst, _ = self.load_data(self.test_dataset)
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
            p_test = Pool(X_tst)

            #Making predictions
            '''
            #Commented part gives probability but 2 columns
            Y_pred = model.predict(p_test, prediction_type="Probability")
            Y_pred = Y_pred[:,1]
            '''
            Y_pred = model.predict(p_test)
            
            print(Y_pred.shape)
            return Y_pred


    #This method loads a model
    #-------------------------
    #path: path to the model
    #-------------------------
    def load_model(self, path):
        if (self.batch is False):
            #Reinitializing model
            self.sround_model = CatBoostClassifier()
            self.sround_model.load_model(path)
            print("Model correctly loaded.\n")    

        else:
            #By loading in this way it is possible to keep on learning            
            self.batch_model = CatBoostClassifier()
            self.batch_model.load_model(path)
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
        #Getting feature importance
        importance = model.get_feature_importance(verbose=verbose)
            
        return importance


    def init_model(self):
        return CatBoostClassifier(loss_function= self.loss_function,
                                  eval_metric= self.eval_metric,
                                  verbose= self.verbose,
                                  iterations= self.iterations,
                                  depth= self.depth,
                                  learning_rate= self.learning_rate,
                                  l2_leaf_reg= self.l2_leaf_reg)



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
