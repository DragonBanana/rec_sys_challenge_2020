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
from Utils.Base.RecommenderBase import RecommenderBase
from Utils.Base.RecommenderGBM import RecommenderGBM
from Utils.Eval.Metrics import ComputeMetrics as CoMe


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
                 num_rounds = 10, 
                 param = {'objective': 'binary:logistic', #outputs the binary classification probability
                          'colsample_bytree': 0.5,
                          'learning_rate': 0.4,
                          'max_depth': 35, #Max depth per tree
                          'alpha': 0.01, #L1 regularization
                          'lambda': 0.01, #L2 regularization
                          'num_parallel_tree': 4, #Number of parallel trees
                          'min_child_weight' : 1,#Minimum sum of instance weight (hessian) needed in a child.
                          'scale_pos_weight' : 1.2,                        
                          'subsample': 0.8},
                 kind = "NO_KIND_GIVEN",
                 batch = False):

        super(XGBoost, self).__init__(
            name="xgboost_classifier", #name of the recommender
            kind=kind,                 #what does it recommends
            batch=batch)               #if it's a batch type

        
        #Inputs
        self.num_rounds = num_rounds
        self.param = param
        self.kind = kind
        self.batch = batch  #The type of learning is now explicitated when you declare the model

        #Dictionary contaning the range of parameters
        #Should be here or in an apposit dicionary file DUNNO
        self.param_dict = {'colsample_bytree': (0,1),
                          'learning_rate': (0.5,0.00001),
                          'max_depth': (5,100),
                          'alpha': (0.01, 0.0001),
                          'lambda': (0.01,0.0001), 
                          'min_child_weight' : (1, 10),
                          'scale_pos_weight' : (1, 1.5),
                          'subsample': (0.6, 1)}

        #CLASS VARIABLES
        #Model
        self.sround_model = None
        self.batch_model = None
        #Prediction
        self.Y_pred = None
        #Train set
        self.X_train = None
        self.Y_train = None
        #Test set
        self.X_test = None
        self.Y_test = None
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
    def fit(self, X=None, Y=None):
        #--------------------------------------
        #Let the loading of the dataset by
        #apposite method work.
        #--------------------------------------
        #Tries to load X and Y if not directly passed        
        if (X is None) and (self.X_train is not None):
            X = self.X_train
        if (Y is None) and (self.Y_train is not None):
            Y = self.Y_train
        #If something miss error message gets displayied
        if ((X is None) or (Y is None)):
            print("Training set not provided.")
            return -1
        #--------------------------------------

        #Learning in a single round
        if self.batch is False:
            #Transforming matrices in DMatrix type
            train = xgb.DMatrix(X, 
                                label=Y)
     	
            #Defining and fitting the models
            self.sround_model = xgb.train(self.param,  #Don't care if overwritten by a new
                                   train,              #run it's single round training
                                   self.num_rounds)
            
        #Learning by consecutive batches
        else:
            #Transforming matrices in DMatrix type
            train = xgb.DMatrix(X, 
                                label=Y)
            #Defining and training the models
            self.batch_model = xgb.train(self.param, #Overwrites the old model with an incremented 
                                         train,      #new one, so new run won't cause trouble
                                         self.num_rounds,
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
        if ((X_tst is None) or (Y_tst is None))and(self.X_test is None)and(self.Y_test is None):
            print("No test set is provided.")
        elif (self.sround_model is None) and (self.batch_model is None):
            print("No model trained yet.")
        else:
            #-----------------------
            #Making work the not yet
            #implmented method for
            #importing the dataset
            #-----------------------
            if (X_tst is None):
                X_tst = self.X_test
            if (Y_tst is None):
                Y_tst = self.Y_test
            #-----------------------
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
            print("PRAUC "+self.kind+": {0}".format(prauc))
            print("RCE "+self.kind+": {0}".format(rce))
            print("MAX: {0}".format(max(Y_pred)))
            print("MIN: {0}\n".format(min(Y_pred)))
        return Y_pred

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
        if (X_tst is None) and (self.X_test is None):
            print("No test set is provided.")
        elif (self.sround_model is None) and (self.batch_model is None):
            print("No model trained yet.")
        else:
            #----------------------
            #Make the loading dataset
            #method work.
            #----------------------
            if (X_tst is None):
                X_tst = self.X_test
            #----------------------
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

    '''
    #This method saves the model
    #------------------------------------------------------
    #path:      path where to save the model
    #filename:  name of the file to save
    #------------------------------------------------------
    #By calling this function the models gets saved anyway
    #------------------------------------------------------
    def save_model(self, path=None, filename=None):
        #Defining the extension
        ext = ".model"
        #Saving the model with premade name in working folder
        if (path is None) and (filename is None):
            date = str(dt.datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
            if (self.batch is False):
                model_name = date+"_sround"+ext
                self.sround_model.save_model(model_name)
            else:
                model_name = date+"_batch"+ext
                self.batch_model.save_model(model_name)
            print("Model {0} saved successfully in working fodler.".format(model_name))

        #Saving model with given name in working folder
        elif (path is None) and (filename is not None):
            model_name = filename+ext
            if (self.batch is False):
                self.sround_model.save_model(model_name)
            else:
                self.batch_model.save_model(model_name)
            print("Model {0} saved successfully in working fodler.".format(model_name))

        #Saving model with given path but no name
        elif (path is not None) and (filename is None):
            date = str(dt.datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
            if (self.batch is False):
                model_name = path+"/"+date+"_sround"+ext
                self.sround_model.save_model(model_name)
            else:
                model_name = path+"/"+date+"_sround"+ext
                self.batch_model.save_model(model_name)
            print("Model {0} saved successfully.".format(model_name))
        
        #Save with given path and filename
        else:
            model_name = path+"/"+filename+ext
            print(model_name)
            if (self.batch is False):
                self.sround_model.save_model(model_name)
            else:
                self.batch_model.save_model(model_name)
            print("Model {0} saved successfully.".format(model_name))
            
        return model_name
    '''

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
    
    #-------------------------------
    #Setting the parameters
    #-------------------------------
    #param:     new parameters pack
    #-------------------------------
    #Unlike the 2019 model this one
    #declares xgb in train phase
    #so it's possible to set again
    #parameters. (maybe useful)
    #-------------------------------
    def set_parameters(self, param):
        self.param = param
    
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


#-----------------------FUTURE IMPLEMENTATION--------------------------------------------------------------
    '''
    #NOTA: PRIMA DI IMPLEMENTARE E' NECESSARIO CREARE UNA CLASSE DATA
    #CON LA TOPOLOGIA DEI DATASET SALVATI.
    def load_train_set():
        #Here declaring an object
        magical_retrieving_object = magical_retrieving_class()
        #Using some voodoo to invoke data BULA BULA
        pandas_data = magical_retrieving_object.retrieve(train, batch) #batch tells which rows are to fetch
        #Perform the split of the data
        self.X_train = pandas_data.drop("label")
        self.Y_train = pandas_data["label"]
        

    #NOTA: PRIMA DI IMPLEMENTARE E' NECESSARIO CREARE UNA CLASSE DATA
    #CONSIDERANDO LA TOPOLOGIA DEI DATASET SALAVATI.
    def load_test_set():
        #Here declaring an object
        magical_retrieving_object = magical_retrieving_class()
        #Using some voodoo to invoke data BULA BULA
        pandas_data = magical_retrieving_object.retrieve(test, batch) #batch tells which rows are to fetch
        #Perform the split of the data
        self.X_test = pandas_data.drop("label")
        self.Y_test = pandas_data["label"]
    '''
#---------------------------------------------------------------------------------------------------------


