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
from Utils.Base.RecommenderBase import RecommenderBase
from Utils.Base.RecommenderGBM import RecommenderGBM
from Utils.Eval.Metrics import ComputeMetrics as CoMe



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
                 num_rounds=1500,
                 batch=False,
                 param={'objective': 'binary',
                        'num_leaves': 31,
                        'learning_rate': 0.2,
                        'num_threads': 4,
                        'num_iterations': 10,
                        #Learning control parameters
                        'max_depth': 150, #Used to deal with overfitting when data is small.
                        'lambda_l1': 0.01,
                        'lambda_l2': 0.01,
                        'colsample_bynode': 0.5,
                        'colsample_bytree': 0.5,
                        'subsample': 0.7,
                        'pos_subsample': 0.5, #In classification positive and negative-
                        'neg_subsample': 0.5, #-subsample ratio.
                        'bagging_freq': 5, #Default 0 perform bagging every k iterations
                        'metric':    ('auc', 'binary_logloss')}): #Multiple metric allowed

        super(LightGBM, self).__init__(
                name="lightgbm_classifier", #name of the recommender
                kind=kind,                 #what does it recommends
                batch=batch)               #if it's a batch type


        self.param_dict={'objective': 'binary',
                        'metric':    ('auc', 'binary_logloss'),
                        'num_leaves': (15,500),
                        'learning_rate': (0.00001,0.1),
                        'num_iterations': (10,600),
                        'max_depth': (15,500),
                        'lambda_l1': (0.00001,0.1),
                        'lambda_l2': (0.00001,0.1),
                        'colsample_bynode': (0,1),
                        'colsample_bytree': (0,1),
                        'subsample': (0,1),
                        'pos_subsample': (0,1),
                        'neg_subsample': (0,1), 
                        'bagging_freq': (0,1)}

        # Inputs
        self.param=param
        self.kind=kind
        self.batch=batch
        self.num_rounds=num_rounds

        # Class variables
        #Models
        self.sround_model = None    #No need to differentiate, but it's
        self.batch_model = None     #way more readable.
        # Train
        self.X_train = None
        self.Y_train = None
        # Test
        self.X_test = None
        self.Y_test = None
        # List of categorical features
        # Must contain name of the column
        self.cat_feat = "auto" #auto is default value
        #Extension of saving file
        self.ext=".txt"

    

    def fit(self, X=None, Y=None, cat_feat=None):
        #------------------------------------------------
        #Let the loading of the dataset by
        #apposite method work.
        #------------------------------------------------
        #Tries to load X and Y if not directly passed        
        if (X is None) and (self.X_train is not None):
            X = self.X_train
        if (Y is None) and (self.Y_train is not None):
            Y = self.Y_train
        #If something miss error message gets displayied
        if ((X is None) or (Y is None)):
            print("Training set not provided.")
            return -1

        #If cat_feat is not null I take cat_feat as a new
        #categorical features list
        if (cat_feat is not None):
            cat_feat = self.cat_feat            
        #------------------------------------------------

        #Learning in a single round
        if self.batch is False:
            #Declaring LightGBM Dataset
            train = lgb.Dataset(data=X,
                                label=Y,
                                categorical_feature=self.cat_feat) 

            #Defining and fitting the model
            self.sround_model = lgb.train(self.param,  
                                          train,       
                                          self.num_rounds)

            
        #Learning by consecutive batches
        else:
            #Declaring LightGBM Dataset
            train = lgb.Dataset(data=X,
                                label=Y,
                                categorical_feature=self.cat_feat)

            #Defining and training the models
            self.batch_model = lgb.train(self.param,  
                                         train,      
                                         self.num_rounds,
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
    def evaluate(self, X_tst=None, Y_tst=None, cat_feat=None):
        Y_pred = None
        if ((X_tst is None) or (Y_tst is None))and(self.X_test is None)and(self.Y_test is None):
            print("No test set is provided.")
        elif (self.sround_model is None) and (self.batch_model is None):
            print("No model trained yet.")
        else:
            #Adapting with the cat_feat
            if (cat_feat is not None):
                self.cat_feat = cat_feat
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
            
            #Preparing LightGBM's Dataset ----------> ERROR: wants row data
            #test = lgb.Dataset(data=X_tst,
            #                   categorical_feature=self.cat_feat)

            #test = X_tst
            #Making predictions
            #Y_pred = model.predict(test)
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
    def get_prediction(self, X_tst=None, cat_feat=None):
        Y_pred = None
        if (X_tst is None) and (self.X_test is None):
            print("No test set is provided.")
        elif (self.sround_model is None) and (self.batch_model is None):
            print("No model trained yet.")
        else:
            if (cat_feat is not None):
                self.cat_feat=cat_feat
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
            
            #Preparing LightGBM's Dataset ----------> ERROR: wants row data
            #test = lgb.Dataset(data=X_tst,
            #                   categorical_feature=self.cat_feat)

            test = X_tst

            #Making predictions
            Y_pred = model.predict(test)
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
        ext = ".txt"
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
            self.sround_model = lgb.Booster(model_file=path)
            print("Model correctly loaded.\n")    

        else:
            #By loading in this way it is possible to keep on learning            
            self.batch_model = lgb.Booster(model_file=path)
            print("Batch model correctly loaded.\n")


    #-------------------------------
    #Setting the parameters
    #-------------------------------
    #param:     new parameters pack
    #-------------------------------
    #Unlike the 2019 model this one
    #declares lgb in train phase
    #so it's possible to set again
    #parameters. (maybe useful)
    #-------------------------------
    def set_parameters(self, param):
        self.param = param


    #Returns/prints the importance of the features
    #-------------------------------------------------
    #verbose:   it also prints the features importance
    #-------------------------------------------------
    # TODO: improve the verbosity, it is possible to 
    # put aside the name of the feature, if we have
    # a list with the feature names.
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
