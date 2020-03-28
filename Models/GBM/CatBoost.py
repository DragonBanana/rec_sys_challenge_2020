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
sys.path.append("../../Utils/Base")
from RecommenderBase import RecommenderBase
sys.path.append("../../Utils/Eval")
from Metrics import ComputeMetrics as CoMe
#---------------------
#DA FINIRE MA FUNZIONA
#---------------------
class CatBoost(RecommenderBase):
    def __init__(self,
                 kind="NO_NAME_GIVEN",
                 batch=False,
                 param={"iterations": 2,    #NOT SURE BOUT THIS
                         "depth":2,
                         "learning_rate": 0.1,
                         "loss_function": "Logloss",
                         "verbose": True}):

        super(CatBoost, self).__init__(
              name="catboost_classifier",
              kind=kind,
              batch=batch)

        #Inputs
        self.param=param
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
        #Train set
        self.X_train = None
        self.Y_train = None
        #Test set
        self.X_test = None
        self.Y_test = None
        #Categorical features
        self.cat_feat = None #Default value --> No categorical features



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
            #Getting the CatBoost Pool matrix format
            train = Pool(data=X, 
                         label=Y,
                         cat_features=self.cat_feat)
     	
            #Defining and fitting the models
            self.sround_model = CatBoostClassifier(iterations=2, depth=2, learning_rate=0.1, loss_function="Logloss", verbose=True)
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
                self.batch_model = CatBoostClassifier(num_trees=10, depth=10, learning_rate=0.1, loss_function="Logloss", verbose=True)
                #Fitting the model 
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
            p_test = Pool(X_tst)

            #Making predictions
            Y_pred = model.predict(p_test)

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
            p_test = Pool(X_tst)

            #Making predictions
            Y_pred = model.predict(p_test)
            return Y_pred




    #This method saves the model
    #------------------------------------------------------
    #path:      path where to save the model
    #filename:  name of the file to save
    #------------------------------------------------------
    #By calling this function the models gets saved anyway
    #------------------------------------------------------
    def save_model(self, path=None, filename=None):
        #Defining the extension
        ext = ".cbm"
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

    #-------------------------------
    #Setting the parameters
    #-------------------------------
    #param:     new parameters pack
    #-------------------------------
    #Unlike the 2019 model this one
    #declares catboost in train phase
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
            model = self.sround_model
        else:
            model = self.batch_model
        #Getting feature importance
        importance = model.get_feature_importance(verbose=verbose)
            
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
#Miao
#Meaw
#Nyan
