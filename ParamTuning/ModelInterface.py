import sys
import os.path
from Models.GBM.XGBoost import XGBoost
from Utils.Base.ParamRangeDict import xgbRange
from Utils.Base.ParamRangeDict import xgbName
from Utils.Data.Data import get_dataset_xgb
from Utils.Data.Data import get_dataset_xgb_batch
from Utils.Data.DataUtils import TRAIN_IDS,  VAL_IDS
import pandas as pd
import datetime as dt
import time
from tqdm import tqdm


class ModelInterface(object):
    def __init__(self, model_name, kind, mode):
        self.model_name = model_name
        self.kind = kind
        self.mode = mode
        # Datasets
        self.X_train = None
        self.Y_train = None
        self.X_val = None
        self.Y_val = None
        # LOGS PARAMS
        # Counter of iterations
        self.iter_count = 0
        # Filename for logs
        self.path = None
        # True make logs, false don't
        self.make_log = True
        
        
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
        if (self.X_val is None) or (self.Y_val is None):
            prauc, rce, confmat, max_pred, min_pred, avg = model.evaluate()
        else:
            prauc, rce, confmat, max_pred, min_pred, avg = model.evaluate(self.X_val, self.Y_val)

        #Make human readable logs here
        if self.make_log is True:
            self.saveLog(param, 
                         prauc, 
                         rce, 
                         confmat, 
                         max_pred, 
                         min_pred, 
                         avg)
        
        #Returning the dumbly combined scores
        return self.metriComb(prauc, rce)


#------------------------------------------------------
#                   BATCH TRAIN
#------------------------------------------------------
#       Use the self.batchLoadSets method
#         In order to load the datasets
#------------------------------------------------------
# Batch may be of different sizes, but considering how
# they're distributed giving them the same weight in
# averaging process shouldn't raise problems.
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
                        #max_delta_step= param[7],
                        scale_pos_weight= param[7],
                        gamma= param[8],                        
                        subsample= param[9],
                        base_score= param[10])

        #Batch train
        for split in tqdm(range(self.tot_train_split)):
            X, Y = get_dataset_xgb_batch(self.tot_train_split, 
                                         split, 
                                         self.train_id, 
                                         self.x_label, 
                                         self.y_label)
            start_time_training_data = time.time()
            #Multistage model fitting
            model.fit(X, Y)

        #TODO: last evaluation set may be smaller so it needs
        # to be weighted according to its dimension.

        #Initializing variables
        tot_prauc = 0
        tot_rce = 0
        tot_confmat = [[0,0],[0,0]]
        max_pred = 0 #Max set to the minimum
        min_pred = 1 #Min set to the maximum
        avg = 0
        #Batch evaluation
        for split in tqdm(range(self.tot_val_split)):
            #Iteratively fetching the dataset
            X, Y = get_dataset_xgb_batch(self.tot_val_split, 
                                         split, 
                                         self.val_id, 
                                         self.x_label, 
                                         self.y_label)
            #Multistage evaluation
            prauc, rce, confmat, max_tmp, min_tmp, avg_tmp= model.evaluate(X, Y)

            #Summing all the evaluations
            tot_prauc = tot_prauc + prauc
            tot_rce = tot_rce + rce
            
            #Computing some statistics for the log
            if self.make_log is True:
                # Getting maximum over iteration
                if max_tmp > max_pred:
                    max_pred = max_tmp
                # Getting minimum over iteration
                if min_tmp < min_pred:
                    min_pred = min_tmp
                # Getting average over itaration
                avg += avg_tmp
                # Computing confusion matrix
                tot_confmat = tot_confmat + confmat           

        #Averaging the evaluations over # of validation splits
        tot_prauc = tot_prauc/self.tot_val_split
        tot_rce = tot_rce/self.tot_val_split
        avg = avg/self.tot_val_split

        #Make human readable logs here
        if self.make_log is True:
            self.saveLog(param, 
                         tot_prauc, 
                         tot_rce, 
                         tot_confmat, 
                         max_pred, 
                         min_pred, 
                         avg)
        
        #Returning the dumbly combined scores
        return self.metriComb(tot_prauc, tot_rce)


#------------------------------------------------------
#        NESTED CROSS VALIDATION TRAIN
#------------------------------------------------------
#        Use the self.ncvLoadSets method
#         In order to load the datasets
#------------------------------------------------------
# Batch may be of different sizes, but considering how
# they're distributed giving them the same weight in
# averaging process shouldn't raise problems.
#------------------------------------------------------
    #Score function for the XGBoost model
    def blackBoxXgbNCV(self, param):
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

        #Iterable returns pair of train - val sets
        id_pairs = zip(TRAIN_IDS, VAL_IDS)

        #Initializing variables
        weight_factor = 0
        averaging_factor = 0
        tot_prauc = 0
        tot_rce = 0
        tot_confmat = [[0,0],[0,0]]
        max_pred = 0
        min_pred = 1
        avg = 0
        #Making iterative train-validations
        for dataset_ids in id_pairs:
            weight_factor += weight_factor+1
            averaging_factor += weight_factor
            #Fetching train set
            X, Y = get_dataset_xgb(dataset_ids[0], 
                                   self.x_label, 
                                   self.y_label)   
            #Multistage model fitting
            model.fit(X, Y)

            #Fetching val set 
            X, Y = get_dataset_xgb(dataset_ids[1], 
                                   self.x_label, 
                                   self.y_label)
            #Multistage evaluation
            prauc, rce, confmat, max_tmp, min_tmp, avg_tmp= model.evaluate(X, Y)

            #Weighting scores (based on how many days are in the train set)
            tot_prauc += prauc * weight_factor
            tot_rce += rce * weight_factor

            #Computing some statistics for the log
            if self.make_log is True:
                #Getting maximum over iteration
                if max_tmp > max_pred:
                    max_pred = max_tmp
                #Getting minimum over iteration
                if min_tmp < min_pred:
                    min_pred = min_tmp
                #Getting average over itaration
                avg += avg_tmp
                #Computing the confusion matrix
                tot_confmat = tot_confmat + confmat

        #Averaging scores
        tot_prauc /= averaging_factor
        tot_rce /= averaging_factor
        #Averaging average (lol)
        avg /= len(TRAIN_IDS)

        #Make human readable logs here
        if self.make_log is True:
            self.saveLog(param, 
                         tot_prauc, 
                         tot_rce, 
                         tot_confmat, 
                         max_pred, 
                         min_pred, 
                         avg)
        
        #Returning the dumbly combined scores
        return self.metriComb(tot_prauc, tot_rce)


#-----------------------------------------
#       TODO:Future implementation
#-----------------------------------------
    # Score function for the lightGBM model
    def blackBoxLGB(self, param):
        #TODO: implement this
        return None

    
    # Score function for the CatBoost model
    def blackBoxCAT(self, param):
        #TODO: implement this
        return None

    # Batch ones    
    # Score function for the lightGBM model
    def blackBoxLgbBatch(self, param):
        #TODO: implement this
        return None

    
    # Score function for the CatBoost model
    def blackBoxCatBatch(self, param):
        #TODO: implement this
        return None

    # NCV ones    
    # Score function for the lightGBM model
    def blackBoxLgbNCV(self, param):
        #TODO: implement this
        return None

    
    # Score function for the CatBoost model
    def blackBoxCatNCV(self, param):
        #TODO: implement this
        return None
#------------------------------------------


#-----------------------------------------------------
#           Parameters informations
#-----------------------------------------------------
    # Returns the ordered parameter dictionary
    def getParams(self):
        # Returning an array containing the hyperparameters
        if self.model_name in "xgboost_classifier":
            param_dict = xgbRange()

        if self.model_name in "lightgbm_classifier":
            param_dict =  []

        if self.model_name in "catboost_classifier":
            param_dict = []

        return param_dict


    # Returns the ordered names parameter dictionary
    def getParamNames(self):
        # Returning the names of the hyperparameters
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
    # This method returns the score function based on model name
    def getScoreFunc(self):
        if self.mode == 0:
            if self.model_name in "xgboost_classifier":
                score_func = self.blackBoxXGB
        
            if self.model_name in "lightgbm_classifier":
                score_func = self.blackBoxLGB

            if self.model_name in "catboost_classifier":
                score_func = self.blackBoxCAT
        elif self.mode == 1:
            if self.model_name in "xgboost_classifier":
                score_func = self.blackBoxXgbBatch

            if self.model_name in "lightgbm_classifier":
                score_func = self.blackBoxLgbBatch

            if self.model_name in "catboost_classifier":
                score_func = self.blackBoxCatBatch
        else:
            if self.model_name in "xgboost_classifier":
                score_func = self.blackBoxXgbNCV

            if self.model_name in "lightgbm_classifier":
                score_func = self.blackBoxLgbNCV

            if self.model_name in "catboost_classifier":
                score_func = self.blackBoxCatNCV         

        return score_func
#---------------------------------------------------------------


#-------------------------------------------------   
#           Combine the two metrics
#-------------------------------------------------
    # Returns a combination of the two metrics
    def metriComb(self, prauc, rce):
        # Superdumb combination
        return -rce
#-------------------------------------------------


#-------------------------------------------------
#           Load dataset methods
#-------------------------------------------------
    # Loads a custom train set
    def loadTrainData(self, X_train, Y_train):
        self.X_train=X_train
        self.Y_train=Y_train

    
    # Loads a custom data set
    def loadValData(self, X_val, Y_val):
        self.X_val=X_val
        self.Y_val=Y_val
#--------------------------------------------------


#--------------------------------------------------
#           Batch/NCV methods
#--------------------------------------------------
    '''
    #SPLITTED IN 3 IN ORDER TO USE setLabels WITH NCV
    def batchLoadSets(self, tot_train_split, tot_val_split, train_id, val_id, x_label, y_label):
        # Parts in which the dataset should be divided
        self.tot_train_split = tot_train_split
        self.tot_val_split = tot_val_split
        # Defining the datasets' ids to work with
        self.train_id = train_id
        self.val_id = val_id
        # Define the labels to import
        self.x_label = x_label
        self.y_label = y_label
    '''
    # Passing train set id and number of batches (batch only)
    def batchTrain(self, tot_train_split, train_id):
        self.tot_train_split = tot_train_split
        self.train_id = train_id
    # Passing val set id and number of batches (batch only)
    def batchVal(self, tot_val_split, val_id):
        self.tot_val_split = tot_val_split
        self.val_id = val_id
    # Setting labels to use in the sets (batch + NCV)
    def setLabels(self, x_label, y_label):
        self.x_label = x_label
        self.y_label = y_label
#--------------------------------------------------


#--------------------------------------------------
#            Save human readable logs
#--------------------------------------------------
    def saveLog(self, param, prauc, rce, confmat, max_arr, min_arr, avg):
        if self.path is None:
            #Taking the path provided
            self.path = str(dt.datetime.now().strftime("%m_%d_%H_%M_%S")) + ".log"
        #Get hyperparameter names
        p_names = self.getParamNames()
        #Maybe check len(p_names) == len(x) here

        #Opening a file and writing into it the logs
        with open(self.path, 'a') as log:
            to_write = "ITERATION NUMBER " + str(self.iter_count) + "\n"
            log.write(to_write)
            for i in range(len(p_names)):
                to_write=str(str(p_names[i])+"= "+str(param[i])+"\n")
                log.write(to_write)
            
            #Writing the log
            tn, fp, fn, tp = confmat.ravel()
            to_write = "-------\n"
            to_write += "PRAUC = "+str(prauc)+"\n"
            to_write += "RCE   = "+str(rce)+"\n"
            to_write += "-------\n"
            to_write += "TN    = "+str(tn)+"\n"
            to_write += "FP    = "+str(fp)+"\n"
            to_write += "FN    = "+str(fn)+"\n"
            to_write += "TP    = "+str(tp)+"\n"
            to_write += "-------\n"
            to_write += "MAX   ="+str(max_arr)+"\n"
            to_write += "MIN   ="+str(min_arr)+"\n"
            to_write += "AVG   ="+str(avg)+"\n\n\n"
            log.write(to_write)

        #Increasing the iteration count
        self.iter_count = self.iter_count + 1
#--------------------------------------------------


#--------------------------------------------------
#             Reset saving parameters
#--------------------------------------------------
# Stop appending to the previos log and make a new
# one.
#--------------------------------------------------
    def resetSaveLog(self):
        self.iter_count = 0
        self.path = None
#--------------------------------------------------


#--------------------------------------------------
#             Setting log saver
#--------------------------------------------------
    def setSaveLog(self, make_log):
        self.make_log=make_log
#--------------------------------------------------


#--------------------------------------------------
#            Setting logs path
#--------------------------------------------------
    def setLogPath(self, path):
        self.path = path
#--------------------------------------------------