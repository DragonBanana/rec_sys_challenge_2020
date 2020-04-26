import functools
import sys
import os.path
from Models.GBM.XGBoost import XGBoost
from Models.GBM.LightGBM import LightGBM
from Utils.Eval.Metrics import ComputeMetrics as CoMe
from Utils.Base.ParamRangeDict import xgbRange
from Utils.Base.ParamRangeDict import xgbName
from Utils.Base.ParamRangeDict import lgbmRange
from Utils.Base.ParamRangeDict import lgbmName
from Utils.Data.Data import get_dataset_xgb
from Utils.Data.Data import get_dataset_xgb_batch
from Utils.Data.DataUtils import TRAIN_IDS,  TEST_IDS
import pandas as pd
import datetime as dt
import time
from tqdm import tqdm
import xgboost as xgb
import numpy as np
import multiprocessing as mp
import xgboost as xgb


class ModelInterface(object):
    def __init__(self, model_name, kind, mode):
        self.model_name = model_name
        self.kind = kind
        self.mode = mode
        # Datasets
        self.dmat_test = None
        self.dmat_train = None
        self.dmat_val = None
        #Batch datasets
        self.val_id = None
        #NCV early stopping param
        self.es_ncv = False
        # LOGS PARAMS
        # Counter of iterations
        self.iter_count = 1
        # Filename for logs
        self.path = None
        # True make logs, false don't
        self.make_log = True
        # Parameters to set
        self.verbosity="1"
        self.process_type="default"
        self.tree_method="auto"
        self.objective="binary:logistic"
        self.num_parallel_tree=4
        self.eval_metric="rmsle"
        self.early_stopping_rounds=5
        
        
#------------------------------------------------------
#                   SINGLE TRAIN
#------------------------------------------------------
    #Score function for the XGBoost model
    def blackBoxXGB(self, param):
        if self.make_log is True:
            self.saveParam(param)
        #Initializing the model it it wasn't already
        model = XGBoost(kind=self.kind,
                        #Not in tuning dict
                        verbosity=self.verbosity,
                        process_type=self.process_type,
                        tree_method=self.tree_method,
                        objective=self.objective,
                        num_parallel_tree=self.num_parallel_tree,
                        eval_metric=self.eval_metric,
                        early_stopping_rounds=self.early_stopping_rounds,
                        #In tuning dict
                        num_rounds = param[0],
                        max_depth = param[1],
                        min_child_weight = param[2],
                        colsample_bytree= param[3],
                        learning_rate= param[4],
                        reg_alpha= param[5],
                        reg_lambda= param[6],
                        scale_pos_weight= param[7],
                        gamma= param[8],                        
                        subsample= param[9],
                        base_score= param[10],
                        max_delta_step= param[11])       
        #Training on custom set
        if (self.dmat_train is None):
            print("No train set passed to the model.")
        else:
            #dmat_train = self.getDMat(self.X_train, self.Y_train) #------------------------------------- DMATRIX GENERATOR
            model.fit(self.dmat_train, self.dmat_val)            
            if self.dmat_val is not None:
                best_iter = model.getBestIter()
            else:
                best_iter = -1

        #Evaluating on custom set
        if (self.dmat_test is None):
            print("No test set provided.")
        else:
            #dmat_test = self.getDMat(self.X_test, self.Y_test) #------------------------------------- DMATRIX GENERATOR
            prauc, rce, confmat, max_pred, min_pred, avg = model.evaluate(self.dmat_test)

        del model
        #Make human readable logs here
        if self.make_log is True:
            self.saveRes(best_iter,
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
        #Saving parameters
        if self.make_log is True:
            self.saveParam(param)
        #Initializing the model it it wasn't already
        model = XGBoost(kind=self.kind,
                        #Not in tuning dict
                        verbosity=self.verbosity,
                        process_type=self.process_type,
                        tree_method=self.tree_method,
                        objective=self.objective,
                        num_parallel_tree=self.num_parallel_tree,
                        eval_metric=self.eval_metric,
                        early_stopping_rounds=self.early_stopping_rounds,
                        #In tuning dict
                        num_rounds = param[0],
                        max_depth = param[1],
                        min_child_weight = param[2],
                        colsample_bytree= param[3],
                        learning_rate= param[4],
                        reg_alpha= param[5],
                        reg_lambda= param[6],
                        scale_pos_weight= param[7],
                        gamma= param[8],                        
                        subsample= param[9],
                        base_score= param[10],
                        max_delta_step= param[11])

        best_iter = []
        #Batch train
        for split in tqdm(range(self.tot_train_split)):
            X, Y = get_dataset_xgb_batch(self.tot_train_split, 
                                         split, 
                                         self.train_id, 
                                         self.x_label, 
                                         self.y_label)
            #start_time_training_data = time.time()
            dmat_train = self.getDMat(X, Y) #------------------------------------- DMATRIX GENERATOR
            del X, Y

            if self.val_id is not None:
                X, Y = get_dataset_xgb_batch(self.tot_train_split, 
                                             split, 
                                             self.val_id, 
                                             self.x_label, 
                                             self.y_label)
                dmat_val = self.getDMat(X, Y)
                del X, Y
            else:
                dmat_val = None

            #Multistage model fitting
            model.fit(dmat_train, dmat_val)
            del dmat_train

            #Get best iteration obtained with es
            if dmat_val is not None:
                best_iter.append(model.getBestIter())
            else:
                best_iter = -1
            del dmat_val


        #Initializing variables
        tot_prauc = 0
        tot_rce = 0
        tot_confmat = [[0,0],[0,0]]
        max_pred = 0 #Max set to the minimum
        min_pred = 1 #Min set to the maximum
        avg = 0
        #Batch evaluation
        for split in tqdm(range(self.tot_test_split)):
            #Iteratively fetching the dataset
            X, Y = get_dataset_xgb_batch(self.tot_test_split, 
                                         split, 
                                         self.test_id, 
                                         self.x_label, 
                                         self.y_label)

            dmat_test = self.getDMat(X, Y) #------------------------------------- DMATRIX GENERATOR
            del X, Y
            #Multistage evaluation
            prauc, rce, confmat, max_tmp, min_tmp, avg_tmp= model.evaluate(dmat_test)
            del dmat_test

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
        del model          

        #Averaging the evaluations over # of validation splits
        tot_prauc = tot_prauc/self.tot_test_split
        tot_rce = tot_rce/self.tot_test_split
        avg = avg/self.tot_test_split

        #Make human readable logs here
        if self.make_log is True:
            self.saveRes(best_iter,
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
        print(param)
        #Saving parameters
        if self.make_log is True:
            self.saveParam(param)
        #Initializing the model it it wasn't already
        model = XGBoost(kind=self.kind,
                        #Not in tuning dict
                        verbosity=self.verbosity,
                        process_type=self.process_type,
                        tree_method=self.tree_method,
                        objective=self.objective,
                        num_parallel_tree=self.num_parallel_tree,
                        eval_metric=self.eval_metric,
                        early_stopping_rounds=self.early_stopping_rounds,
                        #In tuning dict
                        num_rounds = param[0],
                        max_depth = param[1],
                        min_child_weight = param[2],
                        colsample_bytree= param[3],
                        learning_rate= param[4],
                        reg_alpha= param[5],
                        reg_lambda= param[6],
                        scale_pos_weight= param[7],
                        gamma= param[8],                        
                        subsample= param[9],
                        base_score= param[10],
                        max_delta_step= param[11])

        #Iterable returns pair of train - val sets
        id_pairs = zip(TRAIN_IDS, TEST_IDS)

        #Initializing variables
        weight_factor = 0
        averaging_factor = 0
        tot_prauc = 0
        tot_rce = 0
        tot_confmat = [[0,0],[0,0]]
        max_pred = 0
        min_pred = 1
        avg = 0
        best_iter = []
        #Making iterative train-validations
        for dataset_ids in id_pairs:
            weight_factor += weight_factor+1
            averaging_factor += weight_factor
            #Fetching train set
            X, Y = get_dataset_xgb(dataset_ids[0], 
                                   self.x_label, 
                                   self.y_label)
            
            dmat_train = self.getDMat(X, Y) #------------------------------------- DMATRIX GENERATOR
            del X, Y
            # If early stopping is true for ncv fetching validation set
            # by splitting in two the test set with batch method
            if self.es_ncv is True:
                #Fetching val set 
                X, Y = get_dataset_xgb_batch(2, 0,dataset_ids[1], 
                                             self.x_label, 
                                             self.y_label)
                dmat_val = self.getDMat(X, Y) #------------------------------------- DMATRIX GENERATOR
                del X, Y
            else:
                dmat_val = None
            
            #Multistage model fitting
            model.fit(dmat_train, dmat_val)
            del dmat_train
            if dmat_val is not None:
                best_iter.append(model.getBestIter())
            else:
                best_iter = -1
            del dmat_val

            if self.es_ncv is True:
                #Fetching test set 
                X, Y = get_dataset_xgb_batch(2, 1,dataset_ids[1], 
                                             self.x_label, 
                                             self.y_label)
                dmat_test = self.getDMat(X, Y) #------------------------------------- DMATRIX GENERATOR
                del X, Y    
            else:
                X, Y = get_dataset_xgb(dataset_ids[1], 
                                       self.x_label,
                                       self.y_label)
                dmat_test = self.getDMat(X, Y) #------------------------------------- DMATRIX GENERATOR
                del X, Y
            #Multistage evaluation
            prauc, rce, confmat, max_tmp, min_tmp, avg_tmp= model.evaluate(dmat_test)
            del dmat_test

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
        del model

        #Averaging scores
        tot_prauc /= averaging_factor
        tot_rce /= averaging_factor
        #Averaging average (lol)
        avg /= len(TRAIN_IDS)

        #Make human readable logs here
        if self.make_log is True:
            self.saveRes(best_iter,
                         tot_prauc, 
                         tot_rce, 
                         tot_confmat, 
                         max_pred, 
                         min_pred, 
                         avg)
        
        #Returning the dumbly combined scores
        return self.metriComb(tot_prauc, tot_rce)

# ------------------------------------------------------
#        XGB BATCH WITH EXTERNAL MEMORY
# ------------------------------------------------------
#       Use the self.batchLoadSetsWithExtMemory method
#         In order to load the datasets
# ------------------------------------------------------
# Batch may be of different sizes, but considering how
# they're distributed giving them the same weight in
# averaging process shouldn't raise problems.
# ------------------------------------------------------
# Score function for the XGBoost model
    def blackBoxXgbBatchExtMem(self, param):
        queue = mp.Queue()
        sub_process = mp.Process(target=run_xgb_external_memory, args=(param, self, queue))
        sub_process.start()
        sub_process.join()
        return queue.get()
        # with mp.Pool(1) as pool:
        #     return pool.map(functools.partial(run_xgb_external_memory, model_interface=self), [param])[0]

#-----------------------------------------
#       TODO:Future implementation
#-----------------------------------------
    # Score function for the lightGBM model
    def blackBoxLGB(self, param):
        if self.make_log is True:
            self.saveParam(param)
        #Initializing the model it it wasn't already
        model = LightGBM(kind=self.kind,
                        objective=self.objective,
                        #In tuning dict
                        num_iterations =  param[0],
                        num_leaves=       param[1],
                        learning_rate=    param[2],
                        max_depth=        param[3],
                        lambda_l1=        param[4],
                        lambda_l2=        param[5],
                        colsample_bynode= param[6],
                        colsample_bytree= param[7],
                        bagging_fraction= param[8],
                        pos_subsample=    param[9],  
                        neg_subsample=    param[10],  
                        scale_pos_weight= param[11],        #Remember that scale_pos_wiight and is_unbalance are mutually exclusive
                        bagging_freq=     param[12],   
        )
        #Training on custom set
        if (self.dmat_train is None):
            print("No train set passed to the model.")
        else:
            #dmat_train = self.getDMat(self.X_train, self.Y_train) #------------------------------------- DMATRIX GENERATOR
            model.fit(self.X_train, self.Y_train)

        #Evaluating on custom set
        if (self.dmat_test is None):
            print("No test set provided.")
        else:
            #dmat_test = self.getDMat(self.X_test, self.Y_test) #------------------------------------- DMATRIX GENERATOR
            prauc, rce, confmat, max_pred, min_pred, avg = model.evaluate(self.X_test.to_numpy(),self.Y_test.to_numpy())

        del model
        #Make human readable logs here
        if self.make_log is True:
            self.saveRes(-1,
                         prauc, 
                         rce, 
                         confmat, 
                         max_pred, 
                         min_pred, 
                         avg)
        
        #Returning the dumbly combined scores
        return self.metriComb(prauc, rce)


    
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
            param_dict = xgbRange(self.kind)

        if self.model_name in "lightgbm_classifier":
            param_dict =  lgbmRange(self.kind)

        if self.model_name in "catboost_classifier":
            param_dict = []

        return param_dict


    # Returns the ordered names parameter dictionary
    def getParamNames(self):
        # Returning the names of the hyperparameters
        if self.model_name in "xgboost_classifier":
            names_dict = xgbName()

        if self.model_name in "lightgbm_classifier":
            names_dict =  lgbmName()

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
        elif self.mode == 2:
            if self.model_name in "xgboost_classifier":
                score_func = self.blackBoxXgbNCV

            if self.model_name in "lightgbm_classifier":
                score_func = self.blackBoxLgbNCV

            if self.model_name in "catboost_classifier":
                score_func = self.blackBoxCatNCV
        else:
            if self.model_name in "xgboost_classifier":
                score_func = self.blackBoxXgbBatchExtMem

        return score_func
#---------------------------------------------------------------


#-------------------------------------------------   
#           Combine the two metrics
#-------------------------------------------------
    # Returns a combination of the two metrics
    def metriComb(self, prauc, rce):
        # Superdumb combination
        metric = -rce
        if rce > 0:
            metric = - (rce * prauc)
        else:
            metric = - (rce * (1 - prauc))
        if np.isfinite(metric):
            return metric
        else:
            return float(1000)
#-------------------------------------------------


#-------------------------------------------------
#           Load dataset methods
#-------------------------------------------------
    # Loads a custom train set
    def loadTrainData(self, X_train=None, Y_train=None, dmat_train=None):
        self.X_train=X_train
        self.Y_train=Y_train
        if dmat_train is None:
            self.dmat_train = self.getDMat(X_train, Y_train)
        else:
            self.dmat_train = dmat_train

    
    # Loads a custom data set
    def loadValData(self, X_val=None, Y_val=None, dmat_val=None):
        self.X_val=X_val
        self.Y_val=Y_val
        if dmat_val is None:
            self.dmat_val = self.getDMat(X_val, Y_val)
        else:
            self.dmat_val = dmat_val


    # Loads a custom data set
    def loadTestData(self, X_test=None, Y_test=None, dmat_test=None):
        self.X_test=X_test
        self.Y_test=Y_test
        if dmat_test is None:
            self.dmat_test = self.getDMat(X_test, Y_test)
        else:
            self.dmat_test = dmat_test
#--------------------------------------------------


#--------------------------------------------------
#           Batch/NCV methods
#--------------------------------------------------
    # Passing train set id and number of batches (batch only)
    def batchTrain(self, tot_train_split, train_id):
        self.tot_train_split = tot_train_split
        self.train_id = train_id
    # Passing val set id and number of batches (batch only)
    def batchVal(self, val_id):
        self.val_id = val_id
    # Passing val set id and number of batches (batch only)
    def batchTest(self, tot_test_split, test_id):
        self.tot_test_split = tot_test_split
        self.test_id = test_id
    # Setting labels to use in the sets (batch + NCV)
    # es parameter useful only for NCV
    def setLabels(self, x_label, y_label, es_ncv):
        self.x_label = x_label
        self.y_label = y_label
        self.es_ncv = es_ncv
    def setExtMemTrainPaths(self, ext_memory_paths):
        self.ext_memory_train_paths = ext_memory_paths
    def setExtMemValPaths(self, ext_memory_paths):
        self.ext_memory_val_paths = ext_memory_paths
#--------------------------------------------------


#--------------------------------------------------
#            Save human readable logs
#--------------------------------------------------
    #Saves the parameters (called before the training phase)
    def saveParam(self, param):
        if self.path is None:
            #Taking the path provided
            self.path = str(dt.datetime.now().strftime("%m_%d_%H_%M_%S")) + ".log"
        #Get hyperparameter names
        p_names = self.getParamNames()
        #Opening a file and writing into it the logs
        with open(self.path, 'a') as log:
            to_write = "ITERATION NUMBER " + str(self.iter_count) + "\n"
            log.write(to_write)
            for i in range(len(p_names)):
                to_write=str(str(p_names[i])+"= "+str(param[i])+"\n")
                log.write(to_write)


    def saveRes(self, best_iter, prauc, rce, confmat, max_arr, min_arr, avg):
        if self.path is None:
            # Taking the path provided
            self.path = str(dt.datetime.now().strftime("%m_%d_%H_%M_%S")) + ".log"
        # Opening a file and writing into it the logs
        with open(self.path, 'a') as log:
            # Writing the log
            tn, fp, fn, tp = confmat.ravel()
            total = tn + fp + fn + tp
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * ((precision * recall) / (precision + recall))

            obj = self.metriComb(prauc, rce)
            to_write = "-------\n"

            to_write += "best_es_iteration: " + str(best_iter) + "\n"

            to_write += "-------\n"

            to_write += "PRAUC = " + str(prauc) + "\n"
            to_write += "RCE   = " + str(rce) + "\n"

            to_write += "-------\n"

            # confusion matrix in percentages
            to_write += "TN %  = {:0.3f}%\n".format((tn / total) * 100)
            to_write += "FP %  = {:0.3f}%\n".format((fp / total) * 100)
            to_write += "FN %  = {:0.3f}%\n".format((fn / total) * 100)
            to_write += "TP %  = {:0.3f}%\n".format((tp / total) * 100)

            to_write += "-------\n"

            # confusion matrix in percentages
            to_write += "TN    = {:n}\n".format(tn)
            to_write += "FP    = {:n}\n".format(fp)
            to_write += "FN    = {:n}\n".format(fn)
            to_write += "TP    = {:n}\n".format(tp)

            to_write += "-------\n"

            to_write += "PREC    = {:0.5f}%\n".format(precision)
            to_write += "RECALL    = {:0.5f}%\n".format(recall)
            to_write += "F1    = {:0.5f}%\n".format(f1)

            to_write += "-------\n"

            to_write += "MAX   =" + str(max_arr) + "\n"
            to_write += "MIN   =" + str(min_arr) + "\n"
            to_write += "AVG   =" + str(avg) + "\n"

            to_write += "-------\n"

            to_write += "OBJECTIVE: " + str(obj) + "\n\n\n"
            log.write(to_write)

        # Increasing the iteration count
        self.iter_count = self.iter_count + 1

    #--------------------------------------------------


#--------------------------------------------------
#             Reset saving parameters
#--------------------------------------------------
# Stop appending to the previos log and make a new
# one.
#--------------------------------------------------
    def resetSaveLog(self):
        self.iter_count = 1
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


#--------------------------------------------------
#         Setting non tuned parameters
#--------------------------------------------------
    def setParams(self, verbosity, 
                        process_type, 
                        tree_method, 
                        objective, 
                        num_parallel_tree, 
                        eval_metric, 
                        early_stopping_rounds):
        self.verbosity=verbosity
        self.process_type=process_type
        self.tree_method=tree_method
        self.objective=objective
        self.num_parallel_tree=num_parallel_tree
        self.eval_metric=eval_metric
        self.early_stopping_rounds=early_stopping_rounds
#--------------------------------------------------


#--------------------------------------------------
#       Generator of DMatrices
#--------------------------------------------------
    def getDMat(self, X, Y=None):
        return xgb.DMatrix(X, label=Y)
#--------------------------------------------------




#----------------------------------------------------------
def getDMat(X, Y=None):
    return xgb.DMatrix(X, label=Y)


def run_xgb_external_memory(param, model_interface, queue):
    #Saving parameters of the optimization
    if model_interface.make_log is True:
        model_interface.saveParam(param)
    # Initializing the model it it wasn't already
    model = XGBoost(kind=model_interface.kind,
                    # Not in tuning dict
                    objective="binary:logistic",
                    num_parallel_tree=4,
                    eval_metric="auc",
                    # In tuning dict
                    num_rounds=param[0],
                    max_depth=param[1],
                    min_child_weight=param[2],
                    colsample_bytree=param[3],
                    learning_rate=param[4],
                    reg_alpha=param[5],
                    reg_lambda=param[6],
                    scale_pos_weight=param[7],
                    gamma=param[8],
                    subsample=param[9],
                    base_score=param[10],
                    max_delta_step= param[11])

    # Batch train
    for path in tqdm(model_interface.ext_memory_train_paths):
        # Multistage model fitting
        dmat_train = getDMat(path) #------------------------------------- DMATRIX GENERATOR
        model.fit(dmat_train)
        del dmat_train

    # TODO: last evaluation set may be smaller so it needs
    # to be weighted according to its dimension.

    # Initializing variables
    tot_prauc = 0
    tot_rce = 0
    tot_confmat = [[0, 0], [0, 0]]
    max_pred = 0  # Max set to the minimum
    min_pred = 1  # Min set to the maximum
    avg = 0
    model_interface.tot_val_split = len(model_interface.ext_memory_val_paths)
    # Batch evaluation
    for path in tqdm(model_interface.ext_memory_val_paths):
        # Iteratively fetching the dataset
        dmat_test = xgb.DMatrix(path, silent=True)
        
        # Multistage evaluation
        prauc, rce, confmat, max_tmp, min_tmp, avg_tmp = model.evaluate(dmat_test)
        del dmat_test

        # Summing all the evaluations
        tot_prauc = tot_prauc + prauc
        tot_rce = tot_rce + rce

        # Computing some statistics for the log
        if model_interface.make_log is True:
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
    del model

    # Averaging the evaluations over # of validation splits
    tot_prauc = tot_prauc / model_interface.tot_val_split
    tot_rce = tot_rce / model_interface.tot_val_split
    avg = avg / model_interface.tot_val_split

    # Make human readable logs here
    if model_interface.make_log is True:
        model_interface.saveRes(tot_prauc,
                     tot_rce,
                     tot_confmat,
                     max_pred,
                     min_pred,
                     avg)

    # Returning the dumbly combined scores
    queue.put(model_interface.metriComb(tot_prauc, tot_rce))
    return model_interface.metriComb(tot_prauc, tot_rce)
#--------------------------------------------------------------