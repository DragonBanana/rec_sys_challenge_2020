import numpy as np
import skopt
from skopt import gp_minimize
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import sys
import pandas as pd
import time
import datetime as dt
from ParamTuning.ModelInterface import ModelInterface


#----------------------------------------------------
#           Simple skopt optimizer
#----------------------------------------------------
# With this it is possible to run an optimization in 
# just a couple of commands.
#----------------------------------------------------
# logSaver() method provides human readable logs, in
# order to save the results to use in a consecutive
# optimization run, use saveRes() method
#----------------------------------------------------
class Optimizer(object):
    def __init__(self, model_name,
                       kind,
                       auto_save=True, # Saves the results at the end of optimization
                       make_save=True, # Saves the results iteration per iteration
                       make_log=False, # Make a human readable log iteration per iteration
                       path=None):     # Define path in which to save stuff
        #Inputs
        self.model_name = model_name
        self.kind = kind
        self.auto_save=auto_save  #saves the results without explictly calling the method
        self.make_log=make_log
        self.path=path #Providing a path into which write the logs
        #ModelInterface
        self.MI = None
        #Iteration counter
        self.iter_count = 0
                       


    #Setting the parameters of the optimizer  
    def setParameters(self,
                      n_calls=20,
                      n_points = 10000,
                      n_random_starts = 10,
                      n_jobs = 1,
                      # noise = 'gaussian',
                      noise = 1e-5,
                      acq_func = 'gp_hedge',
                      acq_optimizer = 'auto',
                      random_state = None,
                      verbose = True,
                      n_restarts_optimizer = 10,
                      xi = 0.01,
                      kappa = 1.96,
                      x0 = None,
                      y0 = None):

        self.n_point = n_points
        self.n_calls = n_calls
        self.n_random_starts = n_random_starts
        self.n_jobs = n_jobs
        self.acq_func = acq_func
        self.acq_optimizer = acq_optimizer
        self.random_state = random_state
        self.n_restarts_optimizer = n_restarts_optimizer
        self.verbose = verbose
        self.xi = xi
        self.kappa = kappa
        self.noise = noise
        self.x0 = x0
        self.y0 = y0 


    #Setting the model interface
    def defineMI(self, model_name=None, kind=None):
        if model_name is not None:
            self.model_name = model_name
        if kind is not None:
            self.kind = kind

        #Model interface
        self.MI = ModelInterface(model_name=self.model_name,
                                 kind=self.kind)


    #Defining the optimization method
    def optimize(self):

        #Initializing model interface if it's None
        if self.MI is None:
            self.defineMI()

        #Checking if callback has to be called to make logs
        if self.make_log is False:
            #Defining the callback function
            callback_function = self.callback_func
        else:
            callback_function = self.saveLog
            if self.path is None:
                self.path = str("./"+dt.datetime.now().strftime("%m_%d_%H_%M_%S"))

        self.result = gp_minimize(self.MI.getScoreFunc(),
                                  self.MI.getParams(),
                                  base_estimator=None,
                                  n_calls=self.n_calls,
                                  n_random_starts=self.n_random_starts,
                                  acq_func=self.acq_func,
                                  acq_optimizer=self.acq_optimizer,
                                  x0=self.x0,
                                  y0=self.y0,
                                  random_state=self.random_state,
                                  verbose=self.verbose,
                                  callback=callback_function,
                                  n_points=self.n_point,
                                  n_restarts_optimizer=self.n_restarts_optimizer,
                                  xi=self.xi,
                                  kappa=self.kappa,
                                  noise=self.noise,
                                  n_jobs=self.n_jobs)
        
        #Saving the obtained results
        if self.auto_save is True:
            self.saveRes(self.result)
        
        return self.result


    def callback_func(self, res):
        if self.make_res is True:
            self.saveRes(res)
        if self.make_log is True:
            self.saveLog(res)


    #Saving the results of the optimization with built-in method
    def saveRes(self, res):
        path = self.path + ".save"
        #The only way to save this shit
        skopt.dump(self.result, path)
        print("Model {0} successfully saved.".format(path))


    #Fuction to save human readable logs
    def saveLog(self, res):
        #Parameters of the evaluation
        x = res.x_iters
        #Result of the evaluation
        y = res.func_vals
        #Taking the path provided
        path = self.path + ".log"
        #Get hyperparameter names
        p_names = self.MI.getParamNames()
        #Maybe check len(p_names) == len(x) here

        #Opening a file and writing into it the logs
        with open(path, 'a') as log:
            to_write = "ITERATION NUMBER " + str(self.iter_count) + "\n"
            log.write(to_write)
            for i in range(len(p_names)):
                to_write=str(str(p_names[i])+": "+str(x[self.iter_count][i])+"\n")
                log.write(to_write)
            
            #Written this way to be easily found
            to_write="--outcome--: "+str(y[self.iter_count])+"\n\n"
            log.write(to_write)

        #Increasing the iteration count
        self.iter_count = self.iter_count + 1


        #Loading model with built-in method (errors even with pickle)
    def loadModel(self, path = None):
        if (path is None):
            print("File path missing.")
        else:   
            
            #The only way to save this shit
            model = skopt.utils.load(path)

            #Splitting the model
            self.x0 = model.x_iters
            self.y0 = model.func_vals
            print("File {0} loaded successfully.".format(path))

    
    #Load a custom dataset to train for the optimization
    def loadTrainData(self, X_train=None, Y_train=None):
        #Initializing model interface if it's None
        if self.MI is None:
            self.defineMI()
        
        self.MI.loadTrainData(X_train, Y_train)

    
    #Load a custom dataset to test for the optimization
    def loadTestData(self, X_test=None, Y_test=None):
        #Initializing model interface if it's None
        if self.MI is None:
            self.defineMI()
        
        self.MI.loadTestData(X_test, Y_test)


    #Autofetch methods
    #Train
    def autoLoadTrain(self, ids, x_label, y_label):
        #Initializing model interface if it's None
        if self.MI is None:
            self.defineMI()
        
        self.MI.loadTestData(ids, x_label, y_label)


    #Validation/Test
    def autoLoadTest(self):
        None
