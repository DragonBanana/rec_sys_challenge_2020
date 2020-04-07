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
from ParamTuning.Optimizer import Optimizer



def main():
    #Name of the model eg. xgboost_classifier
    model_name="xgboost_classifier"
    #Kind of prediction eg. "like"
    kind = "LIKE"
    
    OP = Optimizer(model_name, kind, make_log=True, make_save=True)
    OP.setParameters(n_calls=3, n_random_starts=3)
    #OP.loadTrainData(X_train, Y_train)
    #OP.loadTestData(X_test, Y_test)
    res=OP.optimize()

    '''
    #Add this for complete routine check
    print(res.func_vals.shape)
    path = OP.saveModel()
    OP.loadModel(path)
    res = OP.optimize()
    print(res.func_vals.shape)
    print("END")
    '''


if __name__ == "__main__":
    main()
