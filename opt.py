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
    model_name = "xgboost_classifier"   
    #Kind of prediction eg. "like"
    kind = "LIKE"




    

    onehot = pd.read_csv("onehot.csv", sep='\x01')
    #u_dict = load_obj("u_dict")
    #i_dict = load_obj("i_dict")
    
    
    #XGBoost part
    test_size = 0.2
    #Dividing the dataset splitting the column i need to predict from the others
    X = onehot[["usr_id", "twt_id", "tmstp_rpl", "tmstp_rtw", "tmstp_rtw_c"]]
    Y = onehot["tmstp_lik"]

    X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                        Y, 
                                                        test_size=test_size, 
                                                        random_state=int(time.time()))







    
    OP = Optimizer(model_name, kind)
    OP.setParameters(n_calls=3, n_random_starts=3)
    OP.loadTrainData(X_train, Y_train)
    OP.loadTestData(X_test, Y_test)
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
