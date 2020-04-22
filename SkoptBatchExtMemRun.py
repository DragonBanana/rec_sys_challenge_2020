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
from Utils.Data import Data
import sklearn.datasets as skd
from tqdm import tqdm
import pathlib as pl

def main():

    # like, retweet, comment, reply
    label = "like"

    #Name of the model eg. xgboost_classifier
    model_name="xgboost_classifier"
    #Kind of prediction eg. "like"
    kind = label

    folder = f"svm_files"

    train_dataset_id = "train_days_123456"
    val_dataset_id = "val_days_7"

    train_batch_n_split = 5
    val_batch_n_split = 1

    #Declaring optimizer
    OP = Optimizer(model_name,
                   kind,
                   mode=3,
                   make_log=True,
                   make_save=True,
                   auto_save=True,
                   path=label,
                   path_log=label)

    if pl.Path(f"{label}.save").is_file():
        OP.loadModel(f"{label}.save")
    OP.setParameters(n_calls=100, n_random_starts=20)
    OP.defineMI()
    OP.MI.setExtMemTrainPaths([
        f"{folder}/{train_dataset_id}_{label}_batch_{i}.svm#{folder}/{label}_train_batch_{i}.cache" for i in range(train_batch_n_split)
    ])
    OP.MI.setExtMemValPaths([
        f"{folder}/{val_dataset_id}_{label}_batch_{i}.svm" for i in range(val_batch_n_split)
    ])
    OP.optimize()
    #------------------------------------------



if __name__ == "__main__":
    main()
