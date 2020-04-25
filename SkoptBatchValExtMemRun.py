from ParamTuning.Optimizer import Optimizer
import pathlib as pl
import xgboost as xgb

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

    #Declaring optimizer
    OP = Optimizer(model_name,
                   kind,
                   mode=0,
                   make_log=True,
                   make_save=True,
                   auto_save=True,
                   path=label,
                   path_log=label)

    if pl.Path(f"{label}.save.npz").is_file():
        OP.loadModel(f"{label}.save.npz")
    OP.setParameters(n_calls=500, n_random_starts=20)
    OP.defineMI()
    OP.loadTrainData(dmat_train=xgb.DMatrix(f"{folder}/{train_dataset_id}_{label}_batch_0.svm#{folder}/{train_dataset_id}_{label}_batch_0.cache"))
    OP.loadTestData(dmat_test=xgb.DMatrix(f"{folder}/{val_dataset_id}_{label}_batch_0.svm"))
    OP.loadValData(dmat_val=xgb.DMatrix(f"{folder}/{val_dataset_id}_{label}_batch_1.svm"))
    OP.setParamsXGB(early_stopping_rounds=5, eval_metric="logloss", tree_method='hist', verbosity=0)

    OP.optimize()
    #------------------------------------------



if __name__ == "__main__":
    main()
