from bayes_opt import BayesianOptimization
from bayes_opt import JSONLogger
from bayes_opt import Events
from ParamTuning.ModelInterface import ModelInterface

#bayes_opt documentation @
#https://github.com/fmfn/BayesianOptimization


def main():

    #Choosing the model to tune
    model_choice = "xgboost_classifier"
    init_pts = 10
    n_itr = 10

    #Declare object
    MI = ModelInterface(model_choice)

    #Retrieve param dictionary
    param = MI.get_param()
    
    #Define the optimizer
    oprimizer = BayesianOptimization(f=MI.black_box(),
                                     pbounds=pbounds,
                                     random_state=1)
    
    #Setting the logger to save log files
    logger = JSONLogger(path="./logs.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    #Optimization of the model
    optimizer.maximize(init_points=init_pts,
                       n_iter=n_itr)


if __name__ == "__main__":
    main()
