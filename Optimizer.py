from bayes_opt import BayesianOptimization
from bayes_opt import JSONLogger
from bayes_opt import Events
from ParamTuning.ModelInterface import ModelInterface

#bayes_opt documentation @
#https://github.com/fmfn/BayesianOptimization


def main():

    #Choosing the model to tune
    model_choice = "xgboost_classifier"
    kind = "LIKE"
    init_pts = 10
    n_itr = 50

    #Declare object
    MI = ModelInterface(model_choice, kind)

    #Retrieve param dictionary
    param = MI.getParams()
    
    #To drop elements from the dictionary
    #del param['key']
    #To fix manually a value should work like this
    #param['key'] = (same_value, same_value)

    #Define the optimizer
    optimizer = BayesianOptimization(f=MI.blackBoxXGB,
                                     pbounds=param,
                                     random_state=5)
    
    #Setting the logger to save log files
    logger = JSONLogger(path="./logs.json")
    optimizer.subscribe('optimization:step', logger)

    #Optimization of the model
    optimizer.maximize(init_points=10, n_iter=10)


if __name__ == "__main__":
    main()
