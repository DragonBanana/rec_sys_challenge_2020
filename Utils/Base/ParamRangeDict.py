from skopt.space import Real
from skopt.space import Integer
from skopt.space import Categorical

#----------------------------------------------------------------
#                   ABOUT XGB PARAMETERS
#----------------------------------------------------------------
# num_rounds:       # of rounds for boosting
# max_depth:        Maximum depth of a tree. Increasing this
#                    will increase the model complexity.
# min_col_weight:   Minimum sum of instance weight needed in
#                    child.
# colsample_bytree: Subsample ratio of columns when constructing
#                    each tree. Occurs once per tree constructed.
# learning_rate:    Step size shrinkage used in update to prevent
#                    overfitting.
# alpha_reg:        L1 regularization term.
# lambda_reg:       L2 regularization term.
# scale_pos_weight: Control the balance of positive and negative 
#                    weights, useful for unbalanced classes.
# gamma:            Minimum loss reduction required to make a 
#                    further partition on a leaf node of the tree.
# subsample:        Subsample ratio of the traning instances.
# base_score:       The initial prediction score of all instances,
#                    global bias.
# max_delta_step:   Maximum delta step we allow each leaf output
#                    to be.
#---------------------------------------------------------------

LIKE = "likeLIKELike"
RETWEET = "retweetRETWEETRetweet"
COMMENT = "commentCOMMENTComment"
REPLY = "replyREPLYReply"

def xgbRange(kind):
    param_range_dict = [Integer(500, 501),                    #num_rounds
                        Integer(5, 40),                    #max_depth
                        Integer(1, 10),                     #min_child_weight
                        Real(0.3, 1),                       #colsample_bytree
                        Real(0.01, 1, 'log-uniform'),     #learning rate
                        Real(0.0001, 1, 'log-uniform'),     #alpha_reg
                        Real(0.0001, 1, 'log-uniform'),     #lambda_reg
                        # SCALE POS WEIGHT FOR LIKE
                        Real(0.9, 1.1),                     #scale_pos_weight
                        Real(0.1, 10),                      #gamma
                        Real(0.3, 1),                       #subsample
                        Real(0.45,0.55),                        #base_score
                        Real(1, 100)]                        #max_delta_step

    return param_range_dict
    #scale_pos_weight ---> good for ranking, bad for predicting probability,
    #use max_delta_step instead


#Names of the hyperparameters that will be optimized
def xgbName():
    param_name_dict = ["n_iterations",
                       "max_depth",
                       "min_child_weight",
                       "colsample_bytree",
                       "learning_rate",
                       "alpha_reg",
                       "lambda_reg",
                       "scale_pos_weight",
                       "gamma",
                       "subsample",
                       "base_score",
                       "max_delta_step"]
    return param_name_dict


####################################################
#                     LIGHTGBM                     #
####################################################

def lgbmRange(kind):
    param_range_dict = [Integer(5, 200),                        #num_iterations
                        Integer(31, 70),                        #num_leaves
                        Real(0.0001, 1, 'log-uniform'),         #learning rate
                        Integer(5, 50),                         #max_depth
                        Real(0.1, 1),                         #lambda_l1
                        Real(0.1, 1),                         #lambda_l2
                        Real(0.1, 1),                           #colsample_bytree
                        Real(0.1, 1),                           #colsample_bynode
                        Real(0.5, 1),                           #bagging_fraction
                        Real(0.1,1),                            #pos_subsample
                        Real(0.1,1),                            #neg_subsample
                        # SCALE POS WEIGHT
                        Real(1,1.000000000001),                         #scale_pos_weight
                        # ALTERNATIVELY IS UMBALANCE MUST BE SET AS TRUE
                        Integer(0,50),                             #bagging_freq
    ]
    return param_range_dict


#Names of the hyperparameters that will be optimized
def lgbmName():
    param_name_dict = [
                       "num_iterations",
                       "num_leaves",
                       "learning rate",
                       "max_depth",
                       "lambda_l1",
                       "lambda_l2",
                       "colsample_bytree",
                       "colsample_bynode",
                       "subsample",
                       "pos_subsample",
                       "neg_subsample",
                       "scale_pos_weight",
                       "bagging_freq"
                       ]
    return param_name_dict


'''
def dictBayOptLGB():
    #BAYESIAN OPTIMIZATION LIBRARY
    #Parameter's range dictionary
    param_range_dict={'num_rounds': (5,1000),
                      'num_leaves': (15,500),
                      'learning_rate': (0.00001,0.1),
                      'num_iterations': (10,600),
                      'max_depth': (15,500),
                      'lambda_l1': (0.00001,0.1),
                      'lambda_l2': (0.00001,0.1),
                      'colsample_bynode': (0,1),
                      'colsample_bytree': (0,1),
                      'subsample': (0,1),
                      'pos_subsample': (0,1),
                      'neg_subsample': (0,1), 
                      'bagging_freq': (0,1)}
    return param_range_dict

def dictBayOptCAT():
    param_range_dict={'iterations':(5,500),
                      'depth':(5,50),
                      'learning_rate':(0.00001,0.1),
                      'l2_leaf_reg':(0.00001,0.1)}
    return param_range_dict
'''
