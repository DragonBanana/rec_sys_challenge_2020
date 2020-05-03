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
    param_range_dict = [Integer(500, 501),                 #num_rounds
                        Integer(5, 40),                    #max_depth
                        Integer(1, 10),                    #min_child_weight
                        Real(0.3, 1),                      #colsample_bytree
                        Real(0.005, 0.5, 'log-uniform'),      #learning rate
                        Real(0.0001, 1, 'log-uniform'),    #alpha_reg
                        Real(0.0001, 1, 'log-uniform'),    #lambda_reg
                        # SCALE POS WEIGHT FOR LIKE
                        Real(0.8, 1.2),                     #scale_pos_weight
                        Real(0.1, 10, 'log-uniform'),                      #gamma
                        Real(0.3, 1),                       #subsample
                        Real(0.4,0.5),                        #base_score
                        Real(0, 100)]                        #max_delta_step
    
    '''
    #PERSONALIZED PARAMETERS---------------SET PROPER RANGE FOR EACH CLASS
    if kind in LIKE:
        param_range_dict[7] = Real(0.9, 1.1)
    elif kind in RETWEET:
        param_range_dict[7] = Real(0.9, 1.1)
    elif kind in COMMENT:
        param_range_dict[7] = Real(0.9, 1.1)
    elif kind in REPLY:
        param_range_dict[7] = Real(0.9, 1.1)
    '''

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
# num_iteration:       # of rounds for boosting
# num_leaves:        Number of leaves of each boosting tree, may be less then 2^(max_depth), significantly lower to reduce overfitting
# colsample_bytree: Subsample ratio of columns when constructing
#                    each tree. Occurs once per tree constructed.
# colsample_bytree: Subsample ratio of columns when constructing
#                    each tree. Occurs once per tree constructed.
# learning_rate:    Step size shrinkage used in update to prevent
#                    overfitting.
# lambda_L1:        L1 regularization term.
# lambda_l2:       L2 regularization term.
# scale_pos_weight: Control the balance of positive and negative
#                    weights, useful for unbalanced classes.
# pos_subsample:        Subsample ratio of the positive traning instances.
# neg_subsample:        Subsample ratio of the negative traning instances.
# bagging_freq:     to perform bagging every k iterations
# max_bin:          for better accuracy, large max_bin can cause overfitting
# extra_trees:       use extremely randomized trees                                 PROVARE SIA CON SIA SENZA EXTRA TREES (TRUE/FALSE)
#                    if set to true, when evaluating node splits LightGBM will check only one randomly-chosen threshold for each feature
#                    can be used to deal with over-fitting
#
#---------------------------------------------------------------
def lgbmRange(kind):
    param_range_dict = [Integer(5, 400),                        #num_iterations
                        Integer(31, 70),                        #num_leaves
                        Real(0.0001, 1, 'log-uniform'),         #learning rate
                        Integer(5, 100),                        #max_depth
                        Real(0.1, 1, 'log-uniform'),            #lambda_l1
                        Real(0.1, 1, 'log-uniform'),            #lambda_l2
                        Real(0.1, 1),                           #colsample_bynode
                        Real(0.1, 1),                           #colsample_bytree
                        Real(0.5, 1),                           #bagging_fraction
                        Real(0.1, 1),                            #pos_subsample
                        Real(0.1, 1),                            #neg_subsample
                        # SCALE POS WEIGHT
                        #Real(1,10),                            #scale_pos_weight
                        # ALTERNATIVELY IS UMBALANCE MUST BE SET AS TRUE
                        Integer(0, 50),                          #bagging_freq
                        Integer(255, 5000)                       #max_bin
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
                       "colsample_bynode",
                       "colsample_bytree",
                       "bagging_fraction",
                       "pos_subsample",
                       "neg_subsample",
                       #"scale_pos_weight",
                       "bagging_freq",
                       "max_bin"
                       ]
    return param_name_dict



def catRange(kind):
    param_range_dict = [Integer(5,200),                     # iterations
                        Integer(1,16),                      # depth
                        Real(0.0001, 1, 'log_uniform'),     # learning_rate
                        Real(0.0001, 20, 'log_uniform'),    # l2_leaf_reg
                        Real(0.1, 0.9),                     # subsample
                        Real(0.001, 30),                    # random_strenght
                        Real(0.01, 1, 'log_uniform'),       # colsample_bylevel
                        Integer(1, 200),                    # leaf_estimation_iterations
                        Real(1,300),                        # scale_pos_weight
                        Real(0.0001,1, 'log_uniform')]      # model_shrink_rate

    '''
    #PERSONALIZED SCALE_POS_WEIGHT---------------SET PROPER RANGE FOR EACH CLASS
    if kind in LIKE:
        param_range_dict[9] = Real(0.9, 1.1)
    elif kind in RETWEET:
        param_range_dict[9] = Real(0.9, 1.1)
    elif kind in COMMENT:
        param_range_dict[9] = Real(0.9, 1.1)
    elif kind in REPLY:
        param_range_dict[9] = Real(0.9, 1.1)
    '''

    return param_range_dict

def catName():
    param_name_dict = ["iterations",
                       "depth",
                       "learning_rate",
                       "l2_leaf_reg",
                       "subsample",
                       "random_strenght",
                       "colsample_bylevel",
                       "leaf_estimation_iterations",
                       "scale_pos_weight",
                       "model_shrink_rate"]
    return param_name_dict
