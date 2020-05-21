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
    param_range_dict = [Categorical([501]),                 #num_rounds
                        Integer(2, 30),                    #max_depth
                        Integer(1, 20),                    #min_child_weight
                        Real(0.1, 1),                      #colsample_bytree
                        Real(0.001, 0.1, 'log-uniform'),      #learning rate
                        Real(0.0001, 1, 'log-uniform'),    #alpha_reg
                        Real(0.0001, 1, 'log-uniform'),    #lambda_reg
                        # SCALE POS WEIGHT FOR LIKE
                        Real(0.7, 1.3),                     #scale_pos_weight
                        Real(0.1, 10, 'log-uniform'),                      #gamma
                        Real(0.1, 1),                       #subsample
                        Categorical([0.5]),                        #base_score
                        Real(0, 200),                      #max_delta_step
                        Integer(2, 20)]                        #num_parallel_tree

    #PERSONALIZED PARAMETERS---------------SET PROPER RANGE FOR EACH CLASS
    if kind in LIKE:
        param_range_dict[7] = Real(0.7, 1.2)
        param_range_dict[11] = Real(0, 5)                      #max_delta_step
    elif kind in RETWEET:
        param_range_dict[7] = Real(0.8, 1.3)
        param_range_dict[11] = Real(0, 25)                      #max_delta_step
    elif kind in COMMENT:
        param_range_dict[7] = Real(0.9, 1.4)
        param_range_dict[11] = Real(0, 100)                      #max_delta_step
    elif kind in REPLY:
        param_range_dict[7] = Real(1, 1.5)
        param_range_dict[11] = Real(0, 200)                      #max_delta_step

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
                       "reg_alpha",
                       "reg_lambda",
                       "scale_pos_weight",
                       "gamma",
                       "subsample",
                       "base_score",
                       "max_delta_step",
                       "parallel_num_tree"]
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
    param_range_dict = [Integer(70, 4095),                      #num_leaves
                        Real(0.005, 1, 'log-uniform'),          #learning rate
                        Integer(2, 70),                         #max_depth
                        Real(0.1, 1, 'log-uniform'),            #lambda_l1
                        Real(0.1, 1, 'log-uniform'),            #lambda_l2
                        Real(0.4, 1),                           #colsample_bynode
                        Real(0.4, 1),                           #colsample_bytree
                        Real(0.1, 1),                           #pos_subsample
                        Real(0.1, 1),                           #neg_subsample
                        #Real(0.1, 1),                          #bagging_positive_over_total_ratio
                        #Real(0.1, 1),                          #dominant_bagging
                        Integer(0, 10),                         #bagging_freq
                        Integer(255, 5000),                     #max_bin
                        Integer(400, 2000),                     #min_data_in_leaf
    ]
    return param_range_dict


#Names of the hyperparameters that will be optimized
def lgbmName():
    param_name_dict = [
                       "num_leaves",
                       "learning rate",
                       "max_depth",
                       "lambda_l1",
                       "lambda_l2",
                       "colsample_bynode",
                       "colsample_bytree",
                       "pos_subsample",
                       "neg_subsample",
                       #"bagging_positive_over_total_ratio",
                       #"dominant_bagging",
                       "bagging_freq",
                       "max_bin",
                       "min_data_in_leaf"
                       ]
    return param_name_dict


#----------------------------------------------------------------
#                   ABOUT CAT PARAMETERS
#----------------------------------------------------------------
# interations:     Maximum number of trees that can be built.
# depth:           Depth of the trees (model set max to 16).
# learning_rate:   Learning rate, reduces the gradient step.
# l2_leaf_reg:     L2 regularization.
# subsample:       Sample rate for begging.
# random_strenght: The amount of randomness to use for scoring 
#                   splits when the tree structure is selected. 
#                   Use this parameter to avoid overfitting the 
#                   model.
# leaf_estimation_iterations: The number of gradient steps when 
#                              calculating the values in leaves.
# scale_pos_weight:           The weight for class 1 in binary 
#                              classification. The value is used 
#                              as a multiplier for the weights of 
#                              objects from class 1.      
# model_shrink_rate:          The constant used to calculate the 
#                              coefficient for multiplying the 
#                              model on each iteration.
#----------------------------------------------------------------
def catRange(kind):
    param_range_dict = [Integer(2000,2001),                 # iterations
                        Integer(1,16),                      # depth
                        Real(0.000001, 1, 'log_uniform'),     # learning_rate
                        Real(0.000001, 40, 'log_uniform'),    # l2_leaf_reg
                        Real(0.1, 0.9),                     # subsample
                        Real(0.0001, 60),                   # random_strenght
                        Real(0.1, 1),                       # colsample_bylevel
                        Integer(5,300),                     # leaf_estimation_iterations
                        Real(1,5),                          # scale_pos_weight
                        Real(0.0001,2, 'log_uniform')]      # model_shrink_rate

    
    # PERSONALIZED SCALE_POS_WEIGHT
    # From documentation put it to neg_samples/pos_samples 
    if kind in LIKE:
        param_range_dict[8] = Real(0.5, 5)
    elif kind in RETWEET:
        param_range_dict[8] = Real(1, 15)
    elif kind in REPLY:
        param_range_dict[8] = Real(15, 55)
    elif kind in COMMENT:
        param_range_dict[8] = Real(80, 170)

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
