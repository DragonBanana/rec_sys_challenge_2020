class DictDispenser(object):
    def __init__(self):
        None

    def xgb(self):
        #PARAMETERS' RANGE DICTIONARY
        param_range_dict = {'colsample_bytree': (0,1),
                            'num_rounds': (5,1000),
                            'learning_rate': (0.5,0.00001),
                            'max_depth': (5,100),
                            'reg_alpha': (0.00001, 0.1),
                            'reg_lambda': (0.00001,0.1), 
                            'min_child_weight' : (1, 10),
                            'scale_pos_weight' : (1, 1.5),
                            'subsample': (0.1, 1)}
        return param_range_dict

    def lgb(self):
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

    def cat(self):
        param_range_dict={'iterations':(5,500),
                          'depth':(5,50),
                          'learning_rate':(0.00001,0.1),
                          'l2_leaf_reg':(0.00001,0.1)}
        return param_range_dict
