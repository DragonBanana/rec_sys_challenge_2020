
def lgbm_get_params():
    params = []
    '''
    LGBM-like-freschi-sovracampione
    ITERATION NUMBER 10

    num_leaves= 139

    learning rate= 0.2667469321398438

    max_depth= 49

    lambda_l1= 43.929733757687444

    lambda_l2= 25.86109747137902

    colsample_bynode= 0.9544155345339177

    colsample_bytree= 0.5349257524842879

    bagging_fractionbagging_freq= 0.14336874823469137

    max_bin= 0

    min_data_in_leaf= 2565

    LGBM-like-freschi-sovracampione
    -------
    EXECUTION TIME: 2033.0731165409088
    -------
    best_es_iteration: 1000
    -------
    PRAUC = 0.8125967126354695
    RCE   = 31.362556041852418
    '''

    params.append({"num_iterations": 1000,
                   "num_leaves": 139,
                   "learning_rate": 0.2667469321398438,
                   "max_depth": 49,
                   "lambda_l1": 43.929733757687444,
                   "lambda_l2": 25.86109747137902,
                   "colsample_bynode": 0.9544155345339177,
                   "colsample_bytree": 0.5349257524842879,
                   "bagging_fraction": 0.14336874823469137,
                   "bagging_freq": 0,
                   "max_bin": 2565,
                   "min_data_in_leaf": 1901})

    return params


def xgb_get_params():
    params = []
    '''
    skopt_result/like
    ITERATION NUMBER 10
    
    n_iterations= 1001
    
    max_depth= 12
    
    min_child_weight= 8
    
    colsample_bytree= 0.5061342342153339
    
    learning_rate= 0.07445195010651855
    
    reg_alpha= 0.03845714736115001
    
    reg_lambda= 0.006126524544308775
    
    scale_pos_weight= 1
    
    gamma= 1.567696726803047
    
    subsample= 0.8634723191706059
    
    base_score= 0.4392
    
    max_delta_step= 40.69001101277949
    
    parallel_num_tree= 3
    
    skopt_result/like
    -------
    EXECUTION TIME: 1212.4507186412811
    -------
    best_es_iteration: 637
    -------
    PRAUC = 0.8000024332958628
    RCE   = 29.06732237134276
    '''

    params.append({"num_rounds": 999,
                    "max_depth": 12,
                    "min_child_weight": 8,
                    "colsample_bytree": 0.5061342342153339,
                    "learning_rate": 0.07445195010651855,
                    "reg_alpha": 0.03845714736115001,
                    "reg_lambda": 0.006126524544308775,
                    "scale_pos_weight": 1,
                    "gamma": 1.567696726803047,
                    "subsample": 0.8634723191706059,
                    "base_score": 0.4392,
                    "max_delta_step": 40.69001101277949,
                    "num_parallel_tree": 3
                    })

    '''
    skopt_result/like
    ITERATION NUMBER 4
    
    n_iterations= 1001
    
    max_depth= 10
    
    min_child_weight= 36
    
    colsample_bytree= 0.24443043292899824
    
    learning_rate= 0.17900935642341087
    
    reg_alpha= 0.0009405978959317872
    
    reg_lambda= 0.03992235159598744
    
    scale_pos_weight= 1
    
    gamma= 0.136778544433405
    
    subsample= 0.48158672176663975
    
    base_score= 0.4392
    
    max_delta_step= 37.48408544495799
    
    parallel_num_tree= 3
    
    skopt_result/like
    -------
    EXECUTION TIME: 573.3331050872803
    -------
    best_es_iteration: 363
    -------
    PRAUC = 0.7968906381669001
    RCE   = 28.539058541226048

    '''

    params.append({"num_rounds": 999,
                   "max_depth": 10,
                   "min_child_weight": 36,
                   "colsample_bytree": 0.24443043292899824,
                   "learning_rate": 0.17900935642341087,
                   "reg_alpha": 0.0009405978959317872,
                   "reg_lambda": 0.03992235159598744,
                   "scale_pos_weight": 1,
                   "gamma": 0.136778544433405,
                   "subsample": 0.48158672176663975,
                   "base_score": 0.4392,
                   "max_delta_step": 37.48408544495799,
                   "num_parallel_tree": 3
                   })

    return params

