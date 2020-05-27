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


def main(): 
    # Defining the dataset used
    train_dataset = "holdout/train"
    test_dataset = "holdout/test"

    # Define the X label
    X_label = [
        "raw_feature_creator_follower_count",                                                               #0                                                                                   
        "raw_feature_creator_following_count",                                                              #1                
        "raw_feature_engager_follower_count",                                                               #2                
        "raw_feature_engager_following_count",                                                              #3                
        "raw_feature_creator_is_verified",                                                                  #4 CATEGORICAL            
        "raw_feature_engager_is_verified",                                                                  #5 CATEGORICAL            
        "raw_feature_engagement_creator_follows_engager",                                                   #6 CATEGORICAL                            
        "tweet_feature_number_of_photo",                                                                    #7            
        "tweet_feature_number_of_video",                                                                    #8            
        "tweet_feature_number_of_gif",                                                                      #9        
        "tweet_feature_number_of_media",                                                                    #10            
        "tweet_feature_is_retweet",                                                                         #11 CATEGORICAL    
        "tweet_feature_is_quote",                                                                           #12 CATEGORICAL    
        "tweet_feature_is_top_level",                                                                       #13 CATEGORICAL        
        "tweet_feature_number_of_hashtags",                                                                 #14            
        "tweet_feature_creation_timestamp_hour",                                                            #15                    
        "tweet_feature_creation_timestamp_week_day",                                                        #16                       
        "tweet_feature_number_of_mentions",                                                                 #17            
        "engager_feature_number_of_previous_like_engagement",                                               #18                                
        "engager_feature_number_of_previous_reply_engagement",                                              #19                                
        "engager_feature_number_of_previous_retweet_engagement",                                            #20                                    
        "engager_feature_number_of_previous_comment_engagement",                                            #21                                  
        "engager_feature_number_of_previous_positive_engagement",                                           #22                                    
        "engager_feature_number_of_previous_negative_engagement",                                           #23                                    
        "engager_feature_number_of_previous_engagement",                                                    #24                            
        "engager_feature_number_of_previous_like_engagement_ratio",                                         #25                                    
        "engager_feature_number_of_previous_reply_engagement_ratio",                                        #26                                        
        "engager_feature_number_of_previous_retweet_engagement_ratio",                                      #27                                        
        "engager_feature_number_of_previous_comment_engagement_ratio",                                      #28                                        
        "engager_feature_number_of_previous_positive_engagement_ratio",                                     #29                                        
        "engager_feature_number_of_previous_negative_engagement_ratio",                                     #30                                        
        "engager_feature_number_of_previous_like_engagement_between_creator_and_engager_by_creator",        #31                                                                        
        "engager_feature_number_of_previous_reply_engagement_between_creator_and_engager_by_creator",       #32                                                                        
        "engager_feature_number_of_previous_retweet_engagement_between_creator_and_engager_by_creator",     #33                                                                        
        "engager_feature_number_of_previous_comment_engagement_between_creator_and_engager_by_creator",     #34                                                                        
        "engager_feature_number_of_previous_negative_engagement_between_creator_and_engager_by_creator",    #35                                                                            
        "engager_feature_number_of_previous_positive_engagement_between_creator_and_engager_by_creator",    #36                                                                            
        "engager_feature_number_of_previous_like_engagement_between_creator_and_engager_by_engager",        #37                                                                        
        "engager_feature_number_of_previous_reply_engagement_between_creator_and_engager_by_engager",       #38                                                                        
        "engager_feature_number_of_previous_retweet_engagement_between_creator_and_engager_by_engager",     #39                                                                        
        "engager_feature_number_of_previous_comment_engagement_between_creator_and_engager_by_engager",     #40                                                                        
        "engager_feature_number_of_previous_negative_engagement_between_creator_and_engager_by_engager",    #41                                                                            
        "engager_feature_number_of_previous_positive_engagement_between_creator_and_engager_by_engager",    #42                                                                            
        "engager_main_language",                                                                            #43 CATEGORICAL    
        "creator_main_language",                                                                            #44 CATEGORICAL    
        "creator_and_engager_have_same_main_language",                                                      #45 CATEGORICAL                        
        "is_tweet_in_creator_main_language",                                                                #46 CATEGORICAL                
        "is_tweet_in_engager_main_language",                                                                #47 CATEGORICAL                
        "statistical_probability_main_language_of_engager_engage_tweet_language_1",                         #48                                                    
        "statistical_probability_main_language_of_engager_engage_tweet_language_2",                         #49                                                    
        #"tweet_feature_dominant_topic_LDA_15"                                                               #50 CATEGORICAL                             
    ]                                                                           
    # Define the Y label
    Y_label = [
        "tweet_feature_engagement_is_retweet"
    ]

    model_name = "catboost_classifier"
    kind = "retweet"

    print("checkpoint L1")
    # Load train data
    loading_data_start_time = time.time()
    X_train, Y_train = Data.get_dataset_xgb_batch(1, 0, train_dataset, X_label, Y_label, 0.05)
    print("checkpoint L2")
    # Load test data
    X_val, Y_val = Data.get_dataset_xgb_batch(2, 0, test_dataset, X_label, Y_label, 1)
    print("checkpoint L3")
    X_test, Y_test = Data.get_dataset_xgb_batch(2, 1, test_dataset, X_label, Y_label, 1)
    print(f"Loading data time: {time.time() - loading_data_start_time} seconds")



    x0 = [[2001, 6, 2.0270100079687366e-05, 0.0008404380539110938, 0.5518035800179055, 37.523996650019534, 0.5148063223891307, 227, 5.931940893899446, 0.36156885017700086],
          [2001, 4, 0.1237256089199201, 0.03441509238453477, 0.562756579201236, 4.862545523753014, 0.21973861400418068, 296, 6.480591838540933, 0.06561641034405197],
          [2000, 9, 1.8571159649001317e-05, 0.00011263651249008102, 0.27567389526249364, 40.99424198386619, 0.4194552995897035, 75, 1.5084253540974055, 0.0025837846746217114],
          [2000, 6, 0.0003660154590650489, 3.057888106705987, 0.6249304988108723, 26.237224022317182, 0.7910644754763927, 102, 9.010600101738897, 0.0022577058737566877],
          [2001, 10, 0.008072797764864201, 0.0008193101369068986, 0.26539626433841157, 35.30530816291813, 0.5273663379507824, 187, 9.931655583010356, 0.4501257511173077],
          [2001, 14, 0.17289877591164327, 0.2165602318513908, 0.8826311711100233, 52.404125001535434, 0.5473563561667688, 103, 13.804186050077462, 0.0029728315605064533],
          [2000, 9, 0.006058101021290835, 0.8536810125425808, 0.47081867704423974, 14.777749034282007, 0.5123290300168486, 84, 12.577540534001052, 0.11622361287752918],
          [2000, 13, 5.616605797912838e-06, 2.9136750856550282e-06, 0.5492829613176046, 8.982425950156827, 0.673334969404701, 157, 4.410272916435371, 0.19122160875523683],
          [2000, 9, 3.9706206971933626e-05, 0.24267887113091707, 0.1832207521059813, 7.501656668882015, 0.5728479377512827, 230, 1.1940888655077735, 1.9528315086921786],
          [2001, 7, 0.004877906618049869, 0.006383755306235181, 0.3387437793872865, 7.874217725320077, 0.9968999115514652, 37, 2.429936270299659, 0.010042864659798924],
          [2001, 13, 0.24858930275267913, 1.7320516812582032e-06, 0.5372325255978666, 46.974535322424416, 0.5433011995786268, 274, 11.81661322801958, 0.004587405007229509],
          [2001, 5, 0.05758772396186155, 7.762624648139777e-06, 0.5507735883357252, 21.74993106744098, 0.18111449224305964, 55, 1.246501973793677, 0.18946159421580266],
          [2000, 16, 2.453519070017055e-06, 0.00044606591787090704, 0.2229878602385725, 41.06880779994542, 0.4039305343834382, 82, 9.001194132963693, 0.20027927133856535],
          [2000, 13, 3.7471390683165616e-05, 5.699690281886826e-06, 0.532015154951811, 38.338661276712, 0.25998557826106417, 26, 9.769455277169152, 0.001301604136710709],
          [2000, 4, 5.978102477902652e-05, 0.0004052871708689183, 0.8477368212266749, 33.95778029136631, 0.7632156517845095, 224, 3.816131816245243, 0.002663580019114624],
          [2000, 3, 0.006263354061962383, 0.10243004363325885, 0.8633292450692686, 33.247937850108606, 0.1482578881832032, 204, 2.9824625909189315, 0.0003571483296991471],
          [2001, 4, 0.07191489896198686, 3.083765687944694e-05, 0.1358796488696421, 16.552420609778228, 0.8012192029484366, 244, 5.983271672051028, 0.25048314939787264],
          [2000, 7, 0.09689284326194433, 0.0017674010032933186, 0.1378009282283463, 39.22296832777054, 0.3230830263759936, 60, 6.794607259300093, 0.0007337759799724515],
          [2000, 12, 1.3573335697838985e-06, 9.189976056514227e-05, 0.6694092732260211, 45.12674339815802, 0.45628133348455513, 83, 1.3342008629551518, 0.014103047999334173],
          [2001, 14, 0.0017382375389771473, 3.065732835587183e-06, 0.1544608259853333, 37.74709002128857, 0.6099672761986662, 268, 1.7066316547393623, 0.025179835082220867],
          [2001, 8, 0.0002006066269717645, 0.0016529558819487992, 0.3110804188833127, 51.65279772197058, 0.33095321849395637, 53, 1.5888210289416467, 0.0016938808026168324],
          [2001, 7, 0.054026513654691743, 0.00018523943477163712, 0.1755639678078755, 16.614638344032446, 0.2898205537097147, 54, 6.086019099068151, 0.022259257437127367],
          [2001, 8, 0.06836974659251013, 13.747564921473108, 0.5784454556953582, 44.397143763447716, 0.11919261703913071, 55, 12.10467095183526, 0.029272873154889727],
          [2001, 16, 0.022885765344143702, 1.0256155970482926e-06, 0.9, 0.0001, 0.1, 34, 1.0, 0.0001],
          [2001, 16, 0.01064868717099391, 0.0002747358190383915, 0.1, 60.0, 0.1, 300, 1.0, 0.0001],
          [2000, 6, 0.7828927605444966, 9.542393882089724e-06, 0.43200603030321294, 54.272550545375026, 0.166460732893411, 187, 2.093786940773919, 0.023955396319324702],
          [2001, 16, 0.1098820773152209, 0.005676868447283466, 0.5834966313339709, 15.900894465425358, 0.6873934119556147, 245, 1.0, 0.0001],
          [2001, 6, 0.009063639021328689, 7.5907354780847225, 0.42309014922493127, 36.818854130431916, 1.0, 42, 1.0, 0.0001],
          [2001, 9, 0.21495360551174472, 40.0, 0.9, 60.0, 1.0, 274, 1.0, 0.0001],
          [2001, 5, 0.13166483101370147, 14.736022402270784, 0.9, 43.38895885505714, 0.1, 5, 1.0, 0.0001],
          [2000, 4, 0.1566878432601928, 9.70047990971767e-05, 0.8151105858499775, 60.0, 1.0, 201, 1.0, 0.831264816846997],
          [2000, 1, 0.0035826116477342797, 40.0, 0.5999632095855046, 57.9869430211834, 0.1, 5, 1.0, 0.0001]
          ]

    y0 = [50.79536810793227, 
          18.262763081842575,
          47.40290204399224,
          41.54639911394005,
          40.04210909236722,
          52.39615227947167,
          45.25184610034129,
          49.403540678023646,
          41.39650183231052,
          -1.7828391223480373,
          51.586679760042166,
          -7.507853921676377,
          51.5534305513194,
          49.72988941354497,
          44.8806772881687,
          1.2635596878436035,
          23.737035231173667,
          29.233576353265626,
          50.81363015737173,
          -3.2232189626687626,
          20.90660304392635,
          19.959077216982116,
          52.79830652102921,
          -6.283777114437548,
          -8.358605153622099,
          -4.511108727951082,
          -8.106003681986078,
          -9.373290806076412,
          -8.179093659558648,
          -9.297132767005053,
          -1.1903745546054656,
          -7.766744731909195]





    print("checkpoint 1")
    OP = Optimizer(model_name, 
                   kind,
                   mode=0,
                   path="CatBoostHoldoutRetweet2",
                   path_log="CatBoostHoldoutRetweet2",
                   make_log=True, 
                   make_save=True, 
                   auto_save=True)

    print("checkpoint 2")
    OP.setParameters(n_calls=50, 
                     n_random_starts=20,
                     x0=x0,
                     y0=y0)
    print("checkpoint 3")
    #Before introducing datasets put the categorical features otherwise they will be ignored.
    OP.setCategoricalFeatures([4,5,6,11,12,13,43,44,45,46,47])
    print("checkpoint 4")
    OP.setParamsCAT(verbosity= 2,
                    boosting_type= "Plain",
                    model_shrink_mode= "Constant",
                    leaf_estimation_method= "Newton",
                    bootstrap_type= "Bernoulli",
                    early_stopping_rounds= 15)
    print("checkpoint 5")
    OP.loadTrainData(X_train, Y_train)
    print("checkpoint 6")
    OP.loadTestData(X_test, Y_test)
    print("checkpoint 7")
    OP.loadValData(X_val, Y_val)
    print("checkpoint 8")
    #OP.setParamsCAT(early_stopping_rounds=15)
    res=OP.optimize()



if __name__ == "__main__":
    main()