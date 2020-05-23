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
        "tweet_feature_engagement_is_like"
    ]

    model_name = "catboost_classifier"
    kind = "like"

    print("checkpoint 1")
    # Load train data
    loading_data_start_time = time.time()
    X_train, Y_train = Data.get_dataset_xgb_batch(1, 0, train_dataset, X_label, Y_label, 0.50)
    print("checkpoint 2")
    # Load test data
    X_val, Y_val = Data.get_dataset_xgb_batch(2, 0, test_dataset, X_label, Y_label, 1)
    print("checkpoint 3")
    X_test, Y_test = Data.get_dataset_xgb_batch(2, 1, test_dataset, X_label, Y_label, 1)
    print(f"Loading data time: {time.time() - loading_data_start_time} seconds")

    #Defining prior knowledge like a n44hb

    x = [[2000, 15, 0.4871524055951343, 0.5656715705638427, 0.8907089079910582, 43.7085693615727, 0.25911385800112063, 299, 3.119289173299644, 0.03278516975122591],
        [2001, 15, 0.0008530550367783653, 7.302960890096405, 0.15568961304925105, 10.208420772573879, 0.500163479753602, 120, 1.7914684006924233, 0.35834711151371385],
        [2000, 16, 0.0008537298852093879, 0.017958185572953995, 0.8207215975343883, 7.061010474467003, 0.43889322667397823, 117, 1.5210657387584867, 0.7067252428247804],
        [2001, 11, 0.0017457932663813873, 0.0017060056183041046, 0.43345651350155967, 2.2336389989550116, 0.7152713670462895, 294, 0.9313859966629652, 0.0006947907186563163],
        [2001, 11, 0.0001777953564463653, 0.016453732126535062, 0.7427900082193563, 55.98545956452194, 0.7118332735344566, 202, 1.3461634764218264, 0.006360233961430392],
        [2001, 11, 0.0007418707332589263, 0.04931312723717593, 0.8525055029846814, 24.676169216629134, 0.6090612147725418, 123, 3.7927438996524394, 0.08193542132994312],
        [2001, 13, 0.6585904465671472, 0.0005923826773071334, 0.2311760793983376, 41.4556186952125, 0.4327203408557466, 223, 1.6062110799676828, 0.000261200639456434],
        [2000, 14, 0.03230442142538747, 0.008855480878883274, 0.38444941462540605, 37.657108032969504, 0.36111029773807213, 169, 2.3625872153966605, 0.14999823155432687],
        [2000, 1, 0.43698703689509405, 0.00014245107143395585, 0.8191067480021194, 33.17175734407814, 0.9424615939824077, 34, 4.652317090898776, 0.6776997804605254],
        [2001, 14, 0.0014775975994935338, 0.8698692673979297, 0.5099459999169196, 55.30457317116035, 0.2806688029777734, 300, 1.7727527886977097, 0.00012360210683420262],
        [2001, 4, 0.35257658821485455, 0.0002628082026599023, 0.5492233949543278, 57.19860049620661, 0.695558425757693, 152, 2.7503531137619186, 0.17321681737610645],
        [2000, 10, 0.0001262306337772611, 8.010896254047815, 0.8119644228657459, 14.991602268648878, 0.6040107559926972, 202, 4.721920512324119, 0.012845525044006876],
        [2001, 10, 0.007770459354153248, 2.744508797384035, 0.12169611161652885, 36.635339376732915, 0.42847840181895847, 136, 1.4360624778285183, 0.07226365938495935],
        [2001, 11, 0.0004369215022612467, 0.0006446940689978312, 0.6147153695481916, 11.006281311592048, 0.6363831519901049, 157, 1.6779075700176949, 0.0025767611845370943],
        [2001, 7, 0.025649087667512, 13.46595876608272, 0.5240813278239512, 35.53261010173976, 0.2160655190706158, 70, 4.018641281874657, 0.49182970154090433],
        [2000, 4, 0.006580264207676148, 0.0010986947743409203, 0.28372902098976316, 33.9163847825487, 0.16256466364574446, 138, 4.070238657304486, 1.0711259800165012],
        [2001, 11, 0.7827085791387419, 8.379734692359671, 0.1315854146198241, 7.453365024284106, 0.6833643747279499, 15, 1.4734273146096593, 0.05702237136514592],
        [2000, 11, 0.005576776276890345, 0.004047191528797974, 0.2829521837966531, 44.53752941171941, 0.5708317222487096, 95, 2.3608938187241844, 0.06323085313922015],
        [2000, 8, 0.00011115198217881887, 0.00012735664743145946, 0.36941434579168875, 4.705582052738818, 0.10756217225315592, 76, 0.8071508964842572, 0.5037827437218717],
        [2000, 2, 0.0026617424358546713, 0.008445222216533751, 0.2542481611341766, 50.665738098581606, 0.8125557865655153, 108, 0.5545333639521819, 0.0716256427161368],
        [2000, 10, 0.13689717952462585, 0.01775054473191654, 0.2921144999548442, 31.84334807504113, 0.3998466010900954, 227, 4.413193360679175, 1.2307458887905078],
        [2001, 11, 0.004731642272712688, 0.00172774765216224, 0.7092149813887147, 34.70668417935658, 0.9218744252128851, 22, 2.823884819953588, 0.0010121910993705905],
        [2001, 8, 0.009430257993131913, 0.010415199960315183, 0.1433212040817224, 57.21209116642497, 0.9261618505500998, 124, 2.5894640629632555, 0.02271854731898392],
        [2000, 11, 0.01809445434380731, 0.8067839601211205, 0.5010647253651234, 56.40886717459845, 0.47329033449850366, 297, 0.5799478838977947, 0.025629697910056433],
        [2000, 14, 0.0001678007851793733, 0.0005434503428529518, 0.8210908635725241, 47.74024918362391, 0.5791953844201266, 14, 3.587880476811137, 0.00043266279419454503],
        [2000, 14, 0.00011940219494331879, 0.0036005412141723227, 0.725469167129109, 47.65982848001294, 0.79978862287093, 89, 3.4046327629820925, 0.13141058100654535],
        [2000, 11, 0.00014446887196618909, 0.008674358344427795, 0.4033210256504741, 53.8387889514807, 0.38693745075501795, 186, 2.868408861563327, 0.49069106459591544],
        [2000, 6, 0.0001031633360873323, 0.018564496544022244, 0.4814786754653839, 8.323413971080983, 0.10777352642827154, 178, 2.040656107380236, 0.6133466889317915],
        [2000, 12, 0.0026543752991205058, 4.5516819338504995, 0.12367598294546502, 43.41893972811237, 0.597484127113699, 13, 2.955477087857908, 0.17093218056299],
        [2000, 12, 0.9677578837500417, 0.19090595105725397, 0.36179171588879055, 38.09451042748725, 0.49020114793309244, 244, 4.896515409952193, 0.0027140243074555183],
        [2000, 13, 0.06660252073444038, 0.22474478714870205, 0.22428398687159734, 13.095799028377684, 0.7271961080906401, 275, 1.7645312805060218, 0.0011523692464387172],
        [2000, 13, 0.0006671730226537297, 34.44745093219744, 0.5106028847804268, 4.349338564989327, 0.28812812741711524, 178, 1.5521580211671915, 0.2412177073476833],
        [2000, 5, 0.0010044264780255253, 0.0006871660140544731, 0.4966554195033194, 44.166506816285974, 0.8970911186009306, 37, 1.633011839425493, 0.9686446040696257],
        [2001, 10, 0.0007240852131543745, 7.69670926152243, 0.3018243624676564, 52.92644697192899, 0.5827450823681866, 24, 2.871197699994699, 0.000623876518110702],
        [2000, 10, 0.0002744997284635283, 27.599515334156745, 0.5531173556496066, 45.75161285214179, 0.13259369763178985, 7, 1.972017438144395, 0.016242087786285123],
        [2000, 16, 0.001678649113671199, 0.0001, 0.9, 0.0001, 1.0, 169, 0.5, 0.0001],
        [2001, 1, 0.0027600547632745604, 0.007100127362985965, 0.3245320018868526, 26.40230207977577, 0.7797581830765842, 183, 0.5, 0.08414416678074642],
        [2000, 2, 0.002723016735027696, 0.004497498635929627, 0.27257290513235455, 49.07877685530312, 0.7970425401228554, 123, 0.5006892097319181, 0.0328623992961772],
        [2000, 3, 0.002562728845596903, 0.010068016114541797, 0.23666796959766073, 50.390385843395784, 0.7097981462924772, 181, 0.5, 1.0908548161091427],
        [2000, 1, 0.0026754050387648998, 0.010488027438046717, 0.20348831476528081, 54.04544392422775, 0.8758108732376878, 98, 0.5061302293505278, 0.031041912386348035],
        [2000, 12, 0.0027893930023464124, 0.00012519131247764817, 0.6296176676407559, 43.70632054635287, 0.30399071338531425, 120, 0.5139337730795831, 0.05775658982075595],
        [2000, 2, 0.00253967509065564, 0.00010646699614719179, 0.4487500937273189, 42.11579359851562, 0.7782528833627118, 88, 0.6709138933525676, 0.1359093464765698],
        [2000, 3, 0.002509787677593961, 0.005123918097380717, 0.16689718879639004, 16.997355963254606, 0.7946105955170114, 108, 0.6800219970561485, 0.29364930464378797],
        [2000, 8, 0.0024088390777762306, 0.30333993160101913, 0.39888673196980984, 21.17722723527967, 0.7949289617134052, 136, 0.7241995315723275, 2.0],
        [2000, 4, 0.0019819997368356033, 0.00047727223709699877, 0.40001744061110966, 50.76035257293238, 0.8358702109587138, 7, 2.118780947938224, 0.046588999511741884],
        [2000, 3, 0.002552270107172929, 0.3373497258259249, 0.30645908191582394, 38.80711226422072, 0.7980861920930158, 144, 3.173127788951823, 0.4559347540108808],
        [2000, 2, 0.002479249404396684, 0.004052796467139349, 0.5901610425022673, 37.00553754794773, 0.7805462870696495, 62, 0.7019509101258722, 0.07029754721515619],
        [2000, 3, 0.0025437086781331117, 0.03100722583213938, 0.330008477969185, 28.183324247319163, 0.8067579698826793, 92, 0.6955516074593835, 0.0006816743335592363],
        [2000, 2, 0.0025417742203957543, 0.02793699637894308, 0.10604922317630608, 19.587098775865677, 0.7625195429448677, 212, 0.5841708045157882, 0.24697316965286253],
        [2000, 2, 0.0024765858762424773, 0.0006648780619791095, 0.47958837557726663, 51.30556672952366, 0.8030575779134773, 173, 0.7378775699297522, 0.34177393465813527],
        [2000, 2, 0.0026462689246714035, 0.0001, 0.527339632960931, 28.531560618498013, 0.38967382662617334, 96, 0.6541949714753166, 0.0011954391904032365],
        [2000, 3, 0.002754059696860472, 0.0007410049304754892, 0.7240355950192048, 4.276179819945341, 0.5470217961137024, 268, 0.608205838908422, 0.06794194569414662],
        [2000, 3, 0.0029580580474422163, 0.028265150992471615, 0.22670965848279273, 31.571955908157047, 0.7468953611347887, 82, 2.3353058819912453, 0.0005238272339350625],
        [2000, 3, 0.0026236443811213828, 1.185420733644226, 0.12367811181995685, 28.740573643563867, 0.9014477369242909, 115, 0.6414443724663856, 0.000348111798295613],
        [2000, 2, 0.002690904655405453, 0.03919976968173449, 0.3437191134967559, 25.73778875054701, 0.7931123726646152, 84, 0.6849001161815252, 0.0006620444039654532],
        [2000, 3, 0.002571158850856258, 0.00046721620486516276, 0.8090001165154918, 13.618829018832542, 0.9917510949397957, 38, 0.6857539023163828, 0.0021118065236363005],
        [2000, 3, 0.002475773049768476, 16.22878540095652, 0.24042860291395862, 39.28891287830051, 0.7721582348807732, 78, 0.7109367272765039, 0.0012543846782260707],
        [2000, 2, 0.0025628609764701995, 0.014552912502831182, 0.7002039776107423, 9.811751409375063, 0.278101199783399, 132, 0.6434630360880008, 0.057246643365792264],
        [2001, 6, 0.0015163836058244406, 0.0001, 0.3995261955753092, 0.0001, 0.9051649573491435, 204, 0.5, 0.0009045327918821139],
        [2000, 3, 0.0025307039766622715, 0.0005929699781047464, 0.4503511433694499, 18.587689225456046, 0.7664844390707904, 56, 0.6894441314767293, 0.0005923569999139181],
        [2000, 3, 0.0025373735575473775, 0.001290152405292928, 0.40140473713480307, 27.149959690275615, 0.7147987308768511, 264, 0.623682516343626, 0.00014695960257956898],
        ]
    y = [3.7397281866150203,
         -1.468065413659672,
         -2.7241662993724614,
         -5.046689608887146,
         -1.132122124343282,
         0.047518096557624866,
         1.6475564588128446,
         0.03869982489475988,
         5.422036573044347,
         -0.5991914691695456,
         1.3378602743002037,
         0.11566891122024506,
         -1.4545192109824077,
         -1.963771816337844,
         0.17994134976340306,
         1.1480502801795958,
         8.417875629815269,
         -0.42022991003184557,
         -0.8783179156732536,
         -6.546987987442397,
         1.4837604733119754,
         -0.03726294818677657,
         -0.015512208786538492,
         -3.784714874826633,
         0.14699076520160903,
         0.18557072784586504,
         0.09828089720398622,
         -0.15399287319396943,
         0.03428972734146929,
         18.72015504563272,
         -3.077725382485707,
         -2.281083241613381,
         -0.7392315113455079,
         0.1049134034119363,
         -0.8226247381258069,
         -2.601757919681432,
         -5.426509636149048,
         -6.379735160965008,
         -6.183587561310541,
         -5.258381650868321,
         -1.7908291356555464,
         -6.735784109448091,
         -6.874157036210871,
         -3.898332315971769,
         -0.596239278558316,
         0.591343375660326,
         -6.660114921934309,
         -6.914212107915035,
         -6.862073880480714,
         -6.384574515302933,
         -6.511703913955726,
         -6.782528733046639,
         -0.010641219622911575,
         -6.8057019455946435,
         -6.648626264734476,
         -6.460208985409169,
         -6.4355757848806885,
         -6.1083410390241735,
         -6.689433569274283,
         -6.887118031889887,
         -7.075982447696991
         ]

    print("checkpoint 1")
    OP = Optimizer(model_name, 
                   kind,
                   mode=0,
                   path="CatBoostHoldoutGLHF",
                   path_log="CatBoostHoldoutGLHF",
                   make_log=True, 
                   make_save=False, 
                   auto_save=True)

    print("checkpoint 2")
    OP.setParameters(n_calls=50, 
                     n_random_starts=20,
                     x0=x,
                     y0=y)
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