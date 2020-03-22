import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, log_loss
import time
import pickle
import os.path

def main():
    #dataframe è uguale a ciò che importa il notebook (in questo caso limitato a 50k di record)
    dataframe = pd.read_csv('dataset.csv', sep='\x01', nrows=50000)
    
    #Creating a new dataframe with core informations: usr_id, twt_id, tmstp
    onehot = pd.DataFrame({'usr_id': dataframe["engaging_user_features_user_id"],
                           'twt_id': dataframe["tweet_features_tweet_id"],
                           'tmstp_rpl': dataframe["engagement_features_reply_engagement_timestamp"],
                           'tmstp_rtw': dataframe["engagement_features_retweet_engagement_timestamp"],
                           'tmstp_rtw_c': dataframe["engagement_features_with_comment_engagement_timestamp"],
                           'tmstp_lik': dataframe["engagement_features_like_engagement_timestamp"]})

    #Removing timestamps leaving 1/0 interactions
    #Setting to 1 non-null timestamps
    #onehot["tmstp_rpl"] = onehot["tmstp_rpl"].map(lambda x: x/x)
    #onehot["tmstp_rtw"] = onehot["tmstp_rtw"].map(lambda x: x/x)
    #onehot["tmstp_rtw_c"] = onehot["tmstp_rtw_c"].map(lambda x: x/x)
    #onehot["tmstp_lik"] = onehot["tmstp_lik"].map(lambda x: x/x)
    onehot[["tmstp_rpl", 
            "tmstp_rtw", 
            "tmstp_rtw_c", 
            "tmstp_lik"]] = onehot[["tmstp_rpl", 
                                    "tmstp_rtw", 
                                    "tmstp_rtw_c", 
                                    "tmstp_lik"]].applymap(lambda x: x/x)
    #Setting to 0 null timestamps
    onehot = onehot.fillna(0)

    #Remapping users and items to integer due to xgboost requirements
    #Preparing dictionaries
    #Users' dictionary
    u_id = onehot["usr_id"].unique()
    u_val = np.arange(len(u_id))
    u_dict = dict(zip(u_id, u_val))
    #Items' dictionary
    i_id = onehot["twt_id"].unique()
    i_val = np.arange(len(i_id))
    i_dict = dict(zip(i_id, i_val))
    #Applying the mapping
    onehot = onehot.replace({"usr_id": u_dict}) #It is possible to use other dictionaries
    onehot = onehot.replace({"twt_id": i_dict}) 

    #Saving the remapped dataframe and the dictionaries
    #onehot.to_csv("obj/onehot.csv", sep='\x01')
    #save_obj(u_dict, "u_dict")
    #save_obj(i_dict, "i_dict")
    
    #Loading everything back (For tuning purposes, 
    #in order to avoid to re-preprocess the data).
    #onehot = pd.read_csv("obj/onehot.csv", sep='\x01')
    #u_dict = load_obj("u_dict")
    #i_dict = load_obj("i_dict")
    

    #XGBoost part
    #Dividing the dataset splitting the column i need to predict from the others
    X = onehot[["usr_id", "twt_id", "tmstp_rpl", "tmstp_rtw", "tmstp_rtw_c"]].to_numpy()
    Y = onehot["tmstp_lik"].to_numpy()
    XGB = XGBoost(name="LIKE")

    XGB.train_model(X, Y)
    XGB.evaluate()

    '''
    X = onehot[["usr_id", "twt_id", "tmstp_rpl", "tmstp_rtw", "tmstp_lik"]].to_numpy()
    Y = onehot["tmstp_rtw_c"].to_numpy()
    XGB = XGBoost(name="RETWEET WITH COMMENT")
    XGB.train_model(X, Y)
    XGB.evaluate()

    X = onehot[["usr_id", "twt_id", "tmstp_rpl", "tmstp_lik", "tmstp_rtw_c"]].to_numpy()
    Y = onehot["tmstp_rtw"].to_numpy()
    XGB = XGBoost(name="RETWEET")
    XGB.train_model(X, Y)
    XGB.evaluate()

    X = onehot[["usr_id", "twt_id", "tmstp_lik", "tmstp_rtw", "tmstp_rtw_c"]].to_numpy()
    Y = onehot["tmstp_rpl"].to_numpy()
    XGB = XGBoost(name="REPLY")
    XGB.train_model(X, Y)
    XGB.evaluate()
    '''
    








class XGBoost(object):
    #---------------------------------------------------------------------------------------------------
    #n_rounds:      Number of rounds for boosting
    #param:         Parameters of the XGB model
    #name:          Name of the kind of prediction to print [LIKE, REPLY, REWTEET, RETWEET WITH COMMENT]
    #---------------------------------------------------------------------------------------------------
    #Not all the parameters are explicitated
    #PARAMETERS DOCUMENTATION:https://xgboost.readthedocs.io/en/latest/parameter.html
    #---------------------------------------------------------------------------------------------------

    def __init__(self, 
                 num_rounds = 10, 
                 param = {'objective': 'binary:logistic', #outputs the binary classification probability
                          'colsample_bytree': 0.5,
                          'learning_rate': 0.4,
                          'max_depth': 35, #Max depth per tree
                          'alpha': 0.01, #L1 regularization
                          'lambda': 0.01, #L2 regularization
                          'num_parallel_tree': 4, #Number of parallel trees
                         },
                 name = "NO_NAME_GIVEN"):
        
        #Inputs
        self.num_rounds = num_rounds
        self.param = param
        self.name = name

        #LOCAL VARIABLES
        #Model
        self.model = None
        #Evaluation
        self.test = None
        self.Y_test = None

    
    #--------------------------------------------------------------------------------------------------
    #                       train_model(...)
    #--------------------------------------------------------------------------------------------------
    #X:         Learning features of the dataset
    #Y:         Target feature of the dataset
    #X_tst:     To provide a personal test set's learning features
    #Y_tst:     To provide a personal test set's target feature
    #user_test:  Is the user providing a custom test set
    #tst_size:  In case user don't provide a custom test set, size of the split to create the test set
    #batch:     [NOT TESTED]Enable learning by batch
    #filename:  [NOT TESTED]To provide in order to save and load the model at each batch learning step
    #---------------------------------------------------------------------------------------------------
    def train_model(self, X, Y, X_tst=None, Y_tst=None, user_test=False, tst_size=0.2, batch=False, filename=None):

        #Splitting train and test set
        if user_test is True and (X_tst is not None) and (Y_tst is not None):
            #The user provides a custom test set
            print("You provided a test set.")
            X_train = X
            X_test = X_tst
            Y_train = Y
            Y_test = Y_tst
        else:
            #Test set gets extrapolated from the provided set
            #print("No test set was provided.")
            print("Test set will be split by your set with size {0}".format(tst_size))
            X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                                Y, 
                                                                test_size=tst_size, 
                                                                random_state=int(time.time()))

        #Learning in a single round
        if batch is False:
            #Transforming matrices in DMatrix type
            train = xgb.DMatrix(X_train, 
                                label=Y_train)
            test = xgb.DMatrix(X_test, 
                            label=Y_test)
     	
            #Defining and fitting the models
            self.model = xgb.train(self.param, 
                                   train, 
                                   self.num_rounds)
            #Variable used in evaluation
            self.Y_test = Y_test
            self.test = test


        #NOTA: self.model tiene traccia del modello all'interno della classe
        #non dovrebbe essere necessario salvarlo, ma trainarlo di nuovo con
        #un altro batch, da approfondire.
        #-------------------------NOT TESTED----------------------------
        #Learning by consecutive batches
        else:
            #Loading model if it does exist
            if filename is not None:
                if os.path.exists(filename):
                    model = load_obj(filename)
                else:
                    print("No previous model available.")
                #Transforming matrices in DMatrix type
                train = xgb.DMatrix(X_train, 
                                    label=Y_train)
                test = xgb.DMatrix(X_test, 
                                label=Y_test)
                #Defining and training the models
                self.model = xgb.train(self.param, 
                                       train, 
                                       self.num_rounds)
                #Saving the model
                save_obj(self.model, filename)


            else:
                print("Filename not valid.")
            
        #---------------------------------------------------------------



    #Call this function after train_model and it will provide you the evaluation
    #---------------------------------------------------------------------------
    #                           evaluate(...)
    #---------------------------------------------------------------------------
    #batch:             [NOT IMPLEMENTED]Enables batch evaluation
    #---------------------------------------------------------------------------
    def evaluate(self, batch=False):
        if (self.Y_test is None) and (batch is False):
            print("There is no available set.")
        elif batch is False:
            #Making predictions
            Y_pred = self.model.predict(self.test)

            #Evaluating
            prauc = self.compute_prauc(Y_pred, self.Y_test)
            rce = self.compute_rce(Y_pred, self.Y_test)
            print("PRAUC "+self.name+": {0}".format(prauc))
            print("RCE "+self.name+": {0}\n".format(rce))

        elif batch is True:
            None #***TO IMPLEMENT***

        return Y_pred

    #Evaluation metrics
    def compute_prauc(self, pred, gt):
        prec, recall, thresh = precision_recall_curve(gt, pred)
        prauc = auc(recall, prec)
        return prauc

    def calculate_ctr(self, gt):
        positive = len([x for x in gt if x == 1])
        ctr = positive/float(len(gt))
        return ctr

    def compute_rce(self, pred, gt):
        cross_entropy = log_loss(gt, pred)
        data_ctr = self.calculate_ctr(gt)
        strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))])
        return (1.0 - cross_entropy/strawman_cross_entropy)*100.0




#To save/load almost everything in pickle
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

   




if __name__ == "__main__":
    main()
