import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, log_loss
import time
import pickle
import os.path

def main():
    '''
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
    #onehot.to_csv("onehot.csv", sep='\x01')
    #save_obj(u_dict, "u_dict")
    #save_obj(i_dict, "i_dict")
    '''

    #Loading everything back (For tuning purposes, 
    #in order to avoid to re-preprocess the data).
    onehot = pd.read_csv("onehot.csv", sep='\x01')
    #u_dict = load_obj("u_dict")
    #i_dict = load_obj("i_dict")
    
    #XGBoost part
    test_size = 0.2
    #Dividing the dataset splitting the column i need to predict from the others
    X = onehot[["usr_id", "twt_id", "tmstp_rpl", "tmstp_rtw", "tmstp_rtw_c"]].to_numpy()
    Y = onehot["tmstp_lik"].to_numpy()
    XGB = XGBoost(name="LIKE")

    X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                        Y, 
                                                        test_size=test_size, 
                                                        random_state=int(time.time()))

    XGB.train_model(X_train, Y_train)
    XGB.evaluate(X_test, Y_test)

    
    X = onehot[["usr_id", "twt_id", "tmstp_rpl", "tmstp_rtw", "tmstp_lik"]].to_numpy()
    Y = onehot["tmstp_rtw_c"].to_numpy()
    XGB = XGBoost(name="RETWEET WITH COMMENT")
    X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                        Y, 
                                                        test_size=test_size, 
                                                        random_state=int(time.time()))

    XGB.train_model(X_train, Y_train)
    XGB.evaluate(X_test, Y_test)

    X = onehot[["usr_id", "twt_id", "tmstp_rpl", "tmstp_lik", "tmstp_rtw_c"]].to_numpy()
    Y = onehot["tmstp_rtw"].to_numpy()
    XGB = XGBoost(name="RETWEET")
    X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                        Y, 
                                                        test_size=test_size, 
                                                        random_state=int(time.time()))

    XGB.train_model(X_train, Y_train)
    XGB.evaluate(X_test, Y_test)

    X = onehot[["usr_id", "twt_id", "tmstp_lik", "tmstp_rtw", "tmstp_rtw_c"]].to_numpy()
    Y = onehot["tmstp_rpl"].to_numpy()
    XGB = XGBoost(name="REPLY")
    X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                        Y, 
                                                        test_size=test_size, 
                                                        random_state=int(time.time()))

    XGB.train_model(X_train, Y_train)
    XGB.evaluate(X_test, Y_test)

    #---------------------------------------------------------------------------------------
    #LEARNING PHASE
    XGB = XGBoost(name="LIKE_BATCH_EXAMPLE", batch=True)
    #Fetching 4 batch of size 10k each
    for i in range(4):
        size = 10**4
        step = i*size
        #Using 10k of rows at time
        onehot = pd.read_csv("onehot.csv", 
                             sep='\x01', 
                             skiprows = range(1, step), 
                             nrows = size)
        #For each iteration split the DataFrame
        X = onehot[["usr_id", "twt_id", "tmstp_rpl", "tmstp_rtw", "tmstp_rtw_c"]].to_numpy()
        Y = onehot["tmstp_lik"].to_numpy()
        #For each iteration train the model
        XGB.train_model(X, Y)    
    
    #EVALUATION PHASE
    #Fetching the last batch of size 10k to use it as test set
    step = 40000
    onehot = pd.read_csv("onehot.csv", 
                             sep='\x01', 
                             skiprows = range(1, step), 
                             nrows = size)
    #Splitting the dataframe
    X = onehot[["usr_id", "twt_id", "tmstp_rpl", "tmstp_rtw", "tmstp_rtw_c"]].to_numpy()
    Y = onehot["tmstp_lik"].to_numpy()
    #Providing the evaluation for the current test set
    XGB.evaluate(X_tst=X, Y_tst=Y)
    #---------------------------------------------------------------------------------------





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
                 name = "NO_NAME_GIVEN",
                 batch = False):
        
        #Inputs
        self.num_rounds = num_rounds
        self.param = param
        self.name = name
        self.batch = batch  #The type of learning is now explicitated when you declare the model

        #LOCAL VARIABLES
        #Model
        self.sround_model = None
        self.batch_model = None

            
    
    #------------------------------------------------
    #               train_model(...)
    #------------------------------------------------
    #X:         Learning features of the dataset
    #Y:         Target feature of the dataset
    #batch:     Enable learning by batch
    #------------------------------------------------
    # sround_model and batch_model are differentiated
    # in order to avoid overwriting.
    #------------------------------------------------
    def train_model(self, X, Y):

        #Learning in a single round
        if self.batch is False:
            #Transforming matrices in DMatrix type
            train = xgb.DMatrix(X, 
                                label=Y)
     	
            #Defining and fitting the models
            self.sround_model = xgb.train(self.param, 
                                   train, 
                                   self.num_rounds)
            
        #Learning by consecutive batches
        else:
            #Transforming matrices in DMatrix type
            train = xgb.DMatrix(X, 
                                label=Y)
            #Defining and training the models
            self.batch_model = xgb.train(self.param, 
                                         train, 
                                         self.num_rounds,
                                         xgb_model=self.batch_model)



    #Call this function after train_model and it will provide you the evaluation
    #---------------------------------------------------------------------------
    #                           evaluate(...)
    #---------------------------------------------------------------------------
    #X_tst:     Features of the test set
    #Y_tst      Ground truth, target of the test set
    #---------------------------------------------------------------------------
    #           Works for both for batch and single training
    #---------------------------------------------------------------------------
    def evaluate(self, X_tst=None, Y_tst=None):
        Y_pred = None
        if (X_tst is None) or (Y_tst is None):
            print("No test set is provided.")
        else:
            #Selecting the coherent model for the evaluation
            #According to the initial declaration (batch/single round)
            if self.batch is False:
                model = self.sround_model
            else:
                model = self.batch_model
            
            #Preparing DMatrix
            d_test = xgb.DMatrix(X_tst,
                                 label = Y_tst)

            #Making predictions
            Y_pred = model.predict(d_test)

            #Evaluating
            prauc = self.compute_prauc(Y_pred, Y_tst)
            rce = self.compute_rce(Y_pred, Y_tst)
            print("PRAUC "+self.name+": {0}".format(prauc))
            print("RCE "+self.name+": {0}\n".format(rce))            

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
