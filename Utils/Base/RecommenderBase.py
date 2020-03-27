from abc import abstractmethod
from abc import ABC
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, log_loss


class RecommenderBase(ABC):
    '''Defines the structure that all the models expose'''

    def __init__ (self, batch=False, name="recommenderbase", kind="NOT_GIVEN"):

        self.name = name
        self.batch = batch
        self.kind = kind

    @abstractmethod
    def fit(self):
        '''
        Fit the model on the data (train). 
        Inherited class should extend this method in appropriate way.
        '''
        pass

    @abstractmethod
    def evaluate(self):
        '''
        Compute the predictions then performs the evaluation of the predicted values. 
        Inherited class should extend this method in appropriate way.
        '''
        pass

    @abstractmethod
    def get_prediction(self):
        '''
        Compute the predictions without performing the evaluation. 
        Inherited class should extend this method in appropriate way.
        '''
        pass

    # Could have made save and load not abstract by using pickle, but in this
    # way specific model methods (that should be optimized) can be exploited
    @abstractmethod
    def save_model(self):
        '''
        Saves the trained model.
        Inherited class should extend this method in appropriate way.
        '''
        pass

    @abstractmethod
    def load_model(self):
        '''
        Load a compatible model. 
        Inherited class should extend this method in appropriate way.
        '''
        pass

    #----------------------------------------------------------------
    # Yet to be implemented, need a data class first
    #----------------------------------------------------------------
    # Commented the @abstractmethod to let the models work
    #@abstractmethod
    def load_train_set(self):
        '''
        Load an appropriately shaped training set for the model. 
        Inherited class should extend this method in appropriate way.
        '''
        pass

    #@abstractmethod
    def load_test_set(self):
        '''
        Load an appropriately shaped test set for the model. 
        Inherited class should extend this method in appropriate way.
        '''
        pass    
    #--------------------------------------------------------------------
    # Still to be implemented, need dictionaries
    #--------------------------------------------------------------------
    # The method to save predictions in a submittable form in a file can 
    # be implemented here
    def save_in_file(self):
        # Retrieve the path to the ditionaries and load them
        # Rebuild the form <usr_id><itm_id><score>
        # Save this shit into a file
        pass
    #--------------------------------------------------------------------


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



        

