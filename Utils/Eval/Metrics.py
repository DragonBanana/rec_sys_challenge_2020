import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, auc, log_loss
from math import log
import xgboost as xgb
#---------------------------------------------------
# In this class are present the methods which define
# the evaluation metrics used in the challenge.
#---------------------------------------------------

class ComputeMetrics(object):
    def __init__(self, pred, gt):
        self.pred=pred.astype(np.float64)
        self.gt=gt.astype(np.float64)
    
    #--------------------------------------------------
    #   Code snippet provided for the challenge
    #--------------------------------------------------
    def compute_prauc(self):
        prec, recall, thresh = precision_recall_curve(self.gt, self.pred)
        prauc = auc(recall, prec)
        return prauc

    def calculate_ctr(self, gt):
        positive = len([x for x in gt if x == 1])
        ctr = positive/float(len(gt))
        return ctr

    def compute_rce(self):
        cross_entropy = log_loss(self.gt, self.pred)
        data_ctr = self.calculate_ctr(self.gt)
        strawman_cross_entropy = log_loss(self.gt, [data_ctr for _ in range(len(self.gt))])
        return (1.0 - cross_entropy/strawman_cross_entropy)*100.0
    #--------------------------------------------------


    #--------------------------------------------------
    #               ABOUT CONFUSION MATRIX
    #--------------------------------------------------
    # labels:        Labels to index the matrix.
    # sample_weight: Sample weights (array).
    # normalize:     Normalizes the confusion matrix
    #                 over the true (rows), predicted
    #                 conditions or all the population.
    #                 If None, no normalization
    #---------------------------------------------------
    def confMatrix(self,
                   labels=None, 
                   sample_weight=None, 
                   normalize=None):

        pred = self.binarize(self.pred)
        return confusion_matrix(self.gt,      
                                pred, 
                                labels=labels, 
                                sample_weight=sample_weight, 
                                normalize=normalize)

    #Makes a probability array binary
    def binarize(self, to_bin):
        threshold = 0.5
        to_bin=np.array(to_bin)
        #Why are symbols inverted, dunno but it works
        to_bin = np.where(to_bin < threshold, to_bin, 1)
        to_bin = np.where(to_bin > threshold, to_bin, 0)
        return to_bin
    #--------------------------------------------------

    #Computes some statistics about the prediction
    def computeStatistics(self):
        return max(self.pred), min(self.pred), np.mean(self.pred)


class CustomEvalXGBoost:

    def __init__(self, every_x_round: int):

        assert every_x_round > 0, f"Parameter 'every_x_round' should be greater than 0"

        self.every_x_round = every_x_round
        self.counter = 0
        self.current_best = np.inf
        self.mode = "every_x_round"

    def custom_eval(self, predt: np.ndarray, dtrain: xgb.DMatrix):
        if self.mode == "every_x_round":
            if self.counter % self.every_x_round == 0:
                self.counter += 1
                eval_metric = float(self.logloss(predt.astype(np.float64), dtrain.get_label().astype(np.bool)))
                if eval_metric > self.current_best:
                    self.mode = "every_round"
                else:
                    self.current_best = eval_metric
                return 'custom_log_loss', eval_metric
            else:
                self.counter += 1
                return 'custom_log_loss', 1000
        else:
            self.counter += 1
            eval_metric = float(self.logloss(predt.astype(np.float64), dtrain.get_label().astype(np.bool)))
            if eval_metric < self.current_best:
                self.mode = "every_x_round"
                self.current_best = eval_metric
            return 'custom_log_loss', eval_metric



    def logloss(self, predicted, target):
        target = [float(x) for x in target]  # make sure all float values
        predicted = [min([max([x, 1e-15]), 1 - 1e-15]) for x in predicted]  # within (0,1) interval
        return -(1.0 / len(target)) * sum([target[i] * log(predicted[i]) + \
                                           (1.0 - target[i]) * log(1.0 - predicted[i]) \
                                           for i in range(len(target))])
