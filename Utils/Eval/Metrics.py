import numpy as np
from sklearn.metrics import precision_recall_curve, auc, log_loss
#---------------------------------------------------
# In this class are present the methods which define
# the evaluation metrics used in the challenge.
#---------------------------------------------------

class ComputeMetrics(object):
    def __init__(self, pred, gt):
        self.pred=pred
        self.gt=gt
        
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