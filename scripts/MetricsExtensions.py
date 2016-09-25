# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 16:38:02 2016

@author: ponomarevgeorgij
"""

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import operator

class MetricsExtensions(object):
    
    def __init__(self):
        pass
        
    def roc_auc_scores(self, y, df_pred):
        roc = {}
        for col in df_pred:
            roc[col] = roc_auc_score(y, df_pred[col])
        return roc
    
    def roc_curves(self, y, df_pred):
        fpr = {}
        tpr = {}
        thresholds = {}
        for col in df_pred:
            fpr[col], tpr[col], thresholds[col] = roc_curve(y, df_pred[col])
        return fpr, tpr, thresholds
    
    def plot_roc_curves(self, y, df_pred):
        fpr, tpr, _ = self.roc_curves(y, df_pred)
        for key in fpr:
            plt.plot(fpr[key], tpr[key], label=key)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.legend(loc="lower left")
    
    def pr_curves(self, y, df_pred):
        precision = {}
        recall = {}
        thresholds = {}
        for col in df_pred:
            precision[col], recall[col], thresholds[col] = \
                precision_recall_curve(y, df_pred[col])
        return precision, recall, thresholds
        
    def plot_pr_curves(self, y, df_pred):
        precision, recall, _ = self.pr_curves(y, df_pred)
        for key in precision:
            plt.plot(recall[key], precision[key], label=key)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.legend(loc="lower left")
    
    def max_precision_given_recall(self, 
                                   precision, 
                                   recall, 
                                   recall_boundary):
        max_prec_dict = {}    
        for key in precision:
            max_prec_dict[key] = 0
            for i in xrange(len(precision[key])):
                if recall[key][i] > recall_boundary \
                   and max_prec_dict[key] < precision[key][i]:
                    max_prec_dict[key] = precision[key][i]
        return max(max_prec_dict.iteritems(), key=operator.itemgetter(1))
    
    def max_recall_given_precision(self, 
                                   precision, 
                                   recall, 
                                   precision_boundary):
        max_rec_dict = {}
        for key in recall:
            max_rec_dict[key] = 0
            for i in xrange(len(recall[key])):
                if precision[key][i] > precision_boundary \
                   and max_rec_dict[key] < recall[key][i]:
                    max_rec_dict[key] = recall[key][i]
        return max(max_rec_dict.iteritems(), key=operator.itemgetter(1))
