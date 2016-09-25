# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 13:56:38 2016

@author: ponomarevgeorgij
"""

import numpy as np

class Logit(object):
    def __init__(self,
                 k=0.1, 
                 C=10.0, 
                 regularization=True, 
                 tol=10**-5, 
                 max_iter=10000):
        self.k = k
        self.C = C
        self.regularization = regularization
        self.tol = tol
        self.max_iter = max_iter
        print "Logit(k =", self.k, \
                     ", C =", self.C, \
                     ", regularization =", self.regularization, \
                     ", tol =", self.tol, \
                     ", max_iter =", self.max_iter, ")"

    def _sigmoid(self, w, x):
        return 1.0/(1.0 + np.exp(-np.dot(x, w)))
        
    def _log_prime(self, y, w, x):
        return 1.0 - 1.0/(1.0 + np.exp(-np.multiply(y, np.dot(x, w))))
        
    def _update_weights(self,
                       weights, 
                       x_matrix, 
                       y_vector):
        new_weights = []
        for j in xrange(len(weights)):
            w_j = weights[j]
            x_j = x_matrix[:, j]
            new_w = w_j  + \
                    self.k * 1.0/len(y_vector) * \
                    np.dot(np.multiply(y_vector, x_j), self._log_prime(y_vector,
                                                                weights,
                                                                x_matrix))
                                                                
            if self.regularization:
                new_w = new_w - self.k * self.C * w_j
            
            new_weights.append(new_w)
        
        return np.array(new_weights)
    
    def fit(self, x, y, initial_weights=None):
        
        if initial_weights != None:
            self.weights = initial_weights
        else:
            self.weights = np.zeros(len(x[0]))
        
        weights_prev = self.weights
        self.weights = self._update_weights(self.weights, x, y)
        i = 1
        while np.linalg.norm(weights_prev-self.weights) > self.tol:
            weights_prev = self.weights
            self.weights = self._update_weights(self.weights, x, y)
            if i > self.max_iter:
                break
            i += 1
    
    def predict(self, x):
        return self._sigmoid(self.weights,x)
