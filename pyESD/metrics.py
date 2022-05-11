# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:34:25 2022

@author: dboateng
"""

#importing models

import numpy as np

from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, explained_variance_score, max_error
from sklearn.metrics import mean_absolute_error, mean_squared_log_error


class Evaluate():
    
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred 
        
    def RMSE(self):
         error = (self.y_true - self.y_pred).dropna()
         score = np.sqrt(np.mean(error **2))
         print("RMSE: {:2f}".format(score))
         
         return score 
     
    def NSE(self):
        score = (1-(np.sum((self.y_pred - self.y_true)**2))/np.sum((self.y_true - np.mean(self.y_true))**2))
        
        print("Nach-Sutcliffe Efficiency(NSE): {:2f}".format(score))
        
        return score 
    
    def MSE(self):
        score = mean_squared_error(self.y_true, self.y_pred, squared=False)
        print("Mean Squared Error): {:.2f}".format(score))
        
        return score 
    
    
    def MAE(self):
        score = mean_absolute_error(self.y_true, self.y_pred)
        print("Mean Absolute Error): {:.2f}".format(score))
        
        return score 
    
    
    
    def R2_score(self):
        score = r2_score(self.y_true, self.y_pred)
        print("R² (Coefficient of determinaiton): {:.2f}".format(score))
        
        return score
    
    def explained_variance(self):
        
        score = explained_variance_score(self.y_true, self.y_pred)
        print("Explained Variance: {:.2f}".format(score))
        
        return score
    

    
    def max_error(self):
        
        score = max_error(self.y_true, self.y_pred)
        print("Maximum error: {:2f}".format(score))
        
        return score 
    
    def adjusted_r2(self):
        
        r2 = r2_score(self.y_true, self.y_pred)
        
        adj_r2 = (1 - (1- r2) * ((self.y_true.shape[0]- 1) / 
                  (self.y_true.shape[0] - self.y_true[1] -1)))
        
        print("Adjusted R²: {:.2f}".format(adj_r2))
        
        return adj_r2



