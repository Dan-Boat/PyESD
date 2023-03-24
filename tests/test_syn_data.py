# -*- coding: utf-8 -*-
"""
Created on Thu May 12 14:57:29 2022

@author: dboateng
"""
import numpy as np
import pandas as pd 

from pyESD.Predictor_Base import Predictor

def generate_syn_data():
    

    np.random.seed(0)
    
    daterange = pd.date_range(start="1979-01-01", end="2000-01-01", freq="MS")
    
    X = np.random.randn(len(daterange), 5)
    
    # offset one predictor by 20
    
    X[:, 2] += 20
    
    #define factors
    
    m = np.asarray([0, 10, 20, -20, -5], dtype=float)
    
    y = 4 + X.dot(m) + 0.01*np.random.randn(len(daterange))
    
    # passing data into dataframe
    
    X = pd.DataFrame(X, index=daterange)
    y = pd.Series(y, index=daterange)
    
    return X , y


class Syn_predictor(Predictor):
    
    def __init__(self, X, i, **kwargs):
        self.X = X.iloc[:, i]
        super().__init__(str(i), **kwargs)
        
    def _generate(self, daterange, dataset, fit, fitted_with, fitted_from):
        return self.X[daterange]
    
    

    
    