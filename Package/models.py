#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:00:11 2022

@author: dboateng

"""

# imporitng modules 
import numpy as np 
import pandas as pd 


#from sklearn
from sklearn.linear_model import LassoCV, LassoLarsCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

#from local
from .splitter import Splitter, MonthlyBooststrapper


class MetaAttributes():
    def get_params(self):
        return self.estimator.get_params()
    
    def set_params(self, **params):
        return self.estimator.set_params()
    
    def alpha(self):
        return self.estimator.alpha_
    
    def best_params(self):
        return self.estimator.best_params_



class HyperparameterOptimize(MetaAttributes):
    
    def __init__(self, method, para_grid, regressor, scoring=None, cv=None):
        self.method = method
        self.para_grid = para_grid
        self.scoring = scoring
        self.cv = cv
        self.regressor = regressor
        
        if self.method == "GridSearchCV":
            self.estimator = GridSearchCV(estimator= self.regressor, param_grid=self.param_grid,
                                          scoring=self.scoring, cv=self.cv)
            
        elif self.method == "RandomizedSearchCV":
            self.estimator = RandomizedSearchCV(estimator= self.regressor, param_grid=self.param_grid,
                                          scoring=self.scoring, cv=self.cv)
    def fit(self, X, y):
        self.estimator.fit(X,y)
        
    def score(self, X,y):
        score = self.estimator.score(X,y)
        return score
    
    def transform(self, X):
        return self.estimator.transform(X)
    
    def predict_log_proba(self, X):
        return self.estimator.predict_log_proba(X)

class Regressors(MetaAttributes):
    pass
