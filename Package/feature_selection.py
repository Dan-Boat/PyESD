#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 17:18:14 2022

@author: dboateng
"""

# importing modules 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2, RFECV, f_classif, r_regression, mutual_info_regression, SelectFromModel, SequentialFeatureSelector
from sklearn.linear_model import Lasso, LassoCV, Ridge, BayesianRidge, ARDRegression
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, LeaveOneOut, LeaveOneGroupOut
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier



# class for recursive feature elimination 
class RecursiveFeatureElimination():
    
    def __init__(self, regressor_name):
        self.regressor_name = regressor_name
        self.cv = TimeSeriesSplit()
        self.n_jobs = -1
        self.min_features = 5
        self.scoring = "r2"
        
        
    def fit(self, X, Y):
        if self.regressor_name == "ARDRegression":
            self.estimator = ARDRegression()
        elif self.regressor_name == "BayesianRidge":
            self.estimator = BayesianRidge()
        elif self.regressor_name == "lasso":
            self.estimator = Lasso()
        elif self.regressor_name == "Ridge":
            self.estimator = Ridge()
        else:
            raise ValueError("Check the regressor if implemented")
        
        
        self.regressor = RFECV(estimator=self.estimator, scoring=self.scoring, cv=self.cv, n_jobs= self.n_jobs,
                      min_features_to_select=self.min_features).fit(X, Y)
        
    
    def print_selected_features(self, X):
        num_features = self.regressor.n_features_
        select_names = X.columns[self.regressor.support_]
        print("{0} : optimal number of predictors and selected variables are {1}".format(num_features, select_names))
    
    def transform(self, X):
        X_new = self.regressor.transform(X)
        return X_new
    
    def cv_test_score(self):
        cv_score = self.regressor.cv_results_["mean_test_score"].mean()
        return cv_score
    
    def score(self,X,Y):
        score = self.regressor.score(X,Y)
        return score
        

class TreeBasedSelection():
    pass

class SequentialFeatureSelection():
    pass