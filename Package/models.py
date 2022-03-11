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
from sklearn.linear_model import ARDRegression, BayesianRidge
from sklearn.linear_model import GammaRegressor, PoissonRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict

#from local
from .splitter import MonthlyBooststrapper


class MetaAttributes():
    def get_params(self):
        return self.estimator.get_params()
    
    def set_params(self, **params):
        return self.estimator.set_params()
    
    def alpha(self):
        return self.estimator.alpha_
    
    def best_params(self):
        return self.estimator.best_params_
    
    def best_estimator(self):
        return self.estimator.best_estimstor_
    
    def coef(self):
        return self.estimator.coef_
    
    def intercept(self):
        return self.estimator.intercept_



class HyperparameterOptimize(MetaAttributes):
    
    def __init__(self, method, param_grid, regressor, scoring=None, cv=None):
        self.method = method
        self.param_grid = param_grid
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
    
    def __init__(self, method, cv=None):
        self.method = method
        self.cv = cv
        if self.cv == None:
            
            print(".....Using monthly bootstrapper as default splitter....")
            cv = MonthlyBooststrapper(n_splits=500, block_size=12)
            
        self.selection = "random"
        
        
    def set_model(self):
        
        # regression with variable selction 
        if self.method == "LassoCV":
            self.estimator = LassoCV(cv=self.cv, selection=self.selection)
            
        elif self.method == "LassoLarsCV":
            self.estimator = LassoLarsCV(cv=self.cv, normalize=False)
        
        #Bayesian regression algorithms 
        elif self.method == "ARD": # Automatic Relevance Determination regression 
            self.estimator = ARDRegression(n_iter=500)
            
        elif self.method == "BayesianRidge":
            self.estimator = BayesianRidge(n_iter=500)
        
        #Generalized Linear Models 
        elif self.method == "GammaRegressor":
            self.estimator = GammaRegressor()
        elif self.method == "PoissonRegressor":
            self.estimator = PoissonRegressor()
            
        # Neural Networks models (Perceptron) 
        elif self.method == "MLPRegressor":
             regressor= MLPRegressor(random_state=42, max_iter=1000, early_stopping=False)
             param_grid = {"hidden_layer_sizes": [1,100,200,300], "alpha": [0.0001, 0.5, 1, 1.5, 2,5, 10],
                              "learning_rate": ["constant", "adaptive"], "solver": ["adam", "sgd"]}
             self.estimator = HyperparameterOptimize(method="GridSearchCV", param_grid= param_grid, regressor=regressor)
             
        #Support Vector Machines
        elif self.method == "SVR":
            regressor = SVR()
            param_grid = {"svr__C":[0.1, 1, 10], "svr__gamma":["auto", 1, 0.1, 0.01, 0.001, 0.0001, 0.2, 0.5, 0.9, 10], 
                         "svr__kernel":["rbf", "poly"]}
            self.estimator = HyperparameterOptimize(method="RandomizedSearchCV", param_grid= param_grid, regressor=regressor)
        
        # Ensemble tree based algorithms    
        elif self.method == "RandomForest":
            self.estimator = RandomForestRegressor(n_estimators=100, random_state=42)
            
        elif self.method == "ExtraTree":
            self.estimator = ExtraTreesRegressor(n_estimators=100, random_state=42)
            
            
        return self.estimator
    
    def fit(self, X,y):
        return self.estimator.fit(X,y)

    
    def predict(self, X):
        yhat = self.estimator.predict(X)
        return yhat
    
    def score(self, X,y):
        return self.estimator.score(X,y)
    
    def cross_val_score(self, X, y):
        return cross_val_score(self.estimator, X, y, cv=self.cv)
    
    def cross_validate(self, X, y):
        return cross_validate(self.estimator, X, y, scoring=["r2", "neg_root_mean_squared_error"],
                                n_jobs=2, verbose=0, cv=self.cv)
    
    
    def cross_val_predict(self, X, y):
        return cross_val_predict(self.estimator, X, y, n_jobs=2, verbose=0)
        
