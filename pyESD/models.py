#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:00:11 2022

@author: dboateng

"""

# imporitng modules 
import numpy as np 
import pandas as pd 
import sys


#from sklearn
from sklearn.linear_model import LassoCV, LassoLarsCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import ARDRegression, BayesianRidge
from sklearn.linear_model import GammaRegressor, PoissonRegressor
from sklearn.linear_model import SGDRegressor, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict

from xgboost import XGBRegressor

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer


try:
    from .splitter import MonthlyBooststrapper
    
except:
    from splitter import MonthlyBooststrapper


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
    
    def __init__(self, method, param_grid, regressor, scoring="r2", cv=10):
        self.method = method
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.regressor = regressor
        
        if self.method == "GridSearchCV":
            self.hyper = GridSearchCV(estimator= self.regressor, param_grid=self.param_grid,
                                          scoring=self.scoring, cv=self.cv)
            
        elif self.method == "RandomizedSearchCV":
            self.hyper = RandomizedSearchCV(estimator= self.regressor, param_distributions=self.param_grid,
                                          scoring=self.scoring, cv=self.cv)
        elif self.method == "BayesSearchCV":
            self.hyper = BayesSearchCV(estimator=self.regressor, search_spaces=self.param_grid, scoring=self.scoring, 
                                       cv=self.cv,)
        else:
            raise ValueError("The defined hyperparameter search can't be recognized")
            
    def fit(self, X, y):
        self.hyper = self.hyper.fit(X,y)
        self.estimator = self.hyper.best_estimator_
        return self.estimator
        
    def score(self, X,y):
        score = self.hyper.score(X,y)
        return score
    
    def transform(self, X):
        return self.hyper.transform(X)
    
    def predict_log_proba(self, X):
        return self.hyper.predict_log_proba(X)
    
    def best_estimator(self):
        return self.hyper.best_estimator_
    
    def cross_val_score(self, X, y):
        return cross_val_score(self.hyper, X, y, cv=self.cv)
    
    def cross_validate(self, X, y):
        return cross_validate(self.hyper, X, y, scoring=["r2", "neg_root_mean_squared_error"],
                                n_jobs=2, verbose=0, cv=self.cv)
    
    
    def cross_val_predict(self, X, y):
        return cross_val_predict(self.estimator, X, y, n_jobs=2, verbose=0)
    
    
    

class Regressors(MetaAttributes):
    
    def __init__(self, method, cv=None, hyper_method=None, scoring=None):
        self.method = method
        self.cv = cv
        self.hyper_method = hyper_method
        self.scoring = scoring 
        
        
        if self.cv == None:
            
            print(".....Using monthly bootstrapper as default splitter....")
            self.cv = MonthlyBooststrapper(n_splits=500, block_size=12)
            
        self.selection = "random"
        
        if hyper_method == None:
            self.hyper_method = "GridSearchCV"
            
        
        if scoring == None:
            self.scoring= ["r2", "neg_root_mean_squared_error"]
            
        
    def set_model(self):
        
        # regression with variable selction 
        if self.method == "LassoCV":
            self.estimator = LassoCV(cv=self.cv, selection=self.selection)
            
        elif self.method == "LassoLarsCV":
            self.estimator = LassoLarsCV(cv=self.cv, normalize=False)
            
        elif self.method == "RidgeCV":
            self.estimator = RidgeCV(cv=self.cv, scoring="r2", alphas=[1e-3, 1e-2, 1e-1, 1, 10])
            
        
        #Bayesian regression algorithms 
        elif self.method == "ARD": # Automatic Relevance Determination regression 
            self.estimator = ARDRegression(n_iter=500)
            
        elif self.method == "BayesianRidge":
            self.estimator = BayesianRidge(n_iter=500)
        
        #Generalized Linear Models 
        elif self.method == "Gamma":
            self.estimator = GammaRegressor()
            
        elif self.method == "Poisson":
            self.estimator = PoissonRegressor()
            
        # Neural Networks models (Perceptron) 
        elif self.method == "MLP":
             regressor= MLPRegressor(random_state=42, max_iter=1000, early_stopping=True, batch_size=50,
                                     n_iter_no_change=20)
             param_grid = {"hidden_layer_sizes": [200,300], "alpha": [0.0001, 1.5, 2,5, 10],
                              "learning_rate": ["adaptive"], "solver": ["adam"]}
             self.hyper = HyperparameterOptimize(method=self.hyper_method , param_grid= param_grid, regressor=regressor)
             
        #Support Vector Machines (it very expensive to do hyperparamiter search)
        elif self.method == "SVR":
            regressor = SVR()
            if self.hyper_method == "BayesSearchCV":
                param_grid =  {"C":Real(0.1, 10, prior="uniform"),
                               "gamma": Real(0.1,10, prior="uniform"),
                             "kernel":Categorical(["linear", "poly", "rbf", "sigmoid"]), 
                             }
            else:
                param_grid = {"C":[0.1, 1, 10], "gamma":[0.1,1, 10], 
                             "kernel":["linear", "poly", "rbf", "sigmoid"]}
            
            self.hyper = HyperparameterOptimize(method=self.hyper_method, param_grid= param_grid, regressor=regressor)
        
        # Ensemble tree based algorithms    
        elif self.method == "RandomForest":
            self.estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
        elif self.method == "ExtraTree":
            self.estimator = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
         
        # Bagging    
        elif self.method == "Bagging":
            self.estimator = BaggingRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        #Boosting
        elif self.method == "AdaBoost":
            self.estimator = AdaBoostRegressor(n_estimators=100, loss= "linear", random_state=42)
            
        elif self.method == "HistGradientBoost":
            self.estimator = HistGradientBoostingRegressor(loss="squared_error", max_iter=200, validation_fraction=0.1,
                                                           random_state=42)
            
        elif self.method == "GradientBoost":
            self.estimator = GradientBoostingRegressor(loss="squared_error", n_estimators=200, validation_fraction=0.1,
                                                           random_state=42, criterion="friedman_mse")
            
        elif self.method == "XGBoost":
            self.estimator = XGBRegressor(n_estimators=100, n_jobs=-1, random_state=42)
            
        
        # gradient descent regression 
        elif self.method == "SGD":
            self.estimator = SGDRegressor(loss = "squared_error", max_iter=2000, early_stopping=True, 
                                          random_state=42, validation_fraction=0.1, learning_rate="invscaling",
                                         )
        else:
            raise ValueError("method name not recognized")
    
    def fit(self, X,y):
        if self.method == "MLP" or self.method=="SVR":
            self.estimator = self.hyper.fit(X,y)
            
            return self.estimator
        else:
            
            self.estimator = self.estimator.fit(X,y)
            return self.estimator

    
    def predict(self, X):
        yhat = self.estimator.predict(X)
        return yhat
    
    def score(self, X,y):
        return self.estimator.score(X,y)
    
    def cross_val_score(self, X, y):
        return cross_val_score(self.estimator, X, y, cv=self.cv, scoring=self.scoring)
    
    def cross_validate(self, X, y):
        return cross_validate(self.estimator, X, y, scoring=self.scoring,
                                n_jobs=2, verbose=0, cv=self.cv)
    
    def cross_val_predict(self, X, y):
        return cross_val_predict(self.estimator, X, y, n_jobs=2, verbose=0)
        
