#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 17:18:14 2022

@author: dboateng
"""

# importing modules 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV, SelectFromModel, SequentialFeatureSelector
from sklearn.linear_model import Lasso, LassoCV, Ridge, BayesianRidge, ARDRegression
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, LeaveOneOut, LeaveOneGroupOut
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier

from pyESD.plot_utils import apply_style



# class for recursive feature elimination 
class RecursiveFeatureElimination():
    
    def __init__(self, regressor_name="ARD"):
        self.regressor_name = regressor_name
        self.cv = TimeSeriesSplit()
        self.n_jobs = -1
        self.min_features = 5
        self.scoring = "r2"
        
        
        if self.regressor_name == "ARD":
            self.estimator = ARDRegression()
        elif self.regressor_name == "BayesianRidge":
            self.estimator = BayesianRidge()
        elif self.regressor_name == "lasso":
            self.estimator = Lasso()
        elif self.regressor_name == "lassocv":
            self.estimator = LassoCV()
        elif self.regressor_name == "Ridge":
            self.estimator = Ridge()
        elif self.regressor_name == "RandomForest":
            self.estimator = RandomForestRegressor()
        else:
            raise ValueError("Check the regressor if implemented")
        
        
    def fit(self, X, y):
        
        self.regressor = RFECV(estimator=self.estimator, scoring=self.scoring, cv=self.cv, n_jobs= self.n_jobs,
                      min_features_to_select=self.min_features).fit(X, y)
        
    
    def print_selected_features(self, X):
        num_features = self.regressor.n_features_
        select_names = X.columns[self.regressor.support_]
        
        self.select_names = select_names
        
        print("{0} : optimal number of predictors and selected variables are {1}".format(num_features, select_names))
    
    def transform(self, X):
        X_new = self.regressor.transform(X)
        return X_new
    
    def cv_test_score(self):
        cv_score = self.regressor.cv_results_["mean_test_score"].mean()
        return cv_score
    
    def score(self,X,y):
        score = self.regressor.score(X,y)
        return score
        
        

class TreeBasedSelection():
    def __init__(self, regressor_name="RandomForest"):
        self.regressor_name = regressor_name
        self.n_jobs = -1
        self.bootstrap = True
        self.criterion = "squared_error"
        self.scoring = "r2"
        
        if self.regressor_name == "RandomForest":
            self.estimator = RandomForestRegressor(n_jobs=self.n_jobs, criterion=self.criterion, bootstrap=self.bootstrap, 
                                                   n_estimators=200)
        elif self.regressor_name == "ExtraTree":
            self.estimator = ExtraTreesRegressor(n_jobs=self.n_jobs, criterion=self.criterion, bootstrap=self.bootstrap)
        else:
            raise ValueError("Tree regressor estimator is not defined properly")
            
        
    def fit(self, X, y):
        
        self.regressor = SelectFromModel(estimator=self.estimator, prefit=False).fit(X,y)
        
    def transform(self, X):
        X_new = self.regressor.transform(X)
        return X_new
    
    def feature_importance(self, X,y, plot=False, fig_path=None, fig_name=None, save_fig=False, station_name=None):
        self.estimator.fit(X,y)
        importance = self.estimator.feature_importances_
        feature_names = X.columns
        forest_importances = pd.Series(importance, index=feature_names)
        
        if plot == True:
            apply_style(fontsize=28, style="seaborn-talk", linewidth=3, usetex=False)  
            std = np.std([tree.feature_importances_ for tree in self.estimator.estimators_], axis=0)
            fig,ax = plt.subplots(figsize=(15, 13))
            forest_importances.plot.bar(yerr=std, ax=ax)
            if station_name is not None:
                ax.set_title("Feature importances using tree regressor (" + station_name + ")",
                             fontweight="bold", fontsize=24)
                
            else:
                ax.set_title("Feature importances using tree regressor")
                
            ax.set_ylabel("Mean Decrease in impurity", fontweight="bold", fontsize=24)
            fig.tight_layout()
            if save_fig:
                plt.savefig(os.path.join(fig_path, fig_name), bbox_inches="tight", format= "png")
                
            else:
                plt.show()
            
        return forest_importances
    
    def permutation_importance_(self, X,y, plot=False, fig_path=None, fig_name=None, save_fig=False):
        self.estimator.fit(X,y)
        importance = permutation_importance(estimator=self.estimator, X=X, y=y, scoring=self.scoring,
                                            n_repeats=10, n_jobs=self.n_jobs)
        sorted_idx = importance.importance_mean.argsort()
        if plot == True:
            fig,ax = plt.subplots(figsize=(15, 13))
            ax.boxplot(importance.importances[sorted_idx].T, vert=False, labels=X.columns[sorted_idx])
            ax.set_title("Permutation Importances (On test data)")
            ax.set_ylabel("Mean accuracy decrease", fontweight="bold", fontsize=20)
            fig.tight_layout()
            if save_fig:
                plt.savefig(os.path.join(fig_path, fig_name), bbox_inches="tight", format= "png")
                
            else:
                plt.show()
            
        return sorted_idx
            
    
    def print_selected_features(self, X):
        select_names = X.columns[self.regressor.get_support(indices=True)]
        num_features = len(select_names)
        print("{0} : optimal number of predictors and selected variables are {1}".format(num_features, select_names))
        
        self.select_names = select_names
        

class SequentialFeatureSelection():
    def __init__(self, regressor_name = "Ridge", n_features= 10, direction="forward"):
        self.regressor_name = regressor_name
        self.n_features =n_features
        self.scoring = "r2"
        self.direction = direction
        
        if self.regressor_name == "ARD":
            self.estimator = ARDRegression()
        elif self.regressor_name == "BayesianRidge":
            self.estimator = BayesianRidge()
        elif self.regressor_name == "lasso":
            self.estimator = Lasso()
        elif self.regressor_name == "lassocv":
            self.estimator = LassoCV()
        elif self.regressor_name == "Ridge":
            self.estimator = Ridge()
        else:
            raise ValueError("Check the regressor if implemented")
            
    def fit(self, X,y):
        self.regressor = SequentialFeatureSelector(estimator=self.estimator, n_features_to_select=self.n_features, scoring=self.scoring,
                                                   direction=self.direction).fit(X,y)
        
    def score(self,X,y):
        score = self.regressor.score(X,y)
        return score
    
    def transform(self, X):
        X_new = self.regressor.transform(X)
        return X_new  
    
    def print_selected_features(self, X):
        select_names = X.columns[self.regressor.get_support(indices=True)]
        num_features = len(select_names)
        print("{0} : optimal number of predictors and selected variables are {1}".format(num_features, select_names))
        
        
        self.select_names = select_names
    
