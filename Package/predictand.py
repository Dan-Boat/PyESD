#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 00:55:22 2021

@author: dboateng
"""

import numpy as np
import pandas as pd
import seaborn as sns
from copy import copy
from collections import OrderedDict

try:
    from .standardizer import MonthlyStandardizer, NoStandardizer
except:
    from standardizer import MonthlyStandardizer, NoStandardizer
    from feature_selection import RecursiveFeatureElimination, TreeBasedSelection, SequentialFeatureSelection
    
    
class PredictandTimeseries():
    
    def __init__(self, data, transform=None, standardizer=None, model=None):
        
        self.data = data
        self.transform = None
        self.standardizer = None

        if transform is not None:
            self.set_transform(transform)
        self.set_standardizer(standardizer)
        if model is not None:
            self.set_model(model)
            
    def get(self, daterange=None, anomalies=False):
        if anomalies:
            if daterange is not None:
                return self.data_st.loc[daterange]
            else:
                return copy(self.data_st)
        else:
            if daterange is not None:
                return self.data.loc[daterange]
            else:
                return copy(self.data)
            
    def set_transform(self, transform):
        if transform == "log":
            self.transform = np.log
            self.backtransform = np.exp
        else:
            raise NotImplementedError("Yet to implement additional transformers to log")
        self.data_st = self.transform(self.data)
        
    def set_standardizer(self, standardizer):
        if standardizer is not None:
            self.standardizer = standardizer
        else:
            self.standardizer = NoStandardizer()
        
        if self.transform is not None:
            self.data_st = self.transform(self.data)
            self.data_st = self.standardizer.fit_transform(self.data_st)
        else:
            self.data_st = self.standardizer.fit_transform(self.data)
            
    def set_model(self, model):
        pass
    
    def set_predictors(self, predictors):
        self.predictors = OrderedDict()
        for p in predictors:
            self.predictors[p.name] = p
            
    def _get_predictor_data(self, daterange, dataset, fit, **predictor_kwargs):
        Xs = []
        
        for p in self.predictors:
            Xs.append(self.predictors[p].get(daterange, dataset, fit, **predictor_kwargs))
            
        return pd.concat(Xs, axis=1)
    
    def fit(self, datarange, predictor_dataset, fit_predictors=True , predictor_selector=True, selector_method="Recursive",
            selector_regressor="Ridge", num_predictors=None, selector_direction=None, **predictor_kwargs):
        
        # checking attributes required before fitting
        
        if not hasattr(self, "model"):
            raise ValueError("...set model before fitting...")
            
        if not hasattr(self, "predictors"):
            raise ValueError("-----define predictor set first with set_predictors method....")
            
            
        X = self._get_predictor_data(datarange, predictor_dataset, fit_predictors, **predictor_kwargs)
        
        y = self.get(datarange, anomalies=fit_predictors)
        
        # dropna values 
        
        X = X.loc[~np.nan(y)]
        
        y = y.dropna()
        
        if predictor_selector ==True:
            
            if selector_method == "Reccursive":
                self.selector = RecursiveFeatureElimination(regressor_name=selector_regressor)
                
            elif selector_method == "TreeBased":
                self.selector == TreeBasedSelection(regressor_name=selector_regressor)
                
            elif selector_method == "Sequential":
                if num_predictors == None and selector_direction == None:
                    self.selector = SequentialFeatureSelection(regressor_name=selector_regressor)
                else:
                    self.selector = SequentialFeatureSelection(regressor_name=selector_regressor, 
                                                               n_features=num_predictors, direction=selector_direction)
                    
            else:
                raise ValueError("....selector method not recognized .....")
                
            self.selector.fit(X, y)
            
            X_selected = self.selector.transform(X)
            
            self.model.fit(X_selected, y)
            
        else:
            self.model.fit(X, y)
            
    def predict(self, datarange, predictor_dataset, anomalies=False, **predictor_kwargs):
        
        X = self._get_predictor_data(datarange, predictor_dataset, **predictor_kwargs)
        
        if not hasattr(self, "selector"):
            
            yhat = pd.Series(data=self.model.predict(X), index=datarange)
            
        else:
            X_selected = self.selector.transform(X)
            
            yhat = pd.Series(data=self.model.predict(X_selected), index=datarange)
            
        if anomalies == True:
            if self.standardizer is not None:
                yhat = self.standardizer.inverse_transform(yhat)
                
            if self.transform is not None:
                yhat = self.backtransform(yhat)
        
        return yhat
    
    
    def cross_validate(self, datarange, predictor_dataset, **predictor_kwargs):
        
        X = self._get_predictor_data(datarange, predictor_dataset, **predictor_kwargs)
        
        y = self.get(datarange, anomalies=True)
        
        X = X.loc[~np.nan(y)]
        
        y = y.dropna()
        
        
        
                
            
                
                    
                
            

            
        
            
        
        
        