#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 00:55:22 2021

@author: dboateng
"""
import sys
import numpy as np
import pandas as pd
import seaborn as sns
from copy import copy
from collections import OrderedDict


try:
    from standardizer import MonthlyStandardizer, NoStandardizer
    from feature_selection import RecursiveFeatureElimination, TreeBasedSelection, SequentialFeatureSelection
    from models import Regressors
    from ensemble_models import EnsembleRegressor
    from metrics import Evaluate
    
except:
    from .standardizer import MonthlyStandardizer, NoStandardizer
    from .feature_selection import RecursiveFeatureElimination, TreeBasedSelection, SequentialFeatureSelection
    from .models import Regressors
    from .ensemble_models import EnsembleRegressor
    from .metrics import Evaluate
    
    
    
    
    
class PredictandTimeseries():
    
    def __init__(self, data, transform=None, standardizer=None):
        
        self.data = data
        self.transform = None
        self.standardizer = None

        if transform is not None:
            self.set_transform(transform)
        self.set_standardizer(standardizer)

            
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
            
    def set_model(self, method, ensemble_learning=False, estimators=None, cv=10, final_estimator_name=None):
        
        self.cv = cv
        
        if ensemble_learning == True:
            if method not in ["stacking", "voting"]:
                raise ValueError("ensemble method used either stacking or voting")
                
            
            if estimators is None:
                raise ValueError("...estimators list must be provided for ensemble models")
                
            else:
                self.estimators = estimators
            
            self.model = EnsembleRegressor(estimators=self.estimators, cv=self.cv, method=method, 
                                           final_estimator_name=final_estimator_name)
        else:
            
            self.model = Regressors(method=method, cv=self.cv)
            
    
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
            selector_regressor="Ridge", num_predictors=None, selector_direction=None, cal_relative_importance=False, 
            **predictor_kwargs):
        
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
            
            
            
        if cal_relative_importance == True:
            
            if not hasattr(self.model, "coef_"):
                raise ValueError("The estimator should have coef_attributes..or must be fitted before this method....")
                            
            else:
                
                coef_ = self.model.coef_
                score = self.model.score(X,y)
                residual = np.sqrt(1 - score)
                
                normalized_coef_ = coef_ * np.std(X, axis=0)
                
                total = residual + np.sum(np.abs(normalized_coef_))
                
                self.predictor_relative_contribution = normalized_coef_ / total
                
                
            
            
            
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
    
    
    def cross_validate_and_predict(self, datarange, predictor_dataset, **predictor_kwargs):
        
        X = self._get_predictor_data(datarange, predictor_dataset, **predictor_kwargs)
        
        y = self.get(datarange, anomalies=True)
        
        X = X.loc[~np.nan(y)]
        
        y = y.dropna()
        
        if hasattr(self, "selector"):
            X_selected = self.selector.transform(X)
            
            val_score = self.model.cross_validate(X_selected, y)
            fit_score = self.model.score(X_selected, y)
            y_pred = self.model.cross_val_predict(X_selected, y)
        else:
            fit_score = self.model.score(X, y)
            val_score = self.model.cross_validate(X, y)
            y_pred = self.model.cross_val_predict(X, y)
            
        scores = {"test_r2": np.mean(val_score["test_r2"]),
                  "test_r2_std": np.std(val_score["test_r2"]),
                  "train_r2": fit_score,
                  "test_rmse": np.mean(val_score["test_neg_root_mean_squared_error"]),
                  "test_rmse_std": np.std(val_score["test_neg_root_mean_squared_error"]), 
                  }
        return scores, y_pred
    
    def evaluate(self, datarnage, predictor_dataset, **predictor_kwargs):
        
        y_true = self.get(datarnage, anomalies=False)
        
        y_pred = self.predict(datarnage, predictor_dataset, anomalies=False, **predictor_kwargs)
        
        self.evaluate = Evaluate(y_true, y_pred)
        
        rmse = self.evaluate.RMSE()
        nse = self.evaluate.NSE()
        mse = self.evaluate.MSE()
        mae = self.evaluate.MAE()
        accuracy = self.evaluate.accuracy()
        exp_var = self.evaluate.explained_variance()
        r2 = self.evaluate.R2_score()
        max_error = self.evaluate.max_error()
        
        scores = {"RMSE": rmse,
                  "MSE": mse,
                  "NSE": nse,
                  "MAE": mae, 
                  "accuracy": accuracy,
                  "explained_variance": exp_var, 
                  "r2": r2, 
                  "max_error": max_error}
        
        return scores 
        
    def predictors_relative_performamce(self, datarange, predictor_datasets, **predictor_kwargs):
        
        if not hasattr(self, "model"):
            raise ValueError("...set model before fitting...")
            
        if not hasattr(self, "predictors"):
            raise ValueError("-----define predictor set first with set_predictors method....")
            
            
        
            
            
            
            
        
        
            
            
        
        
        
                
            
                
                    
                
            

            
        
            
        
        
        