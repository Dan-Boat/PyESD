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
import matplotlib.pyplot as plt
from copy import copy
from collections import OrderedDict
import scipy.stats as stats


try:
    from standardizer import MonthlyStandardizer, NoStandardizer
    from feature_selection import RecursiveFeatureElimination, TreeBasedSelection, SequentialFeatureSelection
    from models import Regressors
    from ensemble_models import EnsembleRegressor
    from metrics import Evaluate
    from MLR_model import BootstrappedForwardSelection, MultipleLSRegression
    
except:
    from .standardizer import MonthlyStandardizer, NoStandardizer
    from .feature_selection import RecursiveFeatureElimination, TreeBasedSelection, SequentialFeatureSelection
    from .models import Regressors
    from .ensemble_models import EnsembleRegressor
    from .metrics import Evaluate
    from .MLR_model import BootstrappedForwardSelection, MultipleLSRegression
    
    
    
    
    
class PredictandTimeseries():
    
    def __init__(self, data, transform=None, standardizer=None):
        
        self.data = data
        self.transform = None
        self.standardizer = None

        if transform is not None:
            self.set_transform(transform)
        self.set_standardizer(standardizer)

            
    def get(self, daterange=None, anomalies=True):
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
            
    def set_model(self, method, ensemble_learning=False, estimators=None, cv=10, final_estimator_name=None,
                  daterange=None, predictor_dataset=None, fit_predictors=True, 
                  scoring=["r2", "neg_root_mean_squared_error"], MLR_learning=False, **predictor_kwargs):
        
        self.cv = cv
        self.scoring = scoring
        
        if ensemble_learning == True:
            if method not in ["Stacking", "Voting"]:
                raise ValueError("ensemble method used either stacking or voting")
                
            
            if estimators is None:
                raise ValueError("...estimators list must be provided for ensemble models")
            
            regressors = []    
            for i in range(len(estimators)) :
                regressor = Regressors(method=estimators[i], cv=self.cv, scoring=self.scoring)
                
                regressor.set_model()
                
                if estimators[i] in ["MLP", "SVR"]:
                    
                    X = self._get_predictor_data(daterange=daterange, dataset=predictor_dataset, 
                                                 fit_predictors = fit_predictors,
                                                 **predictor_kwargs)
                    
                    y = self.get(daterange = daterange, anomalies=fit_predictors)
                    
                    X = X.loc[~np.isnan(y)]
                    
                    y = y.dropna()
                    
                    regressor.fit(X.values, y.values)
                
                regressors.append((estimators[i], regressor.estimator))
                    
                    
                    
                
                
            
            self.model = EnsembleRegressor(estimators=regressors, cv=self.cv, method=method, 
                                           final_estimator_name=final_estimator_name, 
                                           scoring=self.scoring)
        
        elif MLR_learning == True:
            if method == "OLSForward":
                self.model = BootstrappedForwardSelection(MultipleLSRegression(), cv=self.cv)
            
        else:
            
            self.model = Regressors(method=method, cv=self.cv, scoring=self.scoring)
            self.model.set_model()
            
    
    def set_predictors(self, predictors):
        self.predictors = OrderedDict()
        for p in predictors:
            self.predictors[p.name] = p
            
    def _get_predictor_data(self, daterange, dataset, fit_predictors=True, **predictor_kwargs):
        Xs = []
        
        for p in self.predictors:
            Xs.append(self.predictors[p].get(daterange, dataset, fit=fit_predictors))
            
        return pd.concat(Xs, axis=1)
    
    def predictor_correlation(self, daterange, predictor_dataset, fit_predictors=True, fit_predictand=True, 
                              method="pearson", use_scipy=False, **predictor_kwargs):
        
        X = self._get_predictor_data(daterange, predictor_dataset, fit_predictors=fit_predictors, 
                                     **predictor_kwargs)
        
        y = self.get(daterange, anomalies=fit_predictand)
        
        # dropna values 
        
        X = X.loc[~np.isnan(y)]
        
        y = y.dropna()
        
        if use_scipy:
            df_results = pd.DataFrame(index=np.arange(2), columns=X.columns)
            
            for column in X.columns:
                if method =="pearson":
                    corr = stats.pearsonr(y, X[column])
                    df_results[column][0] = corr[0]
                    df_results[column][1] = corr[1]
                    
                elif method == "spearman":
                    corr = stats.spearmanr(y, X[column])
                    df_results[column][0] = corr[0]
                    df_results[column][1] = corr[1]
                else:
                    raise ValueError("The defined method is not accurate")
                    
            return df_results
            
        else:
            corr = X.corrwith(other=y, axis=0, drop=True, method=method)
            
            corr = corr.to_frame()
            
            return  corr.T  # change the code with scipy stats in order to estimate the significance
            
            
    def fit_predictor(self, name, daterange, predictor_dataset):
        
        if type(name) == list:
            
            for n in name:
                self.predictors[n].fit(daterange, predictor_dataset)
        else:
            self.predictors[name].fit(daterange, predictor_dataset)
            
            
            
    
    def fit(self, daterange, predictor_dataset, fit_predictors=True , predictor_selector=True, selector_method="Recursive",
            selector_regressor="Ridge", num_predictors=None, selector_direction=None, cal_relative_importance=False,
            fit_predictand=True, impute=False, impute_method=None, impute_order=None, 
            **predictor_kwargs):
        
        # checking attributes required before fitting
        
        if not hasattr(self, "model"):
            raise ValueError("...set model before fitting...")
            
        if not hasattr(self, "predictors"):
            raise ValueError("-----define predictor set first with set_predictors method....")
            
            
        X = self._get_predictor_data(daterange, predictor_dataset, fit_predictors=fit_predictors, **predictor_kwargs)
        
        y = self.get(daterange, anomalies=fit_predictand)
        
        # dropna values 
        
        
        if impute == False:
            
            X = X.loc[~np.isnan(y)]
            
            y = y.dropna()
        else:
            if impute_method is None:
                raise ValueError("Enter the imputation method, i.e is either linear or spline ..")
            
            
            else:
                y = y.fillna(y.interpolate(method = impute_method, 
                                           order= impute_order))
            
        
        if predictor_selector ==True:
            
            if selector_method == "Recursive":
                self.selector = RecursiveFeatureElimination(regressor_name=selector_regressor)
                
            elif selector_method == "TreeBased":
                self.selector = TreeBasedSelection(regressor_name=selector_regressor)
                
            elif selector_method == "Sequential":
                if num_predictors == None and selector_direction == None:
                    self.selector = SequentialFeatureSelection(regressor_name=selector_regressor)
                else:
                    self.selector = SequentialFeatureSelection(regressor_name=selector_regressor, 
                                                               n_features=num_predictors, direction=selector_direction)
                    
            else:
                raise ValueError("....selector method not recognized .....")
            
                
            X = X.loc[~np.isnan(y)]  # because the imputation won't fill series of nan for more years (it just interpolate)
            
            y = y.dropna()
            
            
            self.selector.fit(X, y)
            self.selector.print_selected_features(X)
            
            
            
            X_selected = self.selector.transform(X)
            
            self.model.fit(X_selected, y)
            
        else:
            self.model.fit(X, y)
            
        # explained variance for OLS model
        if hasattr(self.model, "regressor"):
            if hasattr(self.model.regressor, "explained_variances"):
                self.explained_variance_predictors = self.model.regressor.explained_variances
                
                
            
        if cal_relative_importance == True:
            
            if not hasattr(self.model, "coef"):
                raise ValueError("The estimator should have coef_attributes..or must be fitted before this method....")
                            
            else:
                coef_ = self.model.coef()
                
                if predictor_selector == True:
                
                    score = self.model.score(X_selected,y)
                    residual = np.sqrt(1 - score)
                    normalized_coef_ = coef_ * np.std(X_selected, axis=0)
                else:
                    score = self.model.score(X,y)
                    residual = np.sqrt(1 - score)
                    normalized_coef_ = coef_ * np.std(X, axis=0)
                
                total = residual + np.sum(np.abs(normalized_coef_))
                
                self.predictor_relative_contribution = normalized_coef_ / total
                
                
            
            
            
    def predict(self, daterange, predictor_dataset, fit_predictors=True, fit_predictand=True,
                **predictor_kwargs):
        
        X = self._get_predictor_data(daterange, predictor_dataset, fit_predictors, **predictor_kwargs)
        
        if not hasattr(self, "selector"):
            
            yhat = pd.Series(data=self.model.predict(X), index=daterange)
            
        else:
            X_selected = self.selector.transform(X)
            
            yhat = pd.Series(data=self.model.predict(X_selected), index=daterange)
            
        if fit_predictand == False:
            if self.standardizer is not None:
                yhat = self.standardizer.inverse_transform(yhat)
                
            if self.transform is not None:
                yhat = self.backtransform(yhat)
        
        return yhat
    
    
    def cross_validate_and_predict(self, daterange, predictor_dataset, fit_predictand=True, return_cv_scores=False, 
                                   **predictor_kwargs):
        
        X = self._get_predictor_data(daterange, predictor_dataset, **predictor_kwargs)
        
        y = self.get(daterange, anomalies=fit_predictand)
        
        X = X.loc[~np.isnan(y)]
        
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
            
            
            
        # write code to work for every scoring variable
        if "r2" in self.scoring:
            
            scores = {"test_r2": np.mean(val_score["test_r2"]),
                      "test_r2_std": np.std(val_score["test_r2"]),
                      "train_r2": fit_score,
                      "test_rmse": -np.mean(val_score["test_neg_root_mean_squared_error"]),
                      "test_rmse_std": np.std(val_score["test_neg_root_mean_squared_error"]),
                      "test_mae": -np.mean(val_score["test_neg_mean_absolute_error"]),
                      "test_mae_std": np.std(val_score["test_neg_mean_absolute_error"]),
                      }
            
        else:
            
            scores = {"train_r2": fit_score,
                      "test_rmse": -np.mean(val_score["test_neg_root_mean_squared_error"]),
                      "test_rmse_std": np.std(val_score["test_neg_root_mean_squared_error"]),
                      "test_mae": -np.mean(val_score["test_neg_mean_absolute_error"]),
                      "test_mae_std": np.std(val_score["test_neg_mean_absolute_error"]),
                      "test_mae_precent": -np.mean(val_score["test_neg_mean_absolute_percentage_error"]),
                      "test_mae_precent_std": np.std(val_score["test_neg_mean_absolute_percentage_error"]),
                      }
            
            
        
        y_pred = pd.DataFrame({"obs": y,
                               "prediction" : y_pred})
        
        if return_cv_scores == True:
            return scores, y_pred, val_score
        
        else:
            return scores, y_pred
    
    
    def evaluate(self, daterange, predictor_dataset, fit_predictand=True, **predictor_kwargs):
        
        y_true = self.get(daterange, anomalies=fit_predictand)
        
        y_pred = self.predict(daterange, predictor_dataset, anomalies=fit_predictand, **predictor_kwargs)
        
        y_pred = y_pred.loc[~np.isnan(y_true)]
        
        y_true = y_true.dropna()
        
    
        self.evaluate = Evaluate(y_true, y_pred)
        
        rmse = self.evaluate.RMSE()
        nse = self.evaluate.NSE()
        mse = self.evaluate.MSE()
        mae = self.evaluate.MAE()
        exp_var = self.evaluate.explained_variance()
        r2 = self.evaluate.R2_score()
        max_error = self.evaluate.max_error()
        adj_r2 = self.evaluate.adjusted_r2()
        
        scores = {"RMSE": rmse,
                  "MSE": mse,
                  "NSE": nse,
                  "MAE": mae, 
                  "explained_variance": exp_var, 
                  "r2": r2, 
                  "max_error": max_error,
                  "adj_r2": adj_r2}
        
        return scores 
    
    def ensemble_transform(self, daterange, predictor_dataset, **predictor_kwargs):
        
        X = self._get_predictor_data(daterange, predictor_dataset, **predictor_kwargs)
        
        if not hasattr(self, "selector"):
            
            y_preds = self.model.transform(X)
            
        else:
            X_selected = self.selector.transform(X)
            
            y_preds = self.model.transform(X_selected)
        
        return y_preds
    
    
    def relative_predictor_importance(self):
        
        if not hasattr(self,"predictor_relative_contribution"):
            
            raise ValueError("The relative varince must be calculated during fit")
            
        return self.predictor_relative_contribution
    
    
    def selected_names(self):
        
        if not hasattr(self, "selector"):
            raise ValueError("Predictor selection must be defined when fitting the model")
            
        names = self.selector.select_names
        
        names = names.to_frame()
        
        return names.T
    
    
    
    def tree_based_feature_importance(self, daterange, predictor_dataset, fit_predictand=True, plot=False, **predictor_kwargs):
        
        if not hasattr(self, "selector"):
            raise ValueError("Predictor selection must be defined when fitting the model")
            
        # if not hasattr(self.selector, "feature_importance"):
        #     raise TypeError("the feature selector must be treebased")
            
        X = self._get_predictor_data(daterange, predictor_dataset, **predictor_kwargs)
        
        y = self.get(daterange, anomalies=fit_predictand)
        
        X = X.loc[~np.isnan(y)]
        
        y = y.dropna()
        
        
        return self.selector.feature_importance(X,y, plot=plot)
    
    
    def tree_based_feature_permutation_importance(self, daterange, predictor_dataset, fit_predictand=True, 
                                                  plot=False, **predictor_kwargs):
        
        if not hasattr(self, "selector"):
            raise ValueError("Predictor selection must be defined when fitting the model")
            
        # if not hasattr(self.selector.estimator, "feature_importances_"):
        #     raise TypeError("the feature selector must be treebased")
            
        X = self._get_predictor_data(daterange, predictor_dataset, **predictor_kwargs)
        
        y = self.get(daterange, anomalies=fit_predictand)
        
        
        X = X.loc[~np.isnan(y)]
        
        y = y.dropna()
        
        return self.selector.feature_importance(X,y, plot=plot)
    
    
    def climate_score(self, fit_period, score_period, predictor_dataset, **predictor_kwargs):
        """
        How much better the prediction for the given period is then the
        annual mean.

        Parameters
        ----------
        fit_period : pd.DatetimeIndex
            Range of data that should will be used for creating the reference prediction.
        score_period : pd.DatetimeIndex
            Range of data for that the prediction score is evaluated
        predictor_dataset : stat_downscaling_tools.Dataset
            The dataset that should be used to calculate the predictors
        predictor_kwargs : keyword arguments
            These arguments are passed to the predictor's get function

        Returns
        -------
        cscore : double
            Climate score (similar to rho squared). 1 for perfect fit, 0 for no
            skill, negative for even worse skill than mean prediction.
        """
        if isinstance(self.standardizer, MonthlyStandardizer):
            # the reference prediction is the seasonal cycle
            y_fit_period = self.get(fit_period, anomalies=False)
            new_standardizer = copy(self.standardizer)
            new_standardizer.fit(y_fit_period)
            zero_prediction = pd.Series(np.zeros(len(y_fit_period)), index=fit_period)
            mean_prediction = new_standardizer.inverse_transform(zero_prediction)
            climate_mse = np.mean((y_fit_period - mean_prediction).dropna()**2)
            y_predict_period = self.get(score_period, anomalies=False)
            # Anomalies need to be predicted and the anomalies converted using
            # the new standardizer, otherwise the prediction has more info than the reference
            yhat_anomalies = self.predict(score_period, predictor_dataset, anomalies=True, **predictor_kwargs)
            yhat = new_standardizer.inverse_transform(yhat_anomalies)
            prediction_mse = np.mean((y_predict_period - yhat).dropna()**2)
        else:
            # don't use this together with a transform
            y_fit = self.get(fit_period, anomalies=True).dropna()
            y_predict = self.get(score_period, anomalies=True)
            yhat = self.predict(score_period, predictor_dataset, anomalies=True, **predictor_kwargs)
            error = (y_predict - yhat).dropna()
            prediction_mse = np.mean(error**2)
            climate_mse = np.mean((np.mean(y_fit) - y_fit)**2)
        if climate_mse == 0:
            return 0.0             # identifier for such a case (avoid dividing by 0)
        else:       
            return 1 - prediction_mse/climate_mse
    
    
    
    
    
    
            
            
        
            
            
            
            
        
        
            
            
        
        
        
                
            
                
                    
                
            

            
        
            
        
        
        