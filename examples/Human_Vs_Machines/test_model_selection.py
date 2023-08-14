# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 17:57:37 2023

@author: dboateng
"""

import os 
import pandas as pd 
import numpy as np 
from copy import copy

from pyESD.feature_selection import RecursiveFeatureElimination, TreeBasedSelection, SequentialFeatureSelection
from pyESD.models import Regressors
from pyESD.ensemble_models import EnsembleRegressor
from pyESD.splitter import KFold
from pyESD.metrics import Evaluate
from pyESD.standardizer import StandardScaling
from pyESD.ESD_utils import store_csv, store_pickle
from pyESD.plot import correlation_heatmap
from pyESD.plot_utils import apply_style, correlation_data, count_predictors
import matplotlib.pyplot as plt




stationnames = ["Beograd", "Kikinda", "Novi_Sad", "Palic", "Sombor", "Sremska_Mitrovica", "Vrsac",
                "Zrenjanin"]


path_to_store = "C:/Users/dboateng/Desktop/Datasets/Station/Vojvodina_new/plots/model_selection"
path_to_data = "C:/Users/dboateng/Desktop/Datasets/Station/Vojvodina_new"


predictors = ['dswr', 'lftx', 'mslp', 'p__f', 'p__u', 'p__v', 'p__z', 'p_zh',
       'p5_f', 'p5_u', 'p5_v', 'p5_z', 'p500', 'p5zh', 'p8_f', 'p8_u', 'p8_v',
       'p8_z', 'p850', 'p8zh', 'pottmp', 'pr_wtr', 'prec', 'r500', 'r850',
       'rhum', 'shum', 'temp']



# read train_data 
def get_data(stationname):
    train_X = pd.read_csv(os.path.join(path_to_data, stationname, "train_X.csv"), parse_dates=["date"])
    train_y = pd.read_csv(os.path.join(path_to_data, stationname, "train_y.csv"), parse_dates=["date"])
    
    test_X = pd.read_csv(os.path.join(path_to_data, stationname, "test_X.csv"), parse_dates=["date"])
    test_y = pd.read_csv(os.path.join(path_to_data, stationname, "test_y.csv"), parse_dates=["date"])
    
    # set index
    X = train_X.set_index("date")
    y = train_y.set_index("date")
    
    y = y.squeeze()
    
    
    X = X.loc[~np.isnan(y)]
    y = y.dropna()
    
    # prepare test data 
    X_test = test_X.set_index("date")
    y_test = test_y.set_index("date")

    
    return X, y, X_test, y_test


def evaluate_test(y_true, y_pred):
        
    
        evaluate = Evaluate(y_true, y_pred)
        
        
        # add more metrics for variability and extreme
        rmse = evaluate.RMSE()
        nse = evaluate.NSE()
        mse = evaluate.MSE()
        mae = evaluate.MAE()
        max_error = evaluate.max_error()
        
        scores = {"RMSE": rmse,
                  "MSE": mse,
                  "NSE": nse,
                  "MAE": mae, 
                  "max_error": max_error,}
        
        return scores 
    
    
def cross_validate_predict(X,y, model, return_cv_scores=True):
    
    val_score = model.cross_validate(X, y)
    fit_score = model.score(X, y)
    y_pred = model.cross_val_predict(X, y)
    
    scores = {"test_r2": np.mean(val_score["test_r2"]),
                      "test_r2_std": np.std(val_score["test_r2"]),
                      "train_r2": fit_score,
                      "test_rmse": -np.mean(val_score["test_neg_root_mean_squared_error"]),
                      "test_rmse_std": np.std(val_score["test_neg_root_mean_squared_error"]),
                      "test_mae": -np.mean(val_score["test_neg_mean_absolute_error"]),
                      "test_mae_std": np.std(val_score["test_neg_mean_absolute_error"]),
                      }
    
    y_pred = pd.DataFrame({"obs": y,
                              "prediction" : y_pred})
       
    if return_cv_scores == True:
        return scores, y_pred, val_score
    
    else:
        return scores, y_pred
    
    
def model_fitting_and_predicting(station, estimator, ensemble_learning=False, cv=KFold(n_splits=10),
                                 scoring=None, base_estimators=None, final_estimator=None,
                                 fit_predictand=False):
    
    selector_method = "TreeBased"
    
    selector_regressor="RandomForest"
    selector = TreeBasedSelection(regressor_name=selector_regressor)
    
    
    X, y, X_test, y_test= get_data(station)
     
    y_train = copy(y)
    y_test_ = copy(y_test)
    
    # standardize 
    scaler = StandardScaling(method="standardscaler")
    scaler_x = scaler.fit(X)
    
    X_scaled = scaler_x.transform(X)
    
    X_test_scaled = scaler_x.transform(X_test)
    
    if fit_predictand:
        scaler_y = scaler.fit(y)
        y = scaler_y.transform(y)
        y_test = scaler_y.transform(y_test)
    
    
    
    selector.fit(X_scaled, y)
    X_selected = selector.transform(X_scaled)
    importance = selector.feature_importance(X_scaled, y, plot=True, fig_path=path_to_store, 
                                             fig_name=station + "_feature_importance.png", save_fig=True)
    
    X_test_selected = selector.transform(X_test_scaled)
    
    selector.print_selected_features(X_scaled)
    
    # implement model
    
    if ensemble_learning:
        # modify code if MLP or SVG is added becuase of hpyerparameter optimization
        model = EnsembleRegressor(estimators=base_estimators, cv=cv, method="Stacking", 
                                           final_estimator_name=final_estimator, 
                                           scoring=scoring)
        
    else:
        model = Regressors(method=estimator, cv=cv, scoring=scoring)
        
    model.set_model()
        
    model.fit(X_selected, y)
    score_fit, ypred_fit, scores_all = cross_validate_predict(X_selected,y, model, return_cv_scores=True)
    
    # yhat = pd.Series(data=model.predict(X_test_selected), index=y_test.index)
    # ypred = pd.Series(data=model.predict(X_selected), index=y.index)
    
    # if fit_predictand:
    #     yhat_ = scaler_y.inverse_transform(yhat)
    #     ypred_ = scaler_y.inverse_transform(ypred)
         
    # score = evaluate_test(y_true=y_test_["precipitation"], y_pred=yhat_)
    
    # predictions = pd.DataFrame({
    #     "obs_train": y_train["precipitation"],
    #     "obs_test": y_test_["precipitation"],
    #     "pred_train": ypred_,
    #     "pred_test": yhat_})
    
    #saving files
    store_pickle(station, "validation_score_" + estimator, score_fit, path_to_store)
    store_pickle(station, "CV_scores_" + estimator, scores_all, path_to_store)
    

estimators = ["LassoLarsCV", "ARD", "RandomForest", "XGBoost", "Bagging", "AdaBoost", "RidgeCV"]    

for estimator in estimators:
    for station in stationnames:    
    #station = stationnames[1]   
        model_fitting_and_predicting(station=station, estimator=estimator, ensemble_learning=False,
                                     scoring=["neg_root_mean_squared_error", "r2", "neg_mean_absolute_error"],
                                     fit_predictand=True)
        
        
    
    
    
    

     