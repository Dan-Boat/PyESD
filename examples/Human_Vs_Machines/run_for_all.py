# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 12:57:54 2023

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


path_to_store = "C:/Users/dboateng/Desktop/Datasets/Station/Vojvodina_new/plots/predicted"
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
    
    
    #join data
    X_all = pd.concat([train_X, test_X])
    y_all = pd.concat([train_y, test_y])
    
    X_all = X_all.set_index("date")
    y_all = y_all.set_index("date")
    
    y_all = y_all.squeeze()
    
    
    X_all = X_all.loc[~np.isnan(y_all)]
    y_all = y_all.dropna()
    
    # set index
    X = train_X.set_index("date")
    y = train_y.set_index("date")
    
    y = y.squeeze()
    
    
    X = X.loc[~np.isnan(y)]
    y = y.dropna()
    
    # prepare test data 
    X_test = test_X.set_index("date")
    y_test = test_y.set_index("date")

    
    return X, y, X_test, y_test, X_all, y_all


def evaluate_test(y_true, y_pred):
        
    
        evaluate = Evaluate(y_true, y_pred)
        
        
        # add more metrics for variability and extreme
        rmse = evaluate.RMSE()
        mae = evaluate.MAE()
        
        scores = {"RMSE": rmse,
                  "MAE": mae, 
                  }
        
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
    
    
def model_fitting_and_predicting(station, method, base_estimators=None, cv=KFold(n_splits=10),
                                 scoring=None, final_estimator=None,
                                 fit_predictand=False):
    
    selector_method = "TreeBased"
    
    selector_regressor="RandomForest"
    selector = TreeBasedSelection(regressor_name=selector_regressor)
    
    
    X, y, X_test, y_test, X_all, y_all= get_data(station)
     
    y_train = copy(y_all)
    y_test_ = copy(y_test)
    
    # standardize 
    scaler = StandardScaling(method="standardscaler")
    scaler_x = scaler.fit(X_all)
    
    X_scaled_all = scaler_x.transform(X_all)
    X_scaled = scaler_x.transform(X)
    X_test_scaled = scaler_x.transform(X_test)
    
    if fit_predictand:
        scaler_y = scaler.fit(y_all)
        y_all = scaler_y.transform(y_all)
        y = scaler_y.transform(y)
        y_test = scaler_y.transform(y_test)
    
    
    
    selector.fit(X_scaled_all, y_all)
    X_selected_all = selector.transform(X_scaled_all)
    X_selected = selector.transform(X_scaled)
    # # computed as the mean and standard deviation of accumulation of the impurity decrease within each tree
    # importance = selector.feature_importance(X_scaled, y, plot=True, fig_path=path_to_store, 
    #                                          fig_name=station + "_feature_importance.png", save_fig=True, 
    #                                          station_name=station)
    
    X_test_selected = selector.transform(X_test_scaled)
    
    #selector.print_selected_features(X_scaled)
    
    # implement model
    
    if method == "Stacking":
        regressors = []    
        for i in range(len(base_estimators)) :
            regressor = Regressors(method=base_estimators[i], cv=cv, scoring=scoring)
            
            regressor.set_model()
            
            regressors.append((base_estimators[i], regressor.estimator))
                    
                    
                    
                
                
            
        model = EnsembleRegressor(estimators=regressors, cv=cv, method=method, 
                                       final_estimator_name=final_estimator, 
                                       scoring=scoring)
        
    else:
        model = Regressors(method=method, cv=cv, scoring=scoring)
        
        model.set_model()
   
   
        
    model.fit(X_selected_all, y_all)
    score_fit, ypred_fit, scores_all = cross_validate_predict(X_selected_all, y_all, model, return_cv_scores=True)
    
    yhat = pd.Series(data=model.predict(X_test_selected), index=y_test.index)
    ypred = pd.Series(data=model.predict(X_selected), index=y.index)
    
    if fit_predictand:
        yhat_ = scaler_y.inverse_transform(yhat)
        ypred_ = scaler_y.inverse_transform(ypred)
    
        
    yhat_ = yhat_.mask(yhat_ < 0, 0)
    ypred_ = ypred_.mask(ypred_ < 0, 0)
    
    score_test = evaluate_test(y_true=y_test_["precipitation"], y_pred=yhat_)
    
    predictions = pd.DataFrame({
        "obs_train": y_train,
        "obs_test": y_test_["precipitation"],
        "pred_train": ypred_,
        "pred_test": yhat_})
    
    #saving files
    # store_pickle(station, "validation_score_" + "Stacking", score_fit, path_to_store)
    # store_pickle(station, "CV_scores_" + "Stacking", scores_all, path_to_store)
    store_pickle(station,  "test_score_" + method, score_test, path_to_store)
    store_csv(station, "predictions_" + method, predictions, path_to_store)
    

base_estimators = ["RandomForest", "XGBoost", "Bagging"]  
final_estimator = "ExtraTree"  


estimators = ["LassoLarsCV", "ARD", "RandomForest", "XGBoost", "Bagging", "AdaBoost", "RidgeCV", "Stacking"]

for method in estimators: 
    for station in stationnames:         
        model_fitting_and_predicting(station, method, base_estimators=base_estimators, cv=KFold(n_splits=10),
                                         scoring=["neg_root_mean_squared_error", "r2", "neg_mean_absolute_error"],
                                         final_estimator=final_estimator,
                                         fit_predictand=True)