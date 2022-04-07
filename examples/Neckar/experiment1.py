# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 18:29:07 2022

@author: dboateng

This experiment tackles the inter-method comparison for feature selection options.
The models are set up with all the feature selection approaches (ie. recurssive, sequential and tree-based selection)
to analyse their performance on the station data. A beyesian regressor is used as base model after selection 
of the suitable predictors
"""

import os 
import sys 
import pandas as pd 
import numpy as np 
from collections import OrderedDict


sys.path.append("C:/Users/dboateng/Desktop/Python_scripts/ESD_Package")

from Package.WeatherstationPreprocessing import read_station_csv
from Package.standardizer import MonthlyStandardizer, StandardScaling
from Package.ESD_utils import store_pickle, store_csv

#relative imports 
from read_data import *
from predictor_settings import *



model = "ARD"

variable = "Temperature"

num_of_stations = len(stationnames)

#variable = "Precipitation"

# reading data (loop through all stations)

for i in range(num_of_stations):
    
    stationname = stationnames[i]
    station_dir = os.path.join(station_datadir, stationname + ".csv")
    SO = read_station_csv(filename=station_dir, varname=variable)
    
    
    #setting predictors 
    SO.set_predictors(variable, predictors, predictordir, radius,)
    
    #setting standardardizer
    SO.set_standardizer(variable, standardizer=MonthlyStandardizer(detrending=False,
                                                                    scaling=False))
    #setting model
    SO.set_model(variable, method=model)
    
    #fitting model (with predictor selector optioin)
    
    selector_method = "Recursive"
    
    SO.fit(variable, from1958to2010, ERA5Data, fit_predictors=True, predictor_selector=True, 
            selector_method=selector_method , selector_regressor="ARDRegression", num_predictors=None, 
            selector_direction=None, cal_relative_importance=True)
    
    relative_contribution = SO.relative_predictor_importance(variable)
    
    selected_predictors = SO.selected_names(variable)
    
    score, ypred = SO.cross_validate_and_predict(variable, from1958to2010, ERA5Data)
    
    y_obs = SO.get_var(variable, from1958to2010, anomalies=True).dropna()
    
    
    # just export the ypred due to update
    
    # store selected names also
    
    # store predictions in csv 
    
    predictions = pd.DataFrame({
        "obs": y_obs,
        "ERA5 1958-2010": ypred})
    
    
    #storing results
    
    store_pickle(stationname, "relative_contribution_" + selector_method, relative_contribution, cachedir)
    
    store_pickle(stationname, "validation_score_" + selector_method, score, cachedir)
    
    store_pickle(stationname, "predictions_" + selector_method, predictions, cachedir)
    
    #setting predictors 
    SO.set_predictors(variable, predictors, predictordir, radius,)
    
    #setting standardardizer
    SO.set_standardizer(variable, standardizer=MonthlyStandardizer(detrending=False,
                                                                   scaling=False))
    #setting model
    SO.set_model(variable, method=model)
    
    #fitting model (with predictor selector optioin)
    
    
    selector_method_2 = "TreeBased"
    
    SO.fit(variable, from1958to2010, ERA5Data, fit_predictors=True, predictor_selector=True, 
           selector_method=selector_method_2 , selector_regressor="RandomForest", num_predictors=None, 
           selector_direction=None,)
    
    SO.selected_names(variable)
    
    feature_importance = SO.tree_based_feature_importance(variable, from1958to2010, ERA5Data, plot=True)
    
    permutation_importance = SO.tree_based_feature_permutation_importance(variable, from1958to2010, ERA5Data, plot=True)
    
    score, ypred = SO.cross_validate_and_predict(variable, from1958to2010, ERA5Data)
    
    y_obs = SO.get_var(variable, from1958to2010, anomalies=True).dropna()
    
    predictions = pd.DataFrame({
        "obs": y_obs,
        "ERA5 1958-2010": ypred})
    
    # store permutation importance instance (relatively only temp and tp are selected)
    
    #storing results
    
    store_pickle(stationname, "relative_contribution_" + selector_method_2, relative_contribution, cachedir)
    
    store_pickle(stationname, "validation_score_" + selector_method_2, score, cachedir)
    
    store_pickle(stationname, "predictions_" + selector_method_2, predictions, cachedir)
    
    
    #setting predictors 
    SO.set_predictors(variable, predictors, predictordir, radius,)
    
    #setting standardardizer
    SO.set_standardizer(variable, standardizer=MonthlyStandardizer(detrending=False,
                                                                   scaling=False))
    #setting model
    SO.set_model(variable, method=model)
    
    #fitting model (with predictor selector optioin)
    
    selector_method_3 = "Sequential"
    
    SO.fit(variable, from1958to2010, ERA5Data, fit_predictors=True, predictor_selector=True, 
           selector_method=selector_method_3 , selector_regressor="ARDRegression", num_predictors=10, 
           selector_direction="forward", cal_relative_importance=True)
    
    relative_contribution = SO.relative_predictor_importance(variable)
    
    selected_predictors = SO.selected_names(variable)
    
    score, ypred = SO.cross_validate_and_predict(variable, from1958to2010, ERA5Data)
    
    y_obs = SO.get_var(variable, from1958to2010, anomalies=True).dropna()
    
    predictions = pd.DataFrame({
        "obs": y_obs,
        "ERA5 1958-2010": ypred})
    
    
    #storing results
    
    store_pickle(stationname, "relative_contribution_" + selector_method_3, relative_contribution, cachedir)
    
    store_pickle(stationname, "validation_score_" + selector_method_3, score, cachedir)
    
    store_pickle(stationname, "predictions_" + selector_method_3, predictions, cachedir)
    
    
                 