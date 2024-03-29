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

from pyESD.Weatherstation import read_station_csv
from pyESD.standardizer import MonthlyStandardizer, StandardScaling
from pyESD.ESD_utils import store_pickle, store_csv
from pyESD.splitter import KFold, TimeSeriesSplit
#relative imports 
from read_data import *
from predictor_settings import *


def run_experiment1(variable, regressor, selector_method, cachedir, stationnames,
                    station_datadir):
    
    num_of_stations = len(stationnames)
    



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
        
        
        scoring = ["neg_root_mean_squared_error",
                   "r2", "neg_mean_absolute_error"]
        #setting model
        SO.set_model(variable, method=regressor, scoring=scoring, 
                     cv=TimeSeriesSplit(n_splits=10))
        
        
        #fitting model (with predictor selector optioin)
        
        if selector_method == "Recursive":
            SO.fit(variable, from1958to2010, ERA5Data, fit_predictors=True, predictor_selector=True, 
                    selector_method=selector_method , selector_regressor="ARD", 
                    cal_relative_importance=False)
            
        elif selector_method == "TreeBased":
        
            SO.fit(variable, from1958to2010, ERA5Data, fit_predictors=True, predictor_selector=True, 
                   selector_method=selector_method , selector_regressor="RandomForest",)
        
        elif selector_method == "Sequential":
        
            SO.fit(variable, from1958to2010, ERA5Data, fit_predictors=True, predictor_selector=True, 
                   selector_method=selector_method , selector_regressor="ARD", num_predictors=10, 
                   selector_direction="forward")
        else:
            raise ValueError("Define selector not recognized")
            
        # extracting selected predictors
        
        selected_predictors = SO.selected_names(variable)
        
        # training estimate for the same model
        
        score, ypred = SO.cross_validate_and_predict(variable, from1958to2010, ERA5Data)
        
        # storing results
        
        store_pickle(stationname, "selected_predictors_" + selector_method, selected_predictors,
        cachedir)    
        
        store_pickle(stationname, "validation_score_" + selector_method, score, cachedir)
        
      
    
        
if __name__ == "__main__":
    
    
        regressor = "ARD"
        
        cachedir = [cachedir_temp, cachedir_prec]
        
        variable = ["Temperature", "Precipitation"]
        
        stationnames = [stationnames_temp, stationnames_prec]
        
        station_datadir = [station_temp_datadir, station_prec_datadir]
        
        for i,idx in enumerate(variable):
            
            selector_methods = ["Recursive", "TreeBased", "Sequential"]
        
            for selector_method in selector_methods:
            
                print("------ runing for model: ", selector_method, "----------")
            
                run_experiment1(idx, regressor, selector_method, cachedir[i], stationnames[i], 
                                station_datadir[i])
                     