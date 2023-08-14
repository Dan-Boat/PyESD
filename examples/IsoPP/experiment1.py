# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 21:10:07 2023

@author: dboateng
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
from predictor_setting import *


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
            SO.fit(variable, from1979to2010, ERA5Data, fit_predictors=True, predictor_selector=True, 
                    selector_method=selector_method , selector_regressor="ARD", 
                    cal_relative_importance=False)
            
        elif selector_method == "TreeBased":
        
            SO.fit(variable, from1979to2010, ERA5Data, fit_predictors=True, predictor_selector=True, 
                   selector_method=selector_method , selector_regressor="RandomForest",)
        
        elif selector_method == "Sequential":
        
            SO.fit(variable, from1979to2010, ERA5Data, fit_predictors=True, predictor_selector=True, 
                   selector_method=selector_method , selector_regressor="ARD", num_predictors=10, 
                   selector_direction="forward")
        else:
            raise ValueError("Define selector not recognized")
            
        # extracting selected predictors
        
        selected_predictors = SO.selected_names(variable)
        
        # training estimate for the same model
        
        score, ypred = SO.cross_validate_and_predict(variable, from1979to2010, ERA5Data)
        
        # storing results
        
        store_pickle(stationname, "selected_predictors_" + selector_method, selected_predictors,
        cachedir)    
        
        store_pickle(stationname, "validation_score_" + selector_method, score, cachedir)
        
      
    
        
if __name__ == "__main__":
    
    
        regressor = "ARD"
        
        variable = "O18"
        
        
        
            
        selector_methods = ["Recursive", "TreeBased", "Sequential"]
    
        for selector_method in selector_methods:
        
            print("------ runing for model: ", selector_method, "----------")
        
            run_experiment1(variable, regressor, selector_method, cachedir, stationnames, 
                            station_datadir)