# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 12:13:41 2022

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
from pyESD.splitter import TimeSeriesSplit, MonthlyBooststrapper, KFold, LeaveOneOut

#relative imports 
from read_data import *
from settings import *


def run_test(variable, regressor, selector_method, cachedir, stationnames,
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
            
        #setting model
        scoring = ["neg_root_mean_squared_error",
                   "r2", "neg_mean_absolute_error"]
        
        SO.set_model(variable, method=regressor, cv= KFold(n_splits=20), 
                      scoring=scoring)
        

    
        #fitting model (with predictor selector optioin)
    
            
       
        
        SO.fit(variable, from1981to2017, ERA5Data, fit_predictors=True, predictor_selector=True, 
               selector_method=selector_method , selector_regressor="RandomForest",)
        

        
        # extracting selected predictors
        
        selected_predictors = SO.selected_names(variable)
        
        # training estimate for the same model
        climate_score = SO.climate_score(variable, from1981to2017, from1981to2017, ERA5Data)
        score, ypred = SO.cross_validate_and_predict(variable, from1981to2017, ERA5Data)
        
        # storing results
        
        store_csv(stationname, "predictions_" + selector_method, ypred, cachedir)
        
        store_pickle(stationname, "selected_predictors_" + selector_method, selected_predictors,
        cachedir)    
        
        store_pickle(stationname, "validation_score_" + selector_method, score, cachedir)
        
        
        
        
if __name__ == "__main__":
    
        selector_dir = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Ghana/predictor_selection"
        
        regressor = "LassoLarsCV"
        
        cachedir =selector_dir
        
        variable = "Precipitation"
        
        stationnames = stationnames_prec
        
        station_datadir = station_prec_datadir
        
            
        selector_method = "TreeBased"
        
       
        run_test(variable, regressor, selector_method, cachedir, stationnames, 
                            station_datadir)