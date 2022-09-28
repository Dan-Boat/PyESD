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

#relative imports 
from read_data import *
from settings import *


def run_test(variable, regressor, selector_method, cachedir, stationnames,
                    station_datadir):
    
    #num_of_stations = len(stationnames)
    
    num_of_stations = 1
    



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
        SO.set_model(variable, method=regressor)
        
        #check predictor correlation
        corr = SO.predictor_correlation(variable, from1961to2012, ERA5Data, fit_predictors=True, fit_predictand=True, 
                                  method="pearson")
        
        #fitting model (with predictor selector optioin)
        
        if selector_method == "Recursive":
            SO.fit(variable, from1961to2012, ERA5Data, fit_predictors=True, predictor_selector=True, 
                    selector_method=selector_method , selector_regressor="ARD", 
                    cal_relative_importance=False)
            
        elif selector_method == "TreeBased":
        
            SO.fit(variable, from1961to2012, ERA5Data, fit_predictors=True, predictor_selector=True, 
                   selector_method=selector_method , selector_regressor="RandomForest",)
        
        elif selector_method == "Sequential":
        
            SO.fit(variable, from1961to2012, ERA5Data, fit_predictors=True, predictor_selector=True, 
                   selector_method=selector_method , selector_regressor="ARD", num_predictors=10, 
                   selector_direction="forward")
        else:
            raise ValueError("Define selector not recognized")
            
        # extracting selected predictors
        
        selected_predictors = SO.selected_names(variable)
        
        # training estimate for the same model
        
        score, ypred = SO.cross_validate_and_predict(variable, from1961to2012, ERA5Data)
        
        # storing results
        
        store_csv(stationname, "predictions_" + selector_method, ypred, cachedir)
        
        store_pickle(stationname, "selected_predictors_" + selector_method, selected_predictors,
        cachedir)    
        
        store_pickle(stationname, "validation_score_" + selector_method, score, cachedir)
        
        store_csv(stationname, "corrwith_predictors_" + selector_method, corr, cachedir)
        
        
        
if __name__ == "__main__":
    
    
        regressor = "LassoCV"
        
        cachedir = [cachedir_prec]
        
        variable = ["Precipitation"]
        
        stationnames = [stationnames_prec]
        
        station_datadir = [station_prec_datadir]
        
        run_test(variable[0], regressor, "TreeBased", cachedir[0], stationnames[0], 
                        station_datadir[0])
        
        # for i,idx in enumerate(variable):
            
        #     selector_methods = ["Recursive", "TreeBased", "Sequential"]
        
        #     for selector_method in selector_methods:
            
        #         print("------ runing for model: ", selector_method, "----------")
            
        #         run_test(idx, regressor, selector_method, cachedir[i], stationnames[i], 
        #                         station_datadir[i])