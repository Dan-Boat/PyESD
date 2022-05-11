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

#sys.path.append("C:/Users/dboateng/Desktop/Python_scripts/ESD_Package")

from pyESD.WeatherstationPreprocessing import read_station_csv
from pyESD.standardizer import MonthlyStandardizer, StandardScaling
from pyESD.ESD_utils import store_pickle, store_csv

#relative imports 
from read_data import *
from predictor_settings import *



def run_experiment1(variable, regressor, selector_method):
    
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
        SO.set_model(variable, method=regressor)
        
        #check predictor correlation
        corr = SO.predictor_correlation(variable, from1958to2010, ERA5Data, fit_predictors=True, fit_predictand=True, 
                                  method="pearson")
        
        #fitting model (with predictor selector optioin)
        
        if selector_method == "Recursive":
            SO.fit(variable, from1958to2010, ERA5Data, fit_predictors=True, predictor_selector=True, 
                    selector_method=selector_method , selector_regressor="ARDRegression", 
                    cal_relative_importance=False)
            
        elif selector_method == "TreeBased":
        
            SO.fit(variable, from1958to2010, ERA5Data, fit_predictors=True, predictor_selector=True, 
                   selector_method=selector_method , selector_regressor="RandomForest",)
        
        elif selector_method == "Sequential":
        
            SO.fit(variable, from1958to2010, ERA5Data, fit_predictors=True, predictor_selector=True, 
                   selector_method=selector_method , selector_regressor="ARDRegression", num_predictors=10, 
                   selector_direction="forward")
        else:
            raise ValueError("Define selector not recognized")
            
        # extracting selected predictors
        
        selected_predictors = SO.selected_names(variable)
        
        # training estimate for the same model
        
        score, ypred = SO.cross_validate_and_predict(variable, from1958to2010, ERA5Data)
        
        # storing results
        
        store_csv(stationname, "predictions_" + selector_method, ypred, cachedir)
        
        store_pickle(stationname, "selected_predictors_" + selector_method, selected_predictors,
        cachedir)    
        
        store_pickle(stationname, "validation_score_" + selector_method, score, cachedir)
        
        store_csv(stationname, "corrwith_predictors_" + selector_method, corr, cachedir)
      
    
        
if __name__ == "__main__":
    
    
        regressor = "ARD"

        variable = "Temperature"
        
        selector_methods = ["Recursive", "TreeBased", "Sequential"]
        
        for selector_method in selector_methods:
            
            print("------ runing for model: ", selector_method, "----------")
            
            run_experiment1(variable, regressor, selector_method)
                     