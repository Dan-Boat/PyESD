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

fig_path = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/isoPP/plots/feature_importance"

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
                     cv=TimeSeriesSplit(n_splits=20))
        
        
        #fitting model (with predictor selector optioin)
        
        if selector_method == "Recursive":
            SO.fit(variable, from1979to2012, ERA5Data, fit_predictors=True, predictor_selector=True, 
                    selector_method=selector_method , selector_regressor="LassoLarsCV", 
                    cal_relative_importance=False)
            
        elif selector_method == "TreeBased":
        
            SO.fit(variable, from1979to2012, ERA5Data, fit_predictors=True, predictor_selector=True, 
                   selector_method=selector_method , selector_regressor="RandomForest",)
            
            importance = SO.tree_based_feature_importance(variable, from1979to2012, ERA5Data, fit_predictors=True, 
                                             plot=True, fig_path=fig_path, save_fig=True, station_name=stationname,
                                             fig_name = stationname + "_importance.png")
            corr = SO.predictor_correlation(variable, from1979to2012, ERA5Data, fit_predictor=True, 
                                     fit_predictand=True, method="pearson", use_scipy=True)
            # permutation_importance = SO.tree_based_feature_permutation_importance(variable, from1979to2012, ERA5Data, fit_predictors=True, 
            #                                  plot=True, fig_path=fig_path, save_fig=True, station_name=stationname,
            #                                  fig_name = stationname + "_permutation.png")
        
        else:
            raise ValueError("Define selector not recognized")
            
        # extracting selected predictors
        
        selected_predictors = SO.selected_names(variable)
        
        # training estimate for the same model
        
        score, ypred = SO.cross_validate_and_predict(variable, from1979to2018, ERA5Data)
        
        # storing results
        
        store_pickle(stationname, "selected_predictors_" + selector_method, selected_predictors,
        cachedir)    
        
        store_pickle(stationname, "validation_score_" + selector_method, score, cachedir)
        
        if selector_method == "TreeBased":
            store_csv(stationname, "feature_importance", importance, cachedir)
            store_csv(stationname, varname="corrwith_predictors_scipy", var=corr, cachedir=cachedir)
        
      
    
        
if __name__ == "__main__":
    
    
        regressor = "LassoLarsCV"
        
        variable = "O18"
        
        
        
            
        selector_methods = ["TreeBased", "Recursive", ]
    
        for selector_method in selector_methods:
        
            print("------ runing for model: ", selector_method, "----------")
        
            run_experiment1(variable, regressor, selector_method, cachedir, stationnames, 
                            station_datadir)