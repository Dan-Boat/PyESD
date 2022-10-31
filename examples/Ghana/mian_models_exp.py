# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 10:34:20 2022

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


def run_experiment2(variable, estimator, cachedir, stationnames, 
                    station_datadir, base_estimators=None, 
                    final_estimator=None):



    num_of_stations = len(stationnames)
    
    #num_of_stations = 1
    
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
        
        if estimator == "Stacking":
            
            SO.set_model(variable, method=estimator, ensemble_learning=True, 
                     estimators=base_estimators, final_estimator_name=final_estimator, daterange=from1961to2009,
                     predictor_dataset=ERA5Data)
        else:
            
            
            SO.set_model(variable, method=estimator, daterange=from1961to2009, predictor_dataset=ERA5Data)
        
        #fitting model (with predictor selector optioin)
        
        selector_method = "Recursive"
        
        SO.fit(variable,  from1961to2009, ERA5Data, fit_predictors=True, predictor_selector=True, 
                selector_method=selector_method , selector_regressor="ARD",
                cal_relative_importance=False, impute=True, impute_method="spline", impute_order=5)
        
        score_fit, ypred_fit = SO.cross_validate_and_predict(variable,  from1961to2009, ERA5Data,)
        
        ypred_train = SO.predict(variable, from1961to2009, ERA5Data)
        
        y_obs_train = SO.get_var(variable, from1961to2009, anomalies=True)
        
        
        predictions = pd.DataFrame({
            "obs_train" : y_obs_train,
            "ERA5 1961-2009" : ypred_train})
        
        
        #storing of results
        
        store_pickle(stationname, "validation_score_" + estimator, score_fit, cachedir)
        store_csv(stationname, "validation_predictions_" + estimator, ypred_fit, cachedir)
        store_csv(stationname, "predictions_" + estimator, predictions, cachedir)



if __name__ == "__main__":
    
    cachedir = [cachedir_prec, cachedir_temp ]
       
    variable = ["Precipitation", "Temperature"]
       
    stationnames = [stationnames_prec, stationnames_temp]
       
    station_datadir = [station_prec_datadir,station_temp_datadir]
    
    final_estimator = "ExtraTree"
    
    base_estimators = ["LassoLarsCV", "ARD", "MLP", "RandomForest", "XGBoost", "Bagging"]
    

    estimators = ["LassoLarsCV", "ARD", "MLP", "RandomForest", "XGBoost", "Bagging", "Stacking"]
    
    
    for i,idx in enumerate(variable):
        
        for estimator in estimators:
            
            print("--------- runing model for:", estimator, "-----------")
        
            run_experiment2(idx, estimator, cachedir[i], stationnames[i], station_datadir[i], 
                            base_estimators, final_estimator)