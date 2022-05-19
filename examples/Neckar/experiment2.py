# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 13:31:10 2022

@author: dboateng
"""
import os 
import sys 
import pandas as pd 
import numpy as np 
from collections import OrderedDict


from pyESD.WeatherstationPreprocessing import read_station_csv
from pyESD.standardizer import MonthlyStandardizer, StandardScaling
from pyESD.ESD_utils import store_pickle, store_csv

#relative imports 
from read_data import *
from predictor_settings import *


def run_experiment2(variable, estimator, cachedir, stationnames, 
                    station_datadir, base_estimators=None, 
                    final_estimator=None):



    num_of_stations = 1 #len(stationnames)
    
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
                     estimators=base_estimators, final_estimator_name=final_estimator, daterange=from1958to2010,
                     predictor_dataset=ERA5Data)
        else:
            
            
            SO.set_model(variable, method=estimator)
        
        #fitting model (with predictor selector optioin)
        
        selector_method = "Recursive"
        
        SO.fit(variable,  from1958to2010, ERA5Data, fit_predictors=True, predictor_selector=True, 
                selector_method=selector_method , selector_regressor="ARD",
                cal_relative_importance=False)
        
        score_fit, ypred_fit = SO.cross_validate_and_predict(variable,  from1958to2010, ERA5Data,)
        
        score_test = SO.evaluate(variable,  from1958to2010, ERA5Data,)
        
        ypred_train = SO.predict(variable, from1958to2010, ERA5Data)
        
        ypred_test = SO.predict(variable, from2011to2020, ERA5Data)
        
        y_obs_train = SO.get_var(variable, from1958to2010, anomalies=True)
        
        y_obs_test = SO.get_var(variable, from2011to2020, anomalies=True)
        
        y_obs_full = SO.get_var(variable, from1958to2020, anomalies=True)
        
        
        predictions = pd.DataFrame({
            "obs_full": y_obs_full,
            "obs_train" : y_obs_train,
            "obs_test": y_obs_test,
            "ERA5 1958-2010" : ypred_train,
            "ERA5 2011-2020" : ypred_test})
        
        
        #storing of results
        
        store_pickle(stationname, "validation_score_" + estimator, score_fit, cachedir)
        store_csv(stationname, "validation_predictions_" + estimator, ypred_fit, cachedir)
        store_pickle(stationname, "test_score_" + estimator, score_test, cachedir)
        store_csv(stationname, "predictions_" + estimator, predictions, cachedir)



if __name__ == "__main__":
    
    cachedir = [cachedir_temp, cachedir_prec]
       
    variable = ["Temperature", "Precipitation"]
       
    stationnames = [stationnames_temp, stationnames_prec]
       
    station_datadir = [station_temp_datadir, station_prec_datadir]
    

    estimators = ["LassoLarsCV", "ARD", "MLPRegressor", "RandomForest", "XGBoost", "Bagging", "Stacking"]
    
    #run_experiment2(variable, regressors[5])
    for regressor in regressors:
        
        print("------ runing for model: ", regressor, "----------")
        
        run_experiment2(variable, regressor)
    