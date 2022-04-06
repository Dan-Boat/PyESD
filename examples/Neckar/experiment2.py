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


sys.path.append("C:/Users/dboateng/Desktop/Python_scripts/ESD_Package")

from Package.WeatherstationPreprocessing import read_station_csv
from Package.standardizer import MonthlyStandardizer, StandardScaling
from Package.ESD_utils import store_pickle, store_csv

#relative imports 
from read_data import *
from predictor_settings import *


def run_experiment2(variable, regressor):



    num_of_stations = len(stationnames)
    
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
        
        #fitting model (with predictor selector optioin)
        
        selector_method = "Recursive"
        
        SO.fit(variable, from1958to2010, ERA5Data, fit_predictors=True, predictor_selector=True, 
                selector_method=selector_method , selector_regressor="ARDRegression",
                cal_relative_importance=False)
        
        score_fit, ypred_fit = SO.cross_validate_and_predict(variable, from1958to2010, ERA5Data)
        
        score_test = SO.evaluate(variable, from2011to2020, ERA5Data)
        
        ypred_test = SO.predict(variable, from2011to2020, ERA5Data)
        
        predictions = pd.DataFrame({
            "obs" : y_obs,
            "ERA5 1958-2010" : ypred_fit,
            "ERA5 2011-2020" : ypred_test})
        
        
        #storing of results
        
        store_pickle(stationname, "validation_score_" + regressor, score_fit, cachedir)
        store_pickle(stationname, "test_score_" + regressor, score_test, cachedir)
        store_csv(stationname, "predictions_" + regressor, predictions, cachedir)



if __name__ == "__main__":
    
    variable = "Precipitation"

    regressors = ["LassoLarsCV", "ARD", "MLPRegressor", "RandomForest", "XGBoost", "Bagging"]
    
    #run_experiment2(variable, regressors[5])
    for regressor in regressors:
        
        print("------ runing for model: ", regressor, "----------")
        
        run_experiment2(variable, regressor)
    