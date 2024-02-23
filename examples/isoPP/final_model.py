# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 13:55:23 2023

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

from read_data import *
from predictor_setting import *

def run_main(variable, method, cachedir, stationnames, 
                    station_datadir, base_estimators=None, 
                    final_estimator=None, ensemble_learning=False):



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
        scoring = ["neg_root_mean_squared_error",
                   "r2", "neg_mean_absolute_error"]
        
        
        
            
        SO.set_model(variable, method=method, ensemble_learning=ensemble_learning, 
                 estimators=base_estimators, final_estimator_name=final_estimator, daterange=from1979to2012,
                 predictor_dataset=ERA5Data, cv=KFold(n_splits=20),
                 scoring = scoring)
        
        
        #fitting model (with predictor selector optioin)
        
        selector_method = "TreeBased"
        
        SO.fit(variable,  from1979to2012 , ERA5Data, fit_predictors=True, predictor_selector=True, 
                selector_method=selector_method , selector_regressor="RandomForest",
                cal_relative_importance=False)
        
        
        score_fit, ypred_fit = SO.cross_validate_and_predict(variable,  from1979to2012, ERA5Data,)
        
        
        
        ypred_1979to2012_anomalies = SO.predict(variable, from1979to2012, ERA5Data,
                                      fit_predictand=True)
        
        ypred_2013to2018_anomalies = SO.predict(variable, from2013to2018, ERA5Data,
                                                fit_predictand=True)
        
        ypred_1979to2012 = SO.predict(variable, from1979to2012, ERA5Data,
                                      fit_predictand=False)
        
        ypred_2013to2018 = SO.predict(variable, from2013to2018, ERA5Data,
                                                fit_predictand=False)
        
        
        y_ano_1979to2012 = SO.get_var(variable, from1979to2012, anomalies=True)
        
        y_ano_2013to2018 = SO.get_var(variable, from2013to2018, anomalies=True)
       
        y_ano_full = SO.get_var(variable, from1979to2018, anomalies=True)
        
        
        y_obs_full = SO.get_var(variable, from1979to2018, anomalies=False)
        
        y_obs_1979to2012 = SO.get_var(variable, from1979to2012, anomalies=False)
        
        y_obs_2013to2018 = SO.get_var(variable, from2013to2018, anomalies=False)
        
        
        if len(y_ano_2013to2018.dropna()) > 24:
            score_2013to2018 = SO.evaluate(variable,  from2013to2018, ERA5Data,)
            store_pickle(stationname, "test_score_" + method, score_2013to2018, cachedir)
        
        
        predictions_anomalies = pd.DataFrame({
            "obs": y_ano_full,
            "obs 1979-2012" : y_ano_1979to2012,
            "obs 2013-2018" : y_ano_2013to2018,
            "ERA5 1979-2012" : ypred_1979to2012_anomalies,
            "ERA5 2013-2018" : ypred_2013to2018_anomalies,
            })
        
        
        predictions = pd.DataFrame({
            "obs": y_obs_full,
            "obs 1979-2012" : y_obs_1979to2012,
            "obs 2013-2018" : y_obs_2013to2018,
            "ERA5 1979-2012" : ypred_1979to2012,
            "ERA5 2013-2018" : ypred_2013to2018,
            })
        
        
        #storing of results
        
        store_pickle(stationname, "validation_score_" + method, score_fit, cachedir)
        store_csv(stationname, "validation_predictions_" + method, ypred_fit, cachedir)
        
        store_csv(stationname, "predictions_" + method, predictions, cachedir)
        store_csv(stationname, "predictions_anomalies_" + method, predictions_anomalies,
                  cachedir)
        
        
        
if __name__ == "__main__":
    
    cachedir_data = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/isoPP/final_exp"
    
    variable = "O18"
    
    
    ensemble_method = "Stacking"
 
    base_estimators = ["ARD", "RidgeCV", "LassoLarsCV"]

    final_estimator = "RandomForest"
    
    
    run_main(variable=variable, cachedir=cachedir_data, stationnames=stationnames, 
             station_datadir=station_datadir, method=ensemble_method,
             final_estimator=final_estimator,
             base_estimators=base_estimators, 
             ensemble_learning=True)
        