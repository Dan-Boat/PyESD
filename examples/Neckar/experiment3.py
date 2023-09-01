# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 13:31:23 2022

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
from predictor_settings import *

"""
The downscaling routines used in this final script are based on the results from experiment 1
(predictor selection) and experiment 2 (estimator selection)

This script runs all the necessary routines for statistical downscaling of 
precipitation and temperature for the Elb-neckar subcatachment. 

1. Reads the predictand, predictors and the future predictors datasetsrequired for the 
future climate change estimates. 

2. Trians the selected algorithm based on the station data and the selected 
predictors for the period 1958-2000 and validate the optimized algorithm on the 
indepedent data period of 2001-2020.

3. The model is also retrain with the AMIP simulation that overlaps the time peroids and
predict based on the trained models 

4. The trained models with the perfect predictors are also used to predict the future 
trends of the assumed scenarios (RCP 2.6, 4.5, and 8.5)of the cmip5 climate model simulation

Experiment 3: Feature selection: Recurssive, model: Stacking regression (with 6 based models)

"""

def run_experiment3(variable, cachedir, 
                    stationnames, station_datadir, method, 
                    final_estimator=None,
                    base_estimators=None, ensemble_learning=False):

    num_of_stations = len(stationnames)
    
    
    
        
    for i in range(num_of_stations):
        
        stationname = stationnames[i]
        station_dir = os.path.join(station_datadir, stationname + ".csv")
        SO = read_station_csv(filename=station_dir, varname=variable)
        
        
        # USING ERA5 DATA
        # ================
        
        
        #setting predictors 
        SO.set_predictors(variable, predictors, predictordir, radius,)
        
        #setting standardardizer
        SO.set_standardizer(variable, standardizer=MonthlyStandardizer(detrending=False,
                                                                        scaling=False))
        
        scoring = ["neg_root_mean_squared_error",
                   "r2", "neg_mean_absolute_error"]
        
        #setting model
        SO.set_model(variable, method=method, ensemble_learning=ensemble_learning, 
                      estimators=base_estimators, final_estimator_name=final_estimator, 
                      daterange=from1958to2010,
                      predictor_dataset=ERA5Data, 
                      cv=KFold(n_splits=10),
                      scoring = scoring)
        
        
        # MODEL TRAINING (1958-2000)
        # ==========================
        
        
        SO.fit(variable, from1958to2010, ERA5Data, fit_predictors=True, predictor_selector=True, 
                    selector_method="Recursive" , selector_regressor="ARD",
                    cal_relative_importance=False)
            
        score_1958to2010, ypred_1958to2010 = SO.cross_validate_and_predict(variable, from1958to2010, ERA5Data)
            
        score_2011to2020 = SO.evaluate(variable, from2011to2020, ERA5Data)
        
        ypred_1958to2010 = SO.predict(variable, from1958to2010, ERA5Data)
            
        ypred_2011to2020 = SO.predict(variable, from2011to2020, ERA5Data)
        
        # STORING SCORES
        # ==============
            
        store_pickle(stationname, "validation_score_" + ensemble_method, score_1958to2010, cachedir)
        store_pickle(stationname, "test_score_" + ensemble_method, score_2011to2020, cachedir)
        
        #USING CMIP5 DATA + MODEL 1958-2000
        #====================================
        
        print("fitting the AMIP predictors based on the selected model---all from realisation 1")
        SO.fit_predictor(variable, predictors, fullAMIP, CMIP5_AMIP_R1) 
        
        print("predicting based on the AMIP predictors")
        yhat_CMIP5_AMIP_R1_anomalies = SO.predict(variable, fullAMIP, 
                                        CMIP5_AMIP_R1, fit_predictors=True, fit_predictand=True, 
                                        params_from="CMIP5_AMIP_R1", patterns_from= "CMIP5_AMIP_R1")
        
        print("predicting based on the RCP 2.6 predictors")
        yhat_CMIP5_RCP26_R1_anomalies = SO.predict(variable, fullCMIP5, 
                                        CMIP5_RCP26_R1, fit_predictors=True, fit_predictand=True,
                                        params_from="CMIP5_AMIP_R1", patterns_from= "CMIP5_AMIP_R1")
        
        print("predicting based on the RCP 4.5 predictors")
        yhat_CMIP5_RCP45_R1_anomalies = SO.predict(variable, fullCMIP5, 
                                        CMIP5_RCP45_R1, fit_predictors=True, fit_predictand=True,
                                        params_from="CMIP5_AMIP_R1", patterns_from= "CMIP5_AMIP_R1")
        
        print("predicting based on the RCP 8.5 predictors")
        yhat_CMIP5_RCP85_R1_anomalies = SO.predict(variable, fullCMIP5, 
                                        CMIP5_RCP85_R1, fit_predictors=True, fit_predictand=True,
                                        params_from="CMIP5_AMIP_R1", patterns_from= "CMIP5_AMIP_R1")
        
        
        
        # PREDICTING THE ABSOLUTE VALUES
        # ===============================
        
        print("predicting based on the AMIP predictors")
        yhat_CMIP5_AMIP_R1 = SO.predict(variable, fullAMIP, 
                                        CMIP5_AMIP_R1, fit_predictors=True, fit_predictand=False, 
                                        params_from="CMIP5_AMIP_R1", patterns_from= "CMIP5_AMIP_R1")
        
        print("predicting based on the RCP 2.6 predictors")
        yhat_CMIP5_RCP26_R1 = SO.predict(variable, fullCMIP5, 
                                        CMIP5_RCP26_R1, fit_predictors=True, fit_predictand=False,
                                        params_from="CMIP5_AMIP_R1", patterns_from= "CMIP5_AMIP_R1")
        
        print("predicting based on the RCP 4.5 predictors")
        yhat_CMIP5_RCP45_R1 = SO.predict(variable, fullCMIP5, 
                                        CMIP5_RCP45_R1, fit_predictors=True, fit_predictand=False,
                                        params_from="CMIP5_AMIP_R1", patterns_from= "CMIP5_AMIP_R1")
        
        print("predicting based on the RCP 8.5 predictors")
        yhat_CMIP5_RCP85_R1 = SO.predict(variable, fullCMIP5, 
                                        CMIP5_RCP85_R1, fit_predictors=True, fit_predictand=False,
                                        params_from="CMIP5_AMIP_R1", patterns_from= "CMIP5_AMIP_R1")
        
        
        # STORING OF RESULTS
        # ===================
        
        y_obs_1958to2010 = SO.get_var(variable, from1958to2010, anomalies=True)
            
        y_obs_2011to2020 = SO.get_var(variable, from2011to2020, anomalies=True)
            
        y_obs_1958to2020 = SO.get_var(variable, from1958to2020, anomalies=True)
        
        y_obs_1958to2020_true = SO.get_var(variable, from1958to2020, anomalies=False)
        
        
        predictions = pd.DataFrame({
            "obs": y_obs_1958to2020_true,
            "obs anomalies": y_obs_1958to2020,
            "obs 1958-2010": y_obs_1958to2010,
            "obs 2011-2020": y_obs_2011to2020,
            "ERA5 1958-2010": ypred_1958to2010,
            "ERA5 2011-2020": ypred_2011to2020,
            "CMIP5 AMIP anomalies": yhat_CMIP5_AMIP_R1_anomalies,
            "CMIP5 RCP2.6 anomalies":yhat_CMIP5_RCP26_R1_anomalies,
            "CMIP5 RCP4.5 anomalies":yhat_CMIP5_RCP45_R1_anomalies,
            "CMIP5 RCP8.5 anomalies":yhat_CMIP5_RCP85_R1_anomalies,
            "CMIP5 AMIP": yhat_CMIP5_AMIP_R1,
            "CMIP5 RCP2.6":yhat_CMIP5_RCP26_R1,
            "CMIP5 RCP4.5":yhat_CMIP5_RCP45_R1,
            "CMIP5 RCP8.5":yhat_CMIP5_RCP85_R1,
            })
        
                  
           
        
       
       
        store_csv(stationname, "predictions_" + ensemble_method, predictions, cachedir)
    
    
    
if __name__ == "__main__":
    cachedir = [cachedir_temp, cachedir_prec]
       
    variables = ["Temperature", "Precipitation"]
       
    stationnames = [stationnames_temp, stationnames_prec]
       
    station_datadir = [station_temp_datadir, station_prec_datadir]
    
    
    # experiment with LassoLarsCV (then later use it as  the final estimator for the stacking)

    
    ensemble_method = "Stacking"
 
    base_estimators = ["LassoLarsCV", "ARD", "RandomForest", "Bagging"]

    final_estimator = "ExtraTree"

    for i,variable in enumerate(variables):
        
        
        print("---------- running for variable: ", variable, "-------")
        run_experiment3(variable, cachedir[i], 
                        stationnames[i], station_datadir[i],
                        ensemble_method, final_estimator=final_estimator,
                        base_estimators=base_estimators, 
                        ensemble_learning=True)
        
        
        
    
    