

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 13:41:03 2022

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

def run_main(variable, cachedir, 
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
        #setting model
        
        scoring = ["neg_root_mean_squared_error",
                   "r2", "neg_mean_absolute_error"]
        
        
        SO.set_model(variable, method=method, ensemble_learning=ensemble_learning, 
                      estimators=base_estimators, final_estimator_name=final_estimator, daterange=from1981to2017,
                      predictor_dataset=ERA5Data, scoring=scoring, cv=KFold(n_splits=10))
        
        
        # MODEL TRAINING (1981-2017)
        # ==========================
        
        SO.fit(variable, from1981to2017, ERA5Data, fit_predictors=True, predictor_selector=True, 
                    selector_method="TreeBased" , selector_regressor="RandomForest",
                    cal_relative_importance=False, impute=False, impute_method="spline", impute_order=5)
        
        
        selected_predictors = SO.selected_names(variable)
            
        score_fit, ypred_fit, scores_all = SO.cross_validate_and_predict(variable,  from1981to2017, ERA5Data, 
                                                                         return_cv_scores=True)
        
        
        climate_score_13to17 = SO.climate_score(variable, from1981to2017, from2013to2017, ERA5Data) 
        
        
        score_2013to2017 = SO.evaluate(variable, from2013to2017, ERA5Data)
        
        ypred_1981to2012 = SO.predict(variable, from1981to2012, ERA5Data)
            
        ypred_2013to2017 = SO.predict(variable, from2013to2017, ERA5Data)
        
        # STORING SCORES
        # ==============
            
        store_pickle(stationname, "selected_predictors_" + method, selected_predictors, cachedir)
        store_pickle(stationname, "CV_scores_" + method, scores_all, cachedir)
        store_pickle(stationname, "validation_score_" + method, score_fit, cachedir)
        store_pickle(stationname, "climate_score_13to17_" + method, climate_score_13to17, cachedir)
        store_pickle(stationname, "test_score_13to17_" + method, score_2013to2017, cachedir)
    
        
        #USING CMIP6 DATA + MODEL 1958-2000
        #====================================
        
        print("fitting the AMIP predictors based on the selected model---all from realisation 1")
        SO.fit_predictor(variable, predictors, fullAMIP2, CMIP6_AMIP_R1) 
        
        print("predicting based on the AMIP predictors")
        yhat_CMIP6_AMIP_R1_anomalies = SO.predict(variable, fullAMIP, 
                                        CMIP6_AMIP_R1, fit_predictors=True, fit_predictand=True, 
                                        params_from="CMIP6_AMIP_R1", patterns_from= "CMIP6_AMIP_R1")
        
        print("predicting based on the RCP 2.6 predictors")
        yhat_CMIP6_RCP26_R1_anomalies = SO.predict(variable, fullCMIP6, 
                                        CMIP6_RCP26_R1, fit_predictors=True, fit_predictand=True,
                                        params_from="CMIP6_AMIP_R1", patterns_from= "CMIP6_AMIP_R1")
        
        print("predicting based on the RCP 4.5 predictors")
        yhat_CMIP6_RCP45_R1_anomalies = SO.predict(variable, fullCMIP6, 
                                        CMIP6_RCP45_R1, fit_predictors=True, fit_predictand=True,
                                        params_from="CMIP6_AMIP_R1", patterns_from= "CMIP6_AMIP_R1")
        
        print("predicting based on the RCP 8.5 predictors")
        yhat_CMIP6_RCP85_R1_anomalies = SO.predict(variable, fullCMIP6, 
                                        CMIP6_RCP85_R1, fit_predictors=True, fit_predictand=True,
                                        params_from="CMIP6_AMIP_R1", patterns_from= "CMIP6_AMIP_R1")
        
        
        
        # PREDICTING THE ABSOLUTE VALUES
        # ===============================
        
        print("predicting based on the AMIP predictors")
        yhat_CMIP6_AMIP_R1 = SO.predict(variable, fullAMIP, 
                                        CMIP6_AMIP_R1, fit_predictors=True, fit_predictand=False, 
                                        params_from="CMIP6_AMIP_R1", patterns_from= "CMIP6_AMIP_R1")
        
        print("predicting based on the RCP 2.6 predictors")
        yhat_CMIP6_RCP26_R1 = SO.predict(variable, fullCMIP6, 
                                        CMIP6_RCP26_R1, fit_predictors=True, fit_predictand=False,
                                        params_from="CMIP6_AMIP_R1", patterns_from= "CMIP6_AMIP_R1")
        
        print("predicting based on the RCP 4.5 predictors")
        yhat_CMIP6_RCP45_R1 = SO.predict(variable, fullCMIP6, 
                                        CMIP6_RCP45_R1, fit_predictors=True, fit_predictand=False,
                                        params_from="CMIP6_AMIP_R1", patterns_from= "CMIP6_AMIP_R1")
        
        print("predicting based on the RCP 8.5 predictors")
        yhat_CMIP6_RCP85_R1 = SO.predict(variable, fullCMIP6, 
                                        CMIP6_RCP85_R1, fit_predictors=True, fit_predictand=False,
                                        params_from="CMIP6_AMIP_R1", patterns_from= "CMIP6_AMIP_R1")
        
        
        # STORING OF RESULTS
        # ===================
        
        y_obs_1981to2012 = SO.get_var(variable, from1981to2012, anomalies=True)
            
        y_obs_2013to2017 = SO.get_var(variable, from2013to2017, anomalies=True)
            
        y_obs_1981to2017 = SO.get_var(variable, from1981to2017, anomalies=True)
        
        y_obs_1981to2017_true = SO.get_var(variable, from1981to2017, anomalies=False)
        
        
        predictions = pd.DataFrame({
            "obs": y_obs_1981to2017_true,
            "obs anomalies": y_obs_1981to2017,
            "obs 1981-2012": y_obs_1981to2012,
            "obs 2013-2017": y_obs_2013to2017,
            "ERA5 1981-2012": ypred_1981to2012,
            "ERA5 2013-2017": ypred_2013to2017,
            "CMIP6 AMIP anomalies": yhat_CMIP6_AMIP_R1_anomalies,
            "CMIP6 RCP2.6 anomalies":yhat_CMIP6_RCP26_R1_anomalies,
            "CMIP6 RCP4.5 anomalies":yhat_CMIP6_RCP45_R1_anomalies,
            "CMIP6 RCP8.5 anomalies":yhat_CMIP6_RCP85_R1_anomalies,
            "CMIP6 AMIP": yhat_CMIP6_AMIP_R1,
            "CMIP6 RCP2.6":yhat_CMIP6_RCP26_R1,
            "CMIP6 RCP4.5":yhat_CMIP6_RCP45_R1,
            "CMIP6 RCP8.5":yhat_CMIP6_RCP85_R1,
            })
        
                  
         
        store_csv(stationname, "predictions_" + ensemble_method, predictions, cachedir)
    
    
    
if __name__ == "__main__":
    selector_dir = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Ghana/final_experiment"
    cachedir = selector_dir
    #cachedir = cachedir_prec
       
    variable = "Precipitation"
       
    stationnames = stationnames_prec
       
    station_datadir =  station_prec_datadir
    
    
    ensemble_method = "Stacking"
 
    base_estimators = ["ARD", "RandomForest", "Bagging", "RidgeCV"]

    final_estimator = "LassoLarsCV"

    run_main(variable, cachedir, stationnames, station_datadir, method=ensemble_method,
             final_estimator=final_estimator,
             base_estimators=base_estimators, 
             ensemble_learning=True)