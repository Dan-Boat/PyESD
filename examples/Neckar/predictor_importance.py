# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 13:20:55 2022

@author: dboateng

This script aim to estimate the relative importance of the individual predictors for downscaling 
precipitation and temperature in the Neckar catchment. The use of MLR forward stepwise regression 
is demonstrated to estimate the explained variance and pearson correlation with the predictands.
Usually, this is a necessary step to determine which large-scale atmospheric drivers control the 
predictand variability in a specific location.
This modeling steps support the section 3 (method-predictor selection) of the package manuscript
"""
#----------------Importing Modules -----------------------#

import os 
import sys 
import pandas as pd 
import numpy as np 
from collections import OrderedDict

from pyESD.Weatherstation import read_station_csv
from pyESD.standardizer import MonthlyStandardizer, StandardScaling
from pyESD.ESD_utils import store_pickle, store_csv
from pyESD.splitter import TimeSeriesSplit, MonthlyBooststrapper

#----relative imports------# 
from read_data import *
from predictor_settings import *


def run_predictor_importance(variable, cachedir, stationnames,
                    station_datadir):
    
    num_of_stations = len(stationnames)
    
    # reading data (loop through all stations)
    
    for i in range(num_of_stations):
        
        stationname = stationnames[i]
        station_dir = os.path.join(station_datadir, stationname + ".csv")
        SO = read_station_csv(filename=station_dir, varname=variable)
        
        
        #setting predictors (names from relative imports)
        if variable == "Precipitation":
            SO.set_predictors(variable, predictors_without_tp , predictordir, radius,)
        else:
            SO.set_predictors(variable, predictors_without_t2m , predictordir, radius,)
        
        #setting standardardizer
        SO.set_standardizer(variable, standardizer=MonthlyStandardizer(detrending=False,
                                                                        scaling=False))
        #setting model (using OLForward, with the MLR_learning tag)
        SO.set_model(variable, method="OLSForward", cv=MonthlyBooststrapper(n_splits= 500, block_size= 12), 
                     MLR_learning=True)
        
        #check predictor correlation (just for pearson standard correlation coefficient)
        # corr = SO.predictor_correlation(variable, from1958to2020, ERA5Data, fit_predictors=True, fit_predictand=True, 
        #                           method="pearson")
        
        #fitting model (with the OLS forward approach to calculate the relative importance)
        SO.fit(variable, from1958to2010, ERA5Data, fit_predictors=True, predictor_selector=False, 
                                cal_relative_importance=False)
        
        
        exp_var = SO.get_explained_variance(variable)
        
        #store_csv(stationname, "corrwith_predictors", corr, cachedir)
        if variable == "Precipitation":
            store_pickle(stationname, "exp_var_", OrderedDict(zip(predictors_without_tp , exp_var)), cachedir)
        else:
            store_pickle(stationname, "exp_var_", OrderedDict(zip(predictors_without_t2m , exp_var)), cachedir)
        
if __name__ == "__main__":
    
        
        cachedirs = [cachedir_temp_predictor_importance, cachedir_prec_predictor_importance]
        
        variables = ["Temperature", "Precipitation"]
        
        stationnames = [stationnames_temp, stationnames_prec]
        
        station_datadir = [station_temp_datadir, station_prec_datadir]
        
        for i,variable in enumerate(variables):
    
            run_predictor_importance(variable, cachedirs[i], stationnames[i], 
                                station_datadir[i])



