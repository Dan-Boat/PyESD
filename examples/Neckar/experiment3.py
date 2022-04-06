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


sys.path.append("C:/Users/dboateng/Desktop/Python_scripts/ESD_Package")

from Package.WeatherstationPreprocessing import read_station_csv
from Package.standardizer import MonthlyStandardizer, StandardScaling
from Package.ESD_utils import store_pickle, store_csv

#relative imports 
from read_data import *
from predictor_settings import *

def run_experiment3():
    pass


ensemble_methods = ["Stacking", "Voting"]

base_estimators = ["LassoLarsCV", "ARD", "MLPRegressor", "RandomForest", "XGBoost", "Bagging"]

final_estimator = "ExtraTree"

variable = "Precipitation"

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
    SO.set_model(variable, method=ensemble_methods[1], ensemble_learning=True, 
                 estimators=base_estimators, final_estimator_name=final_estimator, daterange=from1958to2010,
                 predictor_dataset=ERA5Data)