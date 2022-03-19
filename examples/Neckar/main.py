# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 17:31:13 2022.

@author: dboateng

control script for example data using pyESD
"""

# import modules 
import os 
import sys 
import pandas as pd 
import numpy as np 
from collections import OrderedDict


sys.path.append("C:/Users/dboateng/Desktop/Python_scripts/ESD_Package")

from Package.WeatherstationPreprocessing import read_station_csv
from Package.standardizer import MonthlyStandardizer
from Package.ESD_utils import store_pickle, store_csv

#relative imports 
from read_data import *
from predictor_settings import *




models = ["AdaBoost", "LassoLarsCV", "ARD", "GradientBoost", 
          "RandomForest", "ExtraTree", "Bagging", 
          "LassoCV", "RidgeCV", "XGBoost", "MLPRegressor"]

method = "Voting"

num_of_stations = len(stationnames)

variable = "Precipitation"

stationname = stationnames[0]
station_dir = os.path.join(station_datadir, stationname + ".csv")
SO = read_station_csv(filename=station_dir, varname=variable)

SO.set_predictors(variable, predictors, predictordir, radius,)

SO.set_standardizer(variable, standardizer=MonthlyStandardizer(detrending=False, scaling=False))

SO.set_model(variable, method=method, ensemble_learning=True, estimators=models, final_estimator_name=None,
             daterange = from1958to2010, predictor_dataset=ERA5Data)

SO.fit(variable, from1958to2010, ERA5Data, fit_predictors=True, predictor_selector=True, 
       selector_method="Recursive", selector_regressor="Ridge")

score, ypred = SO.cross_validate_and_predict(variable, from1958to2010, ERA5Data)

test_score,test_y_pred = SO.cross_validate_and_predict(variable, from2011to2020 , ERA5Data)

y_true = SO.get_var(variable, from1958to2010, anomalies=True)

y_pred = SO.predict(variable, from1958to2010 , ERA5Data)

scores = SO.evaluate(variable, from2011to2020 , ERA5Data)