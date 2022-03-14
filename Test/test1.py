"""
Created on Sun Nov 21 02:11:09 2021

@author: dboateng
"""

import sys
import os 
import socket
import pandas as pd
import numpy as np

sys.path.append("C:\\Users\dboateng\Desktop\Python_scripts\ESD_Package")

from Package.ESD_utils import Dataset
from Package.WeatherstationPreprocessing import read_weatherstationnames, read_station_csv
from Package.standardizer import MonthlyStandardizer
from Package.feature_selection import RecursiveFeatureElimination, TreeBasedSelection



radius = 200 #km

era5_datadir = "C:/Users/dboateng/Desktop/Datasets/ERA5/monthly_1950_2021"
station_datadir = "C:/Users/dboateng/Desktop/Datasets/Station/Rhine/cdc_download_2021-10-02_11-16_Rhine/processed"
predictordir    = os.path.join(os.path.dirname(__file__), '.predictors_' + str(int(radius/1000)))
cachedir        = os.path.abspath(os.path.join(__file__, os.pardir, 'final_cache'))


ERA5Data = Dataset('ERA5', {
    't2m':os.path.join(era5_datadir, 't2m_monthly.nc'),
    'msl':os.path.join(era5_datadir, 'msl_monthly.nc'),
    'u10':os.path.join(era5_datadir, 'u10_monthly.nc'),
    'v10':os.path.join(era5_datadir, 'v10_monthly.nc'),
    'z500':os.path.join(era5_datadir, 'z500_monthly_new.nc'),
    'z850':os.path.join(era5_datadir, 'z850_monthly_new.nc'),
    'tp':os.path.join(era5_datadir, 'tp_monthly.nc'),
    'q850':os.path.join(era5_datadir, 'q850_monthly_new.nc'),
    'q500':os.path.join(era5_datadir, 'q500_monthly_new.nc'),
    't850':os.path.join(era5_datadir, 't850_monthly_new.nc'),
    't500':os.path.join(era5_datadir, 't500_monthly_new.nc'),
    'r850':os.path.join(era5_datadir, 'r850_monthly_new.nc'),
    'r500':os.path.join(era5_datadir, 'r500_monthly_new.nc'),
    'vo850':os.path.join(era5_datadir, 'vo850_monthly_new.nc'),
    'vo500':os.path.join(era5_datadir, 'vo500_monthly_new.nc'),
    'pv850':os.path.join(era5_datadir, 'pv850_monthly_new.nc'),
    'pv500':os.path.join(era5_datadir, 'pv500_monthly_new.nc'),
    'u850':os.path.join(era5_datadir, 'u850_monthly_new.nc'),
    'u500':os.path.join(era5_datadir, 'u500_monthly_new.nc'),
    'v850':os.path.join(era5_datadir, 'v850_monthly_new.nc'),
    'v500':os.path.join(era5_datadir, 'v500_monthly_new.nc'),
    'sst':os.path.join(era5_datadir, 'sst_monthly.nc'),
    'd2m':os.path.join(era5_datadir, 'd2m_monthly.nc'), })

namedict = read_weatherstationnames(station_datadir)
stationnames = list(namedict.values())

predictors = ["t2m", "tp","msl", "v10", "u10","z500", "z850", "q850","q500", "t850","t500", "r850", "r500",
              "vo850", "vo500", "pv850", "pv500", "u850", "u500", "v850", "v500", "d2m"]

full = pd.date_range(start="1958-01-01", end="2019-12-31", freq="MS")
full_train = pd.date_range(start="1958-01-01", end="2000-12-31", freq="MS")
full_test = pd.date_range(start="2001-01-01", end="2019-12-31", freq="MS")

variable = "Precipitation"
stationname = stationnames[2]
so = read_station_csv(os.path.join(station_datadir, stationname + ".csv"), variable)
#y=so.get_var(variable, full, anomalies=True)
so.set_predictors(variable, predictors, predictordir, radius)
so.set_standardizer(variable, MonthlyStandardizer(detrending= False))

#predictor_data = so._get_predictor_data(variable, full, ERA5Data,fit=True)


# setting data for train and testing
X_train = so._get_predictor_data(variable, full_train, ERA5Data, fit=True)
X_test = so._get_predictor_data(variable, full_test, ERA5Data, fit=True,)
y_train = so.get_var(variable, full_train, anomalies=True)
y_test = so.get_var(variable, full_test, anomalies=True)

# saving data for sample scripts
X_train.to_csv("predictors_train_1958-2000.csv")
y_train.to_csv("precipitation_1958-2000.csv")
X_test.to_csv("predictors_test_2000-2019.csv")
y_test.to_csv("precipitation_2000-2019.csv")


# trying feature class

# rcf = RecursiveFeatureElimination(regressor_name="BayesianRidge")
# rcf.fit(X_train, y_train)
# print(rcf.cv_test_score())
# X_train_new = rcf.transform(X_train)

# tree = TreeBasedSelection(regressor_name="RandomForest")
# tree.feature_importance(X_train,y_train, plot=True)
# tree.fit(X_train, y_train)
# X_train_new = tree.transform(X_train)




