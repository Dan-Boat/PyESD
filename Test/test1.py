#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 02:11:09 2021

@author: dboateng
"""

import sys
import os 
import socket
import pandas as pd
import numpy as np

sys.path.append("/home/dboateng/Python_scripts/ESD_Package")

from Package.ESD_utils import Dataset
from Package.WeatherstationPreprocessing import read_weatherstationnames, read_station_csv
from Package.standardizer import MonthlyStandardizer
from Package.feature_selection import RecursiveFeatureElimination



radius = 200 #km

era5_datadir = "/home/dboateng/Datasets/ERA5/monthly_1950_2021"
station_datadir = "/home/dboateng/Datasets/Station/Rhine/cdc_download_2021-10-02_11-16_Rhine/processed"
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

predictors = ["t2m", "tp","msl", "v10", "u10",'z500', 'z850', 'q850',"q500", "t850","t500", "r850", "r500",
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

# testing different feature selection  
from sklearn.feature_selection import SelectKBest, chi2, RFECV, f_classif, r_regression, mutual_info_regression
from sklearn.linear_model import Lasso, LassoCV
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import  matplotlib.pyplot as plt
from sklearn.svm import SVR

# setting data for train and testing
X_train = so._get_predictor_data(variable, full_train, ERA5Data, fit=True)
X_test = so._get_predictor_data(variable, full_test, ERA5Data, fit=True,)
y_train = so.get_var(variable, full_train, anomalies=True)
y_test = so.get_var(variable, full_test, anomalies=True)


# trying feature class
rcf = RecursiveFeatureElimination(regressor_name="BayesianRidge")
rcf.fit(X_train, y_train)
print(rcf.cv_test_score())
X_train_new = rcf.transform(X_train)





# # checking with univariate
# selector = SelectKBest(mutual_info_regression, k=10).fit(X_train, y_train)
# X_features = X_train.columns[selector.get_support(indices=True)]
# print(X_features)
# scores = selector.scores_
# X_indices = np.arange(X_train.shape[-1])
# plt.bar(X_train.columns, scores, width=0.3, label="Univariate score ($-Log (p_{value})$)")

# #estimator = Lasso()
# estimator = SVR(kernel="linear")
# rfecv = RFECV(estimator= estimator, step=1, cv=5, scoring="r2")
# rfecv = rfecv.fit(X_train, y_train)
# print("Optimal number of features:", rfecv.n_features_)
# print("Best predictors:", X_train.columns[rfecv.support_])

# # visualise the learning curve 

# plt.figure(2)
# plt.xlabel("Number of feature selected")
# plt.ylabel("CV r2 of selected features")
# plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
# #plt.show()


# # using frature importance
# rf = RandomForestRegressor()
# rf = rf.fit(X_train, y_train)
# importances = rf.feature_importances_
# std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
# indices = np.argsort(importances)[::-1]

# # feature ranking 
# for f in range(X_train.shape[1]):
#     print("%d. feature %d (%f)" % (f +1, indices[f], importances[indices[f]]))


# # visualize
# plt.figure(3, figsize=(14, 13))
# plt.title("Feature importances")
# plt.bar(range(X_train.shape[1]), importances[indices], color="g", yerr=std[indices], align="center")
# plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
# plt.xlim([-1, X_train.shape[1]])
# #plt.show()

# # perfumation importance
# result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42,
#                                 n_jobs=2)
# sorted_idx = result.importances_mean.argsort()
# fig, ax = plt.subplots()
# ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=X_test.columns[sorted_idx])
# ax.set_title("Permutation Importances (On test data)")
# fig.tight_layout()
plt.show()
