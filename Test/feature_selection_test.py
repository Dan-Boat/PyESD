#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 19:10:50 2021

@author: dboateng
Goal is to return the selected features plus the fit and train score or importance related to featutures with the selection estimator
"""
# importing modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2, RFECV, f_classif, r_regression, mutual_info_regression, SelectFromModel, SequentialFeatureSelector
from sklearn.linear_model import Lasso, LassoCV, Ridge, BayesianRidge, ARDRegression
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, LeaveOneOut, LeaveOneGroupOut
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier


import  matplotlib.pyplot as plt

full = pd.date_range(start="1958-01-01", end="2019-12-31", freq="MS")
full_train = pd.date_range(start="1958-01-01", end="2000-12-31", freq="MS")
full_test = pd.date_range(start="2001-01-01", end="2019-12-31", freq="MS")

path_to_test_data = "/home/dboateng/Python_scripts/ESD_Package/Test/sample_data.csv"

df = pd.read_csv(path_to_test_data, index_col=["time"], parse_dates=[0])
y = df["Precipitation"]
X = df.drop(["Precipitation"], axis=1)

#removing nans 
X = X.loc[~np.isnan(y)]
y = y.dropna()

# setting train and test data
X_train, X_test = X.loc[full_train], X.loc[full_test]
y_train, y_test = y.loc[full_train], y.loc[full_test]


####### Trying Recursive Feature Elimination using cross-validation (must be implemented with lasso or Ridge)


# estimator = ARDRegression() #BayesianRidge() #SVR(kernel="linear")#RandomForestRegressor()
# scoring = "r2" #neg_root_mean_squared_error"
# cv = TimeSeriesSplit(n_splits=5)
# n_jobs = -1
# min_features =5

# rfecv = RFECV(estimator=estimator, scoring=scoring, cv=cv, n_jobs=n_jobs, min_features_to_select=min_features)
# rfecv = rfecv.fit(X_train, y_train)
# print("Optimal number of features:", rfecv.n_features_)
# print("Best predictors:", X_train.columns[rfecv.support_])

# # extract the names firts and apply it to the 
# X_new = X_train[X_train.columns[rfecv.get_support(indices=True)]]
# rfecv.cv_results_["mean_test_score"]

########### Tree based Feature selection using selectFromModel (use ExtraForest or RandomForest Regressor)

# model = RandomForestRegressor()
# model = model.fit(X_train, y_train)
# importance = model.feature_importances_
# feature_names = X_train.columns
# forest_importances = pd.Series(importance, index=feature_names)
# std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
# fig, ax = plt.subplots()
# forest_importances.plot.bar(yerr=std, ax=ax)
# ax.set_title("Feature importances using tree regressor")


# selector = SelectFromModel(model, prefit=True)
# print(X_train.columns[selector.get_support(indices=True)])


#######Sequential Feature Selection with linear Regressors

estimator = Ridge()
n_features_to_select=5
sfs = SequentialFeatureSelector(estimator=estimator, n_features_to_select=n_features_to_select, scoring="r2", direction="forward")
sfs =sfs.fit(X_train, y_train)
features =[X_train.columns[sfs.get_support(indices=True)]]
print(sfs.score(X_test, y_test))
