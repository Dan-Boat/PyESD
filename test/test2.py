# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 12:41:13 2022

@author: dboateng
"""
import sys
import os 
import pandas as pd
import numpy as np

sys.path.append("C:/Users/dboateng/Desktop/Python_scripts/ESD_Package")

from Package.feature_selection import RecursiveFeatureElimination, TreeBasedSelection
from Package.models import Regressors
from Package.splitter import Splitter

data_path = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/data/"

df_y_train = pd.read_csv(os.path.join(data_path, "precipitation_1958-2000.csv"), index_col="Date")
df_y_test = pd.read_csv(os.path.join(data_path, "precipitation_2000-2019.csv"), index_col="Date")

train_y = df_y_train.squeeze()
test_y = df_y_test.squeeze()

df_X_train = pd.read_csv(os.path.join(data_path, "predictors_train_1958-2000.csv"), index_col="time")
df_X_test = pd.read_csv(os.path.join(data_path, "predictors_test_2000-2019.csv"), index_col="time")

# feature extracting 

# use Recurssive method

selector = RecursiveFeatureElimination(regressor_name="ARDRegression")
selector.fit(df_X_train, df_y_train.squeeze())
selector.print_selected_features(df_X_train)
print(selector.cv_test_score())
train_X_new = selector.transform(df_X_train)
test_X_new = selector.transform(df_X_test)


# using the individual regressors
models = ["LassoCV", "LassoLarsCV", "ARD", "BayesianRidge", "MLPRegressor", 
          "RandomForest", "SVR", "ExtraTree"]



regressor = Regressors(method=models[4], cv=10)
regressor.set_model()
regressor.fit(train_X_new, train_y)
score = regressor.score(train_X_new, train_y)
val = regressor.cross_validate(train_X_new, train_y)
y_pred = regressor.cross_val_predict(train_X_new, train_y)

print(np.mean(val["test_r2"]))
print(np.std(val["test_r2"]))
print(-np.mean(val["test_neg_root_mean_squared_error"]))
print(np.std(val["test_neg_root_mean_squared_error"]))