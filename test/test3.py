# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 11:53:35 2022

@author: dboateng
"""

import sys
import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
sys.path.append("C:/Users/dboateng/Desktop/Python_scripts/ESD_Package")

from Package.feature_selection import RecursiveFeatureElimination, TreeBasedSelection
from Package.models import Regressors
from Package.ensemble_models import EnsembleRegressor


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

models = ["AdaBoost", "LassoLarsCV", "ARD", "GradientBoost", 
          "RandomForest", "SGDRegressor", "ExtraTree", "Bagging", 
          "LassoCV", "MLPRegressor", "RidgeCV", "XGBoost"]

estimators = []

for i in range(len(models)):
    
    regressor = Regressors(method= models[i], cv=10)
    regressor.set_model()
    if models[i] == "MLPRegressor":
        regressor.fit(train_X_new, train_y)
    estimators.append((models[i], regressor.estimator))

methods = ["Voting", "Stacking"] 

from test_utils import plot_regression_results, plot_style
import time

fig, axes = plt.subplots(1,2, sharey=True, sharex=True, figsize=(18, 12))

for i in range(len(methods)) :
    
    ensemble = EnsembleRegressor(estimators=estimators, cv=10, method=methods[i])
    
    start_time = time.time()
    
    ensemble.fit(train_X_new, train_y)
    train_score = ensemble.score(train_X_new, train_y)
    
    score = ensemble.cross_validate(train_X_new, train_y)
    
    test_score = ensemble.cross_validate(test_X_new, test_y)
    
    elapsed_time = time.time() - start_time
    
    y_pred = ensemble.cross_val_predict(train_X_new, train_y)
    
    
    
    axs = np.ravel(axes)
    
    plot_style()
    plot_regression_results(ax=axs[i], y_true=train_y, y_pred=y_pred, 
                                title= methods[i], 
                                scores = (r"$R^2={:.2f} \pm {:.2f}$" + "\n" + r"$RMSE={:.2f} \pm {:.2f}$ " + "\n" + "train " + r"$R^2={:.2f}$").format(
                np.mean(score["test_r2"]),
                np.std(score["test_r2"]),
                -np.mean(score["test_neg_root_mean_squared_error"]),
                np.std(score["test_neg_root_mean_squared_error"]),
                train_score),
                                elapsed_time=elapsed_time, )
plt.suptitle("Ensemble methods performance")
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig("ensemble_models.svg", format= "svg", bbox_inches="tight", dpi=600)
plt.show()

