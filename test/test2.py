# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 12:41:13 2022

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

def plot_style():
    plt.style.use("bmh")
    plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
    mpl.rc('text', usetex=True)
    mpl.rc('font', size=18, family='serif')
    mpl.rc('xtick', labelsize=22)
    mpl.rc('ytick', labelsize=22)
    mpl.rc('legend', fontsize=16)
    mpl.rc('axes', labelsize=22)
    mpl.rc('lines', linewidth=3)

def plot_regression_results(ax, y_true, y_pred, title, scores, elapsed_time):
    """Scatter plot of the predicted vs true targets."""
    ax.plot(
        [y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "--r", linewidth=2
    )
    ax.scatter(y_true, y_pred, alpha=0.2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))
    ax.set_xlim([y_true.min(), y_true.max()])
    ax.set_ylim([y_true.min(), y_true.max()])
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    extra = plt.Rectangle(
        (0, 0), 0, 0, fc="w", fill=False, edgecolor="none", linewidth=0
    )
    ax.legend([extra], [scores], loc="upper left")
    title = title + "\n Evaluation in {:.2f} seconds".format(elapsed_time)
    ax.set_title(title)
    
# using the individual regressors
models = ["LassoCV", "LassoLarsCV", "ARD", "BayesianRidge", 
          "RandomForest", "SGDRegressor", "ExtraTree", "RidgeCV"]

plot_style()
fig, axes = plt.subplots(2, 4, sharey=True, sharex = True, figsize=(23, 15))
axs = np.ravel(axes)

for i in range(len(models)):
    import time 
    start_time = time.time()
    
    regressor = Regressors(method= models[i], cv=10)
    regressor.set_model()
    regressor.fit(train_X_new, train_y)
    
    score = regressor.cross_validate(train_X_new, train_y)
    
    elapsed_time = time.time() - start_time
    
    y_pred = regressor.cross_val_predict(train_X_new, train_y)
    
    plot_regression_results(ax=axs[i], y_true=train_y, y_pred=y_pred, 
                            title= models[i], 
                            scores = (r"$R^2={:.2f} \pm {:.2f}$" + "\n" + r"$RMSE={:.2f} \pm {:.2f}$").format(
            np.mean(score["test_r2"]),
            np.std(score["test_r2"]),
            -np.mean(score["test_neg_root_mean_squared_error"]),
            np.std(score["test_neg_root_mean_squared_error"]),),
                            elapsed_time=elapsed_time, )
plt.suptitle("Base models performance")
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig("models.svg", format= "svg", bbox_inches="tight", dpi=600)
plt.show()

