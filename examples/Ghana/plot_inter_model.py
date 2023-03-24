# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 17:34:07 2022

@author: dboateng
"""

import os 
import sys
import matplotlib.pyplot as plt 
import matplotlib as mpl
import pandas as pd 
import numpy as np 
from collections import OrderedDict
import seaborn as sns


from pyESD.ESD_utils import load_all_stations, load_pickle, load_csv
from pyESD.plot import *
from pyESD.plot_utils import *

from settings import *
from read_data import *
from read_data import *


path_to_data = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Ghana/model_selection/"
path_to_plot = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Ghana/plots"


regressors = ["LassoLarsCV", "ARD", "RandomForest", "XGBoost", "Bagging", "AdaBoost", "RidgeCV"]



    
df_r2 = pd.DataFrame(index=stationnames_prec, columns=regressors)
df_mae = pd.DataFrame(index=stationnames_prec, columns=regressors)
df = pd.DataFrame(index=stationnames_prec, columns= ["r2", "rmse", "mae"])

for regressor in regressors:
    
    scores_df = load_all_stations(varname="CV_scores_" + regressor, path=path_to_data, 
                                  stationnames= stationnames_prec)
    
    for i,stationname in enumerate(stationnames_prec):
        
        r2 = scores_df["test_r2"].loc[stationname]
        mae = -1* scores_df["test_neg_mean_absolute_error"].loc[stationname]
        index_max = r2.argmax()
        index_min = mae.argmin()
        
        df["r2"].loc[stationname] = scores_df["test_r2"].loc[stationname][index_max]
        df["rmse"].loc[stationname] = -1 * scores_df["test_neg_root_mean_squared_error"].loc[stationname][index_max]
        df["mae"].loc[stationname] = -1* scores_df["test_neg_mean_absolute_error"].loc[stationname][index_min]
        
    df_r2[regressor] = df["r2"]
    df_mae[regressor] = df["mae"]    


def box_plot_(data, ax, colors, ylabel=None, xlabel=None):
    color = { "boxes": black,
                  "whiskers": black,
                  "medians": red,
                  "caps": black,
                   }    
    
    
    boxplot = data.plot(kind= "box", rot=90, ax=ax, fontsize=20, color= color, sym="+b", grid=False,
                    widths=0.9, notch=False, patch_artist=True, return_type="dict")


    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        
        
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=20)
        ax.grid(True, linestyle="--", color=gridline_color)
    else:
        ax.grid(True, linestyle="--", color=gridline_color)
        ax.set_yticklabels([])
    
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontweight="bold", fontsize=20)
        ax.grid(True, linestyle="--", color=gridline_color)
    else:
        ax.grid(True, linestyle="--", color=gridline_color)
        ax.set_xticklabels([])
        
        
apply_style(fontsize=23, style="seaborn-talk", linewidth=3,)       
colors = [grey, purple, lightbrown, tomato, skyblue, lightgreen, gold, '#2ca02c',]        
fig, (ax1,ax2) = plt.subplots(2,1, sharex=False, figsize=(20, 15))

box_plot_(df_r2, ax1, colors, ylabel="CV R²")
box_plot_(df_mae, ax2, colors, ylabel="CV MAE", xlabel="Learning Models")
       
plt.tight_layout()
plt.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.05)

plt.savefig(os.path.join(path_to_plot, "estimators_metrics_train1961_2017.svg"), bbox_inches="tight", format= "svg")

