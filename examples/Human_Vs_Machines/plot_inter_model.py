# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 10:34:31 2023

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

stationnames = ["Beograd", "Kikinda", "Novi_Sad", "Palic", "Sombor", "Sremska_Mitrovica", "Vrsac",
                "Zrenjanin"]

regressors = ["LassoLarsCV", "ARD", "RandomForest", "XGBoost", "Bagging", "AdaBoost", "RidgeCV"]    

path_to_data = "C:/Users/dboateng/Desktop/Datasets/Station/Vojvodina_new/plots/model_selection"

df_mae = pd.DataFrame(index=stationnames, columns=regressors)
df_rmse = pd.DataFrame(index=stationnames, columns=regressors)

df = pd.DataFrame(index=stationnames, columns= ["rmse", "mae"])

for regressor in regressors:
    
    scores_df = load_all_stations(varname="CV_scores_" + regressor, path=path_to_data, 
                                  stationnames= stationnames)
    
    for i,stationname in enumerate(stationnames):
        
        r2 = scores_df["test_r2"].loc[stationname]
        mae = -1* scores_df["test_neg_mean_absolute_error"].loc[stationname]
        
        index_max = r2.argmax()
        index_min = mae.argmin()
        
        df["mae"].loc[stationname] = -1* scores_df["test_neg_mean_absolute_error"].loc[stationname][index_min]
        df["rmse"].loc[stationname] = -1 * scores_df["test_neg_root_mean_squared_error"].loc[stationname][index_max]
        
    df_mae[regressor] = df["mae"]   
    df_rmse[regressor] = df["rmse"]  
    
def box_plot_(data, ax, colors, ylabel=None, xlabel=None, ymax=None, ymin=None):
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
        
    if ymax is not None:
        ax.set_ylim(ymin, ymax)
        
        
apply_style(fontsize=23, style="seaborn-talk", linewidth=3,)       
colors = [grey, purple, lightbrown, tomato, skyblue, lightgreen, gold,]        
fig, (ax1,ax2) = plt.subplots(2,1, sharex=False, figsize=(20, 15))

box_plot_(df_rmse, ax1, colors, ylabel="CV RSME", ymin=0.8, ymax=1.5)
box_plot_(df_mae, ax2, colors, ylabel="CV MAE", xlabel="Learning Models", ymax=1, ymin=0.3)
       
plt.tight_layout()
plt.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.05)

plt.savefig(os.path.join(path_to_data, "estimators_metrics.svg"), bbox_inches="tight", format= "svg")