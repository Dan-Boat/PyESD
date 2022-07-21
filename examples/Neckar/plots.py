# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:06:04 2022

@author: dboateng


This script uses the plot and plot utils functions to generate the figures 
for the illustrative case studies

1. Plot the seasonal and annual means of all the stations
2. Plot the performance metrics of the predictor selection method 
3. Plot the performance metrics of the different models experiment 
4. Plot the some prediction examples of the selected algorithm
5. Plot the seasonal trends of the station-based future estimates
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
from pyESD.plot import barplot, correlation_heatmap, boxplot, heatmaps, scatterplot, lineplot
from pyESD.plot_utils import apply_style, correlation_data, count_predictors, boxplot_data, seasonal_mean
from pyESD.plot_utils import *

from predictor_settings import *
from read_data import station_prec_datadir, station_temp_datadir
from read_data import stationnames_prec, stationnames_temp

# DIRECTORY SPECIFICATION
# =======================


path_exp1 = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Neckar/experiment1"
path_exp2 = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Neckar/experiment2"
path_exp3 = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Neckar/experiment3"
path_to_save = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Neckar/plots"


prec_folder_name = "final_cache_Precipitation"
temp_folder_name = "final_cache_Temperature"

# PLOTTING STATION SEASONAL MEANS 
# ===============================



def plot_stations():
    path_to_data_prec = os.path.join(path_exp3, prec_folder_name)
    path_to_data_temp = os.path.join(path_exp3, temp_folder_name)
    
    df_prec = seasonal_mean(stationnames_prec, path_to_data_prec, filename="predictions_Stacking", 
                            daterange=from1958to2020 , id_name="obs")
    
    df_temp = seasonal_mean(stationnames_temp, path_to_data_temp, filename="predictions_Stacking", 
                            daterange=from1958to2020 , id_name="obs")
    
    
    apply_style(fontsize=20, style=None, linewidth=2)
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(20, 15))
    
    heatmaps(data=df_prec, cmap="Blues", label="Precipitation [mm/month]", title= None, 
             ax=ax1, cbar=True)
    
    heatmaps(data=df_temp, cmap="RdBu_r", label="Temperature [°C]", title= None, 
             ax=ax2, cbar=True, vmax=20, vmin=-5, center=0)
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(path_to_save, "Fig1.svg"), bbox_inches="tight", dpi=300)

# PLOTTING METRICS FOR PREDICTOR SELECTOR
# =======================================

def plot_predictor_selector_metrics():
    path_to_data_prec = os.path.join(path_exp1, prec_folder_name)
    path_to_data_temp = os.path.join(path_exp1, temp_folder_name)
    
    selector_methods = ["Recursive", "TreeBased", "Sequential"]
    
    apply_style(fontsize=20, style=None, linewidth=2)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(20, 15), sharex=False)
    
    barplot(methods=selector_methods, stationnames=stationnames_prec, path_to_data=path_to_data_prec,
            xlabel=None, ylabel="CV R²", varname= "test_r2", varname_std ="test_r2_std",
            filename="validation_score_", ax=ax1, legend=False, width=0.7)
    
    barplot(methods=selector_methods, stationnames=stationnames_prec , path_to_data=path_to_data_prec, 
            xlabel="Precipitation Stations", ylabel="CV RMSE", varname= "test_rmse", varname_std ="test_rmse_std",
            filename="validation_score_", ax=ax3, legend=False, width=0.7)
    
    barplot(methods=selector_methods, stationnames=stationnames_temp, path_to_data=path_to_data_temp,
            xlabel=None, ylabel="CV R²", varname= "test_r2", varname_std ="test_r2_std",
            filename="validation_score_", ax=ax2, legend=True, )
    
    barplot(methods=selector_methods, stationnames=stationnames_temp , path_to_data=path_to_data_temp, 
            xlabel="Temperature Stations", ylabel="CV RMSE", varname= "test_rmse", varname_std ="test_rmse_std",
            filename="validation_score_", ax=ax4, legend=False, )
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.94, bottom=0.06)
    plt.savefig(os.path.join(path_to_save, "Fig2.svg"), bbox_inches="tight", dpi=300)


# PLOTTING INTER-ESTIMATOR METRICS
# ================================

def plot_estimators_metrics():
    path_to_data_prec = os.path.join(path_exp2, prec_folder_name)
    path_to_data_temp = os.path.join(path_exp2, temp_folder_name)
    
    regressors = ["LassoLarsCV", "ARD", "MLP", "RandomForest",
                  "XGBoost", "Bagging", "Stacking"]
    
    colors = [grey, purple, lightbrown, tomato, skyblue, lightgreen, gold]
    
    apply_style(fontsize=20, style=None, linewidth=2)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(20, 15), sharex=False)
    
    boxplot(regressors, stationnames_prec, path_to_data_prec, ax=ax1,  
                varname="test_r2", filename="validation_score_", xlabel=None,
                ylabel="CV R²", colors = colors, patch_artist=(True))
    
    boxplot(regressors, stationnames_prec, path_to_data_prec, ax=ax3,  
                varname="test_rmse", filename="validation_score_", xlabel="Estimators",
                ylabel="CV RMSE", colors = colors, patch_artist=(True))
    
    boxplot(regressors, stationnames_temp, path_to_data_temp, ax=ax2,  
                varname="test_r2", filename="validation_score_", xlabel=None,
                ylabel="CV R²", colors = colors, patch_artist=(True))
    
    boxplot(regressors, stationnames_temp, path_to_data_temp, ax=ax4,  
                varname="test_rmse", filename="validation_score_", xlabel="Estimators",
                ylabel="CV RMSE", colors = colors, patch_artist=(True))
    
    
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.05)
    plt.savefig(os.path.join(path_to_save, "Fig3.svg"), bbox_inches="tight", dpi=300)


# PLOTTING PREDICTION EXAMPLES
# ============================

path_to_data_prec = os.path.join(path_exp3, prec_folder_name)
path_to_data_temp = os.path.join(path_exp3, temp_folder_name)

apply_style(fontsize=20, style=None, linewidth=2)

fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(20,15), sharex=False)

scatterplot(station_num=6, stationnames=stationnames_prec, path_to_data=path_to_data_prec, 
            filename="predictions_Stacking", ax=ax1, xlabel="observed", ylabel="predicted")

lineplot(station_num=6, stationnames=stationnames_prec, path_to_data=path_to_data_prec, 
            filename="predictions_Stacking", ax=ax3, fig=fig, ylabel="Precipitation anomalies [mm/month]",
            xlabel= "Years")

scatterplot(station_num=0, stationnames=stationnames_temp, path_to_data=path_to_data_temp, 
            filename="predictions_Stacking", ax=ax2, xlabel="observed", ylabel="predicted")

lineplot(station_num=0, stationnames=stationnames_temp, path_to_data=path_to_data_temp, 
            filename="predictions_Stacking", ax=ax4, fig=fig, ylabel="Temperature anomalies [°C]",
            xlabel= "Years")


plt.show()
# fig, ax = plt.subplots(1,1, figsize=(20,15))

# boxplot(regressors, stationnames_prec, path_data_precipitation, ax=ax,  
#             varname="test_rmse", filename="validation_score_", xlabel="Estimators",
#             ylabel="Validattion RMSE")

# plt.tight_layout()
# plt.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.05)
# plt.savefig("inter_estimators_rmse.png", bbox_inches="tight")

#experiment 2

#regressors = ["LassoLarsCV", "ARD", "MLPRegressor", "RandomForest", "XGBoost", "Bagging"]



# fig, ax = plt.subplots(1,1, figsize=(18,15))

# boxplot(regressors, stationnames_prec, path_data_precipitation, ax=None,  
#             varname="test_r2", filename="validation_score_", xlabel="Regressors",
#             ylabel="Validattion r²")

# plt.tight_layout()
# plt.subplots_adjust(left=0.15, right=0.88, top=0.97, bottom=0.05)
# plt.savefig("inter_model.png")

#plt.show()


#experiment 1 analysis 

#methods = ["Recursive", "TreeBased", "Sequential"]

#count predictors 

# df = count_predictors(methods, stationnames_prec, path_data_precipitation,
#                       "selected_predictors_", predictors)
# df.to_csv("predictors_count.csv")

#plot predictor correlation with station

# df = correlation_data(stationnames_prec, path_data_precipitation, 
#                       filename= "corrwith_predictors_" + methods[0], predictors=predictors)


# apply_style(fontsize=20, style="bmh") 

# fig, ax = plt.subplots(1,1, figsize=(18,15))
                        
# correlation_heatmap(data=df, cmap="RdBu", ax=ax, vmax=1, vmin=-1, center=0, cbar_ax=None, fig=fig,
#                         add_cbar=True, title=None, label= "Correlation Coefficinet", fig_path=None,
#                         xlabel="Predictors", ylabel="Stations")
# plt.tight_layout()
# plt.subplots_adjust(left=0.15, right=0.88, top=0.97, bottom=0.05)
# plt.savefig("predictor_correlation.png")

#plt.show()

#plot CV score for selector method 

# apply_style(fontsize=18, style="bmh")

# fig,(ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(18, 20), sharex=True, sharey=False)

# barplot(methods=methods, stationnames=stationnames_prec , path_to_data=path_data_precipitation, 
#         xlabel="Stations", ylabel="CV RMSE", varname= "test_rmse", varname_std ="test_rmse_std",
#         filename="validation_score_", ax=ax1)

# barplot(methods=methods, stationnames=stationnames_prec , path_to_data=path_data_precipitation, 
#         xlabel="Stations", ylabel="CV score", varname= "test_r2", varname_std ="test_r2_std",
#         filename="validation_score_", ax=ax2, legend=False)

# plt.tight_layout()
# plt.savefig("prec_selectors.png")








