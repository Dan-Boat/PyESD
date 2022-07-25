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
from pyESD.plot import barplot, correlation_heatmap, boxplot, heatmaps, scatterplot, lineplot, plot_time_series
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
    
    df_prec = seasonal_mean(stationnames_prec, path_to_data_prec, filename="predictions_", 
                            daterange=from1958to2020 , id_name="obs", method= "Stacking")
    
    df_temp = seasonal_mean(stationnames_temp, path_to_data_temp, filename="predictions_", 
                            daterange=from1958to2020 , id_name="obs", method= "Stacking")
    
    
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

def plot_prediction_example(station_num_prec, station_num_temp):
    path_to_data_prec = os.path.join(path_exp3, prec_folder_name)
    path_to_data_temp = os.path.join(path_exp3, temp_folder_name)
    
    apply_style(fontsize=20, style=None, linewidth=2)
    
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(20,15), sharex=False)
    
    scatterplot(station_num=station_num_prec, stationnames=stationnames_prec, path_to_data=path_to_data_prec, 
                filename="predictions_", ax=ax1, xlabel="observed", ylabel="predicted",
                method= "Stacking")
    
    lineplot(station_num=station_num_prec, stationnames=stationnames_prec, path_to_data=path_to_data_prec, 
                filename="predictions_", ax=ax3, fig=fig, ylabel="Precipitation anomalies [mm/month]",
                xlabel= "Years", method= "Stacking")
    
    scatterplot(station_num=station_num_temp, stationnames=stationnames_temp, path_to_data=path_to_data_temp, 
                filename="predictions_", ax=ax2, xlabel="observed", ylabel="predicted", 
                method= "Stacking")
    
    lineplot(station_num=station_num_temp, stationnames=stationnames_temp, path_to_data=path_to_data_temp, 
                filename="predictions_", ax=ax4, fig=fig, ylabel="Temperature anomalies [°C]",
                xlabel= "Years", method= "Stacking")
    
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.05)
    plt.savefig(os.path.join(path_to_save, "Fig4.svg"), bbox_inches="tight", dpi=300)


# PLOTTTING SEASONAL TRENDS 
# =========================


def plot_seasonal_climatologies():
    
    
    path_to_data_prec = os.path.join(path_exp3, prec_folder_name)
    
    
    df_prec_26_from2040to2060 = seasonal_mean(stationnames_prec, path_to_data_prec, filename="predictions_", 
                            daterange=from2040to2060 , id_name="CMIP5 RCP2.6 anomalies", method= "Stacking")
    
    
    df_prec_85_from2040to2060 = seasonal_mean(stationnames_prec, path_to_data_prec, filename="predictions_", 
                            daterange=from2040to2060 , id_name="CMIP5 RCP8.5 anomalies", method= "Stacking")
    
    
    
    df_prec_26_from2080to2100 = seasonal_mean(stationnames_prec, path_to_data_prec, filename="predictions_", 
                            daterange=from2080to2100 , id_name="CMIP5 RCP2.6 anomalies", method= "Stacking")
    
    
    df_prec_85_from2080to2100 = seasonal_mean(stationnames_prec, path_to_data_prec, filename="predictions_", 
                            daterange=from2080to2100 , id_name="CMIP5 RCP8.5 anomalies", method= "Stacking")
    
    
    
    apply_style(fontsize=20, style=None, linewidth=2)
    fig, ((ax1,ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(20,15), sharex=False)
    cbar_ax = fig.add_axes([0.90, 0.35, 0.02, 0.25])
    
    heatmaps(data=df_prec_26_from2040to2060, cmap=BrBG, label="Precipitation [mm/month]", title= "RCP 2.6   (2040-2060)", 
             ax=ax1, cbar=True, cbar_ax=cbar_ax, vmax=10, vmin=-10, center=0,)
    
    heatmaps(data=df_prec_85_from2040to2060, cmap=BrBG, label="Precipitation [mm/month]", title= "RCP 8.5", 
             ax=ax3, cbar=False, vmax=10, vmin=-10, center=0, xlabel="Precipitation stations")
    
    heatmaps(data=df_prec_26_from2080to2100, cmap=BrBG, label="Precipitation [mm/month]", title= "RCP 2.6   (2080-2100)", 
             ax=ax2, cbar=False, vmax=10, vmin=-10, center=0,)
    
    heatmaps(data=df_prec_85_from2080to2100, cmap=BrBG, label="Precipitation [mm/month]", title= "RCP 8.5", 
             ax=ax4, cbar=False, vmax=10, vmin=-10, center=0, xlabel="Precipitation stations")
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.02, right=0.90, top=0.94, bottom=0.05,hspace=0.0)
    plt.savefig(os.path.join(path_to_save, "Fig5.svg"), bbox_inches="tight", dpi=300)
    
    
    
    path_to_data_temp = os.path.join(path_exp3, temp_folder_name)
    
    df_temp_26_from2040to2060 = seasonal_mean(stationnames_temp, path_to_data_temp, filename="predictions_", 
                            daterange=from2040to2060 , id_name="CMIP5 RCP2.6 anomalies", method= "Stacking")
    
    
    df_temp_85_from2040to2060 = seasonal_mean(stationnames_temp, path_to_data_temp, filename="predictions_g", 
                            daterange=from2040to2060 , id_name="CMIP5 RCP8.5 anomalies", method= "Stacking")
    
    
    
    df_temp_26_from2080to2100 = seasonal_mean(stationnames_temp, path_to_data_temp, filename="predictions_", 
                            daterange=from2080to2100 , id_name="CMIP5 RCP2.6 anomalies", method= "Stacking")
    
    
    df_temp_85_from2080to2100 = seasonal_mean(stationnames_temp, path_to_data_temp, filename="predictions_", 
                            daterange=from2080to2100 , id_name="CMIP5 RCP8.5 anomalies", method= "Stacking")
    
    apply_style(fontsize=20, style=None, linewidth=2)
    fig, ((ax1,ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(20,15), sharex=False)
    cbar_ax = fig.add_axes([0.90, 0.35, 0.02, 0.25])
    
    heatmaps(data=df_temp_26_from2040to2060, cmap=seismic, label="Temperature [°C]", title= "RCP 2.6   (2040-2060)", 
             ax=ax1, cbar=True, cbar_ax=cbar_ax, vmax=3, vmin=-3, center=0,)
    
    heatmaps(data=df_temp_85_from2040to2060, cmap=seismic, label="Temperature [°C]", title= "RCP 8.5", 
             ax=ax3, cbar=False, vmax=3, vmin=-3, center=0, xlabel="Precipitation stations")
    
    heatmaps(data=df_temp_26_from2080to2100, cmap=seismic, label="Temperature [°C]", title= "RCP 2.6   (2080-2100)", 
             ax=ax2, cbar=False, vmax=3, vmin=-3, center=0,)
    
    heatmaps(data=df_temp_85_from2080to2100, cmap=seismic, label="Temperature [°C]", title= "RCP 8.5", 
             ax=ax4, cbar=False, vmax=3, vmin=-3, center=0, xlabel="Precipitation stations")
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.02, right=0.90, top=0.94, bottom=0.05,hspace=0.0)
    plt.savefig(os.path.join(path_to_save, "Fig6.svg"), bbox_inches="tight", dpi=300)




# extracting time series for all stations 
path_to_data_prec = os.path.join(path_exp3, prec_folder_name)
path_to_data_temp = os.path.join(path_exp3, temp_folder_name)

apply_style(fontsize=20, style=None, linewidth=2)
fig,(ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(15,20),
                                   sharex=True, sharey=True)
plt.subplots_adjust(left=0.02, right=0.90, top=0.94, bottom=0.05,hspace=0.01)


plot_time_series(stationnames_prec, path_to_data_prec, filename="predictions_", 
                 id_name="CMIP5 RCP2.6 anomalies", daterange=fullCMIP5,
                 color=black, label="RCP 2.6", ymax=30, ymin=-20, 
                 ylabel= "Precipitation anomalies [mm/month]", ax=ax1)

plot_time_series(stationnames_prec, path_to_data_prec, filename="predictions_", 
                 id_name="CMIP5 RCP4.5 anomalies", daterange=fullCMIP5,
                 color=red, label="RCP 4.5", ymax=30, ymin=-20,
                 ylabel= "Precipitation anomalies [mm/month]", ax=ax2)

plot_time_series(stationnames_prec, path_to_data_prec, filename="predictions_", 
                 id_name="CMIP5 RCP8.5 anomalies", daterange=fullCMIP5,
                 color=blue, label="RCP 8.5", ymax=30, ymin=-20,
                 ylabel= "Precipitation anomalies [mm/month]", ax=ax3)


plt.tight_layout(h_pad=0.02)
plt.savefig(os.path.join(path_to_save, "Fig7.svg"), bbox_inches="tight", dpi=300)



fig,(ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(15,20),
                                        sharex=True, sharey=True)
plt.subplots_adjust(left=0.02, right=0.90, top=0.94, bottom=0.05,hspace=0.01)

plot_time_series(stationnames_temp, path_to_data_temp, filename="predictions_", 
                 id_name="CMIP5 RCP2.6 anomalies", daterange=fullCMIP5,
                 color=black, label="RCP 2.6", ymax=5, ymin=-5, 
                 ylabel= "Temperature anomalies [°C]", ax=ax1)

plot_time_series(stationnames_temp, path_to_data_temp, filename="predictions_", 
                 id_name="CMIP5 RCP4.5 anomalies", daterange=fullCMIP5,
                 color=red, label="RCP 4.5", ymax=5, ymin=-5,
                 ylabel= "Temperature anomalies [°C]", ax=ax2)

plot_time_series(stationnames_temp, path_to_data_temp, filename="predictions_", 
                 id_name="CMIP5 RCP8.5 anomalies", daterange=fullCMIP5,
                 color=blue, label="RCP 8.5", ymax=5, ymin=-5,
                 ylabel= "Temperature anomalies [°C]", ax=ax3)


plt.tight_layout(h_pad=0.03)
plt.savefig(os.path.join(path_to_save, "Fig8.svg"), bbox_inches="tight", dpi=300)

plt.show()
