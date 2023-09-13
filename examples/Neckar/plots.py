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
from pyESD.plot import *
from pyESD.plot_utils import *
from pyESD.plot_utils import *

from predictor_settings import *
from read_data import *
from read_data import *

# DIRECTORY SPECIFICATION
# =======================

path_mlr = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Neckar/predictor_importance"
path_exp1 = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Neckar/experiment1"
path_exp2 = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Neckar/experiment2"
path_exp3 = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Neckar/experiment3"
path_to_save = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Neckar/plots/final"


prec_folder_name = "final_cache_Precipitation"
temp_folder_name = "final_cache_Temperature"

# PLOTTING STATION SEASONAL MEANS 
# ===============================



def plot_stations():
    path_to_data_prec = os.path.join(path_exp3, prec_folder_name)
    path_to_data_temp = os.path.join(path_exp3, temp_folder_name)
    
    df_prec_sm = seasonal_mean(stationnames_prec, path_to_data_prec, filename="predictions_", 
                            daterange=from1958to2020 , id_name="obs", method= "Stacking")
    
    df_temp_sm = seasonal_mean(stationnames_temp, path_to_data_temp, filename="predictions_", 
                            daterange=from1958to2020 , id_name="obs", method= "Stacking")
    
    
    
    df_prec = monthly_mean(stationnames_prec, path_to_data_prec, filename="predictions_", 
                            daterange=from1958to2020 , id_name="obs", method= "Stacking")
    df_temp = monthly_mean(stationnames_temp, path_to_data_temp, filename="predictions_", 
                            daterange=from1958to2020 , id_name="obs", method= "Stacking")
    
    means_prec, stds_prec = estimate_mean_std(df_prec)
    means_temp, stds_temp = estimate_mean_std(df_temp)
    
    
    
    
    
    apply_style(fontsize=20, style=None, linewidth=2)
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(20, 15))
    
    heatmaps(data=df_prec_sm, cmap="Blues", label="Precipitation [mm/month]", title= None, 
             ax=ax1, cbar=True, xlabel="Precipitation stations")
    
    plot_monthly_mean(means=means_prec, stds=stds_prec, color=seablue, ylabel="Precipitation [mm/month]", 
                      ax=ax2)
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(path_to_save, "StationMonthly_variability_prec.svg"), bbox_inches="tight", dpi=300)
    
    
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(20,15), sharey=False)
    
    heatmaps(data=df_temp_sm, cmap="seismic", label="Temperature [°C]", title= None, 
             ax=ax1, cbar=True, vmax=20, center=0, vmin=-5, xlabel="Temperature stations")

    plot_monthly_mean(means=means_temp, stds=stds_temp, color=indianred, ylabel="Temperature [°C]", 
                      ax=ax2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_save, "StationMonthly_variability_temp.svg"), bbox_inches="tight", dpi=300)
    
    
    
    

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
    plt.savefig(os.path.join(path_to_save, "predictor_selection_metrics.svg"), bbox_inches="tight", dpi=300)


def plot_correlation():
    path_to_data_prec = os.path.join(path_mlr, prec_folder_name)
    path_to_data_temp = os.path.join(path_mlr, temp_folder_name)
    
    
    df_prec = correlation_data(stationnames_prec, path_to_data_prec, "corrwith_predictors", predictors, 
                               use_id=True)
    df_temp = correlation_data(stationnames_temp, path_to_data_temp, "corrwith_predictors", predictors, 
                               use_id=True)
    
    apply_style(fontsize=22, style=None) 
    
    fig, ax = plt.subplots(1,1, figsize=(20,15))
                            
    correlation_heatmap(data=df_prec, cmap="RdBu", ax=ax, vmax=1, vmin=-1, center=0, cbar_ax=None, fig=fig,
                            add_cbar=True, title=None, label= "Pearson Correlation Coefficinet (PCC)", fig_path=path_to_save,
                            xlabel="Predictors", ylabel="Precipitation Stations", fig_name="correlation_prec.svg",)
    
    fig, ax = plt.subplots(1,1, figsize=(20,15))
                            
    correlation_heatmap(data=df_temp, cmap="RdBu", ax=ax, vmax=1, vmin=-1, center=0, cbar_ax=None, fig=fig,
                            add_cbar=True, title=None, label= "Pearson Correlation Coefficinet (PCC)", fig_path=path_to_save,
                            xlabel="Predictors", ylabel="Temperature Stations", fig_name="correlation_temp.svg",)
   




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
    plt.savefig(os.path.join(path_to_save, "Inter-estimator.svg"), bbox_inches="tight", dpi=300)


# PLOTTING PREDICTION EXAMPLES
# ============================

def plot_prediction_example(station_num_prec, station_num_temp):
    path_to_data_prec = os.path.join(path_exp3, prec_folder_name)
    path_to_data_temp = os.path.join(path_exp3, temp_folder_name)
    
    
    print("----plotting for the station:", stationnames_prec[station_num_prec], "for prec and ",
          stationnames_temp[station_num_temp], "for temp----")
    
    
    apply_style(fontsize=22, style=None, linewidth=3)
    
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(22,18), sharex=False)
    
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
    plt.savefig(os.path.join(path_to_save, "prediction_example_for_station.svg"), bbox_inches="tight", dpi=300)


# PLOTTTING SEASONAL TRENDS 
# =========================


def plot_seasonal_climatologies():
    
    
    path_to_data_prec = os.path.join(path_exp3, prec_folder_name)
    
    #obs
    df_prec_sm = seasonal_mean(stationnames_prec, path_to_data_prec, filename="predictions_", 
                            daterange=from1958to2010 , id_name="obs", method= "Stacking",
                            use_id=True)
    
    
    
    
    df_prec_26_from2040to2070 = seasonal_mean(stationnames_prec, path_to_data_prec, filename="predictions_", 
                            daterange=from2040to2070 , id_name="CMIP5 RCP2.6 anomalies", method= "Stacking",
                            use_id=True)
    
    
    df_prec_85_from2040to2070 = seasonal_mean(stationnames_prec, path_to_data_prec, filename="predictions_", 
                            daterange=from2040to2070 , id_name="CMIP5 RCP8.5 anomalies", method= "Stacking",
                            use_id=True)
    
    
    
    df_prec_26_from2070to2100 = seasonal_mean(stationnames_prec, path_to_data_prec, filename="predictions_", 
                            daterange=from2070to2100 , id_name="CMIP5 RCP2.6 anomalies", method= "Stacking",
                            use_id=True)
    
    
    df_prec_85_from2070to2100 = seasonal_mean(stationnames_prec, path_to_data_prec, filename="predictions_", 
                            daterange=from2070to2100 , id_name="CMIP5 RCP8.5 anomalies", method= "Stacking",
                            use_id=True)
    
    
    
    apply_style(fontsize=20, style=None, linewidth=2)
    
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(20,15), sharex=False)
    #cbar_ax = fig.add_axes([0.90, 0.35, 0.02, 0.25])
    
    # end of century
    
    heatmaps(data=df_prec_sm, cmap="YlGnBu", label="Precipitation [mm/month]", title= "obs [1958-2010]", 
             ax=ax1, cbar=True, )
    
    
    heatmaps(data=df_prec_26_from2070to2100, cmap=BrBG, label="Precipitation [mm/month]", title= "RCP 8.5 [2070-2100]", 
             ax=ax2, cbar=True, vmax=20, vmin=-20, center=0, )
    
    heatmaps(data=df_prec_85_from2070to2100, cmap=BrBG, label="Precipitation [mm/month]", title= "RCP 8.5 [2070-2100]", 
             ax=ax3, cbar=True, vmax=20, vmin=-20, center=0, xlabel="Precipitation stations")
    
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.02, right=0.85, top=0.94, bottom=0.05,hspace=0.01)
    plt.savefig(os.path.join(path_to_save, "seasonal_trend_prec_endcentury.svg"), bbox_inches="tight", dpi=300)
    
    
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(20,15), sharex=False)
    #cbar_ax = fig.add_axes([0.90, 0.35, 0.02, 0.25])
    
    # mid of century
    
    heatmaps(data=df_prec_sm, cmap="YlGnBu", label="Precipitation [mm/month]", title= "obs [1958-2010]", 
             ax=ax1, cbar=True, )
    
    
    heatmaps(data=df_prec_26_from2040to2070, cmap=BrBG, label="Precipitation [mm/month]", title= "RCP 8.5 [2040-2070]", 
             ax=ax2, cbar=True, vmax=20, vmin=-20, center=0, )
    
    heatmaps(data=df_prec_85_from2040to2070, cmap=BrBG, label="Precipitation [mm/month]", title= "RCP 8.5 [2040-2070]", 
             ax=ax3, cbar=True, vmax=20, vmin=-20, center=0, xlabel="Precipitation stations")
     
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.02, right=0.85, top=0.94, bottom=0.05,hspace=0.01)
    plt.savefig(os.path.join(path_to_save, "seasonal_trend_prec_midcentury.svg"), bbox_inches="tight", dpi=300)
    
    
    
    path_to_data_temp = os.path.join(path_exp3, temp_folder_name)
    
    #obs
    
    df_temp_sm = seasonal_mean(stationnames_temp, path_to_data_temp, filename="predictions_", 
                            daterange=from1958to2010 , id_name="obs", method= "Stacking",
                            use_id=True)
    
    df_temp_26_from2040to2070 = seasonal_mean(stationnames_temp, path_to_data_temp, filename="predictions_", 
                            daterange=from2040to2070 , id_name="CMIP5 RCP2.6 anomalies", method= "Stacking",
                            use_id=True)
    
    
    df_temp_85_from2040to2070 = seasonal_mean(stationnames_temp, path_to_data_temp, filename="predictions_", 
                            daterange=from2040to2070 , id_name="CMIP5 RCP8.5 anomalies", method= "Stacking",
                            use_id=True)
    
    
    
    df_temp_26_from2070to2100 = seasonal_mean(stationnames_temp, path_to_data_temp, filename="predictions_", 
                            daterange=from2070to2100 , id_name="CMIP5 RCP2.6 anomalies", method= "Stacking",
                            use_id=True)
    
    
    df_temp_85_from2070to2100 = seasonal_mean(stationnames_temp, path_to_data_temp, filename="predictions_", 
                            daterange=from2070to2100 , id_name="CMIP5 RCP8.5 anomalies", method= "Stacking",
                            use_id=True)
    
    apply_style(fontsize=20, style=None, linewidth=2)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(20,15), sharex=False)
    #cbar_ax = fig.add_axes([0.90, 0.35, 0.02, 0.25])
    
    # endofcentury
    heatmaps(data=df_temp_sm, cmap="Spectral_r", label="Temperature [°C]", title= "obs [1958-2010]", 
             ax=ax1, cbar=True, vmax=20, center=0, vmin=-5, )
    
    heatmaps(data=df_temp_26_from2070to2100, cmap=seismic, label="Temperature [°C]", title= "RCP 2.6 [2070-2100]", 
             ax=ax2, cbar=True, vmax=4, vmin=-4, center=0,)
    
    heatmaps(data=df_temp_85_from2070to2100, cmap=seismic, label="Temperature [°C]", title= "RCP 8.5 [2070-2100]", 
             ax=ax3, cbar=True, vmax=4, vmin=-4, center=0, xlabel="Temperature stations")
    
    
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.02, right=0.90, top=0.94, bottom=0.05,hspace=0.1)
    plt.savefig(os.path.join(path_to_save, "seasonal_trend_temp_endofcentury.svg"), bbox_inches="tight", dpi=300)
    
    
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(20,15), sharex=False)
    
    # endofcentury
    heatmaps(data=df_temp_sm, cmap="Spectral_r", label="Temperature [°C]", title= "obs [1958-2010]", 
             ax=ax1, cbar=True, vmax=20, center=0, vmin=-5, )
    
    heatmaps(data=df_temp_26_from2040to2070, cmap=seismic, label="Temperature [°C]", title= "RCP 2.6 [2040-2070]", 
             ax=ax2, cbar=True, vmax=4, vmin=-4, center=0,)
    
    heatmaps(data=df_temp_85_from2040to2070, cmap=seismic, label="Temperature [°C]", title= "RCP 8.5 [2040-2070]", 
             ax=ax3, cbar=True, vmax=4, vmin=-4, center=0, xlabel="Temperature stations")
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.02, right=0.90, top=0.94, bottom=0.05,hspace=0.1)
    plt.savefig(os.path.join(path_to_save, "seasonal_trend_temp_midofcentury.svg"), bbox_inches="tight", dpi=300)


# PLOTTING FUTURE TIMESERIES 
# ==========================


def plot_ensemble_timeseries():

    # extracting time series for all stations 
    path_to_data_prec = os.path.join(path_exp3, prec_folder_name)
    path_to_data_temp = os.path.join(path_exp3, temp_folder_name)
    
    apply_style(fontsize=20, style=None, linewidth=2)
    fig,(ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(15,20),
                                       sharex=True, sharey=True)
    plt.subplots_adjust(left=0.02, right=0.90, top=0.94, bottom=0.05,hspace=0.01)
    
    
    plot_time_series(stationnames_prec, path_to_data_prec, filename="predictions_", 
                     id_name="CMIP5 RCP2.6 anomalies", daterange=fullCMIP5,
                     color=black, label="RCP 2.6", ymax=40, ymin=-40, 
                     ylabel= "Precipitation anomalies [mm/month]", ax=ax1,
                     window=12)
                     
    
    plot_time_series(stationnames_prec, path_to_data_prec, filename="predictions_", 
                     id_name="CMIP5 RCP4.5 anomalies", daterange=fullCMIP5,
                     color=red, label="RCP 4.5", ymax=40, ymin=-40,
                     ylabel= "Precipitation anomalies [mm/month]", ax=ax2, 
                     window=12)
    
    plot_time_series(stationnames_prec, path_to_data_prec, filename="predictions_", 
                     id_name="CMIP5 RCP8.5 anomalies", daterange=fullCMIP5,
                     color=blue, label="RCP 8.5", ymax=40, ymin=-40,
                     ylabel= "Precipitation anomalies [mm/month]", ax=ax3, 
                     window=12)
    
    
    plt.tight_layout(h_pad=0.02)
    plt.savefig(os.path.join(path_to_save, "inter-annual_trend_prec.svg"), bbox_inches="tight", dpi=300)
    
    
    
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
    plt.savefig(os.path.join(path_to_save, "inter-annual_trend_temp.svg"), bbox_inches="tight", dpi=300)


def plot_different_projections(variable= "Precipitation"):
    
    stationloc_dir_prec = os.path.join(station_prec_datadir , "stationloc.csv")
    stationloc_dir_temp = os.path.join(station_temp_datadir , "stationloc.csv")
    
    path_to_data_prec = os.path.join(path_exp3, prec_folder_name)
    path_to_data_temp = os.path.join(path_exp3, temp_folder_name)
    
    datasets_26 = [CMIP5_RCP26_R1, CESM_RCP26, HadGEM2_RCP26, CORDEX_RCP26]
    datasets_85 = [CMIP5_RCP85_R1, CESM_RCP85, HadGEM2_RCP85, CORDEX_RCP85]
    
    
    apply_style(fontsize=20, style=None, linewidth=2)
    
    if variable == "Precipitation":
        fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(20, 15),
                                                    sharex=True, sharey=True)
        
        
        
        plot_projection_comparison(stationnames=stationnames_prec, path_to_data=path_to_data_prec, 
                                            filename="predictions_", id_name="CMIP5 RCP2.6", method="Stacking", 
                                            stationloc_dir=stationloc_dir_prec, daterange=from2040to2060, 
                                            datasets=datasets_26, variable="Precipitation", 
                                            dataset_varname="tp", ax=ax1, legend=False, xlabel= "Precipitation stations",
                                            ylabel="Precipitation [mm/month]",width=0.7, title="RCP 2.6 [2040-2060]")
        
        plot_projection_comparison(stationnames=stationnames_prec, path_to_data=path_to_data_prec, 
                                            filename="predictions_", id_name="CMIP5 RCP2.6", method="Stacking", 
                                            stationloc_dir=stationloc_dir_prec, daterange=from2080to2100, 
                                            datasets=datasets_26, variable="Precipitation", 
                                            dataset_varname="tp", ax=ax2, legend=True, xlabel= "Precipitation stations",
                                            ylabel="Precipitation [mm/month]", width=0.7, title="RCP 2.6 [2080-2100]")
        
        plot_projection_comparison(stationnames=stationnames_prec, path_to_data=path_to_data_prec, 
                                            filename="predictions_", id_name="CMIP5 RCP8.5", method="Stacking", 
                                            stationloc_dir=stationloc_dir_prec, daterange=from2040to2060, 
                                            datasets=datasets_85, variable="Precipitation", 
                                            dataset_varname="tp", ax=ax3, legend=False, xlabel= "Precipitation stations",
                                            ylabel="Precipitation [mm/month]",width=0.7, title="RCP 8.5 [2040-2060]")
        
        plot_projection_comparison(stationnames=stationnames_prec, path_to_data=path_to_data_prec, 
                                            filename="predictions_", id_name="CMIP5 RCP8.5", method="Stacking", 
                                            stationloc_dir=stationloc_dir_prec, daterange=from2080to2100, 
                                            datasets=datasets_85, variable="Precipitation", 
                                            dataset_varname="tp", ax=ax4, legend=False, xlabel= "Precipitation stations",
                                            ylabel="Precipitation [mm/month]", width=0.7, title="RCP 8.5 [2080-2100]")
        
        
        plt.tight_layout(h_pad=0.03)
        plt.savefig(os.path.join(path_to_save, "compare_to_gcm_prec.svg"), bbox_inches="tight", dpi=300)
        
    
    elif variable == "Temperature":    
        fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(20, 15),
                                                   sharex=True, sharey=True)
        
        
        
        plot_projection_comparison(stationnames=stationnames_temp, path_to_data=path_to_data_temp, 
                                            filename="predictions_", id_name="CMIP5 RCP2.6", method="Stacking", 
                                            stationloc_dir=stationloc_dir_temp, daterange=from2040to2060, 
                                            datasets=datasets_26, variable="Temperature", 
                                            dataset_varname="t2m", ax=ax1, legend=False, xlabel= "Temperature stations",
                                            ylabel="Temperature [°C]",width=0.7, title="RCP 2.6 [2040-2060]")
        
        plot_projection_comparison(stationnames=stationnames_temp, path_to_data=path_to_data_temp, 
                                            filename="predictions_", id_name="CMIP5 RCP2.6", method="Stacking", 
                                            stationloc_dir=stationloc_dir_temp, daterange=from2080to2100, 
                                            datasets=datasets_26, variable="Temperature", 
                                            dataset_varname="t2m", ax=ax2, legend=True, xlabel= "Temperature stations",
                                            ylabel="Temperature [°C]", width=0.7, title="RCP 2.6 [2080-2100]")
        
        plot_projection_comparison(stationnames=stationnames_temp, path_to_data=path_to_data_temp, 
                                           filename="predictions_", id_name="CMIP5 RCP8.5", method="Stacking", 
                                           stationloc_dir=stationloc_dir_temp, daterange=from2040to2060, 
                                           datasets=datasets_85, variable="Temperature", 
                                           dataset_varname="t2m", ax=ax3, legend=False, xlabel= "Temperature stations",
                                           ylabel="Temperature [°C]",width=0.7, title="RCP 8.5 [2040-2060]")
        
        plot_projection_comparison(stationnames=stationnames_temp, path_to_data=path_to_data_temp, 
                                           filename="predictions_", id_name="CMIP5 RCP8.5", method="Stacking", 
                                           stationloc_dir=stationloc_dir_temp, daterange=from2080to2100, 
                                           datasets=datasets_85, variable="Temperature", 
                                           dataset_varname="t2m", ax=ax4, legend=False, xlabel= "Temperature stations",
                                           ylabel="Temperature [°C]", width=0.7, title="RCP 8.5 [2080-2100]")
        
        
        plt.tight_layout(h_pad=0.03)
        plt.savefig(os.path.join(path_to_save, "compare_to_gcm_temp.svg"), bbox_inches="tight", dpi=300)

def estimate_mean_std(df):
    means = df.mean(axis=1)
    stds = df.std(axis=1)
    
    return means, stds 

def plot_stations_monthly_mean():
    
    
    path_to_data_prec = os.path.join(path_exp3, prec_folder_name)
    path_to_data_temp = os.path.join(path_exp3, temp_folder_name)
    
    df_prec = monthly_mean(stationnames_prec, path_to_data_prec, filename="predictions_", 
                            daterange=from1958to2020 , id_name="obs", method= "Stacking")
    df_temp = monthly_mean(stationnames_temp, path_to_data_temp, filename="predictions_", 
                            daterange=from1958to2020 , id_name="obs", method= "Stacking")
    
    means_prec, stds_prec = estimate_mean_std(df_prec)
    means_temp, stds_temp = estimate_mean_std(df_temp)
    
    apply_style(fontsize=20, style=None, linewidth=2)
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,15), sharey=False)
    
    plot_monthly_mean(means=means_prec, stds=stds_prec, color=seablue, ylabel="Precipitation [mm/month]", 
                      ax=ax1)
    
    plot_monthly_mean(means=means_temp, stds=stds_temp, color=indianred, ylabel="Temperature [°C]", 
                      ax=ax2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_save, "station_monthly.svg"), bbox_inches="tight", dpi=300)

def save_count_predictors():
    
    path_to_data_prec = os.path.join(path_exp1, prec_folder_name)
    path_to_data_temp = os.path.join(path_exp1, temp_folder_name)
    
    selector_methods = ["Recursive", "TreeBased", "Sequential"]
    
    df_prec = count_predictors(methods=selector_methods , stationnames=stationnames_prec,
                               path_to_data=path_to_data_prec, filename="selected_predictors_",
                               predictors=predictors)
    
    df_temp = count_predictors(methods=selector_methods , stationnames=stationnames_temp,
                               path_to_data=path_to_data_temp, filename="selected_predictors_",
                               predictors=predictors)
    
    df_prec.to_csv(os.path.join(path_to_save, "predictors_prec_count.csv"))
    df_temp.to_csv(os.path.join(path_to_save, "predictors_temp_count.csv"))
    
    
    
    
if __name__ == "__main__":

    #plot_stations()
    #plot_stations_monthly_mean()
    #plot_predictor_selector_metrics()
    #plot_estimators_metrics()
    plot_prediction_example(station_num_prec=6, station_num_temp=0)
    #plot_seasonal_climatologies()
    #plot_ensemble_timeseries()
    #plot_different_projections(variable="Precipitation")
    #plot_different_projections(variable="Temperature")
    #print("--plotting complete --")
    #plot_correlation()
    #save_count_predictors()



