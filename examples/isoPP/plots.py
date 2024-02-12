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

from read_data import *
from predictor_setting import *

# DIRECTORY SPECIFICATION
# =======================


path_exp1 = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/isoPP/final_cache_GNIP"
path_exp2 = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/isoPP/model_selection"
path_exp3 = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Neckar/experiment3"
path_to_save = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/isoPP/plots"


prec_folder_name = "final_cache_Precipitation"
temp_folder_name = "final_cache_Temperature"

# PLOTTING STATION SEASONAL MEANS 
# ===============================



def plot_stations():
    
    
    df_sm = seasonal_mean(stationnames, path_exp2, filename="predictions_", 
                            daterange=from1979to2012 , id_name="obs", method= "Stacking", use_id=True)
    
    df_sm.columns = df_sm.columns.str.replace("_", " ")
    

    
    df = monthly_mean(stationnames, path_exp2, filename="predictions_", 
                            daterange=from1979to2012 , id_name="obs", method= "Stacking")
    df.columns = df.columns.str.replace("_", " ")
   
    
    means, stds = estimate_mean_std(df)
    
    apply_style(fontsize=20, style=None, linewidth=2)
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(20, 15))
    
    heatmaps(data=df_sm, cmap="RdYlBu", label='$\delta^{18}$Op vs SMOW', title= None, 
             ax=ax1, cbar=True, xlabel='$\delta^{18}$Op vs SMOW', vmax=2, vmin=-16)
    
    plot_monthly_mean(means=means, stds=stds, color=seablue, ylabel='$\delta^{18}$Op vs SMOW', 
                      ax=ax2)
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(path_to_save, "StationMonthly_variability.pdf"), bbox_inches="tight", format = "pdf", dpi=300)

    
    
    
    

# PLOTTING METRICS FOR PREDICTOR SELECTOR
# =======================================

def plot_predictor_selector_metrics():
    
    selector_methods = ["Recursive", "TreeBased"]
    
    apply_style(fontsize=20, style=None, linewidth=2)
    
    fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2, figsize=(20, 15), sharex=False)
    
    barplot(methods=selector_methods, stationnames=stationnames, path_to_data=path_exp1,
            xlabel="Precipitation Stations", ylabel="CV R²", varname= "test_r2", varname_std ="test_r2_std",
            filename="validation_score_", ax=ax1, legend=True, width=0.7, rot=90)
    
    barplot(methods=selector_methods, stationnames=stationnames , path_to_data=path_exp1, 
            xlabel="Precipitation Stations", ylabel="CV MAE", varname= "test_mae", varname_std ="test_mae_std",
            filename="validation_score_", ax=ax2, legend=False, width=0.7, rot=90)
    
    
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.94, bottom=0.06)
    plt.savefig(os.path.join(path_to_save, "predictor_selection_metrics.svg"), bbox_inches="tight", dpi=300)


def plot_correlation():
    
    df_cor = correlation_data(stationnames, path_exp1, "corrwith_predictors", predictors, 
                               use_id=True)
    
    
    apply_style(fontsize=22, style=None, linewidth=2) 
    
    fig, ax = plt.subplots(1,1, figsize=(20,15))
                            
    correlation_heatmap(data=df_cor, cmap="RdBu", ax=ax, vmax=1, vmin=-1, center=0, cbar_ax=None, fig=fig,
                            add_cbar=True, title=None, label= "Pearson Correlation Coefficinet (PCC)", fig_path=path_to_save,
                            xlabel="Predictors", ylabel="Precipitation Stations", fig_name="correlation_prec.svg",)
    
    fig, ax = plt.subplots(1,1, figsize=(20,15))
                            
    correlation_heatmap(data=df_temp, cmap="RdBu", ax=ax, vmax=1, vmin=-1, center=0, cbar_ax=None, fig=fig,
                            add_cbar=True, title=None, label= "Pearson Correlation Coefficinet (PCC)", fig_path=path_to_save,
                            xlabel="Predictors", ylabel="GNIP Stations", fig_name="correlation_temp.svg",)
   




# PLOTTING INTER-ESTIMATOR METRICS
# ================================

def plot_estimators_metrics():
   
   
    
    regressors = ["LassoLarsCV", "ARD", "MLP", "RandomForest",
                  "XGBoost", "Bagging", "Stacking"]
    
    colors = [grey, purple, lightbrown, tomato, skyblue, lightgreen, gold]
    
    apply_style(fontsize=20, style=None, linewidth=2)
    
    fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2, figsize=(20, 15), sharex=False)
    
    boxplot(regressors, stationnames, path_exp2, ax=ax1,  
                varname="test_rmse", filename="validation_score_", xlabel="Estimators",
                ylabel="CV RMSE", colors = colors, patch_artist=(True))
    
    boxplot(regressors, stationnames, path_exp2, ax=ax2,  
                varname="test_mae", filename="validation_score_", xlabel="Estimators",
                ylabel="CV MAE", colors = colors, patch_artist=(True))
    
    
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.05)
    plt.savefig(os.path.join(path_to_save, "Inter-estimator.png"), bbox_inches="tight", dpi=300)


# PLOTTING PREDICTION EXAMPLES
# ============================

def plot_prediction_example(station_num):
   
    
   
    
    apply_style(fontsize=22, style=None, linewidth=3)
    
    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(22,18), sharex=False)
    
    scatterplot(station_num=station_num, stationnames=stationnames, path_to_data=path_exp2, 
                filename="predictions_", ax=ax1, xlabel="observed", ylabel="predicted",
                method= "Stacking", 
                obs_train_name="obs_train", 
                obs_test_name=None, 
                val_predict_name="ERA5 1979-2012", 
                test_predict_name=None,
                obs_full_name="obs_full",)
    
    lineplot(station_num=station_num, stationnames=stationnames, path_to_data=path_exp2, 
                filename="predictions_", ax=ax3, fig=fig, ylabel="d18Op",
                xlabel= "Years", method= "Stacking")
    
    
    
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.05)
    plt.savefig(os.path.join(path_to_save, "prediction_example_for_station.svg"), bbox_inches="tight", dpi=300)


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
    
    selector_methods = ["Recursive", "TreeBased"]
    
    df_prec = count_predictors(methods=selector_methods , stationnames=stationnames_prec,
                               path_to_data=path_to_data_prec, filename="selected_predictors_",
                               predictors=predictors)
    
    df_temp = count_predictors(methods=selector_methods , stationnames=stationnames_temp,
                               path_to_data=path_to_data_temp, filename="selected_predictors_",
                               predictors=predictors)
    
    df_prec.to_csv(os.path.join(path_to_save, "predictors_prec_count.csv"))
    df_temp.to_csv(os.path.join(path_to_save, "predictors_temp_count.csv"))
    


def plot_data_available(datarange, stationnames, use_id=False):
    
    apply_style(fontsize=20, style=None, linewidth=2)
    df_obs = pd.DataFrame(index = datarange, columns = stationnames)
    
    n = len(stationnames)
    for i in range(n):
        stationname = stationnames[i]
        df = load_csv(stationname, "predictions_ARD", path_exp2)
        obs = df['obs'][datarange]
        
        df_obs[stationname] = obs
       
    df_obs = df_obs.T
    df_obs.index = df_obs.index.str.replace("_", " ")
    
    if use_id:
        df_obs.reset_index(drop=True, inplace=True)
        df_obs.index +=1
    
    df_obs = df_obs.astype(float)
    df_obs[~df_obs.isnull()] = 1
    df_obs[df_obs==0] = 1
    df_obs[df_obs.isnull()] = 0
    df_obs.columns = df_obs.columns.strftime("%Y")
    
    fig, axes = plt.subplots(1, 1, figsize= (15,12))
    
    sns.heatmap(ax = axes, data= df_obs, cbar = False, linewidth = 0.3, yticklabels=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_save, "weather_station_overview.pdf"), bbox_inches="tight", format = "pdf", dpi=300)
                   
    
if __name__ == "__main__":

    #plot_stations()
    #plot_stations_monthly_mean()
    #plot_predictor_selector_metrics()
    #plot_estimators_metrics()
    #plot_prediction_example(station_num_prec=6, station_num_temp=0)
    #print("--plotting complete --")
    #plot_correlation()
    #save_count_predictors()
    plot_data_available(from1979to2018, stationnames)
    #plot_stations()
    #plot_prediction_example(3)



