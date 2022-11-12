# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 11:01:28 2022

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
from pyESD.plot_utils import *

from settings import *
from read_data import *



path_to_data_prec = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Ghana/final_cache_Precipitation"
path_to_plot = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Ghana/plots"


def estimate_mean_std(df):
    means = df.mean(axis=1)
    stds = df.std(axis=1)
    
    return means, stds 



def plot_stations():
    
    df_prec_sm = seasonal_mean(stationnames_prec, path_to_data_prec, filename="predictions_", 
                            daterange=from1961to2017 , id_name="obs", method= "Stacking")
    
    
    df_prec = monthly_mean(stationnames_prec, path_to_data_prec, filename="predictions_", 
                            daterange=from1961to2017 , id_name="obs", method= "Stacking")
    
    
    means_prec, stds_prec = estimate_mean_std(df_prec)
    
    
    
    
    apply_style(fontsize=20, style=None, linewidth=2)
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(20, 15))
    
    heatmaps(data=df_prec_sm, cmap="Blues", label="Precipitation [mm/month]", title= None, 
             ax=ax1, cbar=True, xlabel="Stations")
    
    plot_monthly_mean(means=means_prec, stds=stds_prec, color=seablue, ylabel="Precipitation [mm/month]", 
                      ax=ax2)
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(path_to_plot, "station_distribution.svg"), bbox_inches="tight", dpi=300)
    
    
def plot_prediction_example():
    
    print("----plotting for the station:", stationnames_prec[2], " and ",
          stationnames_prec[12], "for precipitation ----")
    
    
    apply_style(fontsize=20, style=None, linewidth=2)
    
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(20,15), sharex=False)
    
    scatterplot(station_num=2, stationnames=stationnames_prec, path_to_data=path_to_data_prec, 
                filename="predictions_", ax=ax1, xlabel="observed", ylabel="predicted",
                method= "Stacking", obs_train_name="obs 1961-2012", 
                obs_test_name="obs 2013-2017", 
                val_predict_name="ERA5 1961-2012", 
                test_predict_name="ERA5 2013-2017")
    
    lineplot(station_num=2, stationnames=stationnames_prec, path_to_data=path_to_data_prec, 
                filename="predictions_", ax=ax3, fig=fig, ylabel="Precipitation anomalies [mm/month]",
                xlabel= "Years", method= "Stacking", obs_train_name="obs 1961-2012", 
                obs_test_name="obs 2013-2017", 
                val_predict_name="ERA5 1961-2012", 
                test_predict_name="ERA5 2013-2017")
    
    
    scatterplot(station_num=11, stationnames=stationnames_prec, path_to_data=path_to_data_prec, 
                filename="predictions_", ax=ax2, xlabel="observed", ylabel="predicted",
                method= "Stacking", obs_train_name="obs 1961-2012", 
                obs_test_name="obs 2013-2017", 
                val_predict_name="ERA5 1961-2012", 
                test_predict_name="ERA5 2013-2017")
    
    lineplot(station_num=11, stationnames=stationnames_prec, path_to_data=path_to_data_prec, 
                filename="predictions_", ax=ax4, fig=fig, ylabel="Precipitation anomalies [mm/month]",
                xlabel= "Years", method= "Stacking", obs_train_name="obs 1961-2012", 
                obs_test_name="obs 2013-2017", 
                val_predict_name="ERA5 1961-2012", 
                test_predict_name="ERA5 2013-2017")
    
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.05)
    plt.savefig(os.path.join(path_to_plot, "prediction_examp.svg"), bbox_inches="tight", dpi=300)
    
    
def plot_seasonal_climatologies():
    
    
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
    plt.savefig(os.path.join(path_to_plot, "future_precipitation_trend.svg"), bbox_inches="tight", dpi=300)
    


def plot_ensemble_timeseries():

    stationnames = ["Navrongo", "Bolgatanga", "Wa", "Yendi", "Bole"]
    
    apply_style(fontsize=20, style=None, linewidth=2)
    fig,(ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(15,20),
                                       sharex=True, sharey=True)
    plt.subplots_adjust(left=0.02, right=0.90, top=0.94, bottom=0.05,hspace=0.01)
    
    
    plot_time_series(stationnames, path_to_data_prec, filename="predictions_", 
                     id_name="CMIP5 RCP2.6 anomalies", daterange=fullCMIP5,
                     color=black, label="RCP 2.6", ymax=30, ymin=-30, 
                     ylabel= "Precipitation anomalies [mm/month]", ax=ax1,
                     window=12)
                     
    
    plot_time_series(stationnames, path_to_data_prec, filename="predictions_", 
                     id_name="CMIP5 RCP4.5 anomalies", daterange=fullCMIP5,
                     color=red, label="RCP 4.5", ymax=30, ymin=-30,
                     ylabel= "Precipitation anomalies [mm/month]", ax=ax2, 
                     )
    
    plot_time_series(stationnames, path_to_data_prec, filename="predictions_", 
                     id_name="CMIP5 RCP8.5 anomalies", daterange=fullCMIP5,
                     color=blue, label="RCP 8.5", ymax=30, ymin=-30,
                     ylabel= "Precipitation anomalies [mm/month]", ax=ax3, 
                     )
    
    
    plt.tight_layout(h_pad=0.02)
    plt.savefig(os.path.join(path_to_plot, "time_series_prec_north.svg"), bbox_inches="tight", dpi=300)
    
    
    
    stationnames = ["Wenchi", "Sunyani", "Dormaa-Ahenkro", "Kumasi", "Abetifi", "Dunkwa"]
    
    apply_style(fontsize=20, style=None, linewidth=2)
    fig,(ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(15,20),
                                       sharex=True, sharey=True)
    plt.subplots_adjust(left=0.02, right=0.90, top=0.94, bottom=0.05,hspace=0.01)
    
    
    plot_time_series(stationnames, path_to_data_prec, filename="predictions_", 
                     id_name="CMIP5 RCP2.6 anomalies", daterange=fullCMIP5,
                     color=black, label="RCP 2.6", ymax=30, ymin=-30, 
                     ylabel= "Precipitation anomalies [mm/month]", ax=ax1,
                     window=12)
                     
    
    plot_time_series(stationnames, path_to_data_prec, filename="predictions_", 
                     id_name="CMIP5 RCP4.5 anomalies", daterange=fullCMIP5,
                     color=red, label="RCP 4.5", ymax=30, ymin=-30,
                     ylabel= "Precipitation anomalies [mm/month]", ax=ax2, 
                     )
    
    plot_time_series(stationnames, path_to_data_prec, filename="predictions_", 
                     id_name="CMIP5 RCP8.5 anomalies", daterange=fullCMIP5,
                     color=blue, label="RCP 8.5", ymax=30, ymin=-30,
                     ylabel= "Precipitation anomalies [mm/month]", ax=ax3, 
                     )
    
    
    plt.tight_layout(h_pad=0.02)
    plt.savefig(os.path.join(path_to_plot, "time_series_prec_central.svg"), bbox_inches="tight", dpi=300)
    
    
    
    stationnames = ["Tarkwa", "Axim", "Takoradi", "Saltpond", "Accra", "Tema", "Akuse", "Akim-Oda"]
    
    apply_style(fontsize=20, style=None, linewidth=2)
    fig,(ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(15,20),
                                       sharex=True, sharey=True)
    plt.subplots_adjust(left=0.02, right=0.90, top=0.94, bottom=0.05,hspace=0.01)
    
    
    plot_time_series(stationnames, path_to_data_prec, filename="predictions_", 
                     id_name="CMIP5 RCP2.6 anomalies", daterange=fullCMIP5,
                     color=black, label="RCP 2.6", ymax=30, ymin=-30, 
                     ylabel= "Precipitation anomalies [mm/month]", ax=ax1,
                     window=12)
                     
    
    plot_time_series(stationnames, path_to_data_prec, filename="predictions_", 
                     id_name="CMIP5 RCP4.5 anomalies", daterange=fullCMIP5,
                     color=red, label="RCP 4.5", ymax=30, ymin=-30,
                     ylabel= "Precipitation anomalies [mm/month]", ax=ax2, 
                     )
    
    plot_time_series(stationnames, path_to_data_prec, filename="predictions_", 
                     id_name="CMIP5 RCP8.5 anomalies", daterange=fullCMIP5,
                     color=blue, label="RCP 8.5", ymax=40, ymin=-40,
                     ylabel= "Precipitation anomalies [mm/month]", ax=ax3, 
                     )
    
    
    plt.tight_layout(h_pad=0.02)
    plt.savefig(os.path.join(path_to_plot, "time_series_prec_south.svg"), bbox_inches="tight", dpi=300)
    
    
if __name__ == "__main__":
    plot_stations()
    plot_prediction_example()
    plot_seasonal_climatologies()
    plot_ensemble_timeseries()