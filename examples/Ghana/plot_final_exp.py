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



path_to_data_prec = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Ghana/final_experiment"
path_to_plot = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Ghana/plots"


def estimate_mean_std(df):
    means = df.mean(axis=1)
    stds = df.std(axis=1)
    
    return means, stds 



def plot_stations():
    
    df_prec_sm = seasonal_mean(stationnames_prec, path_to_data_prec, filename="predictions_", 
                            daterange=from1981to2017 , id_name="obs", method= "Stacking")
    
    
    df_prec = monthly_mean(stationnames_prec, path_to_data_prec, filename="predictions_", 
                            daterange=from1981to2017 , id_name="obs", method= "Stacking")
    
    
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
    
    for i,station in enumerate(stationnames_prec):
        
    
        print("----plotting for the station:", station + "precipitation")
        
        apply_style(fontsize=20, style=None, linewidth=2)
        
        fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15,13), sharex=False)
        
        scatterplot(station_num=i, stationnames=stationnames_prec, path_to_data=path_to_data_prec, 
                    filename="predictions_", ax=ax1, xlabel="observed", ylabel="predicted",
                    method= "Stacking", obs_train_name="obs 1961-2012", 
                    obs_test_name="obs 2013-2017", 
                    val_predict_name="ERA5 1961-2012", 
                    test_predict_name="ERA5 2013-2017")
        
        lineplot(station_num=i, stationnames=stationnames_prec, path_to_data=path_to_data_prec, 
                    filename="predictions_", ax=ax2, fig=fig, ylabel="Precipitation anomalies [mm/month]",
                    xlabel= "Years", method= "Stacking", obs_train_name="obs 1961-2012", 
                    obs_test_name="obs 2013-2017", 
                    val_predict_name="ERA5 1961-2012", 
                    test_predict_name="ERA5 2013-2017")
        
        
        # scatterplot(station_num=15, stationnames=stationnames_prec, path_to_data=path_to_data_prec, 
        #             filename="predictions_", ax=ax2, xlabel="observed", ylabel="predicted",
        #             method= "Stacking", obs_train_name="obs 1961-2012", 
        #             obs_test_name="obs 2013-2017", 
        #             val_predict_name="ERA5 1961-2012", 
        #             test_predict_name="ERA5 2013-2017")
        
        # lineplot(station_num=15, stationnames=stationnames_prec, path_to_data=path_to_data_prec, 
        #             filename="predictions_", ax=ax4, fig=fig, ylabel="Precipitation anomalies [mm/month]",
        #             xlabel= "Years", method= "Stacking", obs_train_name="obs 1961-2012", 
        #             obs_test_name="obs 2013-2017", 
        #             val_predict_name="ERA5 1961-2012", 
        #             test_predict_name="ERA5 2013-2017")
        
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.05)
        plt.savefig(os.path.join(path_to_plot, "prediction_examp_" + station + "_.svg"), bbox_inches="tight", dpi=300)
    
    
def plot_seasonal_climatologies():
    apply_style(fontsize=23, style="seaborn-talk", linewidth=3,)
    
    
    df_prec_26_from2040to2060 = seasonal_mean(stationnames_prec, path_to_data_prec, filename="predictions_", 
                            daterange=from2040to2060 , id_name="CMIP6 RCP2.6 anomalies", method= "Stacking")
    
    
    df_prec_85_from2040to2060 = seasonal_mean(stationnames_prec, path_to_data_prec, filename="predictions_", 
                            daterange=from2040to2060 , id_name="CMIP6 RCP8.5 anomalies", method= "Stacking")
    
    
    
    df_prec_26_from2080to2100 = seasonal_mean(stationnames_prec, path_to_data_prec, filename="predictions_", 
                            daterange=from2080to2100 , id_name="CMIP6 RCP2.6 anomalies", method= "Stacking")
    
    
    df_prec_85_from2080to2100 = seasonal_mean(stationnames_prec, path_to_data_prec, filename="predictions_", 
                            daterange=from2080to2100 , id_name="CMIP6 RCP8.5 anomalies", method= "Stacking")
    
    
    
    
    fig, ((ax1,ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(20,15), sharex=False)
    cbar_ax = fig.add_axes([0.90, 0.35, 0.02, 0.25])
    
    heatmaps(data=df_prec_26_from2040to2060, cmap=BrBG, label="Precipitation [mm/month]", title= "RCP 2.6   (2040-2060)", 
             ax=ax1, cbar=True, cbar_ax=cbar_ax, vmax=20, vmin=-20, center=0,)
    
    heatmaps(data=df_prec_85_from2040to2060, cmap=BrBG, label="Precipitation [mm/month]", title= "RCP 8.5", 
             ax=ax3, cbar=False, vmax=20, vmin=-20, center=0, xlabel="Precipitation stations")
    
    heatmaps(data=df_prec_26_from2080to2100, cmap=BrBG, label="Precipitation [mm/month]", title= "RCP 2.6   (2080-2100)", 
             ax=ax2, cbar=False, vmax=20, vmin=-20, center=0,)
    
    heatmaps(data=df_prec_85_from2080to2100, cmap=BrBG, label="Precipitation [mm/month]", title= "RCP 8.5", 
             ax=ax4, cbar=False, vmax=20, vmin=-20, center=0, xlabel="Precipitation stations")
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.02, right=0.90, top=0.94, bottom=0.05,hspace=0.0)
    plt.savefig(os.path.join(path_to_plot, "future_precipitation_trend.svg"), bbox_inches="tight", dpi=300)
    


def plot_ensemble_timeseries():
    apply_style(fontsize=23, style="seaborn-talk", linewidth=3,)

    stationnames = ["Navrongo", "Bolgatanga", "Wa","Bole"]
    
    
    fig,(ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(15,20),
                                       sharex=True, sharey=True)
    plt.subplots_adjust(left=0.02, right=0.90, top=0.94, bottom=0.05,hspace=0.01)
    
    
    plot_time_series(stationnames, path_to_data_prec, filename="predictions_", 
                     id_name="CMIP6 RCP2.6 anomalies", daterange=fullCMIP6,
                     color=black, label="RCP 2.6", ymax=40, ymin=-40, 
                     ylabel= "Precipitation anomalies [mm/month]", ax=ax1,
                     window=12)
                     
    
    plot_time_series(stationnames, path_to_data_prec, filename="predictions_", 
                     id_name="CMIP6 RCP4.5 anomalies", daterange=fullCMIP6,
                     color=red, label="RCP 4.5", ymax=40, ymin=-40,
                     ylabel= "Precipitation anomalies [mm/month]", ax=ax2, 
                     )
    
    plot_time_series(stationnames, path_to_data_prec, filename="predictions_", 
                     id_name="CMIP6 RCP8.5 anomalies", daterange=fullCMIP6,
                     color=blue, label="RCP 8.5", ymax=40, ymin=-40,
                     ylabel= "Precipitation anomalies [mm/month]", ax=ax3, 
                     )
    
    
    plt.tight_layout(h_pad=0.02)
    plt.savefig(os.path.join(path_to_plot, "time_series_prec_north.svg"), bbox_inches="tight", dpi=300)
    
    
    
    stationnames = ["Wenchi", "Sunyani", "Dormaa-Ahenkro", "Kumasi", "Abetifi", "Dunkwa"]
    
    
    fig,(ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(15,20),
                                       sharex=True, sharey=True)
    plt.subplots_adjust(left=0.02, right=0.90, top=0.94, bottom=0.05,hspace=0.01)
    
    
    plot_time_series(stationnames, path_to_data_prec, filename="predictions_", 
                     id_name="CMIP6 RCP2.6 anomalies", daterange=fullCMIP6,
                     color=black, label="RCP 2.6", ymax=40, ymin=-40, 
                     ylabel= "Precipitation anomalies [mm/month]", ax=ax1,
                     window=12)
                     
    
    plot_time_series(stationnames, path_to_data_prec, filename="predictions_", 
                     id_name="CMIP6 RCP4.5 anomalies", daterange=fullCMIP6,
                     color=red, label="RCP 4.5", ymax=40, ymin=-40,
                     ylabel= "Precipitation anomalies [mm/month]", ax=ax2, 
                     )
    
    plot_time_series(stationnames, path_to_data_prec, filename="predictions_", 
                     id_name="CMIP6 RCP8.5 anomalies", daterange=fullCMIP6,
                     color=blue, label="RCP 8.5", ymax=40, ymin=-40,
                     ylabel= "Precipitation anomalies [mm/month]", ax=ax3, 
                     )
    
    
    plt.tight_layout(h_pad=0.02)
    plt.savefig(os.path.join(path_to_plot, "time_series_prec_central.svg"), bbox_inches="tight", dpi=300)
    
    
    
    stationnames = ["Tarkwa", "Axim", "Takoradi", "Saltpond", "Accra", "Tema", "Akuse", "Akim-Oda"]
    
    
    fig,(ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(15,20),
                                       sharex=True, sharey=True)
    plt.subplots_adjust(left=0.02, right=0.90, top=0.94, bottom=0.05,hspace=0.01)
    
    
    plot_time_series(stationnames, path_to_data_prec, filename="predictions_", 
                     id_name="CMIP6 RCP2.6 anomalies", daterange=fullCMIP6,
                     color=black, label="RCP 2.6", ymax=40, ymin=-40, 
                     ylabel= "Precipitation anomalies [mm/month]", ax=ax1,
                     window=12)
                     
    
    plot_time_series(stationnames, path_to_data_prec, filename="predictions_", 
                     id_name="CMIP6 RCP4.5 anomalies", daterange=fullCMIP6,
                     color=red, label="RCP 4.5", ymax=40, ymin=-40,
                     ylabel= "Precipitation anomalies [mm/month]", ax=ax2, 
                     )
    
    plot_time_series(stationnames, path_to_data_prec, filename="predictions_", 
                     id_name="CMIP6 RCP8.5 anomalies", daterange=fullCMIP6,
                     color=blue, label="RCP 8.5", ymax=40, ymin=-40,
                     ylabel= "Precipitation anomalies [mm/month]", ax=ax3, 
                     )
    
    
    plt.tight_layout(h_pad=0.02)
    plt.savefig(os.path.join(path_to_plot, "time_series_prec_south.svg"), bbox_inches="tight", dpi=300)

def plot_different_projections(variable= "Precipitation"):
    apply_style(fontsize=23, style="seaborn-talk", linewidth=3,)
    
    
    datasets_26 = [CMIP6_RCP26_R1, CESM_RCP26, HadGEM2_RCP26]
    datasets_85 = [CMIP6_RCP85_R1, CESM_RCP85, HadGEM2_RCP85]
    
    
    apply_style(fontsize=20, style=None, linewidth=2)
    
    if variable == "Precipitation":
        fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(20, 15),
                                                    sharex=True, sharey=True)
        
        
        stationloc_dir_prec = os.path.join(station_prec_datadir , "stationloc.csv")
        plot_projection_comparison(stationnames=stationnames_prec, path_to_data=path_to_data_prec, 
                                            filename="predictions_", id_name="CMIP6 RCP2.6", method="Stacking", 
                                            stationloc_dir=stationloc_dir_prec, daterange=from2040to2060, 
                                            datasets=datasets_26, variable="Precipitation", 
                                            dataset_varname="tp", ax=ax1, legend=False, xlabel= "Precipitation stations",
                                            ylabel="Precipitation [mm/month]",width=0.7, title="RCP 2.6 [2040-2060]")
        
        plot_projection_comparison(stationnames=stationnames_prec, path_to_data=path_to_data_prec, 
                                            filename="predictions_", id_name="CMIP6 RCP2.6", method="Stacking", 
                                            stationloc_dir=stationloc_dir_prec, daterange=from2080to2100, 
                                            datasets=datasets_26, variable="Precipitation", 
                                            dataset_varname="tp", ax=ax2, legend=True, xlabel= "Precipitation stations",
                                            ylabel="Precipitation [mm/month]", width=0.7, title="RCP 2.6 [2080-2100]")
        
        plot_projection_comparison(stationnames=stationnames_prec, path_to_data=path_to_data_prec, 
                                            filename="predictions_", id_name="CMIP6 RCP8.5", method="Stacking", 
                                            stationloc_dir=stationloc_dir_prec, daterange=from2040to2060, 
                                            datasets=datasets_85, variable="Precipitation", 
                                            dataset_varname="tp", ax=ax3, legend=False, xlabel= "Precipitation stations",
                                            ylabel="Precipitation [mm/month]",width=0.7, title="RCP 8.5 [2040-2060]")
        
        plot_projection_comparison(stationnames=stationnames_prec, path_to_data=path_to_data_prec, 
                                            filename="predictions_", id_name="CMIP6 RCP8.5", method="Stacking", 
                                            stationloc_dir=stationloc_dir_prec, daterange=from2080to2100, 
                                            datasets=datasets_85, variable="Precipitation", 
                                            dataset_varname="tp", ax=ax4, legend=False, xlabel= "Precipitation stations",
                                            ylabel="Precipitation [mm/month]", width=0.7, title="RCP 8.5 [2080-2100]")
        
        
        plt.tight_layout(h_pad=0.03)
        plt.savefig(os.path.join(path_to_plot, "GCMs comparison.svg"), bbox_inches="tight", dpi=300)


def write_metrics(path_to_data, method, stationnames, path_to_save, varname,
                  filename_train = "validation_score_",
                  filename_test="test_score_13to17_"):
    
    train_score = load_all_stations(filename_train + method, path_to_data, stationnames)
    
    test_score = load_all_stations(filename_test + method, path_to_data, stationnames)
    
    #climate_score = load_all_stations("climate_score_13to17_" + method, path_to_data, stationnames)
    
    df = pd.concat([train_score, test_score], axis=1, ignore_index=False)
    
    scores_df = load_all_stations(varname="CV_scores_" + method, path=path_to_data, 
                                  stationnames= stationnames)
    
        
    df_add = pd.DataFrame(index=stationnames_prec, columns= ["r2", "rmse", "mae"])
    for i,stationname in enumerate(stationnames):
        
        r2 = scores_df["test_r2"].loc[stationname]
        mae = -1* scores_df["test_neg_mean_absolute_error"].loc[stationname]
        index_max = r2.argmax()
        index_min = mae.argmin()
        
        df_add["r2"].loc[stationname] = scores_df["test_r2"].loc[stationname][index_max]
        df_add["rmse"].loc[stationname] = -1 * scores_df["test_neg_root_mean_squared_error"].loc[stationname][index_max]
        df_add["mae"].loc[stationname] = -1* scores_df["test_neg_mean_absolute_error"].loc[stationname][index_min]
    
    # save files
    
    df.to_csv(os.path.join(path_to_save, method + "_train_test_metrics.csv"), index=True, header=True)
    df_add.to_csv(os.path.join(path_to_save, method + "_CV_metrics.csv"), index=True, header=True)
    
if __name__ == "__main__":
    #plot_stations()
    #plot_prediction_example()
    #plot_seasonal_climatologies()
    #plot_ensemble_timeseries()
    # plot_different_projections()
    write_metrics(path_to_data_prec, "Stacking", stationnames_prec, path_to_plot, 
              "Precipitation")