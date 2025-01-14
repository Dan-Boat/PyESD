# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 13:34:24 2024

@author: dboateng
"""

# Import modules 
import os 
import pandas as pd 
import numpy as np

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator
import matplotlib.dates as mdates 

from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
import seaborn as sns


from pyClimat.plot_utils import *
from pyClimat.plots import plot_annual_mean 
from pyClimat.data import read_from_path
from pyClimat.analysis import compute_lterm_mean
from pyClimat.variables import extract_var

from pyESD.plot_utils import apply_style, boxplot_data
from pyESD.plot import heatmaps
from pyESD.plot_utils import seasonal_mean

#from pyESD.plot_utils import *
from read_data import *
from predictor_setting import *
from pyESD.ESD_utils import load_csv, haversine, extract_indices_around


from Fig_utils import calculate_regional_means, get_metrics, read_regional_means_from_isoGCM

# Define paths 
path_to_save = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/isoPP/plots"
main_path = "D:/Datasets/Model_output_pst/PD"
path_to_nudging = "E:/Datasets/Nudged_isotopes/"
station_datadir = "C:/Users/dboateng/Desktop/Datasets/Station/GNIP"
station_data = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/isoPP/final_exp"



def get_mae_from_all():
    
    station_info = pd.read_csv(os.path.join(station_datadir, "stationnames_new.csv"), index_col=0)
    
    df_mae_models = pd.DataFrame(index=stationnames, 
                                 columns=["Stacking","ECHAM6-wiso[JRA55]", "MIROC[JRA55]", "IsoGCM[JRA55]", "IsoGCM[ERA5]"],)
    
    df_r2_models = pd.DataFrame(index=stationnames, 
                                 columns=["Stacking","ECHAM6-wiso[JRA55]", "MIROC[JRA55]", "IsoGCM[JRA55]", "IsoGCM[ERA5]"],)
    
    regressors = ["Stacking"]
    
    
    mae_df = boxplot_data(regressors=regressors, stationnames=stationnames,
                          path_to_data=station_data, filename="validation_score_", 
                          varname="test_mae")
    
    
    station_info["Stacking CV"] = mae_df["Stacking"]
    
    
    
    for num, station in enumerate(stationnames):
       
        print(station)
        stationname = station.replace("_", " ")
        
        filename = "predictions_" + "Stacking"
        
        lon = station_info.loc[num+1]["Longitude"]
        lat = station_info.loc[num+1]["Latitude"]
        
        
        df = load_csv(stationname, filename, station_data)
        
        obs = df["obs 2013-2018"]
        
        model = read_regional_means_from_isoGCM(lon, lat, extract_all=True)
        
        if len(model.loc[~np.isnan(obs)]) ==0:
            df_mae_models["ECHAM6-wiso[JRA55]"].loc[station] = np.nan
            df_mae_models["MIROC[JRA55]"].loc[station] = np.nan
            df_mae_models["IsoGCM[JRA55]"].loc[station] = np.nan
            df_mae_models["IsoGCM[ERA5]"].loc[station] = np.nan
            df_mae_models["Stacking"].loc[station] = np.nan
            
            
            df_r2_models["ECHAM6-wiso[JRA55]"].loc[station] = np.nan
            df_r2_models["MIROC[JRA55]"].loc[station] = np.nan
            df_r2_models["IsoGCM[JRA55]"].loc[station] = np.nan
            df_r2_models["IsoGCM[ERA5]"].loc[station] = np.nan
            df_r2_models["Stacking"].loc[station] = np.nan
            
        else:
            
        
            y_pred_models = model.loc[~np.isnan(obs)]
            
            
            common_idx = obs.index.intersection(model.index)
            obs_common = obs.loc[common_idx]
            model = model.loc[common_idx]
            
            y_pred_models = model.loc[~np.isnan(obs_common)]
            
            y_pred_models = y_pred_models.dropna()
            
            y_pred_stacking = df["ERA5 2013-2018"][~np.isnan(obs)]
                
            y_true = obs.dropna()
            
            
            mae_echam, r2_echam = get_metrics(y_true, y_pred_models["echam"])
            
            mae_miroc, r2_miroc = get_metrics(y_true, y_pred_models["miroc"])
            
            mae_isoGCM, r2_isoGCM = get_metrics(y_true, y_pred_models["isoGCM"])
            
            mae_isoGCM_era5, r2_isoGCM_era5 = get_metrics(y_true, y_pred_models["isoGCM_era5"])
            
            mae_stack, r2_stack = get_metrics(y_true, y_pred_stacking)
            
            
            df_mae_models["ECHAM6-wiso[JRA55]"].loc[station] = mae_echam
            df_mae_models["MIROC[JRA55]"].loc[station] = mae_miroc
            df_mae_models["IsoGCM[JRA55]"].loc[station] = mae_isoGCM
            df_mae_models["IsoGCM[ERA5]"].loc[station] = mae_isoGCM_era5
            df_mae_models["Stacking"].loc[station] = mae_stack
            
            
            df_r2_models["ECHAM6-wiso[JRA55]"].loc[station] = r2_echam
            df_r2_models["MIROC[JRA55]"].loc[station] = r2_miroc
            df_r2_models["IsoGCM[JRA55]"].loc[station] = r2_isoGCM
            df_r2_models["IsoGCM[ERA5]"].loc[station] = r2_isoGCM_era5
            df_r2_models["Stacking"].loc[station] = r2_stack
            
            
    
    
    df_mae_models = df_mae_models.reset_index(drop=True)
    df_mae_models.index = df_mae_models.index + 1
    
    df_r2_models = df_r2_models.reset_index(drop=True)
    df_r2_models.index = df_r2_models.index + 1
    
    station_info["ECHAM6-wiso[JRA55] MAE"] = df_mae_models["ECHAM6-wiso[JRA55]"].astype("float")
    station_info["MIROC[JRA55] MAE"] = df_mae_models["MIROC[JRA55]"].astype("float")
    station_info["IsoGCM[JRA55] MAE"] = df_mae_models["IsoGCM[JRA55]"].astype("float")
    station_info["IsoGCM[ERA5] MAE"] = df_mae_models["IsoGCM[ERA5]"].astype("float")
    station_info["Stacking MAE"] = df_mae_models["Stacking"].astype("float")
    
    
    station_info["ECHAM6-wiso[JRA55] R2"] = df_r2_models["ECHAM6-wiso[JRA55]"].astype("float")
    station_info["MIROC[JRA55] R2"] = df_r2_models["MIROC[JRA55]"].astype("float")
    station_info["IsoGCM[JRA55] R2"] = df_r2_models["IsoGCM[JRA55]"].astype("float")
    station_info["IsoGCM[ERA5] R2"] = df_r2_models["IsoGCM[ERA5]"].astype("float")
    station_info["Stacking R2"] = df_r2_models["Stacking"].astype("float")
    
    station_info.to_csv(os.path.join(path_to_save, "metrics.csv"))
    
    
get_mae_from_all()