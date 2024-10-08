# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:50:10 2024

@author: dboateng
"""
# Import modules 
import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.dates import YearLocator
import matplotlib.dates as mdates 

from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats


from pyClimat.plot_utils import *
from pyClimat.plots import plot_annual_mean 
from pyClimat.data import read_ERA_processed, read_ECHAM_processed, read_from_path
from pyClimat.analysis import compute_lterm_mean, compute_lterm_diff
from pyClimat.variables import extract_var

from pyESD.plot_utils import apply_style
from pyESD.plot import heatmaps
from pyESD.plot_utils import seasonal_mean
#from pyESD.plot_utils import *
from read_data import *
from predictor_setting import *
from pyESD.ESD_utils import load_csv, haversine, extract_indices_around

# Define paths 
path_to_save = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/isoPP/plots"
main_path = "D:/Datasets/Model_output_pst/PD"
station_datadir = "C:/Users/dboateng/Desktop/Datasets/Station/GNIP"
station_data = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/isoPP/final_exp"
path_to_nudging = "E:/Datasets/Nudged_isotopes/"







def calculate_regional_means(ds, lon_target, lat_target, radius_deg,):
    """
    Calculate regional means around a specific longitude and latitude location
    with a given radius for a NetCDF dataset using xarray.
    """
    # Find indices of the nearest grid point to the target location
    
    if hasattr(ds, "longitude"):
        ds = ds.rename({"longitude":"lon", "latitude":"lat"})
        
    ds = ds.assign_coords({"lon": (((ds.lon + 180) % 360) - 180)})
    
    indices = extract_indices_around(ds, lat_target, lon_target, radius_deg)
    
    regional_mean = ds.isel(lat=indices[0], lon=indices[1]).mean(dim=("lon", "lat")).data
        
    return np.float64(regional_mean)

def plot_example():
    
    df_mae = pd.DataFrame(index=stationnames, columns=["ECHAM5-wiso", "ECHAM6-wiso[JRA55]", "MIROC[JRA55]", "IsoGCM[JRA55]"])
    
    #df_mae_echam6 = pd.DataFrame(index=stationnames, columns=["ECHAM6-wiso"])
    
    
    from1980to2014 = pd.date_range(start="1979-01-01", end="2014-12-31", freq="MS")
    
    
    
    # for the ECHAM5 data 
    
    # load datasets 
    PD_data = read_from_path(main_path, "PD_1980_2014_monthly.nc", decode=True)
    PD_wiso = read_from_path(main_path, "PD_1980_2014_monthly_wiso.nc", decode=True)
    d18Op = extract_var(Dataset=PD_data, varname="d18op", units="per mil", Dataset_wiso=PD_wiso,
                        )
    
    # for the ECHAM6-JRA5 data 
    from1980to2020 = pd.date_range(start="1979-01-01", end="2020-12-31", freq="MS")
    
    
    PD_echam6 = read_from_path(path_to_nudging, "d18Op_ECHAM6-wiso_JRA55.nc", decode=True)
    PD_echam6["time"] = from1980to2020
    d18Op_echam6 = PD_echam6["d18Op"]
    
    
    PD_miroc = read_from_path(path_to_nudging, "d18Op_MIROC_JRA55.nc", decode=True)
    PD_miroc["time"] = from1980to2020
    d18Op_miroc = PD_miroc["d18Op"]
    
    
    PD_isoGCM = read_from_path(path_to_nudging, "d18Op_IsoGCM_JRA55.nc", decode=True)
    PD_isoGCM["time"] = from1980to2020
    d18Op_isoGCM = PD_isoGCM["d18Op"]
    
    
    
    
    for num, station in enumerate(stationnames):
        print(station)
        stationname = station.replace("_", " ")
    
    
    
   
    
        station_info = pd.read_csv(os.path.join(station_datadir, "stationnames_new.csv"), index_col=0)
        lon = station_info.loc[num+1]["Longitude"]
        lat = station_info.loc[num+1]["Latitude"]
        
        model = pd.DataFrame(columns=["echam"], index=from1980to2014)
        
        model_echam6 = pd.DataFrame(columns=["echam6"], index=from1980to2020)
        
        model_miroc = pd.DataFrame(columns=["miroc"], index=from1980to2020)
        
        model_isoGCM = pd.DataFrame(columns=["isoGCM"], index=from1980to2020)
        
        model["echam"] = calculate_regional_means(ds=d18Op, lon_target=lon, lat_target=lat, radius_deg=200)
        model_echam6["echam6"] = calculate_regional_means(ds=d18Op_echam6, lon_target=lon, lat_target=lat, radius_deg=200)
        
        model_miroc["miroc"] = calculate_regional_means(ds=d18Op_miroc, lon_target=lon, lat_target=lat, radius_deg=200)
        
        model_isoGCM["isoGCM"] = calculate_regional_means(ds=d18Op_isoGCM, lon_target=lon, lat_target=lat, radius_deg=200)
        
       
        
        print("extracting information for the station: ", stationname)
        
        filename = "predictions_" + "Stacking"
        
        df = load_csv(stationname, filename, station_data)
        
        obs = df["obs"]
        
        # estimate metrics
        
        # for echam5
        y_pred = model.loc[~np.isnan(obs)]
        
        
        #for echam6
        
        # Step 1: Find common indices between obs and model_echam6
        common_idx = obs.index.intersection(model_echam6.index)
        
        # Step 2: Subset obs and model_echam6 using the common indices
        obs_common = obs.loc[common_idx]
        
        
        model_echam6_common = model_echam6.loc[common_idx]
        y_pred_echam6 = model_echam6_common.loc[~np.isnan(obs_common)]
        
        model_miroc_common = model_miroc.loc[common_idx]
        y_pred_miroc = model_miroc_common.loc[~np.isnan(obs_common)]
        
        model_isoGCM_common = model_isoGCM.loc[common_idx]
        y_pred_isoGCM = model_isoGCM_common.loc[~np.isnan(obs_common)]

        
            
        y_true = obs.dropna()
        
        mae = mean_absolute_error(y_true=y_true.loc[y_pred.index], y_pred=y_pred)
        
        mae_echam6 = mean_absolute_error(y_true=y_true.loc[y_pred_echam6.index], y_pred=y_pred_echam6)
        
        mae_miroc = mean_absolute_error(y_true=y_true.loc[y_pred_miroc.index], y_pred=y_pred_miroc)
        
        y_pred_isoGCM = y_pred_isoGCM.dropna()
        
        # if station == "GRAZ_UNIVERSITAET":
        #     print(len(y_pred_isoGCM))
        #     y_pred_isoGCM.info()
        mae_isoGCM = mean_absolute_error(y_true=y_true.loc[y_pred_isoGCM.index], y_pred=y_pred_isoGCM)
       
        
        # regression_stats  = stats.linregress(y_true.loc[y_pred.index], y_pred["echam"])
        
        # regression_slope = regression_stats.slope * obs + regression_stats.intercept
        
        # r2 = regression_stats.rvalue
        
        
        df_mae["ECHAM5-wiso"].loc[station] = mae
        
        df_mae["ECHAM6-wiso[JRA55]"].loc[station] = mae_echam6
        
        df_mae["MIROC[JRA55]"].loc[station] = mae_miroc
        
        df_mae["IsoGCM[JRA55]"].loc[station] = mae_isoGCM
        
        #df_mae["r2"].loc[station] = r2
    
    df_mae = df_mae.reset_index(drop=True)
    
    
    
    apply_style(fontsize=22, style=None, linewidth=2)
    
    fig, ax = plt.subplots(1,1, figsize=(15,12)) 
    
    # sns.violinplot(x=' ', y='', data=df, palette='pastel', width=0.7, alpha=0.5, inner_kws=dict(box_width=18, whis_width=2),)
    vio = sns.violinplot(data=df_mae, dodge=False,
                        alpha=0.8,
                        width=0.7, inner_kws=dict(box_width=14, whis_width=2, color="black"),
                        bw_adjust=0.67, ax=ax)
    
    
    ax.tick_params(axis='x', rotation=45)
    
    # Adjust the width of the violin plot to show only half
    # for violin in vio.collections:
    #     bbox = violin.get_paths()[0].get_extents()
    #     x0, y0, width, height = bbox.bounds
    #     violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=vio.transData))
    
    # Adjust the width of the violin plot to show only half
    
    for violin in vio.collections:
        paths = violin.get_paths()
        for path in paths:
            vertices = path.vertices
            vertices[:, 0] = np.clip(vertices[:, 0], min(vertices[:, 0]), np.mean(vertices[:, 0]))
            path.vertices = vertices
    
    
    
    
    
            
            
    old_len_collections = len(vio.collections)
    
    sns.stripplot(data=df_mae, dodge=False, ax=ax, size=10, linewidth=1.5)
    
    
    # Adjust the position of the stripplot
    for dots in vio.collections[old_len_collections:]:
        dots.set_offsets(dots.get_offsets() + np.array([0.17, 0]))
    
    
    ax.set_ylabel("MAE", fontweight="bold", fontsize=20)
    ax.grid(True, linestyle="--")
    
    # ax.legend_.remove()
    plt.grid(True, which='major', axis='y')
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.05)
    plt.savefig(os.path.join(path_to_save, "models_vs_obs_mae.png"), bbox_inches="tight", dpi=300, format="png")
  
    
    
plot_example()