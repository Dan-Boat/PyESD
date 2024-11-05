# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:58:35 2024

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


from Fig_utils import calculate_regional_means, get_metrics

# Define paths 
path_to_save = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/isoPP/plots"
main_path = "D:/Datasets/Model_output_pst/PD"
path_to_nudging = "E:/Datasets/Nudged_isotopes/"
station_datadir = "C:/Users/dboateng/Desktop/Datasets/Station/GNIP"
station_data = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/isoPP/final_exp"




# read isoGCM datasets 

# for the ECHAM6-JRA5 data 


def read_regional_means_from_isoGCM(lon, lat):
    
    from1980to2014 = pd.date_range(start="1979-01-01", end="2014-12-31", freq="MS")
    PD_data = read_from_path(main_path, "PD_1980_2014_monthly.nc", decode=True)
    PD_wiso = read_from_path(main_path, "PD_1980_2014_monthly_wiso.nc", decode=True)
    d18Op_echam5 = extract_var(Dataset=PD_data, varname="d18op", units="per mil", Dataset_wiso=PD_wiso,
                        )
    
    
    model_echam5 = pd.DataFrame(columns=["echam"], index=from1980to2014)
    
    model_echam5["echam"] = calculate_regional_means(ds=d18Op_echam5, lon_target=lon, lat_target=lat, radius_deg=200)
    
    
    
    from1980to2020 = pd.date_range(start="1979-01-01", end="2020-12-31", freq="MS")
    
    PD_echam6 = read_from_path(path_to_nudging, "d18Op_ECHAM6-wiso_JRA55.nc", decode=True)
    PD_echam6["time"] = from1980to2020
    d18Op_echam6 = PD_echam6["d18Op"]
    
    model_echam6 = pd.DataFrame(columns=["echam"], index=from1980to2020)
    
    model_echam6["echam"] = calculate_regional_means(ds=d18Op_echam6, lon_target=lon, lat_target=lat, radius_deg=200)
    
    return model_echam5, model_echam6


def get_mae_from_all():
    
    station_info = pd.read_csv(os.path.join(station_datadir, "stationnames_new.csv"), index_col=0)
    
    df_mae_models = pd.DataFrame(index=stationnames, columns=["ECHAM5-wiso", "ECHAM6-wiso[JRA55]"],)
    
    
    regressors = ["Stacking"]
    
    
    mae_df = boxplot_data(regressors=regressors, stationnames=stationnames,
                          path_to_data=station_data, filename="validation_score_", 
                          varname="test_mae")
    
    
    station_info["Stacking"] = mae_df["Stacking"]
    
    
    for num, station in enumerate(stationnames):
       
        print(station)
        stationname = station.replace("_", " ")
        
        filename = "predictions_" + "Stacking"
        
        lon = station_info.loc[num+1]["Longitude"]
        lat = station_info.loc[num+1]["Latitude"]
        
        
        df = load_csv(stationname, filename, station_data)
        
        obs = df["obs 1979-2012"]
        
        model_echam5, model_echam6 = read_regional_means_from_isoGCM(lon, lat)
        
        y_pred_echam5 = model_echam5.loc[~np.isnan(obs)]
        
        
        common_idx = obs.index.intersection(model_echam6.index)
        obs_common = obs.loc[common_idx]
        model_echam6 = model_echam6.loc[common_idx]
        
        y_pred_echam6 = model_echam6.loc[~np.isnan(obs_common)]
            
        y_true = obs.dropna()
        
        mae_echam5, r2_echam5 = get_metrics(y_true, y_pred_echam5)
        
        mae_echam6, r2_echam6 = get_metrics(y_true, y_pred_echam6)
        
        df_mae_models["ECHAM5-wiso"].loc[station] = mae_echam5
        
        df_mae_models["ECHAM6-wiso[JRA55]"].loc[station] = mae_echam6
    
    
    df_mae_models = df_mae_models.reset_index(drop=True)
    df_mae_models.index = df_mae_models.index + 1
    
    station_info["ECHAM5-wiso"] = df_mae_models["ECHAM5-wiso"].astype("float")
    station_info["ECHAM6-wiso[JRA55]"] = df_mae_models["ECHAM6-wiso[JRA55]"].astype("float")
    
    df = station_info[["Stacking", "ECHAM5-wiso", "ECHAM6-wiso[JRA55]"]]
    
    sources = ["Stacking", "ECHAM5-wiso", "ECHAM6-wiso[JRA55]"]
    colors = ["#168704", "red", "blue"]
    
    apply_style(fontsize=25, style=None, linewidth=2)

    fig, ax = plt.subplots(1,1, figsize=(15,12)) 
    
    
    vio = sns.violinplot(data=df, dodge=False,
                        palette=colors, alpha=0.8,
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
    
    sns.stripplot(data=df, palette=colors, dodge=False, ax=ax, size=10, linewidth=1.5)
    
    
    # Adjust the position of the stripplot
    for dots in vio.collections[old_len_collections:]:
        dots.set_offsets(dots.get_offsets() + np.array([0.17, 0]))
    
    
    ax.set_ylabel("MAE", fontweight="bold", fontsize=20)
    ax.grid(True, linestyle="--")
    
    # ax.legend_.remove()
    plt.grid(True, which='major', axis='y')
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.05)
    plt.savefig(os.path.join(path_to_save, "train_mae_stacking_wiso_fig3.pdf"), bbox_inches="tight", dpi=300, format="pdf")
    
    
    projection = ccrs.Robinson(central_longitude=0, globe=None)
    #projection = ccrs.PlateCarree()
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 14), subplot_kw={"projection":  
                                                                            projection})
    
    ax.set_global()
   
    # add coast lines
    ax.coastlines(resolution = "50m", linewidth=1.5, color="grey")
    
    # add land fratures using gray color
    ax.add_feature(cfeature.LAND, facecolor="gray", alpha=0.3)
    ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth = 0.3)
    
    minLon = -25
    maxLon = 21
    minLat = 42
    maxLat = 65
    
    ax.set_extent([minLon, maxLon, minLat, maxLat], ccrs.PlateCarree())
    
    # add gridlines for latitude and longitude
    gl=ax.gridlines(crs = ccrs.PlateCarree(), draw_labels = True, linewidth = 1,
                         edgecolor = "gray", linestyle = "--", color="gray", alpha=0.5)
    gl.top_labels = False                  # labesl at top
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()     # axis formatter
    gl.yformatter = LatitudeFormatter()
    
    s1 = ax.scatter(station_info['Longitude'], station_info['Latitude'], c= station_info['Stacking'], cmap="YlOrRd",
                    vmax=3, vmin=0, edgecolor="black", s= 300, transform = ccrs.PlateCarree(), linewidth=3)
    
    cbar = fig.colorbar(s1, ax=ax, orientation='vertical', pad=0.05, shrink=0.45, extend="max")
    cbar.set_label('Mean Absolute Error (MAE, ‰)')
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(path_to_save, "MAE_stacking_spatial_fig3.pdf"), 
                bbox_inches="tight", format = "pdf", dpi=300)
    
    
    
    projection = ccrs.Robinson(central_longitude=0, globe=None)
    #projection = ccrs.PlateCarree()
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 14), subplot_kw={"projection":  
                                                                            projection})
    
    ax.set_global()
   
    # add coast lines
    ax.coastlines(resolution = "50m", linewidth=1.5, color="grey")
    
    # add land fratures using gray color
    ax.add_feature(cfeature.LAND, facecolor="gray", alpha=0.3)
    ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth = 0.3)
    
    minLon = -25
    maxLon = 21
    minLat = 42
    maxLat = 65
    
    ax.set_extent([minLon, maxLon, minLat, maxLat], ccrs.PlateCarree())
    
    # add gridlines for latitude and longitude
    gl=ax.gridlines(crs = ccrs.PlateCarree(), draw_labels = True, linewidth = 1,
                         edgecolor = "gray", linestyle = "--", color="gray", alpha=0.5)
    gl.top_labels = False                  # labesl at top
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()     # axis formatter
    gl.yformatter = LatitudeFormatter()
    
    s1 = ax.scatter(station_info['Longitude'], station_info['Latitude'], c= station_info["ECHAM6-wiso[JRA55]"], cmap="YlOrRd",
                    vmax=3, vmin=0, edgecolor="black", s= 300, transform = ccrs.PlateCarree(), linewidth=3)
    
    cbar = fig.colorbar(s1, ax=ax, orientation='vertical', pad=0.05, shrink=0.45, extend="max")
    cbar.set_label('Mean Absolute Error (MAE, ‰)')
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(path_to_save, "MAE_ECHAM6_spatial_fig3.pdf"),
                bbox_inches="tight", format = "pdf", dpi=300)

    
       
# map for voilin..check inter_model plots and set colors, use df
# spatial maps...use station and chech the Fig1 and also plot_available from help function in iso2k
    
    
get_mae_from_all()