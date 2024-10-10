# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:02:54 2024

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


from pyClimat.plot_utils import *
from pyClimat.plots import plot_annual_mean 
from pyClimat.data import read_from_path
from pyClimat.analysis import compute_lterm_mean
from pyClimat.variables import extract_var

from pyESD.plot_utils import apply_style
from pyESD.plot import heatmaps
from pyESD.plot_utils import seasonal_mean

#from pyESD.plot_utils import *
from read_data import *
from predictor_setting import *
from pyESD.ESD_utils import load_csv, haversine, extract_indices_around


from Fig_utils import calculate_regional_means

# Define paths 
path_to_save = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/isoPP/plots"
main_path = "D:/Datasets/Model_output_pst/PD"
path_to_nudging = "E:/Datasets/Nudged_isotopes/"
station_datadir = "C:/Users/dboateng/Desktop/Datasets/Station/GNIP"
station_data = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/isoPP/final_exp"




# read isoGCM datasets 

# for the ECHAM6-JRA5 data 


def read_regional_means_from_isoGCM(lon, lat):
    from1980to2020 = pd.date_range(start="1979-01-01", end="2020-12-31", freq="MS")
    
    
    PD_echam6 = read_from_path(path_to_nudging, "d18Op_ECHAM6-wiso_JRA55.nc", decode=True)
    PD_echam6["time"] = from1980to2020
    d18Op_echam6 = PD_echam6["d18Op"]
    
    
    # PD_miroc = read_from_path(path_to_nudging, "d18Op_MIROC_JRA55.nc", decode=True)
    # PD_miroc["time"] = from1980to2020
    # d18Op_miroc = PD_miroc["d18Op"]
    
    
    # PD_isoGCM = read_from_path(path_to_nudging, "d18Op_IsoGCM_JRA55.nc", decode=True)
    # PD_isoGCM["time"] = from1980to2020
    # d18Op_isoGCM = PD_isoGCM["d18Op"]
    
    model = pd.DataFrame(columns=["echam"], index=from1980to2020)
    
    
    model["echam"] = calculate_regional_means(ds=d18Op_echam6, lon_target=lon, lat_target=lat, radius_deg=200)
    
    # model["miroc"] = calculate_regional_means(ds=d18Op_miroc, lon_target=lon, lat_target=lat, radius_deg=200)
    
    # model["isoGCM"] = calculate_regional_means(ds=d18Op_isoGCM, lon_target=lon, lat_target=lat, radius_deg=200)
    
    return model





# spation plot with stations
def plot_echam_gnip_spatial():

    # load datasets 
    PD_data = read_from_path(main_path, "PD_1980_2014_monthly.nc", decode=True)
    PD_wiso = read_from_path(main_path, "PD_1980_2014_monthly_wiso.nc", decode=True)
    
    
    station_info = pd.read_csv(os.path.join(station_datadir, "stationnames_new.csv"), index_col=0)
    
    # analysis
    
    d18Op = extract_var(Dataset=PD_data, varname="d18op", units="per mil", Dataset_wiso=PD_wiso,
                        ) 
    
    d18Op_alt = compute_lterm_mean(data=d18Op, time="annual")
    
    df_sm = seasonal_mean(stationnames, station_data, filename="predictions_", 
                            daterange=from1979to2012 , id_name="obs", method= "Stacking",
                            use_id=True, transpose=False)
    
    
    #plot the scatter and spatial data (check the style for fonts and line thickness)
    station_info["Annum"] = df_sm["Annum"]
    
    
    #ploting
    
    apply_style(fontsize=22, style=None, linewidth=3, usetex=True)
    
    projection = ccrs.PlateCarree()
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(15, 13), subplot_kw={"projection": projection})
    
    plot_annual_mean(ax=ax1, variable='$\delta^{18}$Op VSMOW', data_alt=d18Op_alt, cmap="RdYlBu", units="‰", vmax=2, vmin=-18, 
                     domain="GNIP_view", levels=22, level_ticks=10, title="", left_labels=True, bottom_labels=True,
                     use_colorbar_default=True,
                     center =False, cbar_pos = [0.20, 0.05, 0.25, 0.02], orientation= "horizontal", coast_resolution='50m',
                     plot_borders=True)
    
    sc = ax1.scatter(station_info['Longitude'], station_info['Latitude'], c= station_info['Annum'], cmap="RdYlBu",
                    vmax=2, vmin=-18, edgecolor="blue", s= 300, transform = projection, linewidth=3)
    
    # for row in station_info[['Longitude', 'Latitude']].itertuples():
    #     ax1.text(row.Longitude - 0.1, row.Latitude + 0.1, row.Index, 
    #               transform = ccrs.Geodetic(), va="center", ha="right", 
    #               fontsize=15, alpha=0.8, fontweight="bold")
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(path_to_save, "GNIP_stations_fig1.pdf"), bbox_inches="tight", format = "pdf", dpi=300)

def get_metrics(y_true, y_pred):
    
    mae = mean_absolute_error(y_true=y_true.loc[y_pred.index], y_pred=y_pred)
    #rmse = mean_squared_error(y_true=y_true.loc[y_pred.index], y_pred=y_pred)
    
    regression_stats  = stats.linregress(y_true.loc[y_pred.index], y_pred["echam"])
    
    r2 = regression_stats.rvalue
    
    return mae, r2


def plot_time_series(station_number):
    
    from1980to2014 = pd.date_range(start="1979-01-01", end="2014-12-31", freq="MS")
    
    # load datasets 
    PD_data = read_from_path(main_path, "PD_1980_2014_monthly.nc", decode=True)
    PD_wiso = read_from_path(main_path, "PD_1980_2014_monthly_wiso.nc", decode=True)
    d18Op = extract_var(Dataset=PD_data, varname="d18op", units="per mil", Dataset_wiso=PD_wiso,
                        )
    
    
    
    stationname = stationnames[station_number-1].replace("_", " ")
    print(stationname)
    
    station_info = pd.read_csv(os.path.join(station_datadir, "stationnames_new.csv"), index_col=0)
    
    lon = station_info.loc[station_number]["Longitude"]
    lat = station_info.loc[station_number]["Latitude"]
    
    model = pd.DataFrame(columns=["echam"], index=from1980to2014)
    
    model["echam"] = calculate_regional_means(ds=d18Op, lon_target=lon, lat_target=lat, radius_deg=50)
    
    model_nudged = read_regional_means_from_isoGCM(lon, lat)
    
   
    
    print("extracting information for the station: ", stationname)
    
    filename = "predictions_" + "Stacking"
    
    df = load_csv(stationname, filename, station_data)
    
    obs = df["obs"]
    
    # estimate metrics
    y_pred = model.loc[~np.isnan(obs)]
    
    
    common_idx = obs.index.intersection(model_nudged.index)
    obs_common = obs.loc[common_idx]
    model_nudged = model_nudged.loc[common_idx]
    
    y_pred_nudged = model_nudged.loc[~np.isnan(obs_common)]
        
    y_true = obs.dropna()
    
    mae_echam5, r2_echam5 = get_metrics(y_true, y_pred)
    
    mae_echam6, r2_echam6 = get_metrics(y_true, y_pred_nudged)
    
    
    model = model.rolling(3, min_periods=1, win_type="hann", center=True).mean()
    obs = obs.rolling(3, min_periods=1, win_type="hann", center=True).mean()
    
    model_nudged = model_nudged.rolling(3, min_periods=1, win_type="hann", center=True).mean()
    
    
    apply_style(fontsize=22, style=None, linewidth=3, usetex=True)
    fig, ax = plt.subplots(1, 1, figsize= (15, 6), sharex=True)
    
    plt.subplots_adjust(left=0.12, right=1-0.01, top=0.98, bottom=0.06, hspace=0.01)
    
    ax.plot(obs, linestyle="-", color=black, label=stationname.capitalize() + "," + str(station_number) + " (GNIP)")
    ax.plot(model, linestyle="-", color=red, 
            label="ECHAM5-wiso[MAE={:.2f} ‰, R²={:.2f}]".format(mae_echam5, r2_echam5))
    ax.plot(model_nudged, color="blue", linestyle="-", 
            label="ECHAM6-wiso(JRA55)[MAE={:.2f} ‰, R²={:.2f}]".format(mae_echam6, r2_echam6))
    
    
    ax.xaxis.set_major_locator(YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.legend(bbox_to_anchor=(0.01, 1.02, 1., 0.102), loc=3, ncol=2, borderaxespad=0., frameon = True, 
              fontsize=18)
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(path_to_save, stationname + "_inter_annual_fig1.pdf"), bbox_inches="tight", format = "pdf", dpi=300)




plot_echam_gnip_spatial()

plot_time_series(station_number=8)