# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:21:50 2024

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



# Define paths 
path_to_save = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/isoPP/plots"
main_path = "D:/Datasets/Model_output_pst/PD"
path_to_nudging = "E:/Datasets/Nudged_isotopes/"
station_datadir = "C:/Users/dboateng/Desktop/Datasets/Station/GNIP"
station_data = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/isoPP/final_exp"



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


def get_metrics(y_true, y_pred):
    
    mae = mean_absolute_error(y_true=y_true.loc[y_pred.index], y_pred=y_pred)
    #rmse = mean_squared_error(y_true=y_true.loc[y_pred.index], y_pred=y_pred)
    
    regression_stats  = stats.linregress(y_true.loc[y_pred.index], y_pred)
    
    r2 = regression_stats.rvalue
    
    return mae, r2


def read_regional_means_from_isoGCM(lon, lat, extract_all=False):
    from1980to2020 = pd.date_range(start="1979-01-01", end="2020-12-31", freq="MS")
    
    
    PD_echam6 = read_from_path(path_to_nudging, "d18Op_ECHAM6-wiso_JRA55.nc", decode=True)
    PD_echam6["time"] = from1980to2020
    d18Op_echam6 = PD_echam6["d18Op"]
    model = pd.DataFrame(columns=["echam"], index=from1980to2020)
    model["echam"] = calculate_regional_means(ds=d18Op_echam6, lon_target=lon, lat_target=lat, radius_deg=200)
    
    if extract_all:
    
        PD_miroc = read_from_path(path_to_nudging, "d18Op_MIROC_JRA55.nc", decode=True)
        PD_miroc["time"] = from1980to2020
        d18Op_miroc = PD_miroc["d18Op"]
        
        
        PD_isoGCM_era5 = read_from_path(path_to_nudging, "d18Op_IsoGCM_ERA5.nc", decode=True)
        PD_isoGCM_era5["time"] = from1980to2020
        d18Op_isoGCM_era5 = PD_isoGCM_era5["d18Op"]
        
        PD_isoGCM = read_from_path(path_to_nudging, "d18Op_IsoGCM_JRA55.nc", decode=True)
        PD_isoGCM["time"] = from1980to2020
        d18Op_isoGCM = PD_isoGCM["d18Op"]
    

        model["miroc"] = calculate_regional_means(ds=d18Op_miroc, lon_target=lon, lat_target=lat, radius_deg=200)
        
        model["isoGCM"] = calculate_regional_means(ds=d18Op_isoGCM, lon_target=lon, lat_target=lat, radius_deg=200)
        
        model["isoGCM_era5"] = calculate_regional_means(ds=d18Op_isoGCM_era5, lon_target=lon, lat_target=lat, radius_deg=200)
        
    
    return model
