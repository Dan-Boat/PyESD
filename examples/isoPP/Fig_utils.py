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

