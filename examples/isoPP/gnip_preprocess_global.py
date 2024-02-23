# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:35:22 2024

@author: dboateng
"""

import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed  # For parallel processing
import multiprocessing  # For determining number of CPU cores

from pyESD.ESD_utils import load_csv, haversine, extract_indices_around
from pyClimat.data import read_from_path
from pyClimat.analysis import compute_lterm_mean
from pyClimat.variables import extract_var

# Function to calculate regional means
def calculate_regional_means(ds, lon_target, lat_target, radius_deg):
    if hasattr(ds, "longitude"):
        ds = ds.rename({"longitude": "lon", "latitude": "lat"})
        
    ds = ds.assign_coords({"lon": (((ds.lon + 180) % 360) - 180)})
    
    indices = extract_indices_around(ds, lat_target, lon_target, radius_deg)
    
    regional_mean = ds.isel(lat=indices[0], lon=indices[1]).mean()
        
    return np.float64(regional_mean)

# Function to process data for each station
def process_station(data, d18Op, name):
    selected_rows = data[data["Sample Site Name"] == name].drop_duplicates()
    selected_rows = selected_rows[selected_rows["Measurand Symbol"] == "O18"]
    
    if len(selected_rows["Measurand Amount"].dropna()) / 12 >= 10:
        lat = selected_rows["Latitude"].iloc[0]
        lon = selected_rows["Longitude"].iloc[0]
        
        years = len(selected_rows["Measurand Amount"].dropna()) / 12
        d18op_mean = selected_rows["Measurand Amount"].mean()
        
        echam = calculate_regional_means(ds=d18Op, lon_target=lon, lat_target=lat, radius_deg=100)
        
        return [name, lat, lon, selected_rows["Altitude"].iloc[0], years, d18op_mean, echam]
    else:
        return None

# Paths and configurations
data_path = "D:/Datasets/GNIP_data/world/raw/file-703845600382738.csv"
path_to_save = "D:/Datasets/GNIP_data/world/scratch/"
main_path = "D:/Datasets/Model_output_pst/PD"
use_cols = ["Sample Site Name", "Latitude", "Longitude", "Altitude", "Sample Date", "Measurand Symbol", "Measurand Amount"]

# Load datasets
PD_data = read_from_path(main_path, "PD_1980_2014_monthly.nc", decode=True)
PD_wiso = read_from_path(main_path, "PD_1980_2014_monthly_wiso.nc", decode=True)
d18Op = extract_var(Dataset=PD_data, varname="d18op", units="per mil", Dataset_wiso=PD_wiso)

# Read data in chunks
chunk_size = 10**6  # Experiment with different chunk sizes
data_chunks = pd.read_csv(data_path, usecols=use_cols, parse_dates=["Sample Date"], index_col=False, chunksize=chunk_size)

# Parallel processing
num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores)(
    delayed(process_station)(chunk, d18Op, name)
    for chunk in data_chunks
    for name in chunk["Sample Site Name"].drop_duplicates().values
)

# Concatenate results and drop NAs
df_info = pd.DataFrame([res for res in results if res is not None], columns=["Name", "lat", "lon", "elev", "years", "d18op", "echam"])
df_info = df_info.dropna()

# Save results
df_info.to_csv(os.path.join(path_to_save, "station_world_overview.csv"), index=False)
