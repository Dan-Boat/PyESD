# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 17:43:20 2024

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

# Paths and configurations
data_path = "D:/Datasets/GNIP_data/world/raw/file-703845600382738.csv"
path_to_save = "D:/Datasets/GNIP_data/world/scratch/"
main_path = "D:/Datasets/Model_output_pst/PD"
use_cols = ["Sample Site Name", "Latitude", "Longitude", "Altitude", "Sample Date", "Measurand Symbol", "Measurand Amount"]



# Function to process data for each station
def process_station(data, name):
    
    
    daterange = pd.date_range(start="1958-01-01", end="2021-12-01", freq="MS") + pd.DateOffset(days=13) # time range for ERA5
    
    df_to_store = pd.DataFrame(columns=["Time", "O18",])
    
    df_to_store["Time"] = daterange
    df_to_store = df_to_store.set_index(["Time"], drop=True)
    
    
    
    selected_rows = data[data["Sample Site Name"] == name].drop_duplicates()
    selected_rows = selected_rows[selected_rows["Measurand Symbol"] == "O18"]
    
    #selected_rows = selected_rows.set_index(["Sample Date"], drop=True)
    
    #selected_rows = selected_rows.asfreq("MS")
    print(name)    
    selected_rows = selected_rows.loc[daterange[0]:daterange[-1]] # selects only available dates
    
    if len(selected_rows["Measurand Amount"].dropna()) / 12 >= 30:
        
        
        df_to_store["O18"][selected_rows.index] = selected_rows["Measurand Amount"][selected_rows.index]
        
        df_to_store = df_to_store.replace(np.nan, -9999)
        
        df_to_store.to_csv(os.path.join(path_to_save, name + ".csv"), index=True)
        
       

# Read data in chunks
# chunk_size = 10**6  # Experiment with different chunk sizes
# data_chunks = pd.read_csv(data_path, usecols=use_cols, index_col=False, chunksize=chunk_size)



data = pd.read_csv(data_path, usecols=use_cols, index_col=False)

# Convert the 'Sampe Date' column to datetime format
data['Sample Date'] = pd.to_datetime(data['Sample Date'], format='%Y-%m-%dT%H:%M:%S.%f%z', errors='coerce', utc=True)

# Normalize the dates to remove the time and timezone information
data['Sample Date'] = data['Sample Date'].dt.tz_convert(None).dt.normalize()

# Set the 'Sampe Date' column as the index
data.set_index('Sample Date', inplace=True)

names = data["Sample Site Name"].drop_duplicates().values


process_station(data, "VALENTIA (OBSERVATORY)")
# for name in names:
#     process_station(data, name)



