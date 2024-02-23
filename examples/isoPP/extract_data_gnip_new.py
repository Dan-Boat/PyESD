# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 20:00:32 2024

@author: dboateng
"""

import pandas as pd 
import os 
import numpy as np


#path to data 

data_path = "D:/Datasets/GNIP_data/Africa/raw/file-702194037013507.csv"

path_to_save = "D:/Datasets/GNIP_data/Africa/scratch/"

use_cols = ["Sample Site Name", "Latitude", "Longitude", "Altitude", "Sample Date", 
            "Measurand Symbol", "Measurand Amount", ]


data = pd.read_csv(data_path, usecols=use_cols, parse_dates=["Sample Date"], index_col=False)

df_info = pd.DataFrame(index=np.arange(200), columns=["Name", "lat", "lon", "elev","years", "d18op"])

stationnames = list(data["Sample Site Name"].drop_duplicates().values)


for i,name in enumerate(stationnames):
    selected_rows = data[data["Sample Site Name"] == name]
    selected_rows = selected_rows.drop_duplicates()
    
    
    # select only O18 variable
    
    selected_rows = selected_rows[selected_rows["Measurand Symbol"] == "O18"]
    
    output_file = f"{name}_output.csv"
    
    selected_rows = selected_rows.reset_index(drop=True)
    print(name)
    if  len(selected_rows["Measurand Amount"].dropna())/12 >= 5:
        
        df_info["Name"][i] = name
        df_info["lat"][i] = selected_rows["Latitude"][0]
        df_info["lon"][i] = selected_rows["Longitude"][0]
        df_info["elev"][i] = selected_rows["Altitude"][0]
        
        df_info["years"][i] = len(selected_rows["Measurand Amount"].dropna())/12
        df_info["d18op"][i] = selected_rows["Measurand Amount"].mean()
        
        
        #selected_rows.to_csv(os.path.join(path_to_save, output_file), index=False)
df_info = df_info.dropna()        
df_info.to_csv(os.path.join(path_to_save, "station_overview.csv"), index=False)