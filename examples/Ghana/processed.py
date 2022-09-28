# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 11:12:17 2022

@author: dboateng
"""

import pandas as pd 
import numpy as np 
import os 
from pathlib import Path as p

path_to_data = "C:/Users/dboateng/Desktop/Datasets/Station/Ghana/Temperature/preprocessed"
path_to_datainfo = "C:/Users/dboateng/Desktop/Datasets/Station/Ghana/Temperature/stationnames.csv"
path_to_processed = "C:/Users/dboateng/Desktop/Datasets/Station/Ghana/Temperature/processed"


filename = "ABE.csv"
varname  = "Temperature"
glob_name = "*.csv"


use_columns = ["ID", "Name", "Longitude", "Latitude"]
data_info = pd.read_csv(path_to_datainfo, usecols=use_columns, index_col=["ID"])



for csv in p(path_to_data).glob(glob_name):
    
    csv_id = csv.name.split(sep=".")[0]
    
    

    csv_info = data_info.loc[csv_id]
    
    data = pd.read_csv(csv)
    
    time_len = len(data["Time"])
    
    df = pd.DataFrame(index=np.arange(800), columns=np.arange(2))
    
    df.at[0,0] = "Station"
    df.at[0,1] = csv_info["Name"]
    
    df.at[1,0] = "Latitude"
    df.at[1,1] = float(csv_info["Latitude"])
    
    df.at[2,0] = "Longitude"
    df.at[2,1] = float(csv_info["Longitude"])
    
    df.at[3,0] = "Elevation"
    df.at[3,1] = -9999
    
    df.at[5,0] = "Time"
    df.at[5,1] = varname
    
    for i in range(time_len):
        df.loc[6+i, 0] = data["Time"][i]
        df.loc[6+i, 1] = data[varname][i]
    
    
    save_name = csv_info["Name"]
    
    print("----------- saving the processed data: " + save_name + " ---------------")
    
    df.to_csv(os.path.join(path_to_processed, save_name + ".csv"), index=False, header=False)
    


data_info = data_info.sort_values(by=["Name"], ascending=True)

data_info = data_info.reset_index(drop=True)

# saving files 

data_info.to_csv(os.path.join(path_to_processed, "stationnames.csv"))

data_info.drop("Name", axis=1, inplace=True)

data_info.to_csv(os.path.join(path_to_processed, "stationloc.csv"))
