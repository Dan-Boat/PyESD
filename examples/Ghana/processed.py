# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 11:12:17 2022

@author: dboateng
"""

import pandas as pd 
import numpy as np 
import os 
from pathlib import Path as p

path_to_data = "C:/Users/dboateng/Desktop/Datasets/Station/Ghana/Update_datasets/preprocessed/monthly"
path_to_datainfo = "C:/Users/dboateng/Desktop/Datasets/Station/Ghana/Update_datasets/stationnames.csv"
path_to_processed_monthly = "C:/Users/dboateng/Desktop/Datasets/Station/Ghana/Update_datasets/processed/monthly"
path_to_processed_daily = "C:/Users/dboateng/Desktop/Datasets/Station/Ghana/Update_datasets/processed/daily"


filename = "ABE.csv"
varname  = "Precipitation"
glob_name = "*.csv"


use_columns = ["Name", "Longitude", "Latitude", "Elevation"]

data_info = pd.read_csv(path_to_datainfo, usecols=use_columns, index_col=["Name"])

for csv in p(path_to_data).glob(glob_name):
    
    csv_id = csv.name.split(sep=".")[0]
    
    
    csv_info = data_info.loc[csv_id]
    
    data = pd.read_csv(csv, parse_dates=["Time"], dayfirst=True)
    
    data = data.rename(columns={csv_id: "Precipitation"})
    
    df_month = data.set_index("Time")
    
    time_len_month = len(df_month)
        
    daterange_month = pd.date_range(start="1961-01-01", end="2018-12-01", freq="MS")
    
    df_month_data = pd.DataFrame(index=np.arange(time_len_month + 20), columns=np.arange(2))
    
    df_month_data.at[0,0] = "Station"
    df_month_data.at[0,1] = csv_id
    
    df_month_data.at[1,0] = "Latitude"
    df_month_data.at[1,1] = float(csv_info["Latitude"])
    
    df_month_data.at[2,0] = "Longitude"
    df_month_data.at[2,1] = float(csv_info["Longitude"])
    
    df_month_data.at[3,0] = "Elevation"
    df_month_data.at[3,1] = float(csv_info["Elevation"])
    
    df_month_data.at[5,0] = "Time"
    df_month_data.at[5,1] = varname
    
    for i in range(time_len_month):
        df_month_data.loc[6+i, 0] = daterange_month[i]
        df_month_data.loc[6+i, 1] = df_month[varname][i]
        
    save_name = csv_id
    
    print("----------- saving the processed data: " + save_name + " ---------------")
    
    df_month_data.to_csv(os.path.join(path_to_processed_monthly, save_name + ".csv"), index=False,
                         header=False)
    
data_info = data_info.sort_values(by=["Latitude"], ascending=False)

data_info = data_info.reset_index(drop=False)

# saving files 

data_info.to_csv(os.path.join(path_to_processed_monthly, "stationnames.csv"))

data_info.drop("Name", axis=1, inplace=True)

data_info.to_csv(os.path.join(path_to_processed_monthly, "stationloc.csv"))


def write_data():

    for csv in p(path_to_data).glob(glob_name):
        
        csv_id = csv.name.split(sep=".")[0]
        
        
    
        csv_info = data_info.loc[csv_id]
        
        data = pd.read_csv(csv, parse_dates=["Time"], dayfirst=True)
        
        data = data.rename(columns={csv_id: "Precipitation"})
        
        data_to_month = data.set_index("Time")
        
        # resample to montly 
        data_to_month[data_to_month == -9999] = np.nan
        df_month = data_to_month.resample("M").sum()
        
        # replace nans with -9999
        df_month[df_month == np.nan] = -9999
        
        # monthly_data 
        time_len_month = len(df_month)
        
        daterange_month = pd.date_range(start="1981-01-01", end="2018-12-01", freq="MS")
        
        df_month_data = pd.DataFrame(index=np.arange(time_len_month + 20), columns=np.arange(2))
        
        df_month_data.at[0,0] = "Station"
        df_month_data.at[0,1] = csv_id
        
        df_month_data.at[1,0] = "Latitude"
        df_month_data.at[1,1] = float(csv_info["Latitude"])
        
        df_month_data.at[2,0] = "Longitude"
        df_month_data.at[2,1] = float(csv_info["Longitude"])
        
        df_month_data.at[3,0] = "Elevation"
        df_month_data.at[3,1] = float(csv_info["Elevation"])
        
        df_month_data.at[5,0] = "Time"
        df_month_data.at[5,1] = varname
        
        for i in range(time_len_month):
            df_month_data.loc[6+i, 0] = daterange_month[i]
            df_month_data.loc[6+i, 1] = df_month[varname][i]
            
            
            
            
        # daily data
        time_len = len(data)
        df = pd.DataFrame(index=np.arange(time_len + 20), columns=np.arange(2))
        
        df.at[0,0] = "Station"
        df.at[0,1] = csv_id
        
        df.at[1,0] = "Latitude"
        df.at[1,1] = float(csv_info["Latitude"])
        
        df.at[2,0] = "Longitude"
        df.at[2,1] = float(csv_info["Longitude"])
        
        df.at[3,0] = "Elevation"
        df.at[3,1] = float(csv_info["Elevation"])
        
        df.at[5,0] = "Time"
        df.at[5,1] = varname
        
        for i in range(time_len):
            df.loc[6+i, 0] = data["Time"][i]
            df.loc[6+i, 1] = data[varname][i]
        
        
        save_name = csv_id
        
        print("----------- saving the processed data: " + save_name + " ---------------")
        
        df.to_csv(os.path.join(path_to_processed_daily, save_name + ".csv"), index=False, header=False)
        df_month_data.to_csv(os.path.join(path_to_processed_monthly, save_name + ".csv"), index=False, header=False)
    
#write_data()	

# data_info = data_info.sort_values(by=["Name"], ascending=True)

# data_info = data_info.reset_index(drop=False)

# # saving files 

# data_info.to_csv(os.path.join(path_to_processed_monthly, "stationnames.csv"))
# data_info.to_csv(os.path.join(path_to_processed_daily, "stationnames.csv"))

# data_info.drop("Name", axis=1, inplace=True)

# data_info.to_csv(os.path.join(path_to_processed_monthly, "stationloc.csv"))
# data_info.to_csv(os.path.join(path_to_processed_daily, "stationloc.csv"))
