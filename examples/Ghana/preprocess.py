# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 13:00:31 2022

@author: dboateng
"""
import pandas as pd 
import numpy as np 
import os 
from pathlib import Path as p


data_path = "C:/Users/dboateng/Desktop/Datasets/Station/Ghana/Stacked_Data/"
path_to_store = "C:/Users/dboateng/Desktop/Datasets/Station/Ghana/Temperature/preprocessed"

path_to_data_new = "C:/Users/dboateng/Desktop/Datasets/Station/Ghana/Update_datasets/"
path_to_store_new = "C:/Users/dboateng/Desktop/Datasets/Station/Ghana/Update_datasets/preprocessed/monthly"

daterange_monthly = pd.date_range(start="1981-01-01", end="2018-12-01", freq="MS")

daterange_daily = pd.date_range(start="1981-01-01", end="2018-12-01", freq="D")

file_name_monthly = "stack_monthly_Ghana_rainfall_stations.csv"

file_name_daily = "stack_daily_Ghana_rainfall_stations.csv"



def daily_preprocess():
    data_daily = pd.read_csv(os.path.join(path_to_data_new, file_name_daily), parse_dates=["Time"], 
                             dayfirst=True)
    
    
    # reindex 
    df = data_daily.set_index(["Time"], drop=True)
    
    df = df.replace(999, np.nan)
    df = df.replace(np.nan, -9999)
    
    #extract range
    colum_names = df.columns.values.tolist()
    
    
    # loop through all the list and extract the stations to store
    
    for i,station in enumerate(colum_names):
        
        data_stn = df[station]
        data_stn.to_csv(os.path.join(path_to_store_new, station + ".csv"))

#monthly 
data_month = pd.read_csv(os.path.join(path_to_data_new, file_name_monthly), 
                         parse_dates=["Time"], dayfirst=True)

df = data_month.set_index(["Time"], drop=True)

df[df > 2000] = -9999
df[df < 0] = -9999

colum_names = df.columns.values.tolist()

for i,station in enumerate(colum_names):
    
    data_stn = df[station]
    data_stn.to_csv(os.path.join(path_to_store_new, station + ".csv"))





# extract values, replace 999 or assert not more than 1000, resample with skipna



def process_data_stacked(path_to_data, path_to_store):
    
    varname  = "Temperature"
    use_colums = ["Time","Tmean"]

    daterange = pd.date_range(start="1961-01-01", end="2013-12-01", freq="MS")
    
    df_to_store = pd.DataFrame(columns = ["Time", varname])
    df_to_store["Time"] = daterange
    df_to_store = df_to_store.set_index(["Time"], drop=True)
    
    
    
    glob_name = "*.csv"
    
    for csv in p(data_path).glob(glob_name):
    
        
        print(csv.name)
    
        # df_stats = df.describe()
        # df_nans = df.isna().sum()
    
        df = pd.read_csv(csv, usecols=use_colums,
                         parse_dates=["Time"], dayfirst=True)
        
        
        df = df.set_index(["Time"], drop=True)
        
        # select the data that is part of the dates
        df = df.loc[daterange[0]:daterange[-1]]
        
        
        df_to_store[varname][df.index] = df["Tmean"][df.index] 
        
        if varname == "Precipitation":
            df_to_store = df_to_store.replace(np.nan, -9999)
        
        elif varname == "Temperature":
            df_to_store = df_to_store.replace(np.nan, -8888)
            
        
        df_to_store.to_csv(os.path.join(path_to_store, csv.name), index=True)    

