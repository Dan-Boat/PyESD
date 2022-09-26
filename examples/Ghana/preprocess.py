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
path_to_store = "C:/Users/dboateng/Desktop/Datasets/Station/Ghana/Precipitation/preprocessed"
filename = "ABE.csv"
storename = "Abetifi"
varname  = "Precipitation"
use_colums = ["Time","Rain"]

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
    
    
    df_to_store[varname][df.index] = df["Rain"][df.index] 
    
    if varname == "Precipitation":
        df_to_store = df_to_store.replace(np.nan, -9999)
    
    
    df_to_store.to_csv(os.path.join(path_to_store, csv.name), index=True)    

