#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 12:03:47 2021

@author: dboateng
"""
# importing modules
import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path as p


# dataframe to store values and time
def extract_DWDdata_with_more_yrs(path_to_data, path_to_store, min_yrs, glob_name, varname="Precipitation", start_date=None,
                          end_date=None, data_freq=None):
    """
    

    Parameters
    ----------
    path_to_data : TYPE: str
        DESCRIPTION. path the directory containing all data 
    path_to_store : TYPE: str
        DESCRIPTION. path to store satisfied data
    yrs : TYPE: int
        DESCRIPTION. The number of years required for analysis
    glob_name : TYPE: str
        DESCRIPTION. The global pattern used to extract data (eg. *.csv)
    varname : TYPE: str, optional
        DESCRIPTION. The default is "Precipitation".
    start_date : TYPE:str, optional
        DESCRIPTION. The default is None. Start date required for the analysis eg. 1958-01-01
    end_date : TYPE: str, optional
        DESCRIPTION. The default is None. End date for data
    data_freq : TYPE:str, optional
        DESCRIPTION. The default is None. The frequency of data (eg. MS), must follow pandas setting

    Returns
    -------
    None.

    """
    if all(par is not None for par in [start_date, end_date, data_freq]):
        daterange = pd.date_range(start=start_date, end=end_date, freq=data_freq)
    else:
        daterange = pd.date_range(start="1958-01-01", end="2020-12-01", freq="MS") # time range for ERA5
    
    use_colums = ["Zeitstempel", "Wert",]
    
    df_to_store = pd.DataFrame(columns=["Time", varname,])
    
    df_to_store["Time"] = daterange
    df_to_store = df_to_store.set_index(["Time"], drop=True)
    
    
    # looping through directory
    for csv in p(path_to_data).glob(glob_name):
        df = pd.read_csv(csv, index_col=False, usecols=use_colums, parse_dates=["Zeitstempel"])
        df = df.set_index(["Zeitstempel"], drop=True)
        df = df.loc[daterange[0]:daterange[-1]] # selects only available dates
        yrs = len(df)/12
        
        if yrs >= min_yrs: # data with more than 60 years
            df_to_store["Precipitation"][df.index] = df["Wert"][df.index]
            df_to_store = df_to_store.replace(np.nan, -9999)
            
            print("saving", csv.name)
            
            df_to_store.to_csv(os.path.join(path_to_store, csv.name), index=True)
# defining paths to data
path_to_data = "/home/dboateng/Datasets/Station/Rhine/cdc_download_2021-10-02_11-16_Rhine/data"
path_to_store = "/home/dboateng/Datasets/Station/Rhine/cdc_download_2021-10-02_11-16_Rhine/considered"
       

extract_DWDdata_with_more_yrs(path_to_data=path_to_data, path_to_store=path_to_store, min_yrs=55, glob_name="data*")