# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 11:15:31 2022

@author: dboateng
"""
# importing packages
import os 
import pandas as pd
import numpy as np
from pathlib import Path as p


def extract_DWDdata_with_more_yrs(path_to_data, path_to_store, min_yrs=60, glob_name="data*.csv", varname="Precipitation", start_date=None,
                          end_date=None, data_freq=None):
    """
    1. make directory to store the data that satisfies the time criteria
    2. Set a glob name that can be used to read all data in directory eg. *data*.csv or *.csv
    The function basically check if the station has more that certain number of year and its then written into another folder (path_to_store)

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
        print("------using the default time of ERA5 dataset: 1958-2020--------")
        
        daterange = pd.date_range(start="1958-01-01", end="2021-12-01", freq="MS") # time range for ERA5
    
    use_colums = ["Zeitstempel", "Wert"]
    
    df_to_store = pd.DataFrame(columns=["Time", varname,])
    
    df_to_store["Time"] = daterange
    df_to_store = df_to_store.set_index(["Time"], drop=True)
    
    
    # looping through directory
    for csv in p(path_to_data).glob(glob_name):
        df = pd.read_csv(csv, index_col=False, usecols=use_colums, parse_dates=["Zeitstempel"])
        df = df.set_index(["Zeitstempel"], drop=True)
        
        df = df.loc[daterange[0]:daterange[-1]] # selects only available dates
        
        if data_freq is None or data_freq =="MS":
            yrs = len(df)/12
        else:
            raise Exception("different time period is not implemented")
        
        if yrs >= min_yrs: # data with more than 60 years
        
        
            df_to_store[varname][df.index] = df["Wert"][df.index]
            
            if varname == "Precipitation":
                df_to_store = df_to_store.replace(np.nan, -9999)
                
                
            elif varname == "Temperature":
                df_to_store = df_to_store.replace(np.nan, -8888)
                
                
            else:
                raise ValueError("Incorrect variable name")
            
            print("saving", csv.name)
            
            df_to_store.to_csv(os.path.join(path_to_store, csv.name), index=True)
            
            
            
            
def add_info_to_data(path_to_info, path_to_data, path_to_store, glob_name="data*", 
                     varname="Precipitation"):
    """
    This function locate the data info in data_to_info by using the station code and then append it to the start of the data 
    This function also stores a summary info file (station names and station loc) that can be used to interate all the stations when applying the downscaling package

    Parameters
    ----------
    path_to_info : TYPE: str
        DESCRIPTION. Path to the data containing all the station infomation
    path_to_data : TYPE: str
        DESCRIPTION. Path to data required for appending info
    glob_name : TYPE: str
        DESCRIPTION. The global pattern used to extract data (eg. *.csv)

    Returns
    -------
    None.

    """
    info_cols = ["SDO_ID","SDO_Name","Geogr_Laenge","Geogr_Breite","Hoehe_ueber_NN"]
    
    
    df_info = pd.DataFrame(columns= info_cols[1:])
    
    for csv_file in p(path_to_data).glob(glob_name):
        sep_filename = csv_file.name.split(sep="_")
        csv_id = int(sep_filename[-1].split(sep=".")[0])
        
       # data = pd.read_csv(csv_file)
        data_info = pd.read_csv(path_to_info, usecols=info_cols)
        data_info = data_info.set_index(["SDO_ID"])
        
        df_info = df_info.append(data_info.loc[csv_id])
        
        csv_info = data_info.loc[csv_id]
        
        # reading data from path (csv_file)
        data_in_glob = pd.read_csv(csv_file)
        
        print(csv_file)
        
        time_len = len(data_in_glob["Time"])
        
        # creating new dataFrame to store the data in required format
        
        df = pd.DataFrame(index=np.arange(800), columns=np.arange(2))
        
        # filling the df up using csv_file
        df.at[0,0] = "Station"
        df.at[0,1] = csv_info["SDO_Name"]
        
        df.at[1,0] = "Latitude"
        df.at[1,1] = float(csv_info["Geogr_Breite"].replace(",", "."))
        
        df.at[2,0] = "Longitude"
        df.at[2,1] = float(csv_info["Geogr_Laenge"].replace(",", "."))
        
        df.at[3,0] = "Elevation"
        df.at[3,1] = csv_info["Hoehe_ueber_NN"]
        
        df.at[5,0] = "Time"
        df.at[5,1] = varname
        
        #adding data from sorted file
        for i in range(time_len):
            df.loc[6+i,0] = data_in_glob["Time"][i]
            df.loc[6+i,1] = data_in_glob[varname][i]
        
        # saving the file with station name
        
        name = csv_info["SDO_Name"]
        
        # replacing special characters eg. German umlaute
        special_char_map = {ord('ä'):'ae', ord('ü'):'ue', ord('ö'):'oe', ord('ß'):'ss', 
                            ord(",") : "", ord(" ") : "_", ord("/"):"_", ord("."):"",}
        name = name.translate(special_char_map)
        
        df.to_csv(os.path.join(path_to_store, name + ".csv"), index=False, header=False)
        
    df_info = df_info.rename(columns = {"SDO_Name":"Name","Geogr_Laenge":"Longitude",
                                        "Geogr_Breite":"Latitude","Hoehe_ueber_NN":"Elevation"})
    
    # replace comma with dot
    
    df_info["Latitude"] = df_info["Latitude"].apply(lambda x:x.replace(",","."))
    
    df_info["Longitude"] = df_info["Longitude"].apply(lambda x:x.replace(",","."))
    
    df_info["Name"] = df_info["Name"].apply(lambda x:x.translate(special_char_map))
    
    df_info = df_info.sort_values(by=["Name"], ascending=True)
    
    df_info = df_info.reset_index(drop=True)
                                 
    
    df_info.to_csv(os.path.join(path_to_store, "stationnames.csv"))
    
    df_info.drop("Name", axis=1, inplace=True)
    
    df_info.to_csv(os.path.join(path_to_store, "stationloc.csv"))
    
    
# add function for Ghana datsets
