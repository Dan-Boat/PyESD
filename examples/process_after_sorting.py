#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 12:14:07 2021

@author: dboateng
"""

# importing modules
import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path as p



    
def add_info(path_to_info, path_to_data, path_to_store, glob_name, varname):
    """
    

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
        
        df.at[5,0] = "Date"
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
    
    #     with open(csv_file, "a") as csv_obj:
    #         writer = csv.DictWriter(csv_obj, delimiter=',', lineterminator="\n", fieldnames=headers)
    #         writer.writeheader()
    #         writer.writerow({"ID":csv_id, "Name":csv_info["SDO_Name"], "lon": csv_info["Geogr_Laenge"], "lat": csv_info["Geogr_Breite"],
    #                          "Elevation": csv_info["Hoehe_ueber_NN"]})
    
            
    # df.to_csv(os.path.join(path_to_data, "data_info_all.csv"))
    
#paths 
path_to_data = "/home/dboateng/Datasets/Station/Rhine/cdc_download_2021-10-02_11-16_Rhine/considered"
path_to_datainfo = "/home/dboateng/Datasets/Station/Rhine/cdc_download_2021-10-02_11-16_Rhine/data/sdo_OBS_DEU_P1M_RR.csv"
path_to_store = "/home/dboateng/Datasets/Station/Rhine/cdc_download_2021-10-02_11-16_Rhine/processed"

add_info(path_to_info=path_to_datainfo, path_to_data=path_to_data, glob_name="data*", 
         varname="Precipitation", path_to_store=path_to_store)
