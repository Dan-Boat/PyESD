# -*- coding: utf-8 -*-
"""
Created on Mon May 16 15:59:40 2022

@author: dboateng
"""

import xarray as xr
import pandas as pd 
import os 


path = "C:/Users/dboateng/Desktop/Datasets/CMIP5/Monthly/AMIP"

path_to_save = "C:/Users/dboateng/Desktop/Datasets/CMIP5/Monthly/AMIP_new"


data = xr.open_dataarray(os.path.join(path, "t500_monthly.nc"))

# formating script for cmip dataset (start month, remove other coords, change coordinate name)

for file in os.listdir(path):

    varname = file.split("_")[0]
      
    data = xr.open_dataset(os.path.join(path, file))
    
    
    if "plev" in data.coords and len(data.plev) == 1:
        data = data.drop("plev")
    
    
    data["time"] = data.time.values.astype("datetime64[M]")
    
    data = data[varname]
    
    
    data.to_netcdf(os.path.join(path_to_save, file))
    
    
