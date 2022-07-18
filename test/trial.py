# -*- coding: utf-8 -*-
"""
Created on Mon May 16 15:59:40 2022

@author: dboateng
"""

import xarray as xr
import pandas as pd 
import os 


path_to_save = "C:/Users/dboateng/Desktop/Datasets/CMIP5/Monthly/RCP85"

path = "C:/Users/dboateng/Desktop/Datasets/CMIP5/Monthly/preprosessed"


#data = xr.open_dataset(os.path.join(path, "z250_monthly.nc"))

# formating script for cmip dataset (start month, remove other coords, change coordinate name)

for file in os.listdir(path):
    
    print(file)

    varname = file.split("_")[0]
      
    data = xr.open_dataset(os.path.join(path, file))
    
    
    if 'lev' in data.coords and len(data.lev) == 1:
        data = data.drop('lev')
    
    
    data["time"] = data.time.values.astype("datetime64[M]")
    
    if hasattr(data, "plev") == True:
        data[varname] = data[varname][:, 0, :, :]
    
    data = data[varname]
    
    
    data.to_netcdf(os.path.join(path_to_save, file))
    
    
