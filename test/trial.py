# -*- coding: utf-8 -*-
"""
Created on Mon May 16 15:59:40 2022

@author: dboateng
"""

import xarray as xr
import pandas as pd 
 
path = "C:/Users/dboateng/Desktop/Datasets/CMIP5/Monthly/AMIP/t850_monthly.nc"

path_era = "C:/Users/dboateng/Desktop/Datasets/ERA5/monthly_1950_2021/z700_monthly.nc"

data = xr.open_dataset(path,  )

if data.time[0].dt.is_month_start == False:
    
    # code it in a nice way in the ESD_utils to solve the different time start problem
    start = str(data.time[0].dt.strftime("%Y-%m-%d"))[40: 48] + "01"
    
    end  = str(data.time[-1].dt.strftime("%Y-%m-%d"))[40: 48] + "31"
    
    time = pd.date_range(start=start, end=end, freq = "MS")
    
    data["time"] = time 
    
    
    

data_era = xr.open_dataset(path_era,)

fullAMIP   = pd.date_range(start='1980-01-01', end='2006-12-31', freq='MS')

da = data.sel(time=fullAMIP)


print(data)