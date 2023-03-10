# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 17:02:27 2023

@author: dboateng
1. clip annual climatologies for Ghana based on the Shape file (if it works)
2. clip the future estimates based on CMIP GCMs and RCM model
"""

# import models 
import xarray as xr
import pandas as pd
import os
import geopandas as gpd
import rioxarray
from shapely.geometry import mapping


# set paths
era_data_path="C:/Users/dboateng/Desktop/Datasets/ERA5/monthly_1950_2021/"
path_shapefile="C:/Users/dboateng/Desktop/Datasets/Station/Ghana/Ghana_ShapeFile"

from1979to2012 = pd.date_range(start="1979-01-01", end="2012-12-31", freq="MS")
# read data

ghana_shape = gpd.read_file(path_shapefile)


tp_monthly= xr.open_dataset(os.path.join(era_data_path, "tp_monthly.nc"))
tp_monthly = tp_monthly["tp"].sel(time=from1979to2012).mean(dim="time")


