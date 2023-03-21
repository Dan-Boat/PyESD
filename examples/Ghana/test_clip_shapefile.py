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
#from shapely.geometry import mapping
import regionmask
import numpy as np

# set paths
era_data_path="C:/Users/dboateng/Desktop/Datasets/ERA5/monthly_1950_2021/"
path_shapefile="C:/Users/dboateng/Desktop/Datasets/Station/Ghana/Ghana_ShapeFile/gh_wgs16dregions.shp"
afr_shape = "C:/Users/dboateng/Desktop/Datasets/Station/Ghana/Africa_shapefile/afr_g2014_2013_0.shp"



#read shapefile
africa_shape = gpd.read_file(path_shapefile)


# countries = list(africa_shape["ISO3"])
# my_list = set(list(countries))
# indexes = [countries.index(x) for x in my_list]

# countries_mask_poly = regionmask.Regions(name="ADM0_NAME", numbers=indexes, 
#                                           names=africa_shape.ADM0_NAME[indexes],
#                                           outlines=list(africa_shape.geometry.values[i] for i in range(0, africa_shape.shape[0])),
#                                           abbrevs=africa_shape["ADM0_CODE"][indexes])
# #select ghana boundaries

# from1979to2012 = pd.date_range(start="1979-01-01", end="2012-12-31", freq="MS")
# tp_monthly= xr.open_dataset(os.path.join(era_data_path, "tp_monthly.nc"))
# tp_monthly = tp_monthly["tp"].sel(time=from1979to2012).mean(dim="time")


# apply boundaries to clip


# # savefile to plot
print(None)



