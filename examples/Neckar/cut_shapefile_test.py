# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 18:14:02 2022

@author: dboateng
"""
import xarray as xr
import rioxarray
import geopandas as gpd
from shapely.geometry import mapping


from read_data import *
from predictor_settings import *


def extract_region(data, datarange, varname, minlat, maxlat, minlon, maxlon):
        
    data = data.get(varname).sel(time=datarange)
    
    if hasattr(data, "longitude"):
        data = data.rename({"longitude":"lon", "latitude":"lat"})
    
    #data = data.assign_coords({"lon": (((data.lon + 180) % 360) - 180)})
    
    data = data.where((data.lat >=minlat) & (data.lat <=maxlat), drop=True)
    data = data.where((data.lon >=minlon) & (data.lon <= maxlon), drop=True)
    
    return data


# read shapefile 
path_to_shapefile ="C:/Users/dboateng/Desktop/Python_scripts/shape_files/b_neckar.shp"
path_to_tiff = "C:/Users/dboateng/Desktop/Python_scripts/shape_files/b_neckar.tif"

geodf = gpd.read_file(path_to_shapefile)
xds = rioxarray.open_rasterio(path_to_tiff)

# read netcdf file 
data_cmip = extract_region(data=CMIP5_AMIP_R1, datarange=fullAMIP, varname="tp", minlat=20, maxlat=80, 
                              minlon=-80, maxlon=60)

#data_cmip.rio.set_spatial_dims()
#clip values 


#plotting