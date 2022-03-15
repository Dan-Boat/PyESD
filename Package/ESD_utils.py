#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 14:02:28 2021

@author: dboateng
This routine contians all the utility classes and functions required for ESD functions 
"""

import xarray as xr
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt



class Dataset():
    def __init__(self, name, variables):
        self.name = name 
        self.variables = variables 
        self.data = {}
        
    def get(self, varname, domain="NH"):
        try:
            data=self.data[varname]
        
        except KeyError:
            
            self.data[varname] = xr.open_dataarray(self.variables[varname])
            
            data = self.data[varname]
            
        if domain == "NH":
            
            minlat, maxlat, minlon, maxlon = 10, 90, -90, 90
            
            if hasattr(data, "longitude"):
                data = data.assign_coords({"longitude": (((data.longitude + 180) % 360) - 180)})
                data = data.where((data.latitude >= minlat) & (data.latitude <= maxlat), drop=True)
                data = data.where((data.longitude >= minlon) & (data.longitude <= maxlon), drop=True)
            else:
                data = data.assign_coords({"lon": (((data.lon + 180) % 360) - 180)})
                data = data.where((data.lat >= minlat) & (data.lat <= maxlat), drop=True)
                data = data.where((data.lon >= minlon) & (data.lon <= maxlon), drop=True)
            
            if hasattr(data, "level"):
                data = data.squeeze(dim="level")
            return data
        
        if hasattr(data, "level"):
            data = data.squeeze(dim="level")
        else:
            return data
    
            
           
        
    
# function to estimate distance 
def haversine(lon1, lat1, lon2, lat2):
# convert decimal degrees to radians 
    lon1 = np.deg2rad(lon1)
    lon2 = np.deg2rad(lon2)
    lat1 = np.deg2rad(lat1)
    lat2 = np.deg2rad(lat2)
    
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371
    return c * r    #km     
        

def extract_indices_around(dataset, lat, lon, radius):
    close_grids = lambda lat_, lon_: haversine(lat, lon, lat_, lon_) <= radius
    
    
    if hasattr(dataset, "longitude"):
        LON, LAT = np.meshgrid(dataset.longitude, dataset.latitude)
    else:
        LON, LAT = np.meshgrid(dataset.lon, dataset.lat)
        
    grids_index = np.where(close_grids(LAT, LON))
    
    return grids_index


def map_to_xarray(X, datarray):
    
    coords = {}
    keys = []
    
    for k in datarray.coords.keys():
        
        if k != "time":
            coords[k] = datarray.coords.get(k)
            keys.append(k)
    
    if "expver" in keys:
        keys.remove("expver")
        
        
    if len(coords[keys[0]]) == np.shape(X)[0]:
        new = xr.DataArray(X, dims=keys, coords=coords)
    else:
        new = xr.DataArray(X, dims=keys[::-1], coords=coords)
        
    return new 
            
    
    
    
    