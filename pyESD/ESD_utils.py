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
import pickle 
import os 
from collections import OrderedDict 

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
            
            if self.data[varname].time[0].dt.is_month_start == False:
                
                # code it in a nice way in the ESD_utils to solve the different time start problem
                start = str(self.data[varname].time[0].dt.strftime("%Y-%m-%d"))[40: 48] + "01"
                
                end  = str(self.data[varname].time[-1].dt.strftime("%Y-%m-%d"))[40: 48] + "31"
                
                time = pd.date_range(start=start, end=end, freq = "MS")
                
                self.data[varname]["time"] = time
                
                if hasattr(self.data[varname], "plev"):
                    
                    self.data[varname] = self.data[varname].drop_vars(["plev"])
                    
            
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
            
    
def store_pickle(stationname, varname, var, cachedir):
    filename = stationname.replace(' ', '_')
    fname = os.path.join(cachedir, filename + '_' + varname + '.pickle') 
    with open(fname, 'wb') as f:
        pickle.dump(var, f)


def load_pickle(stationname, varname, path):
    filename = stationname.replace(' ', '_')
    fname = os.path.join(path, filename + '_' + varname + '.pickle')
    with open(fname, 'rb') as f:
        return pickle.load(f)

def store_csv(stationname, varname, var, cachedir):
    filename = stationname.replace(' ', '_')
    fname = os.path.join(cachedir, filename + '_' + varname + '.csv') 
    var.to_csv(fname)


def load_csv(stationname, varname, path):
    filename = stationname.replace(' ', '_')
    fname = os.path.join(path, filename + '_' + varname + '.csv') 
    return pd.read_csv(fname, index_col=0, parse_dates=True)



def load_all_stations(varname, path, stationnames):
    """
    This assumes that the stored quantity is a dictionary

    Returns a dictionary
    """
    values_dict = OrderedDict()

    for stationname in stationnames:
        values_dict[stationname] = load_pickle(stationname, varname, path)

    df = pd.DataFrame(values_dict).transpose()
   
    # get right order
    columns = list(values_dict[stationname].keys())
    df =  df.loc[stationnames]
    return df[columns]

    


def ranksums_test():
    
    # use scipy.stats ranksums
    pass

def levene_test():
    
    # use scipy.stats.levene
    pass

def ks_test():
    # use scipy.stats.ks_2samp
    
    pass