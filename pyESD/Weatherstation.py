#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 14:01:43 2021


This routine handles the preprocessing of data downloaded directly from DWD. The default time series is monthly, others frequency must be pass to the function
1. Extracting only stations with required number of years 
2. Writing additional information into files (eg. station name, lat, lon and elevation), since its downloaded into a separate file using station codes
3. All utils function to read stations into pyESD Station operator class

Note: This routine is specifically designed for data downloded from DWD (otherwise please contact daniel.boateng@uni-tuebingen.de for assistance on other datasets)

@author: dboateng
"""

# importing packages
import os 
import pandas as pd
import numpy as np

#from StationOperator import StationOperator
# #local packages
try:
    from .StationOperator import StationOperator
except:
    from StationOperator import StationOperator
    



def read_weatherstationnames(path_to_data):
    """
    This function reads all the station names in the data directory

    Parameters
    ----------
    path_to_data : TYPE: str
        DESCRIPTION. The directory path to where all the station data are stored

    Returns
    -------
    namedict : TYPE: dict
        DESCRIPTION.

    """

    nr, name = np.loadtxt(os.path.join(path_to_data, 'stationnames.csv'),
                              delimiter=',', skiprows=1, usecols=(0,1),
                              dtype=str, unpack=True)

    nr = [int(i) for i in nr]
    namedict = dict(zip(nr, name))
    return namedict


def read_station_csv(filename, varname, return_all=False):
    """
    

    Parameters
    ----------
    filename : TYPE: str
        DESCRIPTION. Name of the station in path 
    varname : TYPE: str
        DESCRIPTION. The name of the varibale to downscale (eg. Precipitation, Temperature)

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    ws : TYPE
        DESCRIPTION.

    """
    
    # reading headers info with readline 
    with open(filename, "r") as f:
        name = f.readline().split(',')[1].replace("\n", "")
        lat = float(f.readline().split(',')[1].replace("\n",""))
        lon = float(f.readline().split(',')[1].replace("\n",""))
        elev = float(f.readline().split(',')[1].replace("\n",""))
        
        
        
    data = pd.read_csv(filename, sep=',', skiprows=6, usecols=[0,1,],
                       parse_dates=[0], index_col=0, names=['Time',
                       varname])
    
    data = data.dropna()
    
    if varname == "Precipitation":
        pr = data[varname]
        pr[pr == -9999] = np.nan
        assert not np.any(pr < -1e-2)
        assert not np.any(pr > 2000)
        
        data = {varname:pr}
        
    elif varname == "Temperature":
        t = data[varname]
        t[t == -8888] = np.nan
        assert not np.any(t < -50)
        assert not np.any(t > 80)
        
        data = {varname:t}
        
    elif varname == "O18":
        d18O = data[varname]
        d18O[d18O == -9999] = np.nan
    else:
        raise ValueError("The model does not recognize the variable name")

    if return_all == False:

        so = StationOperator(data, name, lat, lon, elev)
        
        return so
    
    else:
        return data, lat, lon
    
    
    
    
def read_weatherstations(path_to_data):
    """
    Read all the station data in a directory.

    Parameters
    ----------
    path_to_data : TYPE: STR
        DESCRIPTION. relative or absolute path to the station folder

    Returns
    -------
    stations : TYPE: DICT
        DESCRIPTION. Dictionary containing all the datasets

    """
    namedict = read_weatherstationnames(path_to_data)
    stations = {}
    for i in namedict:
        filename = namedict[i].replace(' ', '_') + '.csv'
        print("Reading", filename)
        ws = read_station_csv(os.path.join(path_to_data, filename))
        stations[i] = ws
    return stations