# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:58:59 2022

@author: dboateng
"""

# import modules 
import os 
import sys
import xarray as xr
import numpy as np 
import pandas as pd 
from eofs.xarray import Eof 
from sklearn import decomposition

#from local
sys.path.append("C:/Users/dboateng/Desktop/Python_scripts/ESD_Package")

from Package.Predictor_Base import Predictor
from Package.ESD_utils import map_to_xarray


def eof_analysis(data, neofs, method="eof_package", apply_equal_wtgs=True):
    
    if hasattr(data, name="longitude"):
        data = data.rename({"longitude":"lon", "latitude":"lat"})
    
    if apply_equal_wtgs == True:
        wtgs = np.sqrt(np.abs(np.cos(data.lat * np.pi / 180)))
        data = data * wtgs
    
    if True in data.isnull():
        data = data.dropna(dim="time")
        
    if method == "eof_package":
        
        solver = Eof(data)
        eofs_cov = solver.eofsAsCovariance(neofs=neofs, pscaling=1)
        
        eofs_cov = eofs_cov.sortby(eofs_cov.lon)
        
        pcs = solver.pcs(pcscaling=1, npcs=neofs)
        
    elif method == "sklearn_package":
        
        time_mean = data.mean(dim="time")
        pcs = np.empty(neofs)
        
        PCA = decomposition.PCA(n_components=neofs)
        
        X = np.reshape(data.values, (len(data.time), time_mean.size))
        
        PCA.fit(X)
        
        shape = [neofs] + list(np.shape(time_mean))
        
        np_eofs = np.reshape(PCA.components_, shape)
        pcs = PCA.explained_variance_
        
        ls_eofs= []
        for i in range(neofs):
            ls_eofs.append(map_to_xarray(X=np_eofs[i, ...], datarray=data))
            
        eofs_cov = xr.concat(ls_eofs, pd.Index(range(1, neofs+1), name="eof_number"))
    
    return eofs_cov, pcs 
        

def extract_region(data, varname, minlat, maxlat, minlon, maxlon):
    
    if hasattr(data, name="longitude"):
        data = data.rename({"longitude":"lon", "latitude":"lat"})
        
    
    data = data.assign_coords({"lon": (((data.lon + 180) % 360) - 180)})
    
    data = data.where((data.lat >=minlat) & (data.lat <=maxlat), drop=True)
    data = data.where((data.lon >=minlon) & (data.lon <= maxlon), drop=True)
    
    return data
        
        
        
        
        
    pass

class NAO(Predictor):
    def __init__(self, **kwargs):
        super().__init__(name="NAO", longname= "North Atlantic Oscilation", **kwargs)
        
    def _generate(self, datarange, data, fit, patterns_from, params_from):
        pass


class EA(Predictor):
    def __init__(self, **kwargs):
        super().__init__(name="EA", longname= "East Atlantic Oscilation", **kwargs)
    
    def _generate(self, datarange, data, fit, patterns_from, params_from):
        pass

class SCAN(Predictor):
    def __init__(self, **kwargs):
        super().__init__(name="SCAN", longname= "Scandinavian Oscilation", **kwargs)
    def _generate(self, datarange, data, fit, patterns_from, params_from):
        pass

class EA_WR(Predictor):
    def __init__(self, **kwargs):
        super().__init__(name="EAWR", longname= "East Atlantic_West Russian Oscilation", **kwargs)
    def _generate(self, datarange, data, fit, patterns_from, params_from):
        pass