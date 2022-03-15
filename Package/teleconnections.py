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
import cartopy.crs as ccrs
import cartopy.feature as cfeature 
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from cartopy.util import add_cyclic_point

#from local
sys.path.append("C:/Users/dboateng/Desktop/Python_scripts/ESD_Package")

from Package.Predictor_Base import Predictor
from Package.ESD_utils import map_to_xarray


def eof_analysis(data, neofs, method="eof_package", apply_equal_wtgs=True):
    
    if hasattr(data, "longitude"):
        data = data.rename({"longitude":"lon", "latitude":"lat"})
    
    if apply_equal_wtgs == True:
        wtgs = np.sqrt(np.abs(np.cos(data.lat * np.pi / 180)))
        data = data * wtgs
    
    if True in data.isnull():
        data = data.dropna(dim="time")
        
    if method == "eof_package":
        
        solver = Eof(data)
        eofs_cov = solver.eofsAsCovariance(neofs=neofs, pcscaling=1)
        
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
    if apply_equal_wtgs == True:
        return eofs_cov, pcs, wtgs
    
    else:
        return eofs_cov, pcs 
        

def extract_region(data, datarange, varname, minlat, maxlat, minlon, maxlon):
        
    data = data.get(varname).sel(time=datarange)
    
    if hasattr(data, "longitude"):
        data = data.rename({"longitude":"lon", "latitude":"lat"})
    
    data = data.assign_coords({"lon": (((data.lon + 180) % 360) - 180)})
    
    data = data.where((data.lat >=minlat) & (data.lat <=maxlat), drop=True)
    data = data.where((data.lon >=minlon) & (data.lon <= maxlon), drop=True)
    
    return data
        
        


class NAO(Predictor):
    def __init__(self, **kwargs):
        super().__init__(name="NAO", longname= "North Atlantic Oscilation", **kwargs)
        
    def _generate(self, datarange, dataset, fit, patterns_from, params_from):
        
        params = self.params[params_from]
        
        data = extract_region(dataset, datarange, varname="msl", minlat=20, maxlat=80, 
                              minlon=-80, maxlon=60)
        
        # removing monthly cycle
        group = data.groupby("time.month")
        
        if fit == True:
            params["monthly_means"] = group.mean(dim="time")
        
        anomalies = group - params["monthly_means"]
        anomalies = anomalies.drop("month")
        
        if fit ==True:
            params["std_field"] = anomalies.std(dim="time")
            
        anomalies /= params["std_field"]
        
        eofs, pcs, wtgs  = eof_analysis(data=anomalies, neofs=1, method="eof_package", apply_equal_wtgs=True)
        
        if fit ==True:
            self.patterns[dataset.name] = {}
            self.patterns[dataset.name]["eof"] = eofs[0]
        nao = (self.patterns[patterns_from]["eof"]* wtgs).sum(dim = ("lat", "lon"))
        nao.name = "NAO"
        
        if fit ==True:
            self.patterns[dataset.name]["std"] = nao.std()
        nao /= self.patterns[patterns_from]["std"]
        nao_series = nao.to_series()
        
        
        return nao_series
        


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