#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 14:03:09 2021

@author: dboateng
"""
# importing modules

import numpy as np
import xarray as xr
import pandas as pd




try:
    from .Predictor_Base import Predictor
    from .ESD_utils import extract_indices_around
except:
    from Predictor_Base import Predictor
    from ESD_utils import extract_indices_around
    
    
class RegionalAverage(Predictor):
    
    def __init__(self, name, lat, lon, standardizer_constructor=None, 
                 radius=250, **kwargs):
        
        self.lon = lon
        self.lat = lat
        self.varname = name 
        self.radius = radius
        self.standardizer_constructor = standardizer_constructor
        
        super().__init__(name, name + '_{:2.2f}N_{:2.2f}E'.format(self.lat, self.lon),
                         **kwargs)
        
    
    def _generate(self, daterange, dataset, fit, patterns_from, params_from):
        
        params = self.params[params_from]
        
        da = dataset.get(self.varname)
        
        da = da.sel(time=daterange)
        
        #da = da[self.varname]
        
        if hasattr(da, "longitude"):
            da = da.rename({"longitude":"lon", "latitude":"lat"})
            
        if "indices" not in params or fit is True:
            params["indices"] = extract_indices_around(da, self.lat, self.lon, self.radius)
        values = da.isel(lat=params["indices"][0], lon=params["indices"][1])
        
        data = values.mean(dim=("lat", "lon")).to_series().astype(np.double)
        
        if self.standardizer_constructor is not None:
            if fit:
                params["standardizer"] = self.standardizer_constructor()
                params["standardizer"].fit(data)
            data = params["standardizer"].transform(data)
            
        return pd.Series(data)
    
    