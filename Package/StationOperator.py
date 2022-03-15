#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 00:55:37 2021

@author: dboateng
"""

import pickle
import numpy as np
import pandas as pd

try:
    from .Predictor_Generator import *
    from .standardizer import MonthlyStandardizer
    from .predictand import PredictandTimeseries
    from .teleconnections import NAO
except:
    from Predictor_Generator import *
    from standardizer import MonthlyStandardizer
    from predictand import PredictandTimeseries
    from teleconnections import NAO
    
    
class StationOperator():
    
    def __init__(self, data, name, lat, lon, elevation):
        
        self.variables = {}
        for varname in data:
            self.variables[varname] = PredictandTimeseries(data[varname])
        self.name = name
        self.lat = lat
        self.lon = lon
        self.elevation = elevation
        print(name, lat, lon, elevation)
        
    def get_var(self, variable, daterange, anomalies=False):
        y = self.variables[variable].get(daterange, anomalies)
        return y
    
    def set_predictors(self, variable, predictors, cachedir, radius=250, detrending=False, scaling=False):
        predictor_list = []
        
        for name in predictors:
            if name == "NAO":
                predictor_list.append(NAO(cachedir=cachedir))
            else:
                
                predictor_list.append(RegionalAverage(name, self.lat, self.lon, radius=radius, cachedir=cachedir,
                                                      standardizer_constructor=lambda:
                                                          MonthlyStandardizer(detrending=detrending, scale=scaling)))
        
        self.variables[variable].set_predictors(predictor_list)
        
    def set_transform(self, variable, transform):
        self.variables[variable].set_transform(transform)
        
    def set_standardizer(self, variable, standardizer):
        self.variables[variable].set_standardizer(standardizer)
        
    def set_model(self, variable, model):
        self.variables[variable].set_model(model)
    
    def _get_predictor_data(self,variable, datarange, dataset, fit, **predictor_kwargs):
        return self.variables[variable]._get_predictor_data(datarange, dataset, fit, **predictor_kwargs)
    
    
    def save(self, directory=None, fname=None):
        """
        Saves the weatherstation object to a file (pickle).

        Parameters
        ----------
        directory : str, optional (default : None)
            Directory name where the pickle-file should be stored. Defaults to
            the current directory.
        fname : str, optional (default: None)
            Filename of the file where the station should be stored. Defaults
            to ``self.name.replace(' ', '_') + '.pickle'``.
        """
        if directory is None:
            directory = './'
        if fname is None:
            fname = self.name.replace(' ', '_') + '.pickle'
        filename = os.path.join(directory, fname)
        with open(f, 'wb') as f:
            pickle.dump(self, f)
            
            
def load_station(fname):
    """
    Loads a pickled station from the given file
    """
    with open(fname, 'rb') as f:
        so = pickle.load(f)
    return so
        
        
        