#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 14:02:45 2021

@author: dboateng
"""

# importing modules 
from abc import ABC, abstractmethod
import os 
import sys
import pickle
import numpy as np
import pandas as pd

def _check_data_available(data, daterange):
    for d in daterange:
        data.loc[d]
        
        
class Predictor(ABC):
    
    def __init__(self, name, longname=None, cachedir=None):
        
        self.name = name
        if longname is not None:
            self.longname = longname
        else:
            self.longname = name

        if cachedir is not None:
            self.cachedir = cachedir
        else:
            self.cachedir = '.predictors'
            
        self.data = {}
        self.params = {}
        self.patterns = {}
        
        
    def save(self):
        
        if not os.path.isdir(self.cachedir):
            try:
                os.makedirs(self.cachedir)
            except:
                print("There might be problem making the directory" + self.cachedir + 
                      "which is required to store predictors", file=sys.stderr)
                raise 
                
        
        filename_to_store = os.path.join(self.cachedir, self.longname + ".pickle")
        
        with open(filename_to_store, "wb") as f:
            predictordata = {"data":self.data, "params":self.params, "patterns":self.patterns}
            
            # serialize the the predictor data with dump()
            pickle.dump(predictordata, f)
    
    def load(self):
        filename_to_store = os.path.join(self.cachedir, self.longname + ".pickle")
        
        if not os.path.exists(filename_to_store):
            raise FileNotFoundError("Predictor data may not be available in serialize form")
            
            
        with open(filename_to_store, "rb") as f:
            
            predictordata = pickle.load(f)
            self.data = predictordata["data"]
            self.params = predictordata["params"]
            self.patterns = predictordata["patterns"]
            
            
    def get(self, daterange, dataset, fit, regenerate=False, patterns_from=None, params_from=None):
        
        if patterns_from is None:
            patterns_from = dataset.name
        if params_from is None:
            params_from = dataset.name
        
        data_key = "data=" + dataset.name + "_patterns=" +params_from + "_params=" + params_from
        
        if not self.data and not self.params and not regenerate:
            try:
                self.load()
            except FileNotFoundError:
                pass
            if not self.data:
                regenerate = True 
                
        if not regenerate:
            try:
                _check_data_available(self.data[data_key], daterange)
            except KeyError:
                regenerate=True 
                
        if regenerate:
            print("Regenerating predictor data for", self.name, "using dataset", dataset.name,
                  "with loading patterns and params from", patterns_from, "and", params_from)
            
            
            if dataset.name not in self.params:
                self.params[dataset.name] = {}
            data = self._generate(daterange, dataset, fit, patterns_from, params_from)
                
            if data_key in self.data:
                self.data[data_key] = self.data[data_key].combine_first(data)
            else:
                self.data[data_key] = data
            
            self.save()
            
        data = self.data[data_key].loc[daterange]
        
        try:
            _check_data_available(data, daterange)
        except KeyError:
            print("Predictor data for", self.name, "could not be generated for all required timesteps",
                  file=sys.stderr)
            raise 
        return data
    
    def fit(self, daterange, dataset):
        self.get(daterange, dataset, True)
    
    
    @abstractmethod
    def _generate(self, daterange, dataset, fit, patterns_from, params_from):
        ...
        
        
        
        
    def plot(self, daterange, dataset, fit, regenerate=False, patterns_from=None, params_from=None, 
             **plot_kwargs):
        data = self.get(daterange, dataset, fit, regenerate, patterns_from, params_from)
        handle = data.plot(**plot_kwargs)
        return handle
        
            
            
        
        
        
        
        
