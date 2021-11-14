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


class Predictor(ABC):
    
    def __init__(self, name, longname=None, cachedir=None, resampler=None):
        
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
        
        if not os.exists(filename_to_store):
            raise FileNotFoundError("Predictor data may not be available in serialize form")
