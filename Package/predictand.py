#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 00:55:22 2021

@author: dboateng
"""

import numpy as np
import pandas as pd
import seaborn as sns
from copy import copy
from collections import OrderedDict

try:
    from .standardizer import MonthlyStandardizer, NoStandardizer
except:
    from standardizer import MonthlyStandardizer, NoStandardizer
    
    
class PredictandTimeseries():
    
    def __init__(self, data, transform=None, standardizer=None, model=None):
        
        self.data = data
        self.transform = None
        self.standardizer = None

        if transform is not None:
            self.set_transform(transform)
        self.set_standardizer(standardizer)
        if model is not None:
            self.set_model(model)
            
    def get(self, daterange=None, anomalies=False):
        if anomalies:
            if daterange is not None:
                return self.data_st.loc[daterange]
            else:
                return copy(self.data_st)
        else:
            if daterange is not None:
                return self.data.loc[daterange]
            else:
                return copy(self.data)
            
    def set_transform(self, transform):
        if transform == "log":
            self.transform = np.log
            self.backtransform = np.exp
        else:
            raise NotImplementedError("Yet to implement additional transformers to log")
        self.data_st = self.transform(self.data)
        
    def set_standardizer(self, standardizer):
        if standardizer is not None:
            self.standardizer = standardizer
        else:
            self.standardizer = NoStandardizer()
        
        if self.transform is not None:
            self.data_st = self.transform(self.data)
            self.data_st = self.standardizer.fit_transform(self.data_st)
        else:
            self.data_st = self.standardizer.fit_transform(self.data)
            
    def set_model(self, model):
        pass
    
    def set_predictors(self, predictors):
        self.predictors = OrderedDict()
        for p in predictors:
            self.predictors[p.name] = p
            
    def _get_predictor_data(self, daterange, dataset, fit, **predictor_kwargs):
        Xs = []
        
        for p in self.predictors:
            Xs.append(self.predictors[p].get(daterange, dataset, fit, **predictor_kwargs))
            
        return pd.concat(Xs, axis=1)
    

            
        
            
        
        
        