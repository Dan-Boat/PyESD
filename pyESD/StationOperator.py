#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 00:55:37 2021

@author: dboateng
"""

import pickle
import numpy as np
import pandas as pd
import os 



try:
    from .Predictor_Generator import *
    from .standardizer import MonthlyStandardizer
    from .predictand import PredictandTimeseries
    from .teleconnections import NAO, SCAN, EA, EAWR
except:
    from Predictor_Generator import *
    from standardizer import MonthlyStandardizer
    from predictand import PredictandTimeseries
    from teleconnections import NAO, SCAN, EA, EAWR
    
    
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
        
    def get_var(self, variable, daterange, anomalies=True):
        
        y = self.variables[variable].get(daterange, anomalies=anomalies)
       
        return y
    
    def set_predictors(self, variable, predictors, cachedir, radius=250, detrending=False, scaling=False,
                       standardizer=None):
        predictor_list = []
        
        for name in predictors:
            if name == "NAO":
                predictor_list.append(NAO(cachedir=cachedir))
            elif name == "EA":
                predictor_list.append(EA(cachedir=cachedir))
            
            elif name == "SCAN":
                predictor_list.append(SCAN(cachedir=cachedir))
            
            elif name == "EAWR":
                predictor_list.append(EAWR(cachedir=cachedir))
            else:
                
                if standardizer == None: 
                    predictor_list.append(RegionalAverage(name, self.lat, self.lon, radius=radius, cachedir=cachedir,
                                                      standardizer_constructor=lambda:
                                                          MonthlyStandardizer(detrending=detrending, scaling=scaling)))
                else: 
                    predictor_list.append(RegionalAverage(name, self.lat, self.lon, radius=radius, cachedir=cachedir,
                                                      standardizer_constructor=lambda:
                                                          standardizer))
        
        self.variables[variable].set_predictors(predictor_list)
        
    def set_transform(self, variable, transform):
        
        self.variables[variable].set_transform(transform)
        
    def set_standardizer(self, variable, standardizer):
        
        self.variables[variable].set_standardizer(standardizer)
        
        
        
    def set_model(self, variable, method, ensemble_learning=False, estimators=None, cv=10, final_estimator_name=None, 
                  daterange =None, predictor_dataset=None, fit_predictors=True, **predictor_kwargs):
        
        
        self.variables[variable].set_model(method, ensemble_learning=ensemble_learning, estimators=estimators, cv=cv, final_estimator_name=final_estimator_name, 
                                           daterange =daterange , predictor_dataset=predictor_dataset, 
                                           fit_predictors=fit_predictors, **predictor_kwargs)
        
        
    
    def _get_predictor_data(self,variable, daterange , dataset, fit_predictors=True, **predictor_kwargs):
       
        return self.variables[variable]._get_predictor_data(daterange , dataset, fit_predictors=fit_predictors, **predictor_kwargs)
    
    
    
    def predictor_correlation(self, variable, daterange, predictor_dataset, fit_predictors=True, fit_predictand=True, 
                              method="pearson", **predictor_kwargs):
        
        return self.variables[variable].predictor_correlation(daterange, predictor_dataset, fit_predictors=fit_predictors, 
                                                              fit_predictand=fit_predictand, 
                                  method= method, **predictor_kwargs)
    
    
    
    def fit_predictor(self, variable, name, daterange, predictor_dataset):
        
        self.variables[variable].fit_predictor(name, daterange, predictor_dataset)
        
        
        
    def fit(self, variable, daterange , predictor_dataset, fit_predictors=True , predictor_selector=True, selector_method="Recursive",
            
            selector_regressor="Ridge", num_predictors=None, selector_direction=None, cal_relative_importance=False, 
            fit_predictand=True, **predictor_kwargs):
        
        
        
        return self.variables[variable].fit(daterange , predictor_dataset, fit_predictors=fit_predictors , predictor_selector=predictor_selector, 
                                            selector_method=selector_method,
                selector_regressor= selector_regressor,
                num_predictors=num_predictors,
                selector_direction= selector_direction,
                cal_relative_importance = cal_relative_importance, 
                fit_predictand = fit_predictand,
                **predictor_kwargs)
    
    
    def predict(self, variable, daterange , predictor_dataset, fit_predictand=True, fit_predictors=True, **predictor_kwargs):
        
        return self.variables[variable].predict(daterange , predictor_dataset, fit_predictand=fit_predictand,
                                                fit_predictors=fit_predictors,
                                                **predictor_kwargs)
    
    
    def cross_validate_and_predict(self, variable, daterange , predictor_dataset, fit_predictand=True,
                                   **predictor_kwargs):
        
        return self.variables[variable].cross_validate_and_predict(daterange , predictor_dataset, 
                                                                   fit_predictand=fit_predictand,
                                                                    **predictor_kwargs)
    
    def evaluate(self, variable, daterange, predictor_dataset,  fit_predictand=True, **predictor_kwargs):
        
        return self.variables[variable].evaluate(daterange, predictor_dataset,  fit_predictand=fit_predictand, **predictor_kwargs)
    
    
    def ensemble_transform(self, variable, daterange, predictor_dataset, **predictor_kwargs):
        
        return self.variables[variable].ensemble_transform(daterange, predictor_dataset, **predictor_kwargs)
    
    def relative_predictor_importance(self, variable):
        
        return self.variables[variable].relative_predictor_importance()
    
    def selected_names(self, variable):
        
        return self.variables[variable].selected_names()
    
    
    def tree_based_feature_importance(self, variable, daterange, predictor_dataset, fit_predictand=True,
                                      plot=False, **predictor_kwargs):
        
        return self.variables[variable].tree_based_feature_importance(daterange, predictor_dataset, fit_predictand=fit_predictand,
                                                                      plot=plot, **predictor_kwargs)
    
    
    def tree_based_feature_permutation_importance(self, variable, daterange, predictor_dataset, fit_predictand=True, 
                                                  plot=False, **predictor_kwargs):
        
        return self.variables[variable].tree_based_feature_permutation_importance(daterange, predictor_dataset, fit_predictand=fit_predictand,
                                                                                  plot=plot, **predictor_kwargs)
    
    
    
    
    
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
        
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            
            
def load_station(fname):
    """
    Loads a pickled station from the given file
    """
    with open(fname, 'rb') as f:
        so = pickle.load(f)
    return so
        
        
        