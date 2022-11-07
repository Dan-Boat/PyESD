# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 10:00:08 2022

@author: dboateng
"""

# import libraries
import os
from collections import OrderedDict
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


# import pyESD modules 
from pyESD.Weatherstation import read_station_csv
from pyESD.standardizer import StandardScaling, NoStandardizer
from pyESD.ESD_utils import store_csv, store_pickle
from pyESD.plot import correlation_heatmap
from pyESD.plot_utils import apply_style, correlation_data

# relative files import 
from read_data import *
from settings import *

#directories
corr_dir = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Ghana/correlation_data"

radius = 50  #km
variable = "Precipitation"

def generate_correlation():
    num_of_stations = len(stationnames_prec)
    
    for i in range(num_of_stations):
        
        stationname = stationnames_prec[i]
        station_dir = os.path.join(station_prec_datadir, stationname + ".csv")
        
        SO = read_station_csv(filename=station_dir, varname=variable)
        
        # set predictors 
        SO.set_predictors(variable, predictors, predictordir, radius, 
                          standardizer=StandardScaling(method="standardscaling"))
        
        # set standardizer 
        SO.set_standardizer(variable, standardizer=StandardScaling(method="standardscaling"))
        
        corr = SO.predictor_correlation(variable, from1961to2012, ERA5Data, fit_predictor=True, 
                                 fit_predictand=True, method="pearson")
        # get the time series
        
        y_obs = SO.get_var(variable, from1961to2012, anomalies=False)
        
        predictors_obs = SO._get_predictor_data(variable, from1961to2012, ERA5Data, fit_predictors=True,)
        
        #save values
        store_csv(stationname, varname="obs", var=y_obs, cachedir=corr_dir)
        store_csv(stationname, varname="corrwith_predictors", var=corr, cachedir=corr_dir)
        store_csv(stationname, varname="predictors_data", var=predictors_obs, cachedir=corr_dir)
          
    
# ploting of correlations
def plot_correlation():
    df = correlation_data(stationnames_prec, corr_dir, "corrwith_predictors", predictors)
    
    apply_style(fontsize=22, style=None) 
    
    fig, ax = plt.subplots(1,1, figsize=(20,15))
                            
    correlation_heatmap(data=df, cmap="RdBu", ax=ax, vmax=1, vmin=-1, center=0, cbar_ax=None, fig=fig,
                            add_cbar=True, title=None, label= "Correlation Coefficinet", fig_path=corr_dir,
                            xlabel="Predictors", ylabel="Stations", fig_name="correlation_prec.svg",)
    
if __name__ == "__main__":
    generate_correlation()
    plot_correlation()