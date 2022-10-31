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


# import pyESD modules 
from pyESD.Weatherstation import read_station_csv
from pyESD.standardizer import StandardScaling, NoStandardizer
from pyESD.ESD_utils import store_csv, store_pickle

# relative files import 
from read_data import *
from settings import *

#directories
corr_dir = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Ghana/correlation_data"

radius = 250  #km
variable = "Precipitation"
num_of_stations = len(stationnames_prec)

# for i in range(num_of_stations):
    
stationname = stationnames_prec[2]
station_dir = os.path.join(station_prec_datadir, stationname + ".csv")

SO = read_station_csv(filename=station_dir, varname=variable)

# set predictors 
SO.set_predictors(variable, predictors, predictordir, radius, 
                  standardizer=StandardScaling(method="standardscaling"))

# set standardizer 
SO.set_standardizer(variable, standardizer=StandardScaling(method="standardscaling"))

corr = SO.predictor_correlation(variable, from1961to2012, ERA5Data, fit_predictor=True, 
                         fit_predictand=True, method="pearson")

#save values

store_csv(stationname, varname="corrwith_predictors", var=corr, cachedir=corr_dir)
  