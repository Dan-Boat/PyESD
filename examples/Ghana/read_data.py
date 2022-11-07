# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 12:13:08 2022

@author: dboateng

Read all the required datasets for downscaling
"""

import sys
import os 
import socket
import pandas as pd
import numpy as np

from pyESD.ESD_utils import Dataset
from pyESD.Weatherstation import read_weatherstationnames

radius = 50 #km

# DEFINING PATHS TO DATA
# ======================
            
era5_datadir = "C:/Users/dboateng/Desktop/Datasets/ERA5/monthly_1950_2021/"
station_prec_datadir = "C:/Users/dboateng/Desktop/Datasets/Station/Ghana/Update_datasets/processed/monthly"
station_temp_datadir = "C:/Users/dboateng/Desktop/Datasets/Station/Ghana/Temperature/processed"

predictordir = os.path.join(os.path.dirname(__file__), '.predictors_' + str(int(radius)))
cachedir_prec = os.path.abspath(os.path.join(__file__, os.pardir, 'final_cache_Precipitation'))
cachedir_temp = os.path.abspath(os.path.join(__file__, os.pardir, 'final_cache_Temperature'))

namedict_prec = read_weatherstationnames(station_prec_datadir)
stationnames_prec = list(namedict_prec.values())

namedict_temp = read_weatherstationnames(station_temp_datadir)
stationnames_temp = list(namedict_temp.values())

ERA5Data = Dataset('ERA5', {
    't2m':os.path.join(era5_datadir, 't2m_monthly.nc'),
    'msl':os.path.join(era5_datadir, 'msl_monthly.nc'),
    'u10':os.path.join(era5_datadir, 'u10_monthly.nc'),
    'v10':os.path.join(era5_datadir, 'v10_monthly.nc'),
    'z250':os.path.join(era5_datadir, 'z250_monthly.nc'),
    'z500':os.path.join(era5_datadir, 'z500_monthly.nc'),
    'z700':os.path.join(era5_datadir, 'z700_monthly.nc'),
    'z850':os.path.join(era5_datadir, 'z850_monthly.nc'),
    'z1000':os.path.join(era5_datadir, 'z1000_monthly.nc'),
    'tp':os.path.join(era5_datadir, 'tp_monthly.nc'),
    'q850':os.path.join(era5_datadir, 'q850_monthly.nc'),
    'q500':os.path.join(era5_datadir, 'q500_monthly.nc'),
    't250':os.path.join(era5_datadir, 't250_monthly.nc'),
    't500':os.path.join(era5_datadir, 't500_monthly.nc'),
    't700':os.path.join(era5_datadir, 't700_monthly.nc'),
    't850':os.path.join(era5_datadir, 't850_monthly.nc'),
    't1000':os.path.join(era5_datadir, 't1000_monthly.nc'),
    'r250':os.path.join(era5_datadir, 'r250_monthly.nc'),
    'r500':os.path.join(era5_datadir, 'r500_monthly.nc'),
    'r700':os.path.join(era5_datadir, 'r700_monthly.nc'),
    'r850':os.path.join(era5_datadir, 'r850_monthly.nc'),
    'r1000':os.path.join(era5_datadir, 'r1000_monthly.nc'),
    'vo850':os.path.join(era5_datadir, 'vo850_monthly.nc'),
    'vo500':os.path.join(era5_datadir, 'vo500_monthly.nc'),
    'pv850':os.path.join(era5_datadir, 'pv850_monthly.nc'),
    'pv500':os.path.join(era5_datadir, 'pv500_monthly.nc'),
    'u250':os.path.join(era5_datadir, 'u250_monthly.nc'),
    'u500':os.path.join(era5_datadir, 'u500_monthly.nc'),
    'u700':os.path.join(era5_datadir, 'u700_monthly.nc'),
    'u850':os.path.join(era5_datadir, 'u850_monthly.nc'),
    'u1000':os.path.join(era5_datadir, 'u1000_monthly.nc'),
    'v250':os.path.join(era5_datadir, 'v250_monthly.nc'),
    'v500':os.path.join(era5_datadir, 'v500_monthly.nc'),
    'v700':os.path.join(era5_datadir, 'v700_monthly.nc'),
    'v850':os.path.join(era5_datadir, 'v850_monthly.nc'),
    'v1000':os.path.join(era5_datadir, 'v1000_monthly.nc'),
    'sst':os.path.join(era5_datadir, 'sst_monthly.nc'),
    'dtd250':os.path.join(era5_datadir, 'dtd250_monthly.nc'), 
    'dtd500':os.path.join(era5_datadir, 'dtd500_monthly.nc'),
    'dtd700':os.path.join(era5_datadir, 'dtd700_monthly.nc'),
    'dtd850':os.path.join(era5_datadir, 'dtd850_monthly.nc'),
    'dtd1000':os.path.join(era5_datadir, 'dtd1000_monthly.nc'),},
    domain_name= "Africa")
