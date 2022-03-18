# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 16:43:14 2022

@author: dboateng

Example script that handles reading of large-scale reanalysis datasets (eg. ERA5) and station data

"""

import sys
import os 
import socket
import pandas as pd
import numpy as np

sys.path.append("C:/Users/dboateng/Desktop/Python_scripts/ESD_Package")

from Package.ESD_utils import Dataset
from Package.WeatherstationPreprocessing import read_weatherstationnames

radius = 200 # km


# paths 
era5_datadir = "C:/Users/dboateng/Desktop/Datasets/ERA5/monthly_1950_2021/"
station_datadir = "C:/Users/dboateng/Desktop/Datasets/Station/Neckar_Enz/Precipitation/cdc_download_2022-03-17_13/processed/"
predictordir    = os.path.join(os.path.dirname(__file__), '.predictors_' + str(int(radius)))
cachedir        = os.path.abspath(os.path.join(__file__, os.pardir, 'final_cache'))


ERA5Data = Dataset('ERA5', {
    't2m':os.path.join(era5_datadir, 't2m_monthly.nc'),
    'msl':os.path.join(era5_datadir, 'msl_monthly.nc'),
    'u10':os.path.join(era5_datadir, 'u10_monthly.nc'),
    'v10':os.path.join(era5_datadir, 'v10_monthly.nc'),
    'z500':os.path.join(era5_datadir, 'z500_monthly_new.nc'),
    'z850':os.path.join(era5_datadir, 'z850_monthly_new.nc'),
    'tp':os.path.join(era5_datadir, 'tp_monthly.nc'),
    'q850':os.path.join(era5_datadir, 'q850_monthly_new.nc'),
    'q500':os.path.join(era5_datadir, 'q500_monthly_new.nc'),
    't850':os.path.join(era5_datadir, 't850_monthly_new.nc'),
    't500':os.path.join(era5_datadir, 't500_monthly_new.nc'),
    'r850':os.path.join(era5_datadir, 'r850_monthly_new.nc'),
    'r500':os.path.join(era5_datadir, 'r500_monthly_new.nc'),
    'vo850':os.path.join(era5_datadir, 'vo850_monthly_new.nc'),
    'vo500':os.path.join(era5_datadir, 'vo500_monthly_new.nc'),
    'pv850':os.path.join(era5_datadir, 'pv850_monthly_new.nc'),
    'pv500':os.path.join(era5_datadir, 'pv500_monthly_new.nc'),
    'u850':os.path.join(era5_datadir, 'u850_monthly_new.nc'),
    'u500':os.path.join(era5_datadir, 'u500_monthly_new.nc'),
    'v850':os.path.join(era5_datadir, 'v850_monthly_new.nc'),
    'v500':os.path.join(era5_datadir, 'v500_monthly_new.nc'),
    'sst':os.path.join(era5_datadir, 'sst_monthly.nc'),
    'd2m':os.path.join(era5_datadir, 'd2m_monthly.nc'), })




namedict = read_weatherstationnames(station_datadir)
stationnames = list(namedict.values())