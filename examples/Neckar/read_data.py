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


from pyESD.ESD_utils import Dataset
from pyESD.WeatherstationPreprocessing import read_weatherstationnames

radius = 200 # km


# paths 
era5_datadir = "C:/Users/dboateng/Desktop/Datasets/ERA5/monthly_1950_2021/"
amip_datadir = "C:/Users/dboateng/Desktop/Datasets/CMIP5/Monthly/AMIP"
cmip5_26_datadir = "C:/Users/dboateng/Desktop/Datasets/CMIP5/Monthly/RCP26"
cmip5_45_datadir = "C:/Users/dboateng/Desktop/Datasets/CMIP5/Monthly/RCP45"
cmip5_85_datadir = "C:/Users/dboateng/Desktop/Datasets/CMIP5/Monthly/RCP85"


station_temp_datadir = "C:/Users/dboateng/Desktop/Datasets/Station/Neckar_Enz/Temperature/cdc_download_2022-03-17_13-38/processed"
station_prec_datadir = "C:/Users/dboateng/Desktop/Datasets/Station/Neckar_Enz/Precipitation/cdc_download_2022-03-17_13/processed"
predictordir    = os.path.join(os.path.dirname(__file__), '.predictors_' + str(int(radius)))
cachedir_temp        = os.path.abspath(os.path.join(__file__, os.pardir, 'final_cache_Temperature'))
cachedir_prec        = os.path.abspath(os.path.join(__file__, os.pardir, 'final_cache_Precipitation'))


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
    'd2m':os.path.join(era5_datadir, 'd2m_monthly.nc'), })


AMIPData = Dataset('AMIP', {
    't2m':os.path.join(amip_datadir, 't2m_monthly.nc'),
    'msl':os.path.join(amip_datadir, 'msl_monthly.nc'),
    'u10':os.path.join(amip_datadir, 'u10_monthly.nc'),
    'v10':os.path.join(amip_datadir, 'v10_monthly.nc'),
    'z250':os.path.join(amip_datadir, 'z250_monthly.nc'),
    'z500':os.path.join(amip_datadir, 'z500_monthly.nc'),
    'z700':os.path.join(amip_datadir, 'z700_monthly.nc'),
    'z850':os.path.join(amip_datadir, 'z850_monthly.nc'),
    'z1000':os.path.join(amip_datadir, 'z1000_monthly.nc'),
    'tp':os.path.join(amip_datadir, 'tp_monthly.nc'),
    't250':os.path.join(amip_datadir, 't250_monthly.nc'),
    't500':os.path.join(amip_datadir, 't500_monthly.nc'),
    't700':os.path.join(amip_datadir, 't700_monthly.nc'),
    't850':os.path.join(amip_datadir, 't850_monthly.nc'),
    't1000':os.path.join(amip_datadir, 't1000_monthly.nc'),
    'r250':os.path.join(amip_datadir, 'r250_monthly.nc'),
    'r500':os.path.join(amip_datadir, 'r500_monthly.nc'),
    'r700':os.path.join(amip_datadir, 'r700_monthly.nc'),
    'r850':os.path.join(amip_datadir, 'r850_monthly.nc'),
    'r1000':os.path.join(amip_datadir, 'r1000_monthly.nc'),
    'u250':os.path.join(amip_datadir, 'u250_monthly.nc'),
    'u500':os.path.join(amip_datadir, 'u500_monthly.nc'),
    'u700':os.path.join(amip_datadir, 'u700_monthly.nc'),
    'u850':os.path.join(amip_datadir, 'u850_monthly.nc'),
    'u1000':os.path.join(amip_datadir, 'u1000_monthly.nc'),
    'v250':os.path.join(amip_datadir, 'v250_monthly.nc'),
    'v500':os.path.join(amip_datadir, 'v500_monthly.nc'),
    'v700':os.path.join(amip_datadir, 'v700_monthly.nc'),
    'v850':os.path.join(amip_datadir, 'v850_monthly.nc'),
    'v1000':os.path.join(amip_datadir, 'v1000_monthly.nc'),
    })


CMIP5_26_Data = Dataset('CMIP5_26', {
    't2m':os.path.join(cmip5_26_datadir, 't2m_monthly.nc'),
    'msl':os.path.join(cmip5_26_datadir, 'msl_monthly.nc'),
    'u10':os.path.join(cmip5_26_datadir, 'u10_monthly.nc'),
    'v10':os.path.join(cmip5_26_datadir, 'v10_monthly.nc'),
    'z250':os.path.join(cmip5_26_datadir, 'z250_monthly.nc'),
    'z500':os.path.join(cmip5_26_datadir, 'z500_monthly.nc'),
    'z700':os.path.join(cmip5_26_datadir, 'z700_monthly.nc'),
    'z850':os.path.join(cmip5_26_datadir, 'z850_monthly.nc'),
    'z1000':os.path.join(cmip5_26_datadir, 'z1000_monthly.nc'),
    'tp':os.path.join(cmip5_26_datadir, 'tp_monthly.nc'),
    't250':os.path.join(cmip5_26_datadir, 't250_monthly.nc'),
    't500':os.path.join(cmip5_26_datadir, 't500_monthly.nc'),
    't700':os.path.join(cmip5_26_datadir, 't700_monthly.nc'),
    't850':os.path.join(cmip5_26_datadir, 't850_monthly.nc'),
    't1000':os.path.join(cmip5_26_datadir, 't1000_monthly.nc'),
    'r250':os.path.join(cmip5_26_datadir, 'r250_monthly.nc'),
    'r500':os.path.join(cmip5_26_datadir, 'r500_monthly.nc'),
    'r700':os.path.join(cmip5_26_datadir, 'r700_monthly.nc'),
    'r850':os.path.join(cmip5_26_datadir, 'r850_monthly.nc'),
    'r1000':os.path.join(cmip5_26_datadir, 'r1000_monthly.nc'),
    'u250':os.path.join(cmip5_26_datadir, 'u250_monthly.nc'),
    'u500':os.path.join(cmip5_26_datadir, 'u500_monthly.nc'),
    'u700':os.path.join(cmip5_26_datadir, 'u700_monthly.nc'),
    'u850':os.path.join(cmip5_26_datadir, 'u850_monthly.nc'),
    'u1000':os.path.join(cmip5_26_datadir, 'u1000_monthly.nc'),
    'v250':os.path.join(cmip5_26_datadir, 'v250_monthly.nc'),
    'v500':os.path.join(cmip5_26_datadir, 'v500_monthly.nc'),
    'v700':os.path.join(cmip5_26_datadir, 'v700_monthly.nc'),
    'v850':os.path.join(cmip5_26_datadir, 'v850_monthly.nc'),
    'v1000':os.path.join(cmip5_26_datadir, 'v1000_monthly.nc'),
    })



CMIP5_45_Data = Dataset('CMIP_45', {
    't2m':os.path.join(cmip5_45_datadir, 't2m_monthly.nc'),
    'msl':os.path.join(cmip5_45_datadir, 'msl_monthly.nc'),
    'u10':os.path.join(cmip5_45_datadir, 'u10_monthly.nc'),
    'v10':os.path.join(cmip5_45_datadir, 'v10_monthly.nc'),
    'z250':os.path.join(cmip5_45_datadir, 'z250_monthly.nc'),
    'z500':os.path.join(cmip5_45_datadir, 'z500_monthly.nc'),
    'z700':os.path.join(cmip5_45_datadir, 'z700_monthly.nc'),
    'z850':os.path.join(cmip5_45_datadir, 'z850_monthly.nc'),
    'z1000':os.path.join(cmip5_45_datadir, 'z1000_monthly.nc'),
    'tp':os.path.join(cmip5_45_datadir, 'tp_monthly.nc'),
    't250':os.path.join(cmip5_45_datadir, 't250_monthly.nc'),
    't500':os.path.join(cmip5_45_datadir, 't500_monthly.nc'),
    't700':os.path.join(cmip5_45_datadir, 't700_monthly.nc'),
    't850':os.path.join(cmip5_45_datadir, 't850_monthly.nc'),
    't1000':os.path.join(cmip5_45_datadir, 't1000_monthly.nc'),
    'r250':os.path.join(cmip5_45_datadir, 'r250_monthly.nc'),
    'r500':os.path.join(cmip5_45_datadir, 'r500_monthly.nc'),
    'r700':os.path.join(cmip5_45_datadir, 'r700_monthly.nc'),
    'r850':os.path.join(cmip5_45_datadir, 'r850_monthly.nc'),
    'r1000':os.path.join(cmip5_45_datadir, 'r1000_monthly.nc'),
    'u250':os.path.join(cmip5_45_datadir, 'u250_monthly.nc'),
    'u500':os.path.join(cmip5_45_datadir, 'u500_monthly.nc'),
    'u700':os.path.join(cmip5_45_datadir, 'u700_monthly.nc'),
    'u850':os.path.join(cmip5_45_datadir, 'u850_monthly.nc'),
    'u1000':os.path.join(cmip5_45_datadir, 'u1000_monthly.nc'),
    'v250':os.path.join(cmip5_45_datadir, 'v250_monthly.nc'),
    'v500':os.path.join(cmip5_45_datadir, 'v500_monthly.nc'),
    'v700':os.path.join(cmip5_45_datadir, 'v700_monthly.nc'),
    'v850':os.path.join(cmip5_45_datadir, 'v850_monthly.nc'),
    'v1000':os.path.join(cmip5_45_datadir, 'v1000_monthly.nc'),
    })

CMIP5_85_Data = Dataset('CMIP_85', {
    't2m':os.path.join(cmip5_85_datadir, 't2m_monthly.nc'),
    'msl':os.path.join(cmip5_85_datadir, 'msl_monthly.nc'),
    'u10':os.path.join(cmip5_85_datadir, 'u10_monthly.nc'),
    'v10':os.path.join(cmip5_85_datadir, 'v10_monthly.nc'),
    'z250':os.path.join(cmip5_85_datadir, 'z250_monthly.nc'),
    'z500':os.path.join(cmip5_85_datadir, 'z500_monthly.nc'),
    'z700':os.path.join(cmip5_85_datadir, 'z700_monthly.nc'),
    'z850':os.path.join(cmip5_85_datadir, 'z850_monthly.nc'),
    'z1000':os.path.join(cmip5_85_datadir, 'z1000_monthly.nc'),
    'tp':os.path.join(cmip5_85_datadir, 'tp_monthly.nc'),
    't250':os.path.join(cmip5_85_datadir, 't250_monthly.nc'),
    't500':os.path.join(cmip5_85_datadir, 't500_monthly.nc'),
    't700':os.path.join(cmip5_85_datadir, 't700_monthly.nc'),
    't850':os.path.join(cmip5_85_datadir, 't850_monthly.nc'),
    't1000':os.path.join(cmip5_85_datadir, 't1000_monthly.nc'),
    'r250':os.path.join(cmip5_85_datadir, 'r250_monthly.nc'),
    'r500':os.path.join(cmip5_85_datadir, 'r500_monthly.nc'),
    'r700':os.path.join(cmip5_85_datadir, 'r700_monthly.nc'),
    'r850':os.path.join(cmip5_85_datadir, 'r850_monthly.nc'),
    'r1000':os.path.join(cmip5_85_datadir, 'r1000_monthly.nc'),
    'u250':os.path.join(cmip5_85_datadir, 'u250_monthly.nc'),
    'u500':os.path.join(cmip5_85_datadir, 'u500_monthly.nc'),
    'u700':os.path.join(cmip5_85_datadir, 'u700_monthly.nc'),
    'u850':os.path.join(cmip5_85_datadir, 'u850_monthly.nc'),
    'u1000':os.path.join(cmip5_85_datadir, 'u1000_monthly.nc'),
    'v250':os.path.join(cmip5_85_datadir, 'v250_monthly.nc'),
    'v500':os.path.join(cmip5_85_datadir, 'v500_monthly.nc'),
    'v700':os.path.join(cmip5_85_datadir, 'v700_monthly.nc'),
    'v850':os.path.join(cmip5_85_datadir, 'v850_monthly.nc'),
    'v1000':os.path.join(cmip5_85_datadir, 'v1000_monthly.nc'),
    })



namedict_prec = read_weatherstationnames(station_prec_datadir)
namedict_temp = read_weatherstationnames(station_temp_datadir)

stationnames_prec = list(namedict_prec.values())
stationnames_temp = list(namedict_temp.values())