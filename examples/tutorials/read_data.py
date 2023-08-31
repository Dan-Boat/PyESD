# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 17:07:07 2023

@author: dboateng
"""

import sys
import os 
import socket
import pandas as pd
import numpy as np


from pyESD.ESD_utils import Dataset
from pyESD.Weatherstation import read_weatherstationnames

radius = 200 # km


# DEFINING PATHS TO DATA
# ======================


            
era5_datadir = "C:/Users/dboateng/Desktop/Datasets/ERA5/monthly_1950_2021/"

cmip5_26_datadir = "D:/Datasets/CMIP5/Monthly/MIPESM/RCP26"
cmip5_45_datadir = "D:/Datasets/CMIP5/Monthly/MIPESM/RCP45"
cmip5_85_datadir = "D:/Datasets/CMIP5/Monthly/MIPESM/RCP85"



station_prec_datadir = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/tutorials/data"

predictordir = os.path.join(os.path.dirname(__file__), '.predictors_' + str(int(radius)))


cachedir_prec = os.path.abspath(os.path.join(__file__, os.pardir, 'final_cache_Precipitation'))

if not os.path.exists(cachedir_prec):
    os.makedirs(cachedir_prec)

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
    'd2m':os.path.join(era5_datadir, 'd2m_monthly.nc'), 
    'dtd250':os.path.join(era5_datadir, 'dtd250_monthly.nc'), 
    'dtd500':os.path.join(era5_datadir, 'dtd500_monthly.nc'),
    'dtd700':os.path.join(era5_datadir, 'dtd700_monthly.nc'),
    'dtd850':os.path.join(era5_datadir, 'dtd850_monthly.nc'),
    'dtd1000':os.path.join(era5_datadir, 'dtd1000_monthly.nc'),},
    domain_name= "NH") # select domain for Northern Hemisphere




CMIP5_RCP26_R1 = Dataset('CMIP5_RCP26_R1', {
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
    'dtd250':os.path.join(cmip5_26_datadir, 'dtd250_monthly.nc'), 
    'dtd500':os.path.join(cmip5_26_datadir, 'dtd500_monthly.nc'),
    'dtd700':os.path.join(cmip5_26_datadir, 'dtd700_monthly.nc'),
    'dtd850':os.path.join(cmip5_26_datadir, 'dtd850_monthly.nc'),
    'dtd1000':os.path.join(cmip5_26_datadir, 'dtd1000_monthly.nc')
    }, domain_name= "NH")



CMIP5_RCP45_R1 = Dataset('CMIP5_RCP45_R1', {
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
    'dtd250':os.path.join(cmip5_45_datadir, 'dtd250_monthly.nc'), 
   'dtd500':os.path.join(cmip5_45_datadir, 'dtd500_monthly.nc'),
   'dtd700':os.path.join(cmip5_45_datadir, 'dtd700_monthly.nc'),
   'dtd850':os.path.join(cmip5_45_datadir, 'dtd850_monthly.nc'),
   'dtd1000':os.path.join(cmip5_45_datadir, 'dtd1000_monthly.nc')
    }, domain_name= "NH")

CMIP5_RCP85_R1 = Dataset('CMIP5_RCP85_R1', {
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
    'dtd250':os.path.join(cmip5_85_datadir, 'dtd250_monthly.nc'), 
    'dtd500':os.path.join(cmip5_85_datadir, 'dtd500_monthly.nc'),
    'dtd700':os.path.join(cmip5_85_datadir, 'dtd700_monthly.nc'),
    'dtd850':os.path.join(cmip5_85_datadir, 'dtd850_monthly.nc'),
    'dtd1000':os.path.join(cmip5_85_datadir, 'dtd1000_monthly.nc')
    }, domain_name= "NH")





namedict_prec = read_weatherstationnames(station_prec_datadir)
stationnames_prec = list(namedict_prec.values())
