# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 13:40:47 2022

@author: dboateng
"""

import pandas as pd 
import numpy as np
from scipy import stats 
import matplotlib.pyplot as plt
from pyESD.Weatherstation import read_station_csv
from pyESD.plot_utils import*

from read_data import *
from settings import *

# reading datasets 
variable = "Precipitation"
stationname = stationnames_prec[9
                                ]
station_dir = os.path.join(station_prec_datadir, stationname + ".csv")
data, lat, lon = read_station_csv(filename=station_dir, varname=variable, 
                                  return_all=True)

data = data.get(variable)

# get sum of missing values 
missing_values = data.isna().sum()
data_stats = data.describe()


# extract closest grid to compare with

era_tp = ERA5Data.get("tp")
era_tp = era_tp.sortby("longitude")
data_tp = era_tp.sel(longitude=lon, latitude=lat, method= "nearest").to_series()


# extract the time time series range 
station = data[from1961to2013]
reanalysis = data_tp[from1961to2013] * 1000 *30

# perform correlation 

reg = stats.spearmanr(station.values , reanalysis.values)
r2 = reg.correlation
pvalue = reg.pvalue

# group into monthly then plot with cor and no. of missing values

station_mon = resample_monthly(station, from1961to2013)
reanalysis_mon = resample_monthly(reanalysis, from1961to2013)

# plot the distribution

import calendar	
month_names = [calendar.month_abbr[im+1] for im in np.arange(12)]
df = pd.DataFrame(index=stationnames_prec, columns=month_names)
df_reanalysis = pd.DataFrame(index=stationnames_prec, columns=month_names)

for i,month in enumerate(month_names):
    df.loc[stationname][month] = station_mon[i]
    df_reanalysis.loc[stationname][month] = reanalysis[i]
    


fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 15))
ax2 = ax.twinx()

width = 0.4
df.loc[stationname].plot(kind="bar", ax=ax, fontsize=20, color= red, position=1,
                         width=width)
df_reanalysis.loc[stationname].plot(kind="bar", ax=ax2, fontsize=20, color= black, position=0, 
                                    width=width,)

ax.set_ylabel("Station Prec Avg [mm]", fontweight="bold", fontsize=20)
ax2.set_ylabel("ERA5 Total Prec Avg [mm]", fontweight="bold", fontsize=20)
plt.title(stationname + ", No. of missing months: " + str(missing_values), fontweight="bold", fontsize=20)
plt.tight_layout()
plt.savefig(stationname + ".png")
plt.show()