# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 20:52:18 2024

@author: dboateng
"""

import os 
import pandas as pd 
import numpy as np

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.dates import YearLocator
import matplotlib.dates as mdates


from pyClimat.plot_utils import *
from pyClimat.plots import plot_annual_mean 
from pyClimat.data import read_ERA_processed, read_ECHAM_processed, read_from_path
from pyClimat.analysis import compute_lterm_mean, compute_lterm_diff
from pyClimat.variables import extract_var

main_path = "D:/Datasets/Model_output_pst/PI"
gnip_path = "D:/Datasets/GNIP_data/world/scratch/station_world_overview.csv" 


df = pd.read_csv(gnip_path)
#load datasets 
PD_data = read_from_path(main_path, "PI_1003_1017_monthly.nc", decode=True)
PD_wiso = read_from_path(main_path, "PI_1003_1017_monthly_wiso.nc", decode=True)
d18Op = extract_var(Dataset=PD_data, varname="d18op", units="per mil", Dataset_wiso=PD_wiso,
                    )

# load datasets 
# PD_data = read_from_path(main_path, "PD_1980_2014_monthly.nc", decode=True)
# PD_wiso = read_from_path(main_path, "PD_1980_2014_monthly_wiso.nc", decode=True)
# d18Op = extract_var(Dataset=PD_data, varname="d18op", units="per mil", Dataset_wiso=PD_wiso,
#                     )

d18Op_alt = compute_lterm_mean(data=d18Op, time="annual")

apply_style(fontsize=22, style=None, linewidth=2) 

projection = ccrs.PlateCarree()
fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(13, 13), subplot_kw={"projection":  
                                                                        projection})

# d18Op
plot_annual_mean(ax=ax1, variable='$\delta^{18}$Op vs SMOW', data_alt=d18Op_alt, cmap="Spectral_r", 
                 units="‰", vmax=0, vmin=-20, domain=None, 
                  levels=22, level_ticks=10, GNIP_data=df, title=None, left_labels=True, bottom_labels=True, 
                  use_colorbar_default=True, center=False)

fig.canvas.draw()   # the only way to apply tight_layout to matplotlib and cartopy is to apply canvas first 
plt.tight_layout() 
plt.subplots_adjust(left=0.05, right=0.95, top=0.94, bottom=0.06)
