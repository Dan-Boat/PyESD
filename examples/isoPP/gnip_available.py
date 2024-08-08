# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 13:45:58 2024

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
gnip_30_path = "D:/Datasets/GNIP_data/world/scratch/station_world_overview_30years.csv" 
gnip_10_path = "D:/Datasets/GNIP_data/world/scratch/station_world_overview_10years.csv" 
gnip_5_path = "D:/Datasets/GNIP_data/world/scratch/station_world_overview_5years.csv" 

path_to_store = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/isoPP/plots/GNIP"
 


df = pd.read_csv(gnip_30_path)



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

apply_style(fontsize=25, style=None, linewidth=2) 

projection = ccrs.Robinson(central_longitude=0, globe=None)
#projection = ccrs.PlateCarree()

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(18, 14), subplot_kw={"projection":  
                                                                        projection})
    
scatter = ax1.scatter(x=df["lon"], y=df["lat"], c=df["d18op"], 
           cmap="Spectral_r", vmax=0, vmin=-25, edgecolor="black", s= 140,
           transform=ccrs.Geodetic(),linewidth=2)

ax1.coastlines(resolution = "50m", linewidth=1.5, color="grey")
ax1.add_feature(cfeature.BORDERS, edgecolor="black", linewidth = 0.3)
minLon = -120
maxLon = 120
minLat = 25
maxLat = 85
#ax1.set_extent([minLon, maxLon, minLat, maxLat], ccrs.PlateCarree())

ax1.set_global()

gl=ax1.gridlines(crs = ccrs.PlateCarree(), draw_labels = True, linewidth = 1,
                     edgecolor = "gray", linestyle = "--", color="gray", alpha=0.5)
gl.top_labels = False                  # labesl at top
gl.right_labels = False
gl.xformatter = LongitudeFormatter()     # axis formatter
gl.yformatter = LatitudeFormatter()
cbar = fig.colorbar(scatter, ax=ax1, orientation='vertical', pad=0.05, shrink=0.25)
cbar.set_label('$\delta^{18}$Op vs SMOW [‰]')

#uncoment

gl.xlabel_style = {"fontsize": 20, "color": "black", "fontweight": "semibold"}   #axis style 
gl.ylabel_style = {"fontsize": 20, "color": "black", "fontweight": "semibold"}


ax1.set_title(
    f"GNIP stations (n={len(df)})")

# plot_annual_mean(ax=ax1, variable='$\delta^{18}$Op vs SMOW', data_alt=d18Op_alt, cmap="Spectral_r", 
#                   units="‰", vmax=0, vmin=-25, 
#                   levels=22, level_ticks=10, GNIP_data=df, title=None, left_labels=True, bottom_labels=True, 
#                   use_colorbar_default=True, center=False,
#                   plot_projection=projection, domain="NH Wide",)

fig.canvas.draw()   # the only way to apply tight_layout to matplotlib and cartopy is to apply canvas first 
plt.tight_layout() 
plt.subplots_adjust(left=0.05, right=0.95, top=0.94, bottom=0.06)
plt.savefig(os.path.join(path_to_store, "gnip_30years.png"), format= "png", bbox_inches="tight")

