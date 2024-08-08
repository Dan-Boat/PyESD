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

path_to_store = "C:/Users/dboateng/Desktop/Python_scripts/ClimatPackage_repogit/examples/Africa/plots"

#gnip_path = "D:/Datasets/GNIP_data/Africa/scratch/station_africa_overview.csv" 


df = pd.read_csv(gnip_path)


#df = df.drop(df[df.echam > 5].index)  # what is the reason for this code?


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
#projection_p = ccrs.PlateCarree()
fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(13, 13), subplot_kw={"projection":  
                                                                        projection})




plot_annual_mean(ax=ax1, variable='$\delta^{18}$Op vs SMOW', data_alt=d18Op_alt, cmap="Spectral_r", 
                  units="‰", vmax=0, vmin=-20, domain=None, 
                  levels=22, level_ticks=10, GNIP_data=df, title=None, left_labels=False, bottom_labels=False, 
                  use_colorbar_default=True, center=False,
                  plot_projection=projection)

fig.canvas.draw()   # the only way to apply tight_layout to matplotlib and cartopy is to apply canvas first 
plt.tight_layout() 
plt.subplots_adjust(left=0.05, right=0.95, top=0.94, bottom=0.06)
plt.savefig(os.path.join(path_to_store, "global_echam_gnip.pdf"), format= "pdf", bbox_inches="tight")




apply_style(fontsize=24, style=None, linewidth=3)
fig,ax = plt.subplots(1,1, figsize=(13,13))

gnip = df["d18op"]
echam = df["echam"]

from scipy import stats
  
regression_stats  = stats.linregress(gnip, echam)

regression_slope = regression_stats.slope * gnip + regression_stats.intercept

r2 = regression_stats.rvalue

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true= gnip, y_pred= echam) 

ax.scatter(gnip, echam, alpha=0.6, c="black", s=150, marker="o")
ax.plot(gnip, regression_slope, color="#06c4be", label="r² = {:.2f}, MAE = {:.2f} (‰)".format(r2, mae))
ax.legend(loc= "upper left", fontsize=22)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.spines["left"].set_position(("outward", 22))
ax.spines["bottom"].set_position(("outward", 22))

ax.set_ylabel("ECHAM5-wiso", fontweight="bold", fontsize=20)
ax.grid(True, linestyle="--", color="gray")

ax.set_xlabel("GNIP", fontweight="bold", fontsize=20)
ax.grid(True, linestyle="--", color="gray")

plt.tight_layout()
plt.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.05)
    
   
plt.savefig(os.path.join(path_to_store, "compare_echam_gnip_scatter.pdf"), bbox_inches="tight", format= "pdf")

