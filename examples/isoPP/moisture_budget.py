# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 21:20:45 2024

@author: dboateng
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
from scipy.integrate import trapz

main_path = "D:/Datasets/Model_output_pst/PI/MONTHLY_MEANS/"

data = xr.open_dataset(main_path + "1003_1017_1m_mlterm_dynamics.nc")

lats = data.lat.data
lons = data.lon.data
levs = data.plev.data

dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats)

divergence_all_levels = []

for lev in levs:
    uq = data["q"]*1000 * data["u"]*(1/(9.81*1000))
    vq = data["q"]*1000 * data["v"]*(1/(9.81*1000))

    vq = vq.mean(dim="time")
    uq = uq.mean(dim="time")

    HMC_LE = (np.array(mpcalc.divergence(uq.sel(plev=lev), vq.sel(plev=lev), dx=dx, dy=dy)))
    divergence_all_levels.append(HMC_LE)

divergence_all_levels = np.array(divergence_all_levels)

# Integrate the divergence values across all levels
integrated_divergence = trapz(divergence_all_levels, x=levs, axis=0)

# Plot the integrated divergence
plt.figure(figsize=(10, 6))
plt.contourf(lons, lats, integrated_divergence *86400, cmap=plt.cm.RdBu_r, levels=20, extend='both')
plt.colorbar(label='Integrated Moisture Flux Divergence (10^-5)')
plt.title('Integrated Moisture Flux Divergence across all Vertical Levels')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()
