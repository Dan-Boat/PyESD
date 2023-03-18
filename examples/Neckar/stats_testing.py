# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 12:48:43 2022

@author: dboateng
"""
import os 
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as colors


from read_data import *
from predictor_settings import *

from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator)

from pyESD.ESD_utils import extract_region, _get_month, StatTest, plot_ks_stats
        

def ks_stats_amip_era5(varname, center=False, standardize=False, plot=False):
    
    
    data_cmip = extract_region(data=CMIP5_AMIP_R1, datarange=fullAMIP, varname=varname, minlat=30, maxlat=70, 
                                  minlon=-20, maxlon=50)
    
    data_era = extract_region(data=ERA5Data, datarange=fullAMIP, varname=varname, minlat=30, maxlat=70, 
                                  minlon=-20, maxlon=50)


    # convert data to the same physical units 
    if varname == "tp":
        data_cmip = data_cmip *60*60*24*30 #mm/month
        
        data_era = data_era * 1000 * 30  #mm/month

    
    # interpolate to the same coordinate (to cmip gridsize)
    
    data_era = data_era.interp(lat=data_cmip.lat).interp(lon=data_cmip.lon)


    # compute anomalies
    group_era = data_era.groupby("time.month")
    
    group_cmip = data_cmip.groupby("time.month")
    
    monthly_means_era = group_era.mean(dim="time")
    
    monthly_means_cmip = group_cmip.mean(dim="time")
    
    anomalies_era = group_era.apply(
               lambda x: x - monthly_means_era.sel(month=_get_month(x[0].time.values))
           )
    
    anomalies_cmip = group_cmip.apply(
               lambda x: x - monthly_means_era.sel(month=_get_month(x[0].time.values))
           )
    
    # apply standardize (optional for testing)
    if center ==True:
    # centered 
        anomalies_era = anomalies_era - anomalies_era.mean("time")
        
        anomalies_cmip = anomalies_cmip - anomalies_cmip.mean("time")
    
    if standardize ==True:
        anomalies_era = anomalies_era / anomalies_era.std(dim="time")
        
        anomalies_cmip = anomalies_cmip / anomalies_cmip.std(dim="time")

        
    
    pvalx, svalx = StatTest(x=anomalies_era, y=anomalies_cmip, test="KS", dim="time")
    
    sig_loc  = xr.where(pvalx < 0.05, pvalx, pvalx*np.nan)
    
    sig_loc = sig_loc.sortby("lon")
    svalx = svalx.sortby("lon")  
    
    return sig_loc, svalx



        
        
#plotting
path_to_store = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Neckar/plots"

varname = "u850"

sig_loc, svalx = ks_stats_amip_era5(varname=varname, center=False, standardize=False)
sig_loc_c, svalx_c = ks_stats_amip_era5(varname=varname, center=True, standardize=False)
sig_loc_s, svalx_s = ks_stats_amip_era5(varname=varname, center=True, standardize=True)



projection = ccrs.PlateCarree()
fig, (ax1,ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize=(28, 13), subplot_kw={"projection": projection})

plot_ks_stats(data=svalx, ax=ax1, cmap="YlGnBu", vmax=1, vmin=0, levels=22, level_ticks=6, plot_stats=True, 
              stats_results=sig_loc, domain="Europe", orientation="horizontal", cbar_pos = [0.35, 0.25, 0.35, 0.02],
              title= varname + " raw (ks_test(MIP-ESM, ERA5))", add_colorbar=False, fig=fig) #, path_to_store=path_to_store, output_format="svg", output_name="r850_raw.svg")

plot_ks_stats(data=svalx_c, ax=ax2, cmap="YlGnBu", vmax=1, vmin=0, levels=22, level_ticks=6, plot_stats=True, 
              stats_results=sig_loc_c, domain="Europe", orientation="horizontal", cbar_pos = [0.35, 0.25, 0.35, 0.02],
              title= varname + " centered (ks_test(MIP-ESM, ERA5))", fig=fig) #, path_to_store=path_to_store, output_format="svg", output_name="r850_raw.svg")

plot_ks_stats(data=svalx_s, ax=ax3, cmap="YlGnBu", vmax=1, vmin=0, levels=22, level_ticks=6, plot_stats=True, 
              stats_results=sig_loc_s, domain="Europe", orientation="horizontal", cbar_pos = [0.35, 0.25, 0.35, 0.02],
              title= varname + " standardize (ks_test(MIP-ESM, ERA5))", add_colorbar=False, fig=fig) #, path_to_store=path_to_store, output_format="svg", output_name="r850_raw.svg")

fig.canvas.draw()   # the only way to apply tight_layout to matplotlib and cartopy is to apply canvas firt 
plt.tight_layout()
plt.subplots_adjust(left=0.05, right=0.95, top=0.94, bottom=0.15)
plt.savefig(os.path.join(path_to_store, "u850_ks_test.svg"), format= "svg", bbox_inches="tight", dpi=300)


# projection = ccrs.PlateCarree()
# fig, ax = plt.subplots(1, 1, sharex=False, figsize= (15, 13), subplot_kw= {"projection":projection})
# score.plot(cmap="YlGnBu", transform=projection, add_colorbar=True)
# p = ax.contourf(stats.lon.data, stats.lat.data, stats.data, colors="none", hatches=["xx"])
# plot_background(p, domain="Europe", ax=ax)
plt.show()


