# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 15:26:56 2024

@author: dboateng

Plot the long-term means of PD ECHAM5-wiso simulation and compare with the GNIP stations (+ inter-annual variability)
"""
# Import modules 
import os 
import pandas as pd 

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

from pyESD.plot_utils import apply_style
from pyESD.plot import heatmaps
from pyESD.plot_utils import seasonal_mean
#from pyESD.plot_utils import *
from read_data import *
from predictor_setting import *
from pyESD.ESD_utils import load_csv, haversine, extract_indices_around

# Define paths 
path_to_save = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/isoPP/plots"
main_path = "D:/Datasets/Model_output_pst/PD"
station_datadir = "C:/Users/dboateng/Desktop/Datasets/Station/GNIP"
station_data = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/isoPP/model_selection"




def calculate_regional_means(ds, lon_target, lat_target, radius_deg,):
    """
    Calculate regional means around a specific longitude and latitude location
    with a given radius for a NetCDF dataset using xarray.
    """
    # Find indices of the nearest grid point to the target location
    
    if hasattr(ds, "longitude"):
        ds = ds.rename({"longitude":"lon", "latitude":"lat"})
        
    ds = ds.assign_coords({"lon": (((ds.lon + 180) % 360) - 180)})
    
    indices = extract_indices_around(ds, lat_target, lon_target, radius_deg)
    
    regional_mean = ds.isel(lat=indices[0], lon=indices[1]).mean(dim=("lon", "lat")).data
        
    return np.float64(regional_mean)

def plot_example(num):
    from1980to2014 = pd.date_range(start="1979-01-01", end="2014-12-31", freq="MS")
    
    # load datasets 
    PD_data = read_from_path(main_path, "PD_1980_2014_monthly.nc", decode=True)
    PD_wiso = read_from_path(main_path, "PD_1980_2014_monthly_wiso.nc", decode=True)
    d18Op = extract_var(Dataset=PD_data, varname="d18op", units="per mil", Dataset_wiso=PD_wiso,
                        )
    
    
    
    stationname = stationnames[num-1].replace("_", " ")
    
    station_info = pd.read_csv(os.path.join(station_datadir, "stationnames_new.csv"), index_col=0)
    lon = station_info.loc[num]["Longitude"]
    lat = station_info.loc[num]["Latitude"]
    
    model = pd.DataFrame(columns=["echam"], index=from1980to2014)
    
    model["echam"] = calculate_regional_means(ds=d18Op, lon_target=lon, lat_target=lat, radius_deg=50)
    
    model = model.rolling(3, min_periods=1, win_type="hann", center=True).mean()
    
    print("extracting information for the station: ", stationname)
    
    filename = "predictions_" + "Stacking"
    
    df = load_csv(stationname, filename, station_data)
    
    obs = df["obs"]
    obs = obs.rolling(3, min_periods=1, win_type="hann", center=True).mean()
    
    apply_style(fontsize=22, style=None, linewidth=3, usetex=True)
    fig, ax = plt.subplots(1, 1, figsize= (12, 4), sharex=True)
    
    plt.subplots_adjust(left=0.12, right=1-0.01, top=0.98, bottom=0.06, hspace=0.01)
    
    ax.plot(obs, linestyle="-", color=black, label=stationname.capitalize() + "," + str(num) + " (GNIP)")
    ax.plot(model, linestyle="-", color=red, label="ECHAM5-wiso")
    
    
    ax.xaxis.set_major_locator(YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.legend(bbox_to_anchor=(0.01, 1.02, 1., 0.102), loc=3, ncol=3, borderaxespad=0., frameon = True, 
              fontsize=20)
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(path_to_save, stationname + "_inter_annual.pdf"), bbox_inches="tight", format = "pdf", dpi=300)



def plot_echam_gnip_spatial():

    # load datasets 
    PD_data = read_from_path(main_path, "PD_1980_2014_monthly.nc", decode=True)
    PD_wiso = read_from_path(main_path, "PD_1980_2014_monthly_wiso.nc", decode=True)
    
    df_sm = seasonal_mean(stationnames, station_data, filename="predictions_", 
                            daterange=from1979to2012 , id_name="obs", method= "Stacking",
                            use_id=True, transpose=False)
    
    df = seasonal_mean(stationnames, station_data, filename="predictions_", 
                            daterange=from1979to2012 , id_name="obs", method= "Stacking", use_id=True)
    
    
    
    
    station_info = pd.read_csv(os.path.join(station_datadir, "stationnames_new.csv"), index_col=0)
    
    # analysis
    
    d18Op = extract_var(Dataset=PD_data, varname="d18op", units="per mil", Dataset_wiso=PD_wiso,
                        ) 
    
    d18Op_alt = compute_lterm_mean(data=d18Op, time="annual")
    
    #plot the scatter and spatial data (check the style for fonts and line thickness)
    station_info["Annum"] = df_sm["Annum"]
    df = df.drop(index="Annum", axis=0)
    
    
    
    #ploting
    
    apply_style(fontsize=22, style=None, linewidth=3, usetex=True)
    
    projection = ccrs.PlateCarree()
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15, 13), subplot_kw={"projection": projection})
    
    plot_annual_mean(ax=ax2, variable='$\delta^{18}$Op vs SMOW', data_alt=d18Op_alt, cmap="RdYlBu", units="‰", vmax=2, vmin=-16, domain="Europe", 
                      levels=22, level_ticks=10, title="", left_labels=True, bottom_labels=True, use_colorbar_default=True,
                      center =False, cbar_pos = [0.20, 0.05, 0.25, 0.02], orientation= "horizontal", coast_resolution='50m',
                      plot_borders=True)
    
    sc = ax2.scatter(station_info['Longitude'], station_info['Latitude'], c= station_info['Annum'], cmap="RdYlBu",
                    vmax=2, vmin=-16, edgecolor="k", s= 160, transform = projection)
    
    # for row in station_info[['Longitude', 'Latitude']].itertuples():
    #     ax2.text(row.Longitude, row.Latitude, row.Index, transform = projection, fontsize=10, alpha=0.5)
    
    heatmaps(data=df, cmap="RdYlBu", label='$\delta^{18}$Op vs SMOW', title= None, 
             ax=ax1, cbar=False, xlabel='$\delta^{18}$Op vs SMOW', vmax=2, vmin=-16, rot=0)
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(path_to_save, "echam_gnip.pdf"), bbox_inches="tight", format = "pdf", dpi=300)
    
if __name__ == "__main__":
    plot_example(num=39)    
