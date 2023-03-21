# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 12:13:49 2022

@author: dboateng
"""

import os 
import sys
import matplotlib.pyplot as plt 
import matplotlib as mpl
import pandas as pd 
import numpy as np 
from collections import OrderedDict
import seaborn as sns
import geopandas as gpd 



from pyESD.ESD_utils import load_all_stations, load_pickle, load_csv
from pyESD.plot import *
from pyESD.plot_utils import *
from pyESD.plot_utils import *

from settings import *
from read_data import *

path_to_data = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Ghana/model_selection"
path_to_plot = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Ghana/plots"
shape_file_dir = "C:/Users/dboateng/Desktop/Datasets/Station/Ghana/Ghana_ShapeFile"

station_info = "C:/Users/dboateng/Desktop/Datasets/Station/Ghana/Update_datasets/processed/monthly/stationnames.csv"

def plot_stations(stationnames):
    
    df_prec_sm = seasonal_mean(stationnames, path_to_data, filename="predictions_", 
                            daterange=from1961to2012 , id_name="obs", method= "Stacking")
    
    
    df_prec = monthly_mean(stationnames, path_to_data, filename="predictions_", 
                            daterange=from1961to2012 , id_name="obs", method= "Stacking")
    
    means_prec, stds_prec = estimate_mean_std(df_prec)
    
    
    
    apply_style(fontsize=22, style=None, linewidth=3)
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(20, 15))
    
    heatmaps(data=df_prec_sm, cmap="Blues", label="Precipitation [mm/month]", title= None, 
             ax=ax1, cbar=True, xlabel="Precipitation stations")
    
    plot_monthly_mean(means=means_prec, stds=stds_prec, color=seablue, ylabel="Precipitation [mm/month]", 
                      ax=ax2)
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(path_to_plot, "stations.svg"), bbox_inches="tight", dpi=300)
    
    
# ploting the locations of stations 

def plot_location_map():
    data = gpd.read_file(shape_file_dir)
    
    apply_style(fontsize=24, style=None, linewidth=2)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(24, 16))
    
    data.boundary.plot(ax=ax, color="black", alpha=0.8, linewidth=2)
    data.plot(ax=ax, column="REGION", cmap="tab20", alpha=0.7, legend=True,
              legend_kwds={"loc":"upper right", "fontsize": 18, "bbox_to_anchor":(1.15, 0.96)}) #add bbhox anchor to put it outside the fig
    
    df_info = pd.read_csv(station_info)
    
    gdf = gpd.GeoDataFrame(df_info, geometry=gpd.points_from_xy(x=df_info.Longitude, y=df_info.Latitude))
    
    gdf.plot(ax=ax, color="black", markersize=80)
    
    for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf.Name):
        ax.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points", color="black", fontsize=24, 
                    fontweight="bold")
    
    ax.set_ylabel("Latitude [°N]", fontweight="bold", fontsize=24)
    ax.grid(True, linestyle="--", color="grey")
    
    ax.set_xlabel("Longitude [°E]", fontweight="bold", fontsize=24)
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.05)
    plt.savefig(os.path.join(path_to_plot, "stations_map.svg"), bbox_inches="tight", format= "svg", dpi=600)



