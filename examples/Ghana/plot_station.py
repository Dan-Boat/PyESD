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

def plot_stations():
    
    df_prec_sm = seasonal_mean(stationnames_prec, path_to_data, filename="predictions_", 
                            daterange=from1961to2012 , id_name="obs", method= "Stacking")
    
    
    df_prec = monthly_mean(stationnames_prec, path_to_data_prec, filename="predictions_", 
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
data = gpd.read_file(shape_file_dir)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 15))

data.boundary.plot(ax=ax, color="black", alpha=0.8, linewidth=2)
data.plot(ax=ax, column="REGION", cmap="tab20", alpha=0.7, legend=True)