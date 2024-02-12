# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 17:28:01 2023

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


from pyESD.ESD_utils import load_all_stations, load_pickle, load_csv
from pyESD.plot import *
from pyESD.plot_utils import *
from pyESD.plot_utils import *

from settings import *
from read_data import *



path_to_data_prec = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Ghana/final_experiment"
path_to_plot = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Ghana/plots"
path_to_data_metrics = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Ghana/plots/Stacking_final_metrics.csv"

def plot_seasonal_climatologies():
    
    
    #obs
    df_prec_sm = seasonal_mean(stationnames_prec, path_to_data_prec, filename="predictions_", 
                            daterange=from1981to2012 , id_name="obs", method= "Stacking",
                            use_id=False)
    
    
    
    
    df_prec_26_from2040to2070 = seasonal_mean(stationnames_prec, path_to_data_prec, filename="predictions_", 
                            daterange=from2040to2070 , id_name="CMIP6 RCP2.6 anomalies", method= "Stacking",
                            use_id=False)
    
    
    df_prec_85_from2040to2070 = seasonal_mean(stationnames_prec, path_to_data_prec, filename="predictions_", 
                            daterange=from2040to2070 , id_name="CMIP6 RCP8.5 anomalies", method= "Stacking",
                            use_id=False)
    
    
    
    df_prec_26_from2070to2100 = seasonal_mean(stationnames_prec, path_to_data_prec, filename="predictions_", 
                            daterange=from2070to2100 , id_name="CMIP6 RCP2.6 anomalies", method= "Stacking",
                            use_id=False)
    
    
    df_prec_85_from2070to2100 = seasonal_mean(stationnames_prec, path_to_data_prec, filename="predictions_", 
                            daterange=from2070to2100 , id_name="CMIP6 RCP8.5 anomalies", method= "Stacking",
                            use_id=False)
    
    
    
    apply_style(fontsize=23, style="seaborn-talk", linewidth=3,)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(20,15), sharex=False)
    #cbar_ax = fig.add_axes([0.90, 0.35, 0.02, 0.25])
    
    # end of century
    
    heatmaps(data=df_prec_sm, cmap="YlGnBu", label="Precipitation [mm/month]", title= "obs [1981-2012]", 
             ax=ax1, cbar=True, )
    
    
    heatmaps(data=df_prec_26_from2070to2100, cmap=BrBG, label="Precipitation [mm/month]", title= "RCP 2.6 [2070-2100]", 
             ax=ax2, cbar=True, vmax=20, vmin=-20, center=0, )
    
    heatmaps(data=df_prec_85_from2070to2100, cmap=BrBG, label="Precipitation [mm/month]", title= "RCP 8.5 [2070-2100]", 
             ax=ax3, cbar=True, vmax=20, vmin=-20, center=0, xlabel="Stations")
    
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.02, right=0.85, top=0.94, bottom=0.05,hspace=0.01)
    plt.savefig(os.path.join(path_to_plot, "seasonal_trend_prec_endcentury.svg"), bbox_inches="tight", dpi=600)
    
    apply_style(fontsize=23, style="seaborn-talk", linewidth=3,)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(20,15), sharex=False)
    #cbar_ax = fig.add_axes([0.90, 0.35, 0.02, 0.25])
    
    # mid of century
    
    heatmaps(data=df_prec_sm, cmap="YlGnBu", label="Precipitation [mm/month]", title= "obs [1981-2012]", 
             ax=ax1, cbar=True, )
    
    
    heatmaps(data=df_prec_26_from2040to2070, cmap=BrBG, label="Precipitation [mm/month]", title= "RCP 2.6 [2040-2070]", 
             ax=ax2, cbar=True, vmax=20, vmin=-20, center=0, )
    
    heatmaps(data=df_prec_85_from2040to2070, cmap=BrBG, label="Precipitation [mm/month]", title= "RCP 8.5 [2040-2070]", 
             ax=ax3, cbar=True, vmax=20, vmin=-20, center=0, xlabel="Stations")
     
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.02, right=0.85, top=0.94, bottom=0.05,hspace=0.01)
    plt.savefig(os.path.join(path_to_plot, "seasonal_trend_prec_midcentury.svg"), bbox_inches="tight", dpi=600)
    
    
plot_seasonal_climatologies()
    
    
