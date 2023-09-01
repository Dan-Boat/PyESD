# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 11:29:59 2023

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


def plot_data_available(datarange, stationnames):
    df_obs = pd.DataFrame(index = datarange, columns = stationnames)
    
    n = len(stationnames)
    for i in range(n):
        stationname = stationnames[i]
        df = load_csv(stationname, "predictions_Stacking", path_to_data_prec)
        obs = df['obs'][datarange]
        
        df_obs[stationname] = obs
       
    df_obs = df_obs.T
    df_obs.index = df_obs.index.str.replace("_", " ")
    df_obs = df_obs.astype(float)
    df_obs[~df_obs.isnull()] = 1
    df_obs[df_obs==0] = 1
    df_obs[df_obs.isnull()] = 0
    df_obs.columns = df_obs.columns.strftime("%Y")
    
    fig, axes = plt.subplots(1, 1, figsize= (15,13))
    
    sns.heatmap(ax = axes, data= df_obs, cmap= "Blues", cbar = False, linewidth = 0.2)
    
    #plt.tight_layout()
    plt.savefig(os.path.join(path, "weather_station_overview" + variable +".svg"),
                format = "svg", dpi = 1200)
    plt.close()
    plt.clf()
    
plot_data_available(from1981to2017, stationnames_prec)