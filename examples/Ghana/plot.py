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


from pyESD.ESD_utils import load_all_stations, load_pickle, load_csv
from pyESD.plot import *
from pyESD.plot_utils import *
from pyESD.plot_utils import *

from predictor_settings import *
from read_data import *
from read_data import *


def plot_stations():
    
    df_prec_sm = seasonal_mean(stationnames_prec, path_to_data, filename="predictions_", 
                            daterange=from1961to2012 , id_name="obs", method= "Recursive")
    
    
    df_prec = monthly_mean(stationnames_prec, path_to_data_prec, filename="predictions_", 
                            daterange=from1961to2012 , id_name="obs", method= "Recursive")
    
    means_prec, stds_prec = estimate_mean_std(df_prec)
    
    
    
    apply_style(fontsize=20, style=None, linewidth=2)
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(20, 15))
    
    heatmaps(data=df_prec_sm, cmap="Blues", label="Precipitation [mm/month]", title= None, 
             ax=ax1, cbar=True, xlabel="Precipitation stations")
    
    plot_monthly_mean(means=means_prec, stds=stds_prec, color=seablue, ylabel="Precipitation [mm/month]", 
                      ax=ax2)
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(path_to_save, "Fig1a.svg"), bbox_inches="tight", dpi=300)