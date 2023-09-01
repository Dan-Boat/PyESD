# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:56:39 2023

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


def plot_prediction_example(axes, stationnames):
    
    for i,station in enumerate(stationnames):
        
    
        print("----plotting for the station:", station + "precipitation")
        
        
        
        scatterplot(station_num=i, stationnames=stationnames, path_to_data=path_to_data_prec, 
                    filename="predictions_", ax=axes[i], xlabel="observed", ylabel="predicted",
                    method= "Stacking", obs_train_name="obs 1981-2012", 
                    obs_test_name="obs 2013-2017", 
                    val_predict_name="ERA5 1981-2012", 
                    test_predict_name="ERA5 2013-2017")
        
        
        axes[i].set_title(station, fontsize=20, weight="bold", loc="left")    
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.05)
    plt.savefig(os.path.join(path_to_plot, "prediction_examp_" + station + "_.svg"), bbox_inches="tight", dpi=300)
        
        
stationnames = ["Navrongo", "Sunyani", "Axim"]

apply_style(fontsize=23, style="seaborn-talk", linewidth=3,)
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(23, 15),)
axes = [ax1, ax2, ax3]

plot_prediction_example(stationnames=stationnames, axes=axes)
plt.tight_layout()
plt.savefig(os.path.join(path_to_plot, "Prediction_example_txt.svg"), bbox_inches="tight", dpi=600)