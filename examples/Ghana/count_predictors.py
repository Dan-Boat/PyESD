# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 10:14:16 2023

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
from matplotlib.ticker import FormatStrFormatter


from pyESD.ESD_utils import load_all_stations, load_pickle, load_csv
from pyESD.plot import *
from pyESD.plot_utils import *
from pyESD.plot_utils import *

from settings import *
from read_data import *



path_to_data_prec = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Ghana/predictor_selection"
path_to_plot = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Ghana/plots"


def save_count_predictors():
    
    
    selector_methods = ["TreeBased"]
    
    df_prec = count_predictors(methods=selector_methods , stationnames=stationnames_prec,
                               path_to_data=path_to_data_prec, filename="selected_predictors_",
                               predictors=predictors)
    
    
    df_prec.to_csv(os.path.join(path_to_plot, "predictors_prec_count.csv"))
    
selected_predictors = ["tp", "u700", "v850", "u850", "r250", "v700", "u500", "r500", "v1000", "NAO",]
number = [19, 16, 13, 13, 13, 9, 8, 7, 6, 6]

apply_style(fontsize=23, style="seaborn-talk", linewidth=3,)    
fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 12))
ax.bar(x=selected_predictors, height=number)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

plt.tight_layout()
plt.savefig(os.path.join(path_to_plot, "predictors_count.svg"), bbox_inches="tight", dpi=600)