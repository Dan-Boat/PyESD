# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 16:46:03 2022

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

from settings import *
from read_data import *
from read_data import *


path_to_data = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Ghana/predictor_selection"
path_to_plot = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Ghana/plots"


def plot_predictor_selector_metrics():
 
    selector_methods = ["Recursive", "TreeBased", "Sequential"]
    
    apply_style(fontsize=20, style=None, linewidth=2)
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(20, 15), sharex=False)
    
    barplot(methods=selector_methods, stationnames=stationnames_prec, path_to_data=path_to_data,
            xlabel=None, ylabel="CV MAE", varname= "test_mae", varname_std ="test_mae_std",
            filename="validation_score_", ax=ax1, legend=True, width=0.8, rot=90)
    
    barplot(methods=selector_methods, stationnames=stationnames_prec , path_to_data=path_to_data, 
            xlabel="Stations", ylabel="Fit R²", varname= "train_r2", varname_std =None,
            filename="validation_score_", ax=ax2, legend=False, width=0.8, rot=90)
    
  
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.94, bottom=0.06)
    plt.savefig(os.path.join(path_to_plot, "predictor_selection.svg"), bbox_inches="tight", dpi=300)


plot_predictor_selector_metrics()

#extract selected predictors