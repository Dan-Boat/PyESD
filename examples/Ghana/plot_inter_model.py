# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 17:34:07 2022

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


path_to_data = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Ghana/model_selection"
path_to_plot = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Ghana/plots"



def plot_estimators_metrics():
    
    regressors = ["LassoLarsCV", "ARD", "MLP", "RandomForest",
                  "XGBoost", "Bagging", "Stacking"]
    
    colors = [grey, purple, lightbrown, tomato, skyblue, lightgreen, gold]
    
    apply_style(fontsize=20, style=None, linewidth=2)
    
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(20, 15), sharex=False)
    
    
    boxplot(regressors, stationnames_prec, path_to_data, ax=ax1,  
                varname="test_mae", filename="validation_score_", xlabel="Estimators",
                ylabel="CV MAE", colors = colors, patch_artist=(True))
    
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.05)
    plt.savefig(os.path.join(path_to_save, "inter_model_metrics.svg"), bbox_inches="tight", dpi=300)
    
    
plot_estimators_metrics()