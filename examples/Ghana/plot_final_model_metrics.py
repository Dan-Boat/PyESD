# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:08:45 2023

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

def write_metrics(path_to_data, method, stationnames, path_to_save, varname,
                  filename_train = "validation_score_",
                  filename_test="test_score_13to17_"):
    
    train_score = load_all_stations(filename_train + method, path_to_data, stationnames)
    
    test_score = load_all_stations(filename_test + method, path_to_data, stationnames)
    
    #climate_score = load_all_stations("climate_score_13to17_" + method, path_to_data, stationnames)
    
    df = pd.concat([train_score, test_score], axis=1, ignore_index=False)
    
    scores_df = load_all_stations(varname="CV_scores_" + method, path=path_to_data, 
                                  stationnames= stationnames)
    
        
    df_add = pd.DataFrame(index=stationnames_prec, columns= ["r2", "rmse", "mae"])
    for i,stationname in enumerate(stationnames):
        
        r2 = scores_df["test_r2"].loc[stationname]
        mae = -1* scores_df["test_neg_mean_absolute_error"].loc[stationname]
        index_max = r2.argmax()
        index_min = mae.argmin()
        
        df_add["r2"].loc[stationname] = scores_df["test_r2"].loc[stationname][index_max]
        df_add["rmse"].loc[stationname] = -1 * scores_df["test_neg_root_mean_squared_error"].loc[stationname][index_max]
        df_add["mae"].loc[stationname] = -1* scores_df["test_neg_mean_absolute_error"].loc[stationname][index_min]
    
    # save files
    
    df.to_csv(os.path.join(path_to_save, method + "_train_test_metrics.csv"), index=True, header=True)
    df_add.to_csv(os.path.join(path_to_save, method + "_CV_metrics.csv"), index=True, header=True)
    
    
# write_metrics(path_to_data_prec, "Stacking", stationnames_prec, path_to_plot, 
#            "Precipitation")


# read metrics 

data = pd.read_csv(path_to_data_metrics, usecols=["CV R2", "CV MAE"])

# plot R2 and MAE for the final model
apply_style(fontsize=23, style="seaborn-talk", linewidth=3,)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 12))

color = { "boxes": skyblue,
              "whiskers": black,
              "medians": red,
              "caps": black,
               }

box1 = data["CV R2"].plot(kind="box", ax= ax1, fontsize=23, color=color, 
                   sym="+b", grid=False,widths=0.9, notch=False, return_type="dict", patch_artist=True)
ax1.set_ylabel("CV R²", fontweight="bold", fontsize=20)
ax1.grid(True, linestyle="--", color=gridline_color)


box2 = data["CV MAE"].plot(kind="box", ax= ax2, fontsize=23, color=color, 
                   sym="+b", grid=False,widths=0.9, notch=False, return_type="dict", patch_artist=True)
ax2.yaxis.tick_right()
ax2.set_ylabel("CV MAE", fontweight="bold", fontsize=20)
ax2.grid(True, linestyle="--", color=gridline_color)


boxes = [box1, box2]

            
plt.tight_layout()
plt.savefig(os.path.join(path_to_plot, "final_model_metrics.svg"), bbox_inches="tight", format= "svg",  dpi=600)

#plot.show()
