# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 19:47:22 2023

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

def plot_different_projections(variable= "Precipitation"):
    
    
    datasets_26 = [CMIP5_RCP26_R1, CESM_RCP26, HadGEM2_RCP26, CORDEX_RCP26]
    datasets_85 = [CMIP5_RCP85_R1, CESM_RCP85, HadGEM2_RCP85, CORDEX_RCP85]
    
    
    apply_style(fontsize=23, style="seaborn-talk", linewidth=3,)
    
    if variable == "Precipitation":
        fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(24, 18),
                                                    sharex=True, sharey=True)
        
        
        stationloc_dir_prec = os.path.join(station_prec_datadir , "stationloc.csv")
        plot_projection_comparison(stationnames=stationnames_prec, path_to_data=path_to_data_prec, 
                                            filename="predictions_", id_name="CMIP5 RCP2.6", method="Stacking", 
                                            stationloc_dir=stationloc_dir_prec, daterange=from2040to2070, 
                                            datasets=datasets_26, variable="Precipitation", 
                                            dataset_varname="tp", ax=ax1, legend=False, xlabel= "Precipitation stations",
                                            ylabel="Precipitation [mm/month]",width=0.7, title="RCP 2.6 [2040-2070]",
                                            vmax=250, vmin=50, use_id=False)
        
        plot_projection_comparison(stationnames=stationnames_prec, path_to_data=path_to_data_prec, 
                                            filename="predictions_", id_name="CMIP5 RCP2.6", method="Stacking", 
                                            stationloc_dir=stationloc_dir_prec, daterange=from2070to2100, 
                                            datasets=datasets_26, variable="Precipitation", 
                                            dataset_varname="tp", ax=ax2, legend=True, xlabel= "Precipitation stations",
                                            ylabel="Precipitation [mm/month]", width=0.7, title="RCP 2.6 [2070-2100]",
                                            vmax=250, vmin=50, use_id=False)
        
        plot_projection_comparison(stationnames=stationnames_prec, path_to_data=path_to_data_prec, 
                                            filename="predictions_", id_name="CMIP5 RCP8.5", method="Stacking", 
                                            stationloc_dir=stationloc_dir_prec, daterange=from2040to2070, 
                                            datasets=datasets_85, variable="Precipitation", 
                                            dataset_varname="tp", ax=ax3, legend=False, xlabel= "Precipitation stations",
                                            ylabel="Precipitation [mm/month]",width=0.7, title="RCP 8.5 [2040-2070]", 
                                            vmax=250, vmin=50, use_id=False)
        
        plot_projection_comparison(stationnames=stationnames_prec, path_to_data=path_to_data_prec, 
                                            filename="predictions_", id_name="CMIP5 RCP8.5", method="Stacking", 
                                            stationloc_dir=stationloc_dir_prec, daterange=from2070to2100, 
                                            datasets=datasets_85, variable="Precipitation", 
                                            dataset_varname="tp", ax=ax4, legend=False, xlabel= "Precipitation stations",
                                            ylabel="Precipitation [mm/month]", width=0.7, title="RCP 8.5 [2070-2100]", 
                                            vmax=250, vmin=50, use_id=False)
        
        
        plt.tight_layout(h_pad=0.03)
        plt.subplots_adjust(left=0.02, right=0.85, top=0.94, bottom=0.05)
        plt.savefig(os.path.join(path_to_plot, "GCMs_comparison.svg"), bbox_inches="tight", dpi=600)
        
plot_different_projections()       