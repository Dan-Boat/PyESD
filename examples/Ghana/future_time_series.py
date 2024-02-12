# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 17:41:25 2023

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

def plot_ensemble_timeseries():

    stationnames = ["Navrongo", "Bolgatanga", "Wa","Bole"]
    
    apply_style(fontsize=23, style="seaborn-talk", linewidth=3,)
    fig,(ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(15,20),
                                       sharex=True, sharey=True)
    plt.subplots_adjust(left=0.02, right=0.90, top=0.94, bottom=0.05,hspace=0.01)
    
    
    plot_time_series(stationnames, path_to_data_prec, filename="predictions_", 
                     id_name="CMIP6 RCP2.6 anomalies", daterange=fullCMIP6,
                     color=black, label="RCP 2.6", ymax=40, ymin=-40, 
                     ylabel= "Precipitation anomalies [mm/month]", ax=ax1,
                     window=12)
                     
    
    plot_time_series(stationnames, path_to_data_prec, filename="predictions_", 
                     id_name="CMIP6 RCP4.5 anomalies", daterange=fullCMIP6,
                     color=red, label="RCP 4.5", ymax=40, ymin=-40,
                     ylabel= "Precipitation anomalies [mm/month]", ax=ax2, 
                     )
    
    plot_time_series(stationnames, path_to_data_prec, filename="predictions_", 
                     id_name="CMIP6 RCP8.5 anomalies", daterange=fullCMIP6,
                     color=blue, label="RCP 8.5", ymax=40, ymin=-40,
                     ylabel= "Precipitation anomalies [mm/month]", ax=ax3, 
                     )
    
    
    plt.tight_layout(h_pad=0.02)
    plt.savefig(os.path.join(path_to_plot, "time_series_prec_north.svg"), bbox_inches="tight", dpi=300)
    
    
    
    stationnames = ["Wenchi", "Sunyani", "Dormaa-Ahenkro",] 
    
    apply_style(fontsize=23, style="seaborn-talk", linewidth=3,)
    fig,(ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(15,20),
                                       sharex=True, sharey=True)
    plt.subplots_adjust(left=0.02, right=0.90, top=0.94, bottom=0.05,hspace=0.01)
    
    
    plot_time_series(stationnames, path_to_data_prec, filename="predictions_", 
                     id_name="CMIP6 RCP2.6 anomalies", daterange=fullCMIP6,
                     color=black, label="RCP 2.6", ymax=40, ymin=-40, 
                     ylabel= "Precipitation anomalies [mm/month]", ax=ax1,
                     window=12)
                     
    
    plot_time_series(stationnames, path_to_data_prec, filename="predictions_", 
                     id_name="CMIP6 RCP4.5 anomalies", daterange=fullCMIP6,
                     color=red, label="RCP 4.5", ymax=40, ymin=-40,
                     ylabel= "Precipitation anomalies [mm/month]", ax=ax2, 
                     )
    
    plot_time_series(stationnames, path_to_data_prec, filename="predictions_", 
                     id_name="CMIP6 RCP8.5 anomalies", daterange=fullCMIP6,
                     color=blue, label="RCP 8.5", ymax=40, ymin=-40,
                     ylabel= "Precipitation anomalies [mm/month]", ax=ax3, 
                     )
    
    
    plt.tight_layout(h_pad=0.02)
    plt.savefig(os.path.join(path_to_plot, "time_series_prec_central.svg"), bbox_inches="tight", dpi=300)
    
    
    
    stationnames = ["Kumasi", "Abetifi", "Dunkwa", "Tarkwa", "Axim", "Takoradi", 
                          "Accra", "Tema", "Akuse", "Akim-Oda"]
    
    apply_style(fontsize=23, style="seaborn-talk", linewidth=3,)
    fig,(ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(15,20),
                                       sharex=True, sharey=True)
    plt.subplots_adjust(left=0.02, right=0.90, top=0.94, bottom=0.05,hspace=0.01)
    
    
    plot_time_series(stationnames, path_to_data_prec, filename="predictions_", 
                     id_name="CMIP6 RCP2.6 anomalies", daterange=fullCMIP6,
                     color=black, label="RCP 2.6", ymax=40, ymin=-40, 
                     ylabel= "Precipitation anomalies [mm/month]", ax=ax1,
                     window=12)
                     
    
    plot_time_series(stationnames, path_to_data_prec, filename="predictions_", 
                     id_name="CMIP6 RCP4.5 anomalies", daterange=fullCMIP6,
                     color=red, label="RCP 4.5", ymax=40, ymin=-40,
                     ylabel= "Precipitation anomalies [mm/month]", ax=ax2, 
                     )
    
    plot_time_series(stationnames, path_to_data_prec, filename="predictions_", 
                     id_name="CMIP6 RCP8.5 anomalies", daterange=fullCMIP6,
                     color=blue, label="RCP 8.5", ymax=40, ymin=-40,
                     ylabel= "Precipitation anomalies [mm/month]", ax=ax3, 
                     )
    
    
    plt.tight_layout(h_pad=0.02)
    plt.savefig(os.path.join(path_to_plot, "time_series_prec_south.svg"), bbox_inches="tight", dpi=300)
    
    
plot_ensemble_timeseries()