# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 16:46:27 2022

@author: dboateng
"""
import os 
import sys
import matplotlib.pyplot as plt 
import matplotlib as mpl




sys.path.append("C:/Users/dboateng/Desktop/Python_scripts/ESD_Package")

from pyESD.ESD_utils import load_all_stations, load_pickle, load_csv
from pyESD.WeatherstationPreprocessing import read_weatherstationnames

from pyESD.plot import barplot, correlation_heatmap, boxplot
from pyESD.plot_utils import apply_style, correlation_data, count_predictors, boxplot_data

from predictor_settings import predictors

temp_datadir = "C:/Users/dboateng/Desktop/Datasets/Station/Neckar_Enz/Temperature/cdc_download_2022-03-17_13-38/processed"

prec_datadir = "C:/Users/dboateng/Desktop/Datasets/Station/Neckar_Enz/Precipitation/cdc_download_2022-03-17_13/processed"


namedict_prec  = read_weatherstationnames(prec_datadir)
stationnames_prec = list(namedict_prec.values())

namedict_temp  = read_weatherstationnames(temp_datadir)
stationnames_temp = list(namedict_temp.values())


selector_methods = ["Recursive", "TreeBased", "Sequential"]


exp_dir_prec = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Neckar/experiment2/final_cache_Precipitation"
exp_dir_temp = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Neckar/experiment2/final_cache_Temperature"

fig_path = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Neckar/plots"



regressors = ["LassoLarsCV", "ARD", "MLP", "RandomForest", "XGBoost",
              "Bagging", "Stacking"]


# font = {'weight' : 'semibold',
#     'size'   : 20}

# mpl.rc('font', **font)

apply_style(fontsize=22, style=None)


fig, ax = plt.subplots(1,1, figsize=(20,15))

boxplot(regressors, stationnames_prec, exp_dir_prec, ax=ax,  
            varname="test_r2", filename="validation_score_", xlabel="Estimators",
            ylabel="CV R²")

plt.tight_layout()
plt.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.05)
plt.savefig(os.path.join(fig_path, "fig 3.svg"), bbox_inches="tight")



fig, ax = plt.subplots(1,1, figsize=(20,15))

boxplot(regressors, stationnames_prec, exp_dir_prec, ax=ax,  
            varname="test_rmse", filename="validation_score_", xlabel="Estimators",
            ylabel="CV RMSE")

plt.tight_layout()
plt.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.05)
plt.savefig(os.path.join(fig_path, "fig4.svg"), bbox_inches="tight")


fig, ax = plt.subplots(1,1, figsize=(20,15))

boxplot(regressors, stationnames_temp, exp_dir_temp, ax=ax,  
            varname="test_r2", filename="validation_score_", xlabel="Estimators",
            ylabel="CV R²")

plt.tight_layout()
plt.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.05)
plt.savefig(os.path.join(fig_path, "fig5.svg"), bbox_inches="tight")



fig, ax = plt.subplots(1,1, figsize=(20,15))

boxplot(regressors, stationnames_temp, exp_dir_temp, ax=ax,  
            varname="test_rmse", filename="validation_score_", xlabel="Estimators",
            ylabel="CV RMSE")

plt.tight_layout()
plt.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.05)
plt.savefig(os.path.join(fig_path, "fig6.svg"), bbox_inches="tight")