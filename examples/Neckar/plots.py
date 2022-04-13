# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:06:04 2022

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


sys.path.append("C:/Users/dboateng/Desktop/Python_scripts/ESD_Package")

from Package.ESD_utils import load_all_stations, load_pickle, load_csv
from Package.WeatherstationPreprocessing import read_weatherstationnames

from Package.plot import barplot, correlation_heatmap, boxplot
from Package.plot_utils import apply_style, correlation_data, count_predictors, boxplot_data

from predictor_settings import predictors

temp_datadir = "C:/Users/dboateng/Desktop/Datasets/Station/Neckar_Enz/Temperature/cdc_download_2022-03-17_13-38/processed"

prec_datadir = "C:/Users/dboateng/Desktop/Datasets/Station/Neckar_Enz/Precipitation/cdc_download_2022-03-17_13/processed"


namedict_prec  = read_weatherstationnames(prec_datadir)
stationnames_prec = list(namedict_prec.values())


num_stations_prec = len(stationnames_prec)
stationname_prec = stationnames_prec


path_data_precipitation = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Neckar/experiment2/final_cache_Precipitation"
path_data_temperature = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Neckar/experiment2/final_cache_Temperature"

#experiment 3

regressors = ["LassoLarsCV", "ARD", "MLPRegressor", "RandomForest", "XGBoost", "Bagging", "Stacking", "Voting"]


font = {'weight' : 'semibold',
    'size'   : 18}

mpl.rc('font', **font)


fig, ax = plt.subplots(1,1, figsize=(20,15))

boxplot(regressors, stationnames_prec, path_data_precipitation, ax=ax,  
            varname="test_r2", filename="validation_score_", xlabel="Estimators",
            ylabel="Validattion r²")

plt.tight_layout()
plt.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.05)
plt.savefig("inter_estimators_r2.png", bbox_inches="tight")



fig, ax = plt.subplots(1,1, figsize=(20,15))

boxplot(regressors, stationnames_prec, path_data_precipitation, ax=ax,  
            varname="test_rmse", filename="validation_score_", xlabel="Estimators",
            ylabel="Validattion RMSE")

plt.tight_layout()
plt.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.05)
plt.savefig("inter_estimators_rmse.png", bbox_inches="tight")

#experiment 2

#regressors = ["LassoLarsCV", "ARD", "MLPRegressor", "RandomForest", "XGBoost", "Bagging"]



# fig, ax = plt.subplots(1,1, figsize=(18,15))

# boxplot(regressors, stationnames_prec, path_data_precipitation, ax=None,  
#             varname="test_r2", filename="validation_score_", xlabel="Regressors",
#             ylabel="Validattion r²")

# plt.tight_layout()
# plt.subplots_adjust(left=0.15, right=0.88, top=0.97, bottom=0.05)
# plt.savefig("inter_model.png")

#plt.show()


#experiment 1 analysis 

#methods = ["Recursive", "TreeBased", "Sequential"]

#count predictors 

# df = count_predictors(methods, stationnames_prec, path_data_precipitation,
#                       "selected_predictors_", predictors)
# df.to_csv("predictors_count.csv")

#plot predictor correlation with station

# df = correlation_data(stationnames_prec, path_data_precipitation, 
#                       filename= "corrwith_predictors_" + methods[0], predictors=predictors)


# apply_style(fontsize=20, style="bmh") 

# fig, ax = plt.subplots(1,1, figsize=(18,15))
                        
# correlation_heatmap(data=df, cmap="RdBu", ax=ax, vmax=1, vmin=-1, center=0, cbar_ax=None, fig=fig,
#                         add_cbar=True, title=None, label= "Correlation Coefficinet", fig_path=None,
#                         xlabel="Predictors", ylabel="Stations")
# plt.tight_layout()
# plt.subplots_adjust(left=0.15, right=0.88, top=0.97, bottom=0.05)
# plt.savefig("predictor_correlation.png")

#plt.show()

#plot CV score for selector method 

# apply_style(fontsize=18, style="bmh")

# fig,(ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(18, 20), sharex=True, sharey=False)

# barplot(methods=methods, stationnames=stationnames_prec , path_to_data=path_data_precipitation, 
#         xlabel="Stations", ylabel="CV RMSE", varname= "test_rmse", varname_std ="test_rmse_std",
#         filename="validation_score_", ax=ax1)

# barplot(methods=methods, stationnames=stationnames_prec , path_to_data=path_data_precipitation, 
#         xlabel="Stations", ylabel="CV score", varname= "test_r2", varname_std ="test_r2_std",
#         filename="validation_score_", ax=ax2, legend=False)

# plt.tight_layout()
# plt.savefig("prec_selectors.png")








