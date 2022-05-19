# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 16:46:03 2022

@author: dboateng
"""

import os 
import sys
import matplotlib.pyplot as plt  
 

# setting model path

from pyESD.WeatherstationPreprocessing import read_weatherstationnames

from pyESD.plot import barplot, correlation_heatmap
from pyESD.plot_utils import apply_style, correlation_data, count_predictors

from predictor_settings import predictors

temp_datadir = "C:/Users/dboateng/Desktop/Datasets/Station/Neckar_Enz/Temperature/cdc_download_2022-03-17_13-38/processed"

prec_datadir = "C:/Users/dboateng/Desktop/Datasets/Station/Neckar_Enz/Precipitation/cdc_download_2022-03-17_13/processed"


namedict_prec  = read_weatherstationnames(prec_datadir)
stationnames_prec = list(namedict_prec.values())

namedict_temp  = read_weatherstationnames(temp_datadir)
stationnames_temp = list(namedict_temp.values())


selector_methods = ["Recursive", "TreeBased", "Sequential"]


exp_dir_prec = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Neckar/experiment1/final_cache_Precipitation"
exp_dir_temp = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Neckar/experiment1/final_cache_Temperature"

fig_path = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Neckar/plots"


#ploting the correlation map for precipitation and temperature 

# extracting correlation data

df_prec = correlation_data(stationnames_prec, exp_dir_prec, 
                    filename= "corrwith_predictors_" + selector_methods[0], predictors=predictors)

df_temp = correlation_data(stationnames_temp, exp_dir_temp, 
                    filename= "corrwith_predictors_" + selector_methods[0], predictors=predictors)


apply_style(fontsize=22, style=None) 

fig, ax = plt.subplots(1,1, figsize=(20,15))
                        
correlation_heatmap(data=df_prec, cmap="RdBu", ax=ax, vmax=1, vmin=-1, center=0, cbar_ax=None, fig=fig,
                        add_cbar=True, title=None, label= "Correlation Coefficinet", fig_path=fig_path,
                        xlabel="Predictors", ylabel="Stations", fig_name="fig s1a.svg",)


fig, ax = plt.subplots(1,1, figsize=(20,15))

correlation_heatmap(data=df_temp, cmap="RdBu", ax=ax, vmax=1, vmin=-1, center=0, cbar_ax=None, fig=fig,
                        add_cbar=True, title=None, label= "Correlation Coefficinet", fig_path=fig_path,
                        xlabel="Predictors", ylabel="Stations", fig_name="fig s1b.svg",)

# plotting model performance based on predictor selector approach

fig,(ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(18, 20), sharex=True, sharey=False)

barplot(methods=selector_methods, stationnames=stationnames_prec , path_to_data=exp_dir_prec, 
        xlabel="Stations", ylabel="CV RMSE", varname= "test_rmse", varname_std ="test_rmse_std",
        filename="validation_score_", ax=ax1)

barplot(methods=selector_methods, stationnames=stationnames_prec , path_to_data=exp_dir_prec, 
        xlabel="Stations", ylabel="CV R²", varname= "test_r2", varname_std ="test_r2_std",
        filename="validation_score_", ax=ax2, legend=False)

plt.tight_layout()
plt.savefig(os.path.join(fig_path,"fig1.svg"))



fig,(ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(18, 20), sharex=True, sharey=False)

barplot(methods=selector_methods, stationnames=stationnames_temp , path_to_data=exp_dir_temp, 
        xlabel="Stations", ylabel="CV RMSE", varname= "test_rmse", varname_std ="test_rmse_std",
        filename="validation_score_", ax=ax1)

barplot(methods=selector_methods, stationnames=stationnames_temp , path_to_data=exp_dir_temp, 
        xlabel="Stations", ylabel="CV R²", varname= "test_r2", varname_std ="test_r2_std",
        filename="validation_score_", ax=ax2, legend=False)

plt.tight_layout()
plt.savefig(os.path.join(fig_path,"fig2.svg"))



# extrating the frequency of predictors selected


df_prec = count_predictors(selector_methods, stationnames_prec, exp_dir_prec,
                      "selected_predictors_", predictors)

df_temp = count_predictors(selector_methods, stationnames_temp, exp_dir_temp,
                      "selected_predictors_", predictors)

df_prec.to_csv(os.path.join(fig_path, "predictors_count_prec.csv"))
df_temp.to_csv(os.path.join(fig_path, "predictors_count_temp.csv"))
