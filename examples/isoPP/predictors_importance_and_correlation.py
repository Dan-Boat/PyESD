# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 17:23:09 2024

@author: dboateng
"""

import os 
import pandas as pd 
import matplotlib.pyplot as plt 


from pyESD.plot_utils import apply_style, correlation_data, count_predictors
from pyESD.plot import correlation_heatmap



from read_data import *
from predictor_setting import *
from pyESD.ESD_utils import load_csv, load_all_stations

path_to_save = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/isoPP/plots"
path_to_data = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/isoPP/final_cache_GNIP"

selector_methods = ["Recursive", "TreeBased",]

df = count_predictors(methods=selector_methods , stationnames=stationnames,
                           path_to_data=path_to_data, filename="selected_predictors_",
                           predictors=predictors)



df.to_csv(os.path.join(path_to_save, "predictors_count.csv"))




def plot_fract_importance():
    df = pd.DataFrame(index=predictors, columns=stationnames)
    
    for i,idx in enumerate(stationnames):
        
        df_imp = load_csv(stationname=idx, varname="feature_importance", path=path_to_data)
        df[idx] = df_imp
    
    df.columns = df.columns.str.replace("_", " ")
    
    df_mean = df.mean(axis=1).sort_values()
    df_std = df.std(axis=1)
    
    apply_style(fontsize=22, style=None, linewidth=2) 
    
    fig, ax = plt.subplots(1,1, figsize=(5,15))
    
    df_mean.plot(kind="barh", xerr=df_std)
    
    ax.set_xlabel("Mean Decrease in impurity", fontweight="bold", fontsize=22)
    fig.tight_layout()
    plt.savefig(os.path.join(path_to_save, "fract_importance.pdf"), bbox_inches="tight", format= "pdf")

def convert_pvalue(pvalue):
    if pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    elif pvalue > 0.05:
        return ""
    
def plot_corr_with():    
    df_cor, df_pvalue = correlation_data(stationnames, path_to_data, "corrwith_predictors_scipy", predictors, 
                               use_id=True, use_scipy=True)
    
    
        
    df_annot = df_pvalue.applymap(convert_pvalue)
    
    
    
    apply_style(fontsize=22, style=None, linewidth=2) 
    
    fig, ax = plt.subplots(1,1, figsize=(18,15))
                            
    correlation_heatmap(data=df_cor, cmap="RdBu", ax=ax, vmax=1, vmin=-1, center=0, cbar_ax=None, fig=fig,
                            add_cbar=True, title=None, label= "Pearson Correlation Coefficinet (PCC)", fig_path=path_to_save,
                            xlabel="Predictors", ylabel="GNIP Stations", fig_name="correlation_prec.pdf", annot=df_annot,
                            fmt="")
    
    
if __name__ == "__main__":
    
    plot_fract_importance()
    #plot_corr_with()


    