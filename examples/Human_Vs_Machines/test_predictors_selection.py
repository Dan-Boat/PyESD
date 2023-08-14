# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 17:57:17 2023

@author: dboateng
"""
import os 
import pandas as pd 
import numpy as np 

from pyESD.feature_selection import RecursiveFeatureElimination, TreeBasedSelection, SequentialFeatureSelection
from pyESD.models import Regressors
from pyESD.standardizer import StandardScaling
from pyESD.ESD_utils import store_csv, store_pickle
from pyESD.plot import correlation_heatmap
from pyESD.plot_utils import apply_style, correlation_data, count_predictors
import matplotlib.pyplot as plt




stationnames = ["Beograd", "Kikinda", "Novi_Sad", "Palic", "Sombor", "Sremska_Mitrovica", "Vrsac",
                "Zrenjanin"]


path_to_store = "C:/Users/dboateng/Desktop/Datasets/Station/Vojvodina_new/plots"
path_to_data = "C:/Users/dboateng/Desktop/Datasets/Station/Vojvodina_new"


predictors = ['dswr', 'lftx', 'mslp', 'p__f', 'p__u', 'p__v', 'p__z', 'p_zh',
       'p5_f', 'p5_u', 'p5_v', 'p5_z', 'p500', 'p5zh', 'p8_f', 'p8_u', 'p8_v',
       'p8_z', 'p850', 'p8zh', 'pottmp', 'pr_wtr', 'prec', 'r500', 'r850',
       'rhum', 'shum', 'temp']



# read train_data 
def get_data(stationname):
    train_X = pd.read_csv(os.path.join(path_to_data, stationname, "train_X.csv"), parse_dates=["date"])
    train_y = pd.read_csv(os.path.join(path_to_data, stationname, "train_y.csv"), parse_dates=["date"])
    
    # set index
    X = train_X.set_index("date")
    y = train_y.set_index("date")
    
    y = y.squeeze()
    
    
    X = X.loc[~np.isnan(y)]
    y = y.dropna()
    
    
    # standardize 
    
    scaler = StandardScaling(method="standardscaler")
    scaler_x = scaler.fit(X)
    X_scaled = scaler_x.transform(X)
    
    scaler_y = scaler.fit(y)
    y_scaled = scaler_y.transform(y)
    
    return X_scaled, y_scaled



# perform correlation

def perform_cor():
    for station in stationnames:

        X, y = get_data(station)
        
        corr = X.corrwith(other=y, axis=0, drop=True, method="spearman")
        corr = corr.to_frame()
        corr = corr.T

        store_csv(station, varname="corrwith_predictors", var=corr, cachedir=path_to_store)




def plot_corr_with():
    perform_cor()

    df = correlation_data(stationnames, path_to_store, "corrwith_predictors",
                              predictors, use_scipy=False)
    
    df.index = df.index.str.replace("_", " ")
    df.columns = df.columns.str.replace("_", "")
        
    apply_style(fontsize=22, style=None) 
    
    fig, ax = plt.subplots(1,1, figsize=(20,15))
                            
    correlation_heatmap(data=df, cmap="RdBu", ax=ax, vmax=0.5, vmin=-0.5, center=0, cbar_ax=None, fig=fig,
                            add_cbar=True, title=None, label= "Spearman Correlation Coefficinet", fig_path=path_to_store,
                            xlabel="Predictors", ylabel="Stations", fig_name="correlation_prec.svg",)




# predictor selection
def select_predictors(selector, selector_method, station):

    X, y = get_data(station)
    
    selector.fit(X, y)
    selector.transform(X)
    selector.print_selected_features(X)
    selected_features = selector.select_names

    store_pickle(station, "selected_predictors_" + selector_method, selected_features,
            path_to_store) 
    
def perform_predictor_selection():
    selector_method = "TreeBased"
    
    selector_regressor="RandomForest"
    
    regressor_name_other = "Ridge"
    
    tree_based = TreeBasedSelection(regressor_name=selector_regressor)
    
    sequential = SequentialFeatureSelection(regressor_name = regressor_name_other)
    
    recursive = RecursiveFeatureElimination(regressor_name = regressor_name_other)
    
    for station in stationnames:
        select_predictors(selector=tree_based, selector_method="TreeBased", station=station)
        select_predictors(selector=sequential, selector_method="Sequential", station=station)
        select_predictors(selector=recursive, selector_method="Recursive", station=station)
        


#plot_corr_with()
#perform_predictor_selection()

        
selector_methods = ["TreeBased", "Sequential", "Recursive"]
df = count_predictors(methods=selector_methods , stationnames=stationnames,
                               path_to_data=path_to_store, filename="selected_predictors_",
                               predictors=predictors)
df.columns = df.columns.str.replace("_", " ")

apply_style(fontsize=23, style="seaborn-talk", linewidth=3,)    
fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 12))

df.T.plot(kind="bar", rot=45, fontsize=20, ax=ax)

ax.set_ylabel("Number of Stations selected", fontweight="bold", fontsize=20)
ax.grid(True, linestyle="--", color="grey")
plt.tight_layout()
plt.savefig(os.path.join(path_to_store, "predictors_count.png"), bbox_inches="tight", dpi=600)



