# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:34:25 2022

@author: dboateng
"""

import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import seaborn as sns
import os 

try:  
    from plot_utils import *
    
except:
    from .plot_utils import *


def correlation_heatmap(data, cmap, ax=None, vmax=None, vmin=None, center=0, cbar_ax=None, 
                        add_cbar=True, title=None, label= "Correlation Coefficinet", fig_path=None,
                        xlabel=None, ylabel=None, fig=None):
    
    if ax is None:
        fig,ax = plt.subplots(1,1, sharex=False, figsize=(15, 13))
        
    
    if add_cbar == True:
        if cbar_ax is None:
            cbar_ax = [0.90, 0.4, 0.02, 0.25]
        
        
        cbar_ax = fig.add_axes(cbar_ax)
        cbar_ax.get_xaxis().set_visible(False)
        cbar_ax.yaxis.set_ticks_position('right')
        cbar_ax.set_yticklabels([])
        cbar_ax.tick_params(size=0)
        
    sns.set(font_scale=1.2)
    if all(parameter is not None for parameter in [vmin, vmax]):
        
        sns.heatmap(ax=ax, data=data, cmap=cmap, vmax=vmax, vmin=vmin, center=center, cbar=add_cbar,
                    square=True, cbar_ax = cbar_ax, cbar_kws={"label": label, "shrink":0.5,
                                                              "drawedges": False},
                    linewidth=0.5, linecolor="black")
    else:
        
        sns.heatmap(ax=ax, data=data, cmap=cmap, robust=True, cbar=add_cbar,
                    square=True, cbar_ax = cbar_ax, cbar_kws={"label": label, "shrink":0.5,
                                                              "drawedges": False},
                    linewidth=0.5, linecolor="black")
    
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
    
    plt.tight_layout()
    
    plt.subplots_adjust(left=0.15, right=0.88, top=0.97, bottom=0.05)
    
    if fig_path is not None:
        plt.savefig(os.path.join(fig_path, "corr_fig.png"), bbox_inches="tight")
        

def barplot(methods, stationnames, path_to_data, ax=None, xlabel=None, ylabel=None, 
            varname="test_r2", varname_std="test_r2_std", filename="validation_score_", legend=True):
    
    if ax is None:
        fig,ax = plt.subplots(1,1, sharex=False, figsize=(18, 15))
        
    df, df_std = barplot_data(methods, stationnames, path_to_data, varname=varname, varname_std=varname_std, 
                     filename=filename)
    
    colors = [selector_method_colors[m] for m in methods]
    mpl.rcParams["axes.prop_cycle"] = cycler("color", colors)
    
    df.plot(kind="bar", yerr=df_std, rot=0, ax=ax, legend = legend, fontsize=20)
    
    if xlabel is not None:
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=20)
        ax.set_xlabel(xlabel, fontweight="bold", fontsize=20)
        
    if legend ==True:    
        ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1), borderaxespad=0., frameon=True)
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.88, top=0.97, bottom=0.05)
        
