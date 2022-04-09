# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:34:25 2022

@author: dboateng
"""

import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import searbon as sns 
import os 


def correlation_heatmap(data, cmap, ax=None, vmax=None, vmin=None, center=0, levels=10, cbar_ax=None, 
                        add_cbar=True, title=None, label= "Correlation Coefficinet", fig_path=None):
    
    if ax is None:
        fig,ax = plt.subplots(1,1, sharex=False, figsize=(15, 13))
        
    
    if add_cbar == True:
        if cbar_ax is None:
            cbar_ax = [0.90, 0.30, 0.03, 0.45]
        
        
        cbar_ax = fig.add_axes(cbar_ax)
        cbar_ax.get_xaxis().set_visible(False)
        cbar_ax.yaxis.set_ticks_position('right')
        cbar_ax.set_yticklabels([])
        cbar_ax.tick_params(size=0)
        
    
    if all(parameter is not None for parameter in [vmin, vmax, levels]):
        
        sns.heatmap(ax=ax, data=data, cmap=cmap, vmax=vmax, vmin=vmin, center=center, cbar=add_cbar,
                    square=True, cbar_ax = cbar_ax, cbar_kws={"label": label, "shrink":0.50})
    
    
    plt.tight_layout()
    
    if fig_path is not None:
        plt.savefig(os.path.join(fig_path, "corr_fig.png"), bbox_inches="tight")
        


