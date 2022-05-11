# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 09:03:49 2022

@author: dboateng
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import pandas as pd
from cycler import cycler
from matplotlib import rc

try:
    from ESD_utils import load_all_stations, load_csv, load_pickle
except:
    from .ESD_utils import load_all_stations, load_csv, load_pickle

# colors 

orange = 'orangered'
lightblue = 'teal'
brown = 'sienna'
red = '#a41a36'
blue = '#006c9e'
green = '#55a868'
purple = '#8172b2'
lightbrown = '#ccb974'
pink = 'fuchsia'
lightgreen = 'lightgreen'
skyblue = "skyblue"
tomato = "tomato"
gold = "gold"
magenta = "magenta"
black = "black"
grey = "grey"

RdBu_r = plt.cm.RdBu_r
RdBu = plt.cm.RdBu

selector_method_colors = {
    "Recursive": green,
    "TreeBased": orange,
    "Sequential": blue,
    }



def apply_style(fontsize=20, style=None, linewidth=2):
    """
    

    Parameters
    ----------
    fontsize : TYPE, optional
        DESCRIPTION. The default is 10.
    style : TYPE, optional
        DESCRIPTION. The default is "bmh". ["seaborn", "fivethirtyeight",]

    Returns
    -------
    None.

    """
    if style is not None:
        plt.style.use(style)  
        
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    mpl.rc('text', usetex=True)
    mpl.rc('xtick', labelsize=fontsize)
    mpl.rc('ytick', labelsize=fontsize)
    mpl.rc('legend', fontsize=fontsize)
    mpl.rc('axes', labelsize=fontsize)
    mpl.rc('lines', linewidth=linewidth)
    mpl.rc("font", weight="bold")
    

    
    
def barplot_data(methods, stationnames, path_to_data, varname="test_r2", varname_std="test_r2_std", 
                 filename="validation_score_"):
    
    df = pd.DataFrame(index=stationnames, columns=methods,)
    df_std = pd.DataFrame(index=stationnames, columns=methods,)
    
    for method in methods:
        scores = load_all_stations(filename + method, path_to_data, stationnames)
        df[method] = scores[varname]
        df_std[method] = scores[varname_std]
        
        
    df.reset_index(drop=True, inplace=True)
    df.index += 1
    df_std.reset_index(drop=True, inplace=True)
    df_std.index += 1
    
    return df, df_std
    

def correlation_data(stationnames, path_to_data, filename, predictors):
    
    df = pd.DataFrame(index=stationnames, columns=predictors)
    
    for i,idx in enumerate(stationnames):
        
        df.iloc[i] = load_csv(idx, filename, path_to_data)
    
    df = df.astype(float)
    df.reset_index(drop=True, inplace=True)
    df.index += 1
    
    return df 

def count_predictors(methods, stationnames, path_to_data, filename, predictors):
    
    df_count =  pd.DataFrame(index=methods, columns=predictors)
    

    for method in methods:
        
        df = pd.DataFrame()
        for i, idx in enumerate(stationnames):
            
            selected = load_pickle(idx, filename + method, path_to_data)
            
            df = pd.concat([df, selected], axis=0)
        
        for predictor in predictors:
            
            if predictor in df:
                df_count.loc[method, predictor] = len(df[predictor].dropna())
        
    df_count = df_count.astype(float)
    
    return df_count

def boxplot_data(regressors, stationnames,  path_to_data, filename="validation_score_", 
                 varname="test_r2"):
    
    df = pd.DataFrame(index=stationnames, columns=regressors)
    
    for regressor in regressors:
        scores = load_all_stations(filename + regressor, path_to_data, stationnames)
        
        df[regressor] = scores[varname]
        
    df.reset_index(drop=True, inplace=True)
    df.index += 1
    
    return df 
        