# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 09:03:49 2022

@author: dboateng
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np 
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



def resample_seasonally(data, daterange):
    df = data[daterange].resample("Q-NOV").mean()
    winter = df[df.index.quarter == 1].mean()
    spring = df[df.index.quarter == 2].mean()
    summer = df[df.index.quarter == 3].mean()
    autumn = df[df.index.quarter == 4].mean()
    
    return winter, spring, summer, autumn

def resample_monthly(data, daterange):
    
    df = data.resample("MS").mean()
    df = df[daterange]
    
    Jan = df[df.index.quarter == 1].mean()
    Feb = df[df.index.quarter == 2].mean()
    Mar = df[df.index.quarter == 3].mean()
    Apr = df[df.index.quarter == 4].mean()
    May = df[df.index.quarter == 5].mean()
    Jun = df[df.index.quarter == 6].mean()
    Jul = df[df.index.quarter == 7].mean()
    Aug = df[df.index.quarter == 8].mean()
    Sep = df[df.index.quarter == 9].mean()
    Oct = df[df.index.quarter == 10].mean()
    Nov = df[df.index.quarter == 11].mean()
    Dec = df[df.index.quarter == 12].mean()
    
    
    month_means = [Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec]
    
    return month_means


def seasonal_mean(stationnames, path_to_data, filename, daterange, id_name):

    columns = ["DJF", "MAM", "JJA", "SON", "Annum"]
    
    df_stations = pd.DataFrame(index=stationnames, columns=columns)
    
    for i,stationname in enumerate(stationnames):
        df = load_csv(stationname, filename, path_to_data)
        obs = df[id_name]
        winter, spring, summer, autumn = resample_seasonally(obs, daterange)
        
        obs_mean = obs[daterange].mean()
        
        means = [ winter, spring, summer, autumn, obs_mean]
        
        for j,season in enumerate(columns):
            df_stations.loc[stationname][season] = means[j]
    
    df_stations.reset_index(drop=True, inplace=True)
    df_stations.index +=1
    df_stations = df_stations.T.astype(float)
    
    return df_stations
             
def monthly_mean(stationnames, path_to_data, filename, daterange, id_name):
    
    import calendar	
    month_names = [calendar.month_abbr[im+1] for im in np.arange(12)]
    df_stations = pd.DataFrame(index=stationnames, columns=month_names)
    
    for i,stationname in enumerate(stationnames):
        df = load_csv(stationname, filename, path_to_data)
        obs = df[id_name]
        month_means = resample_monthly(obs, daterange)
        
        for j,month in enumerate(month_names):
            df_stations.loc[stationname][month] = month_means[j]
            
    df_stations.reset_index(drop=True, inplace=True)
    df_stations.index +=1
    df_stations = df_stations.T.astype(float)
    
    return df_stations
        
    
    
    
    