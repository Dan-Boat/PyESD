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

# A4 paper size: 210 mm X 297 mm

cm = 0.3937   # 1 cm in inch for plot size
pt = 1/72.27  # pt in inch from latex geometry package
textwidth = 345*pt
big_width = textwidth + 2*3*cm



# colors 

orange = 'orangered'
lightblue = 'teal'
brown = 'sienna'
red = 'red'
blue = 'blue'
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
BrBG = plt.cm.BrBG
seismic = plt.cm.seismic

selector_method_colors = {
    "Recursive": orange,
    "TreeBased": black,
    "Sequential": grey,
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
    

def correlation_data(stationnames, path_to_data, filename, predictors, method):
    
    df = pd.DataFrame(index=stationnames, columns=predictors)
    
    filename = filename + method
    
    for i,idx in enumerate(stationnames):
        
        df.iloc[i] = load_csv(idx, filename, path_to_data)
    
    df = df.astype(float)
    df.reset_index(drop=True, inplace=True)
    df.index += 1
    
    return df 

def count_predictors(methods, stationnames, path_to_data, filename, predictors,
                     ):
    
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


def seasonal_mean(stationnames, path_to_data, filename, daterange, id_name,
                  method):

    columns = ["DJF", "MAM", "JJA", "SON", "Annum"]
    
    df_stations = pd.DataFrame(index=stationnames, columns=columns)
    
    filename = filename + method
    
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
             
def monthly_mean(stationnames, path_to_data, filename, daterange, id_name,
                 method):
    
    import calendar	
    month_names = [calendar.month_abbr[im+1] for im in np.arange(12)]
    df_stations = pd.DataFrame(index=stationnames, columns=month_names)
    filename = filename + method
    
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
        
    
def prediction_example_data(station_num, stationnames, path_to_data, filename,
                            obs_train_name="obs 1958-2010", 
                            obs_test_name="obs 2011-2020", 
                            val_predict_name="ERA5 1958-2010", 
                            test_predict_name="ERA5 2011-2020",
                            method = "Stacking"
                            ):
    
    stationname = stationnames[station_num]
    print("extracting information for the station: ", stationname)
    
    filename = filename + method
    df = load_csv(stationname, filename, path_to_data)
    
    obs_train = df[obs_train_name].dropna()
    obs_test = df[obs_test_name].dropna()
    ypred_validation = df[val_predict_name][~np.isnan(df[obs_train_name])]
    ypred_test = df[test_predict_name][~np.isnan(df[obs_test_name])]
    obs_full = df["obs anomalies"].dropna()
    
    
    
    validation_score_filename = "validation_score_" + method
    test_score_filename = "test_score_" + method
    
    
    #load scores
    validation_score = load_pickle(stationname, varname=validation_score_filename ,
                                   path=path_to_data)
    test_score = load_pickle(stationname, varname=test_score_filename,
                             path=path_to_data)
    
    station_info = {"obs_train": obs_train,
                    "obs_test": obs_test,
                    "ypred_train": ypred_validation,
                    "ypred_test": ypred_test,
                    "train_score": validation_score, 
                    "test_score": test_score,
                    "obs": obs_full}
    
    return station_info
    

    
def extract_time_series(stationnames, path_to_data, filename, id_name, method,
                        daterange):
    
    df = pd.DataFrame(columns=stationnames)
    
    filename = filename + method
    
    for i,stationname in enumerate(stationnames):
        
        stn_data = load_csv(stationname, filename, path_to_data)
        
        pred_data = stn_data[id_name][daterange].dropna()
        
        df[stationname] = pred_data
        
    # calculation of statistics 
    
    df["mean"] = df.mean(axis=1)
    df["std"] = df.std(axis=1)
    df["max"] = df.max(axis=1)
    df["min"] = df.min(axis=1)
    
    return df
        
        
    
    
    