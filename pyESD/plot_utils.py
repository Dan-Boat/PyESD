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
gridline_color = "#d5dbdb"
indianred = "#ec7063"
seablue = "#5dade2"
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

#specific colors

recursive_color = "#D35400"
treebased_color = "#1abc9c"
sequential_color = "#1C2833"

esd_color = "#212F3D"
mpi_color = "#52BE80"
cesm_color = "#CB4335"
had_color = "#D4AC0D"
cordex_color = "#8E44AD"


RdBu_r = plt.cm.RdBu_r
RdBu = plt.cm.RdBu
BrBG = plt.cm.BrBG
seismic = plt.cm.seismic

selector_method_colors = {
    
    "Recursive": recursive_color,
    "TreeBased": treebased_color,
    "Sequential": sequential_color,
    }


Models_colors = {
    "ESD" : esd_color,
    "MPIESM": mpi_color,
    "CESM5": cesm_color,
    "HadGEM2": had_color,
    "CORDEX": cordex_color}





# colors for individual predictors 


predictor_colors = {
        'r250':green, 'r500':lightgreen, 'r700':green, 'r850':green, 'r1000':green,
        'z250':pink, 'z500':green, 'z700':green, 'z850':green, 'z1000':green,
        't250':tomato, 't500':green, 't700':green, 't850':purple, 't1000':green,
        'v250':green, 'v500':green, 'v700':green, 'v850':green, 'v1000':green,
        'u250':green, 'u500':green, 'u700':green, 'u850':green, 'u1000':green,
        'dt250':green, 'dt500':green, 'dt700':green, 'dt850':lightblue, 'dt1000':green,
        'NAO':magenta,
        'EA':brown,
        'EAWR':brown,
        'SCAN':brown,
        't2m':gold,
        'msl':blue,
        "u10": skyblue
        }

def apply_style(fontsize=20, style=None, linewidth=2, usetex=True):
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
    mpl.rc('text', usetex=usetex)
    mpl.rc('xtick', labelsize=fontsize)
    mpl.rc('ytick', labelsize=fontsize)
    mpl.rc('legend', fontsize=fontsize)
    mpl.rc('axes', labelsize=fontsize)
    mpl.rc('lines', linewidth=linewidth)
    mpl.rc("font", weight="bold")
    

    
    
def barplot_data(methods, stationnames, path_to_data, varname="test_r2", varname_std="test_r2_std", 
                 filename="validation_score_", use_id=False):
    
    df = pd.DataFrame(index=stationnames, columns=methods,)
    df_std = pd.DataFrame(index=stationnames, columns=methods,)
    
    for method in methods:
        scores = load_all_stations(filename + method, path_to_data, stationnames)
        df[method] = scores[varname]
        
        if varname_std is not None:
            df_std[method] = scores[varname_std]
        
    if use_id == True:
        
        df.reset_index(drop=True, inplace=True)
        df.index += 1
        df_std.reset_index(drop=True, inplace=True)
        df_std.index += 1
    
    return df, df_std
    

def correlation_data(stationnames, path_to_data, filename, predictors, use_id=False, use_scipy=False):
    
    df = pd.DataFrame(index=stationnames, columns=predictors)
    
    if use_scipy:
        df_pval = pd.DataFrame(index=stationnames, columns=predictors)
        
    
    
    for i,idx in enumerate(stationnames):
        
        if use_scipy:
            df_corr = load_csv(idx, filename, path_to_data)
            
            df.iloc[i] = df_corr.loc[0,:]
            df_pval.iloc[i] = df_corr.loc[1,:]
            
        else:
            df.iloc[i] = load_csv(idx, filename, path_to_data)
    
    
    if use_scipy:
        df = df.astype(float)
        df_pval = df_pval.astype(float)
        
        if use_id == True:
            df.reset_index(drop=True, inplace=True)
            df.index += 1
            
            df_pval.reset_index(drop=True, inplace=True)
            df_pval.index += 1
        
        return df, df_pval
    else:
        
        df = df.astype(float)
        
        if use_id == True:
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
            
            if isinstance(selected, pd.Index):
                selected = selected.to_frame().T
                
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
    
    Jan = df[df.index.month == 1].mean()
    Feb = df[df.index.month == 2].mean()
    Mar = df[df.index.month == 3].mean()
    Apr = df[df.index.month == 4].mean()
    May = df[df.index.month == 5].mean()
    Jun = df[df.index.month == 6].mean()
    Jul = df[df.index.month == 7].mean()
    Aug = df[df.index.month == 8].mean()
    Sep = df[df.index.month == 9].mean()
    Oct = df[df.index.month == 10].mean()
    Nov = df[df.index.month == 11].mean()
    Dec = df[df.index.month == 12].mean()
    
    
    month_means = [Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec]
    
    return month_means


def seasonal_mean(stationnames, path_to_data, filename, daterange, id_name,
                  method, use_id=False):

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
    
    
    if use_id == True:
        df_stations.reset_index(drop=True, inplace=True)
        df_stations.index +=1
    
    df_stations = df_stations.T.astype(float)
    
    return df_stations
             
def monthly_mean(stationnames, path_to_data, filename, daterange, id_name,
                 method, use_id=False):
    
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
    
    if use_id == True:
        df_stations.reset_index(drop=True, inplace=True)
        df_stations.index +=1
        
        
    df_stations = df_stations.T.astype(float)
    
    return df_stations
        
    
def prediction_example_data(station_num, stationnames, path_to_data, filename,
                            obs_train_name="obs 1958-2010", 
                            obs_test_name="obs 2011-2020", 
                            val_predict_name="ERA5 1958-2010", 
                            test_predict_name="ERA5 2011-2020",
                            method = "Stacking",
                            use_cv_all=False):
    
    stationname = stationnames[station_num]
    print("extracting information for the station: ", stationname)
    
    filename = filename + method
    df = load_csv(stationname, filename, path_to_data)
    
    obs_train = df[obs_train_name].dropna()
    obs_test = df[obs_test_name].dropna()
    ypred_validation = df[val_predict_name][~np.isnan(df[obs_train_name])]
    ypred_test = df[test_predict_name][~np.isnan(df[obs_test_name])]
    obs_full = df["obs anomalies"].dropna()
    
    
    # if use_cv_all == True:
        
    #     scores_df = load_pickle(stationname, varname="CV_scores_" + method, path=path_to_data)
        
            
    #     test_score = scores_df["test_r2"].loc[stationname].max()
        
    # else:
    #     test_score_filename = "test_score_" + method
        
    #     test_score = load_pickle(stationname, varname=test_score_filename,
    #                              path=path_to_data)
        
        
    # load score
    
    # validation_score_filename = "validation_score_" + method
    

    # validation_score = load_pickle(stationname, varname=validation_score_filename ,
    #                                path=path_to_data)
    
    
    station_info = {"obs_train": obs_train,
                    "obs_test": obs_test,
                    "ypred_train": ypred_validation,
                    "ypred_test": ypred_test,
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
        
        
def extract_comparison_data_means(stationnames, path_to_data,
                                  filename, id_name, method, stationloc_dir,
                                  daterange, datasets, variable, dataset_varname,
                                  use_id=True):
    
    models_col_names = ["ESD", "MPIESM", "CESM5", "HadGEM2", "CORDEX"]
    
    df = pd.DataFrame(index=stationnames, columns=models_col_names)
    
    df_stn_loc = pd.read_csv(stationloc_dir, index_col=False, 
                             usecols=["Latitude", "Longitude"])
    
    filename = filename + method
    for i,stationname in enumerate(stationnames):
        stn_data = load_csv(stationname, filename, path_to_data)
        
        df["ESD"].loc[stationname] = stn_data[id_name][daterange].dropna().mean()
        
        lon = df_stn_loc.iloc[i]["Longitude"]
        lat = df_stn_loc.iloc[i]["Latitude"]
        
        for j in range(len(datasets)):
            
            data = datasets[j].get(dataset_varname, is_Dataset=True)
            if hasattr(data, "rlat"):
                df_proj = data.sel(rlat=lat, rlon=lon, method= "nearest").to_series()
            
            else:
                data = data.sortby("lon")
                df_proj = data.sel(lat=lat, lon=lon, method= "nearest").to_series()
            
            df_proj_mean = df_proj[daterange].mean()
            
            if variable == "Temperature":
                df[models_col_names[j+1]].loc[stationname] = df_proj_mean - 273.15
            elif variable == "Precipitation":
                df[models_col_names[j+1]].loc[stationname] = df_proj_mean *60*60*24*30
    
    if use_id == True:
        df.reset_index(drop=True, inplace=True)
        df.index +=1
    
    df = df.astype(float)
    
    return df
            
        
        
        
    
    