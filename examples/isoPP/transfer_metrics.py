# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:43:47 2024

@author: dboateng
"""

import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib.dates import YearLocator
import matplotlib.dates as mdates 


from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats


from pyClimat.plot_utils import *
from pyClimat.plots import plot_annual_mean 
from pyClimat.data import read_ERA_processed, read_ECHAM_processed, read_from_path
from pyClimat.analysis import compute_lterm_mean, compute_lterm_diff
from pyClimat.variables import extract_var
from pyESD.ESD_utils import load_csv, haversine, extract_indices_around

from pyESD.plot_utils import apply_style, correlation_data, count_predictors
from pyESD.plot import scatterplot


from read_data import *
from predictor_setting import *

from pyESD.plot import boxplot
from pyESD.plot_utils import *

from Fig_utils import calculate_regional_means, get_metrics, read_regional_means_from_isoGCM

main_path = "D:/Datasets/Model_output_pst/PD"
station_datadir = "C:/Users/dboateng/Desktop/Datasets/Station/GNIP"
#path_to_data = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/isoPP/final_exp"
path_to_transfer = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/isoPP/final_exp_transfer"
path_to_save = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/isoPP/plots"



def plot_line_example(num):
    
    from1980to2014 = pd.date_range(start="1979-01-01", end="2014-12-31", freq="MS")
    
    # load datasets 
    PD_data = read_from_path(main_path, "PD_1980_2014_monthly.nc", decode=True)
    PD_wiso = read_from_path(main_path, "PD_1980_2014_monthly_wiso.nc", decode=True)
    d18Op = extract_var(Dataset=PD_data, varname="d18op", units="per mil", Dataset_wiso=PD_wiso,
                        )
    
    
    stationname = stationnames[num-1]
    station = stationname.replace("_", " ")
    print(stationname)
    
    station_info = pd.read_csv(os.path.join(station_datadir, "stationnames_new.csv"), index_col=0)
    lon = station_info.loc[num]["Longitude"]
    lat = station_info.loc[num]["Latitude"]
    
    model = pd.DataFrame(columns=["echam"], index=from1980to2014)
    
    model["echam"] = calculate_regional_means(ds=d18Op, lon_target=lon, lat_target=lat, radius_deg=50)
    
    model_nudged = read_regional_means_from_isoGCM(lon, lat)
    
    print("extracting information for the station: ", stationname)
    
    filename = "predictions_" + "Stacking"
    
    df = load_csv(stationname, filename, path_to_transfer)
    
    obs = df["obs"]
    
    # estimate metrics
    #y_pred = model.loc[~np.isnan(obs)]
    
    
    common_idx = obs.index.intersection(model_nudged.index)
    obs_common = obs.loc[common_idx]
    model_nudged = model_nudged.loc[common_idx]
    
    y_pred_nudged = model_nudged.loc[~np.isnan(obs_common)]
    
    y_pred_stacking = df["ERA5 2013-2018"].dropna()[~np.isnan(obs)]
    y_pred_train_stacking = df["ERA5 1979-2012"].dropna()[~np.isnan(obs)]
    
        
    y_true = obs.dropna()
    
    #mae_echam5, r2_echam5 = get_metrics(y_true, y_pred)
    
    mae_echam6, r2_echam6 = get_metrics(y_true, y_pred_nudged["echam"])
    mae_stack, r2_stack = get_metrics(y_true, y_pred_stacking)
    mae_train_stack, r2_train_stack = get_metrics(y_true, y_pred_train_stacking)
    
    
    #model = model.rolling(3, min_periods=1, win_type="hann", center=True).mean()
    obs = obs.rolling(3, min_periods=1, win_type="hann", center=True).mean()
    
    model_nudged = model_nudged.rolling(3, min_periods=1, win_type="hann", center=True).mean()
    
    
    ypred_train = df["ERA5 1979-2012"].rolling(3, min_periods=1, win_type="hann", center=True).mean()
    ypred_test = df["ERA5 2013-2018"].rolling(3, min_periods=1, win_type="hann", center=True).mean()
    
    
    apply_style(fontsize=22, style=None, linewidth=3, usetex=True)
    fig, ax = plt.subplots(1, 1, figsize= (15, 6), sharex=True)
    
    plt.subplots_adjust(left=0.12, right=1-0.01, top=0.98, bottom=0.06, hspace=0.01)
    
    ax.plot(obs, linestyle="-", color=black, label=station.capitalize() + "," + str(num) + " (GNIP)")
    ax.plot(model, linestyle="-", color=red, 
            label="ECHAM5-wiso[MAE={:.2f} ‰, R²={:.2f}]".format(mae_echam5, r2_echam5))
    
    ax.plot(model_nudged, color="blue", linestyle="-", 
            label="ECHAM6-wiso(JRA55)[MAE={:.2f} ‰]".format(mae_echam6))
    
    ax.plot(ypred_train, linestyle="--", color=grey, 
            label="ERA5 1979-2012 (stacking-37*)[MAE={:.2f} ‰]".format(mae_train_stack))
    ax.plot(ypred_test, linestyle="--", color="#06c4be", 
            label="ERA5 2013-2018 (stacking-37*)[MAE={:.2f} ‰]".format(mae_stack))
    
    ax.set_ylabel('$\delta^{18}$Op VSMOW [‰]', fontweight="bold", fontsize=20)
    ax.xaxis.set_major_locator(YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.legend(bbox_to_anchor=(0.01, 1.02, 1., 0.102), loc=3, ncol=2, borderaxespad=0., frameon = True, 
              fontsize=18)
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(path_to_save, stationname + "_transfer_fig5.pdf"), bbox_inches="tight", format = "pdf", dpi=300)

transfer_stations = [36, 8, 21, 10, 12, 35]

#transfer_stations = [27, 35] # comment out the predict for testing period

for num in transfer_stations:
    
    plot_line_example(num=num)