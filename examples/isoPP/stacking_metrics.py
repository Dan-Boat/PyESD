# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 17:45:31 2024

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
from pyClimat.plot_utils import red


main_path = "D:/Datasets/Model_output_pst/PD"
station_datadir = "C:/Users/dboateng/Desktop/Datasets/Station/GNIP"
path_to_data = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/isoPP/final_exp"
path_to_save = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/isoPP/plots"




def calculate_regional_means(ds, lon_target, lat_target, radius_deg,):
    """
    Calculate regional means around a specific longitude and latitude location
    with a given radius for a NetCDF dataset using xarray.
    """
    # Find indices of the nearest grid point to the target location
    
    if hasattr(ds, "longitude"):
        ds = ds.rename({"longitude":"lon", "latitude":"lat"})
        
    ds = ds.assign_coords({"lon": (((ds.lon + 180) % 360) - 180)})
    
    indices = extract_indices_around(ds, lat_target, lon_target, radius_deg)
    
    regional_mean = ds.isel(lat=indices[0], lon=indices[1]).mean(dim=("lon", "lat")).data
        
    return np.float64(regional_mean)


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
    
       
    
    print("extracting information for the station: ", station)
    
    filename = "predictions_" + "Stacking"
    
    df = load_csv(stationname, filename, path_to_data)
    
    obs = df["obs"]
    ypred_train = df["ERA5 1979-2012"].rolling(3, min_periods=1, win_type="hann", center=True).mean()
    ypred_test = df["ERA5 2013-2018"].rolling(3, min_periods=1, win_type="hann", center=True).mean()
    
    
    model = model.rolling(3, min_periods=1, win_type="hann", center=True).mean()
    obs = obs.rolling(3, min_periods=1, win_type="hann", center=True).mean()
    
    
    # regression_stats  = stats.linregress(obs_test, ypred_test)
    
    # regression_slope = regression_stats.slope * obs + regression_stats.intercept
    
    # r2 = regression_stats.rvalue
    
    apply_style(fontsize=22, style=None, linewidth=3, usetex=True)
    fig, ax1 = plt.subplots(1, 1, figsize= (12, 4), sharex=True)
    
    plt.subplots_adjust(left=0.12, right=1-0.01, top=0.98, bottom=0.06, hspace=0.01)
    
    ax1.plot(obs, linestyle="-", color=black, label=station.capitalize() + "," + str(num) + " (GNIP)")
    ax1.plot(model, linestyle="-", color=red, label="ECHAM5-wiso")
    ax1.plot(ypred_train, linestyle="--", color=grey, label="ERA5 1979-2012 (stacking)")
    ax1.plot(ypred_test, linestyle="--", color="#06c4be", label="ERA5 2013-2018 (stacking)")
    
    
    ax1.xaxis.set_major_locator(YearLocator(5))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.legend(bbox_to_anchor=(0.01, 1.02, 1., 0.102), loc=3, ncol=2, borderaxespad=0., frameon = True, 
              fontsize=20)
    
    ax1.set_ylabel("$\delta^{18}$Op vs SMOW", fontweight="bold", fontsize=20)
    ax1.grid(True, linestyle="--", color=gridline_color)
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(path_to_save, station + "_timeseries.pdf"), bbox_inches="tight", format = "pdf", dpi=600)

def plot_prediction_example(axes, stationnames):
    
    for i,station in enumerate(stationnames):
        
    
        print("----plotting for the station:", station + "precipitation")
        
        
        station = station.replace("_", " ")
        scatterplot(station_num=i, stationnames=stationnames, path_to_data=path_to_data, 
                    filename="predictions_", ax=axes[i], xlabel="observed (GNIP)", ylabel="predicted (Stacking)",
                    method= "Stacking", obs_train_name="obs 1979-2012", 
                    obs_test_name="obs 2013-2018", 
                    val_predict_name="ERA5 1979-2012", 
                    test_predict_name="ERA5 2013-2018", obs_full_name="obs",
                    test_color="#06c4be", show_mae=True)
        
        
        axes[i].set_title(station.capitalize(), fontsize=20, weight="bold", loc="left")    
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.05)
    #plt.savefig(os.path.join(path_to_plot, "prediction_examp_" + station + "_.png"), bbox_inches="tight", dpi=300)
        
def plot_scatter_stations():        
    stationnames = ['HOHENPEISSENBERG', 'WUERZBURG', 'GENOA_SESTRI']
    
    apply_style(fontsize=24, style="seaborn-talk", linewidth=3,)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12, 18),)
    axes = [ax1, ax2, ax3]
    
    plot_prediction_example(stationnames=stationnames, axes=axes)
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_save, "Prediction_examples.pdf"), bbox_inches="tight", dpi=600, format="pdf")



def plot_stacking_cv():
    regressors = ["Stacking"]
    
    colors = ["#168704"]
    
    mae_df = boxplot_data(regressors=regressors, stationnames=stationnames,
                          path_to_data=path_to_data, filename="validation_score_", 
                          varname="test_mae")
    
    
    apply_style(fontsize=22, style=None, linewidth=3)
    
    fig, ax = plt.subplots(1,1, figsize=(5,12)) 
    
    # sns.violinplot(x=' ', y='', data=df, palette='pastel', width=0.7, alpha=0.5, inner_kws=dict(box_width=18, whis_width=2),)
    vio = sns.violinplot(data=mae_df, dodge=False,
                        palette=colors, alpha=0.7,
                        width=0.7, inner_kws=dict(box_width=14, whis_width=2, color="black"),
                        bw_adjust=0.67, ax=ax)
    
    
    ax.tick_params(axis='x', rotation=45)
    
    # Adjust the width of the violin plot to show only half
    # for violin in vio.collections:
    #     bbox = violin.get_paths()[0].get_extents()
    #     x0, y0, width, height = bbox.bounds
    #     violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=vio.transData))
    
    # Adjust the width of the violin plot to show only half
    
    for violin in vio.collections:
        paths = violin.get_paths()
        for path in paths:
            vertices = path.vertices
            vertices[:, 0] = np.clip(vertices[:, 0], min(vertices[:, 0]), np.mean(vertices[:, 0]))
            path.vertices = vertices
    
    
    
    
    
            
            
    old_len_collections = len(vio.collections)
    
    sns.stripplot(data=mae_df, palette=colors, dodge=False, ax=ax, size=10, linewidth=1.5)
    
    
    # Adjust the position of the stripplot
    for dots in vio.collections[old_len_collections:]:
        dots.set_offsets(dots.get_offsets() + np.array([0.17, 0]))
    
    
    ax.set_ylabel("CV MAE", fontweight="bold", fontsize=20)
    #ax.set_ylim([ymin, ymax])
    ax.grid(True, linestyle="--")
    
    # ax.legend_.remove()
    plt.grid(True, which='major', axis='y')
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.05)
    plt.savefig(os.path.join(path_to_save, "stacking_cv_mae.pdf"), bbox_inches="tight", dpi=300, format="pdf")

plot_scatter_stations()

#plot_line_example(num=8)
#plot_line_example(num=14)
#plot_line_example(num=39)
#plot_stacking_cv()