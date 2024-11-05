# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:27:27 2024

@author: dboateng
"""

# Import modules 
import os 
import pandas as pd 
import numpy as np

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator
import matplotlib.dates as mdates 

from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
import seaborn as sns


from pyClimat.plot_utils import *
from pyClimat.plots import plot_annual_mean 
from pyClimat.data import read_from_path
from pyClimat.analysis import compute_lterm_mean
from pyClimat.variables import extract_var

from pyESD.plot_utils import apply_style, boxplot_data
from pyESD.plot import scatterplot
from pyESD.plot_utils import seasonal_mean

#from pyESD.plot_utils import *
from read_data import *
from predictor_setting import *
from pyESD.ESD_utils import load_csv, haversine, extract_indices_around


from Fig_utils import calculate_regional_means, get_metrics, read_regional_means_from_isoGCM

# Define paths 
path_to_save = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/isoPP/plots"
main_path = "D:/Datasets/Model_output_pst/PD"
path_to_nudging = "E:/Datasets/Nudged_isotopes/"
station_datadir = "C:/Users/dboateng/Desktop/Datasets/Station/GNIP"
station_data = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/isoPP/final_exp"






def plot_time_series(station_number):
    
    from1980to2014 = pd.date_range(start="1979-01-01", end="2014-12-31", freq="MS")
    
    # load datasets 
    PD_data = read_from_path(main_path, "PD_1980_2014_monthly.nc", decode=True)
    PD_wiso = read_from_path(main_path, "PD_1980_2014_monthly_wiso.nc", decode=True)
    d18Op = extract_var(Dataset=PD_data, varname="d18op", units="per mil", Dataset_wiso=PD_wiso,
                        )
    
    
    
    stationname = stationnames[station_number-1].replace("_", " ")
    print(stationname)
    
    station_info = pd.read_csv(os.path.join(station_datadir, "stationnames_new.csv"), index_col=0)
    
    lon = station_info.loc[station_number]["Longitude"]
    lat = station_info.loc[station_number]["Latitude"]
    
    #model = pd.DataFrame(columns=["echam"], index=from1980to2014)
    
    #model["echam"] = calculate_regional_means(ds=d18Op, lon_target=lon, lat_target=lat, radius_deg=50)
    
    model_nudged = read_regional_means_from_isoGCM(lon, lat)
    
   
    
    print("extracting information for the station: ", stationname)
    
    filename = "predictions_" + "Stacking"
    
    df = load_csv(stationname, filename, station_data)
    
    obs = df["obs"]
    
    # estimate metrics
    #y_pred = model.loc[~np.isnan(obs)]
    
    
    common_idx = obs.index.intersection(model_nudged.index)
    obs_common = obs.loc[common_idx]
    model_nudged = model_nudged.loc[common_idx]
    
    y_pred_nudged = model_nudged.loc[~np.isnan(obs_common)]
        
    y_true = obs.dropna()
    
    #mae_echam5, r2_echam5 = get_metrics(y_true, y_pred)
    
    mae_echam6, r2_echam6 = get_metrics(y_true, y_pred_nudged)
    
    
    #model = model.rolling(3, min_periods=1, win_type="hann", center=True).mean()
    obs = obs.rolling(3, min_periods=1, win_type="hann", center=True).mean()
    
    model_nudged = model_nudged.rolling(3, min_periods=1, win_type="hann", center=True).mean()
    
    
    ypred_train = df["ERA5 1979-2012"].rolling(3, min_periods=1, win_type="hann", center=True).mean()
    ypred_test = df["ERA5 2013-2018"].rolling(3, min_periods=1, win_type="hann", center=True).mean()
    
    
    apply_style(fontsize=22, style=None, linewidth=3, usetex=True)
    fig, ax = plt.subplots(1, 1, figsize= (15, 6), sharex=True)
    
    plt.subplots_adjust(left=0.12, right=1-0.01, top=0.98, bottom=0.06, hspace=0.01)
    
    ax.plot(obs, linestyle="-", color=black, label=stationname.capitalize() + "," + str(station_number) + " (GNIP)")
    #ax.plot(model, linestyle="-", color=red, 
            #label="ECHAM5-wiso[MAE={:.2f} ‰, R²={:.2f}]".format(mae_echam5, r2_echam5))
    ax.plot(model_nudged, color="blue", linestyle="-", 
            label="ECHAM6-wiso(JRA55)[MAE={:.2f} ‰, R²={:.2f}]".format(mae_echam6, r2_echam6))
    
    ax.plot(ypred_train, linestyle="--", color=grey, label="ERA5 1979-2012 (stacking)")
    ax.plot(ypred_test, linestyle="--", color="#06c4be", label="ERA5 2013-2018 (stacking)")
    
    ax.set_ylabel('$\delta^{18}$Op VSMOW [‰]', fontweight="bold", fontsize=20)
    ax.xaxis.set_major_locator(YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.legend(bbox_to_anchor=(0.01, 1.02, 1., 0.102), loc=3, ncol=2, borderaxespad=0., frameon = True, 
              fontsize=18)
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(path_to_save, stationname + "_final_inter_annual_fig4.pdf"), bbox_inches="tight", format = "pdf", dpi=300)


def plot_prediction_example(axes, stationnames):
    
    for i,station in enumerate(stationnames):
        
    
        print("----plotting for the station:", station + "precipitation")
        
        
        station = station.replace("_", " ")
        scatterplot(station_num=i, stationnames=stationnames, path_to_data=station_data, 
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
    
    apply_style(fontsize=25, style="seaborn-talk", linewidth=3,)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12, 18),)
    axes = [ax1, ax2, ax3]
    
    plot_prediction_example(stationnames=stationnames, axes=axes)
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_save, "Prediction_examples.pdf"), bbox_inches="tight", dpi=600, format="pdf")

        
def get_mae_from_all():
    
    station_info = pd.read_csv(os.path.join(station_datadir, "stationnames_new.csv"), index_col=0)
    
    df_mae_models = pd.DataFrame(index=stationnames, 
                                 columns=["Stacking","ECHAM6-wiso[JRA55]", "MIROC[JRA55]", "IsoGCM[JRA55]", "IsoGCM[ERA5]"],)
    
    
    
    for num, station in enumerate(stationnames):
       
        print(station)
        stationname = station.replace("_", " ")
        
        filename = "predictions_" + "Stacking"
        
        lon = station_info.loc[num+1]["Longitude"]
        lat = station_info.loc[num+1]["Latitude"]
        
        
        df = load_csv(stationname, filename, station_data)
        
        obs = df["obs 2013-2018"]
        
        model = read_regional_means_from_isoGCM(lon, lat, extract_all=True)
        
        if len(model.loc[~np.isnan(obs)]) ==0:
            df_mae_models["ECHAM6-wiso[JRA55]"].loc[station] = np.nan
            df_mae_models["MIROC[JRA55]"].loc[station] = np.nan
            df_mae_models["IsoGCM[JRA55]"].loc[station] = np.nan
            df_mae_models["IsoGCM[ERA5]"].loc[station] = np.nan
            df_mae_models["Stacking"].loc[station] = np.nan
            
        else:
            
        
            y_pred_models = model.loc[~np.isnan(obs)]
            
            
            common_idx = obs.index.intersection(model.index)
            obs_common = obs.loc[common_idx]
            model = model.loc[common_idx]
            
            y_pred_models = model.loc[~np.isnan(obs_common)]
            
            y_pred_models = y_pred_models.dropna()
            
            y_pred_stacking = df["ERA5 2013-2018"][~np.isnan(obs)]
                
            y_true = obs.dropna()
            
            
            mae_echam, r2_echam = get_metrics(y_true, y_pred_models["echam"])
            
            mae_miroc, r2_miroc = get_metrics(y_true, y_pred_models["miroc"])
            
            mae_isoGCM, r2_isoGCM = get_metrics(y_true, y_pred_models["isoGCM"])
            
            mae_isoGCM_era5, r2_isoGCM_era5 = get_metrics(y_true, y_pred_models["isoGCM_era5"])
            
            mae_stack, r2_stack = get_metrics(y_true, y_pred_stacking)
            
            
            df_mae_models["ECHAM6-wiso[JRA55]"].loc[station] = mae_echam
            
            df_mae_models["MIROC[JRA55]"].loc[station] = mae_miroc
            df_mae_models["IsoGCM[JRA55]"].loc[station] = mae_isoGCM
            df_mae_models["IsoGCM[ERA5]"].loc[station] = mae_isoGCM_era5
            df_mae_models["Stacking"].loc[station] = mae_stack
            
            
    
    
    df_mae_models = df_mae_models.reset_index(drop=True)
    df_mae_models.index = df_mae_models.index + 1
    
    station_info["ECHAM6-wiso[JRA55]"] = df_mae_models["ECHAM6-wiso[JRA55]"].astype("float")
    station_info["MIROC[JRA55]"] = df_mae_models["MIROC[JRA55]"].astype("float")
    station_info["IsoGCM[JRA55]"] = df_mae_models["IsoGCM[JRA55]"].astype("float")
    station_info["IsoGCM[ERA5]"] = df_mae_models["IsoGCM[ERA5]"].astype("float")
    station_info["Stacking"] = df_mae_models["Stacking"].astype("float")
    
    
    
    
    
    sources = ["Stacking", "ECHAM6-wiso[JRA55]", "MIROC[JRA55]", "IsoGCM[JRA55]", "IsoGCM[ERA5]"]
    colors = ["#168704", "blue", "#008080", "#FF7F50", "#bb8fce"]
    
    df = station_info[sources]
    apply_style(fontsize=25, style=None, linewidth=2)

    fig, ax = plt.subplots(1,1, figsize=(12,10)) 
    
    
    vio = sns.violinplot(data=df, dodge=False,
                        palette=colors, alpha=0.8,
                        width=0.7, inner_kws=dict(box_width=20, whis_width=2, color="black"),
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
    
    sns.stripplot(data=df, palette=colors, dodge=False, ax=ax, size=10, linewidth=1.5)
    
    
    # Adjust the position of the stripplot
    for dots in vio.collections[old_len_collections:]:
        dots.set_offsets(dots.get_offsets() + np.array([0.17, 0]))
    
    
    ax.set_ylabel("MAE", fontweight="bold", fontsize=20)
    ax.grid(True, linestyle="--")
    
    # ax.legend_.remove()
    plt.grid(True, which='major', axis='y')
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.05)
    plt.savefig(os.path.join(path_to_save, "test_mae_stacking_wiso_fig4.pdf"), bbox_inches="tight", dpi=300, format="pdf")
#plot_time_series(station_number=8)
#plot_scatter_stations()


get_mae_from_all()