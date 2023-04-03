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
import matplotlib as mpl
import seaborn as sns 
from matplotlib.dates import YearLocator
import matplotlib.dates as mdates 
from cycler import cycler


try:  
    from plot_utils import *
    
except:
    from .plot_utils import *

def plot_monthly_mean(means, stds, color, ylabel=None, ax=None, 
                      fig_path=None, fig_name=None, lolims=False):
    
    if ax is None:
        fig,ax = plt.subplots(1,1, sharex=False, figsize=(20, 15))

    plot = means.plot(kind="bar", yerr=stds, rot=0, ax=ax, fontsize=20, capsize=4,
            width=0.8, color=color, edgecolor=black, 
            error_kw=dict(ecolor='black',elinewidth=0.5, lolims=lolims))

    for ch in plot.get_children():
        if str(ch).startswith("Line2D"):
            ch.set_marker("_")
            ch.set_markersize(10)
            break
    
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=20)
        ax.grid(True, linestyle="--", color=gridline_color)
    else:
        ax.grid(True, linestyle="--", color=gridline_color)
        ax.set_yticklabels([])
    
    plt.tight_layout()
    if fig_path is not None:
        plt.savefig(os.path.join(fig_path, fig_name), bbox_inches="tight", format= "svg")

def correlation_heatmap(data, cmap, ax=None, vmax=None, vmin=None, center=0, cbar_ax=None, 
                        add_cbar=True, title=None, label= "Correlation Coefficinet", fig_path=None, fig_name=None,
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
                                                              "drawedges": False,},
                    linewidth=0.5, linecolor="black",)
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
        plt.savefig(os.path.join(fig_path, fig_name), bbox_inches="tight", format= "svg")
        

def barplot(methods, stationnames, path_to_data, ax=None, xlabel=None, ylabel=None, 
            varname="test_r2", varname_std="test_r2_std", filename="validation_score_", legend=True,
            fig_path=None, fig_name=None, show_error=False, width=0.5, rot=0):
    
    if ax is None:
        fig,ax = plt.subplots(1,1, sharex=False, figsize=(18, 15))
        
    df, df_std = barplot_data(methods, stationnames, path_to_data, varname=varname, varname_std=varname_std, 
                     filename=filename, use_id=True)
    
    colors = [selector_method_colors[m] for m in methods]
    mpl.rcParams["axes.prop_cycle"] = cycler("color", colors)
    
    if show_error == True:
        
        df.plot(kind="bar", yerr=df_std, rot=rot, ax=ax, legend = legend, fontsize=20, capsize=4,
                width=width, edgecolor=black)
    else:
        
        df.plot(kind="bar", rot=rot, ax=ax, legend = legend, fontsize=20, width=width, edgecolor=black)
        
        
        
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=20)
        ax.grid(True, linestyle="--", color=gridline_color)
    else:
        ax.grid(True, linestyle="--", color=gridline_color)
        ax.set_yticklabels([])
    
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontweight="bold", fontsize=20)
        ax.grid(True, linestyle="--", color=gridline_color)
    else:
        ax.grid(True, linestyle="--", color=gridline_color)
        ax.set_xticklabels([])
        
        
        
    if legend ==True:    
        ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1), borderaxespad=0., frameon=True, fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.05)
    
    if fig_path is not None:
        plt.savefig(os.path.join(fig_path, fig_name), bbox_inches="tight", format= "svg")
        

def boxplot(regressors, stationnames, path_to_data, ax=None, xlabel=None, ylabel=None, 
            varname="test_r2", filename="validation_score_",
            fig_path=None, fig_name=None, colors=None, patch_artist=False, rot=45):
    
    if ax is None:
        fig,ax = plt.subplots(1,1, sharex=False, figsize=(20, 15))
    
    
    scores = boxplot_data(regressors, stationnames, path_to_data, filename=filename, 
                     varname=varname)
    
    color = { "boxes": black,
              "whiskers": black,
              "medians": red,
              "caps": black,
               }
    
    
    boxplot = scores.plot(kind= "box", rot=rot, ax=ax, fontsize=20, color= color, sym="+b", grid=False,
                widths=0.9, notch=False, patch_artist=patch_artist, return_type="dict")
    
    
    if colors is not None:
        
        for patch, color in zip(boxplot["boxes"], colors):
            patch.set_facecolor(color)
            
            
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=20)
        ax.grid(True, linestyle="--", color=gridline_color)
    else:
        ax.grid(True, linestyle="--", color=gridline_color)
        ax.set_yticklabels([])
    
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontweight="bold", fontsize=20)
        ax.grid(True, linestyle="--", color=gridline_color)
    else:
        ax.grid(True, linestyle="--", color=gridline_color)
        ax.set_xticklabels([])
        
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.05)
    
    if fig_path is not None:
        plt.savefig(os.path.join(fig_path, fig_name), bbox_inches="tight", format= "svg")


def heatmaps(data, cmap, label=None, title=None, vmax=None, vmin=None, center=None, ax=None,
             cbar=True, cbar_ax=None, xlabel=None):
    
    if ax is None:
        fig,ax = plt.subplots(1,1, figsize=(20,15))
        plt.subplots_adjust(left=0.02, right=1-0.02, top=0.94, bottom=0.45, hspace=0.25)
    
    if all(parameter is not None for parameter in [vmax, vmin, center]):
        
        if cbar == False:
            sns.heatmap(data=data, ax=ax, cmap=cmap, vmax=vmax, vmin=vmin, center=center, 
                        square=True, cbar=cbar, linewidth=0.3, linecolor="black")
        else:
            
            if cbar_ax is not None:
                sns.heatmap(data=data, ax=ax, cmap=cmap, vmax=vmax, vmin=vmin, center=center, 
                            square=True, cbar=cbar, cbar_kws={"label":label, 
                                                              "shrink":.80, "extend":"both"},
                            linewidth=0.3, linecolor="black", cbar_ax=cbar_ax)
            else:
                sns.heatmap(data=data, ax=ax, cmap=cmap, vmax=vmax, vmin=vmin, center=center, 
                            square=True, cbar=cbar, cbar_kws={"label":label, 
                                                              "shrink":.80, "extend":"both"},
                            linewidth=0.3, linecolor="black")
    else:
        if cbar == False:
            sns.heatmap(data=data, ax=ax, cmap=cmap, square=True, cbar=cbar, linewidth=0.3,
                        linecolor="black")
        else:
            
            if cbar_ax is not None:
                sns.heatmap(data=data, ax=ax, cmap=cmap, square=True, cbar=cbar,
                            cbar_kws={"label":label,"shrink":.80,"extend":"both"},
                            linewidth=0.3, linecolor="black", cbar_ax=cbar_ax)
            else:
                sns.heatmap(data=data, ax=ax, cmap=cmap, square=True, cbar=cbar,
                            cbar_kws={"label":label,"shrink":.80,"extend":"both"},
                            linewidth=0.3, linecolor="black")
        
    if title is not None:
        ax.set_title(title, fontsize=20, fontweight="bold", loc="left")
    
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontweight="bold", fontsize=20)
        
    else:
        ax.set_xticklabels([])
        
        
        
        
        
def scatterplot(station_num, stationnames, path_to_data, filename, ax=None, 
                obs_train_name="obs 1958-2010", 
                obs_test_name="obs 2011-2020", 
                val_predict_name="ERA5 1958-2010", 
                test_predict_name="ERA5 2011-2020",
                method = "Stacking", ylabel=None, xlabel=None,
                fig_path=None, fig_name=None,
                ):
    
    if ax is None:
        fig,ax = plt.subplots(1,1, figsize=(20,15))
        plt.subplots_adjust(left=0.02, right=1-0.02, top=0.94, bottom=0.45, hspace=0.25)
        
    station_info = prediction_example_data(station_num, stationnames, path_to_data, filename,
                                           obs_test_name=obs_test_name, obs_train_name=obs_train_name,
                                           val_predict_name=val_predict_name, test_predict_name=test_predict_name,
                                           method=method)
    
    obs_train = station_info["obs_train"]
    obs_test = station_info["obs_test"]
    ypred_train = station_info["ypred_train"]
    ypred_test = station_info["ypred_test"]
    obs  = station_info["obs"]
    
    from scipy import stats
    
    regression_stats  = stats.linregress(obs_test, ypred_test)
    
    regression_slope = regression_stats.slope * obs + regression_stats.intercept
    
    r2 = regression_stats.rvalue 
    
    ax.scatter(obs_train, ypred_train, alpha=0.3, c=black, s=100, label=val_predict_name)
    
    ax.scatter(obs_test, ypred_test, alpha=0.3, c=red, s=100, label=test_predict_name)
    
    ax.plot(obs, regression_slope, color=red, label="PCC = {:.2f}".format(r2))
    
    ax.legend(loc= "upper left", fontsize=20)
    
    # Plot design 
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines["left"].set_position(("outward", 20))
    ax.spines["bottom"].set_position(("outward", 20))
    
    
    
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=20)
        ax.grid(True, linestyle="--", color=gridline_color)
    else:
        ax.grid(True, linestyle="--", color=gridline_color)
        ax.set_yticklabels([])
    
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontweight="bold", fontsize=20)
        ax.grid(True, linestyle="--", color=gridline_color)
    else:
        ax.grid(True, linestyle="--", color=gridline_color)
        ax.set_xticklabels([])
        
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.05)
    
    if fig_path is not None:
        plt.savefig(os.path.join(fig_path, fig_name), bbox_inches="tight", format= "svg")
        
        
        

def lineplot(station_num, stationnames, path_to_data, filename, ax=None, fig=None,
                obs_train_name="obs 1958-2010", 
                obs_test_name="obs 2011-2020", 
                val_predict_name="ERA5 1958-2010", 
                test_predict_name="ERA5 2011-2020",
                method = "Stacking", ylabel=None, xlabel=None,
                fig_path=None, fig_name=None,
                ):
    
    
    
    if ax is None:
       fig, ax = plt.subplots(1, 1, figsize= (20, 15), sharex=True)
    
    plt.subplots_adjust(left=0.12, right=1-0.01, top=0.98, bottom=0.06, hspace=0.01)
        
    station_info = prediction_example_data(station_num, stationnames, path_to_data, filename,
                                           obs_test_name=obs_test_name, obs_train_name=obs_train_name,
                                           val_predict_name=val_predict_name, test_predict_name=test_predict_name,
                                           method=method)
    

    ypred_train = station_info["ypred_train"].rolling(3, min_periods=1, win_type="hann",
                                                  center=True).mean()
    ypred_test = station_info["ypred_test"].rolling(3, min_periods=1, win_type="hann",
                                                  center=True).mean()
    obs  = station_info["obs"].rolling(3, min_periods=1, win_type="hann",
                                                  center=True).mean()
    

    
    ax.plot(obs, linestyle="-", color=green, label="Obs")
    ax.plot(ypred_train, linestyle="-.", color= blue, label = val_predict_name)
    ax.plot(ypred_test, linestyle="--", color=red, label=test_predict_name)
    
    
    ax.xaxis.set_major_locator(YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.axhline(y=0, linestyle="--", color=grey, linewidth=2)
    ax.legend(bbox_to_anchor=(0.01, 1.02, 1., 0.102), loc=3, ncol=3, borderaxespad=0., frameon = True, 
              fontsize=20)
    
    
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=20)
        ax.grid(True, linestyle="--", color=gridline_color)
    else:
        ax.grid(True, linestyle="--", color=gridline_color)
        ax.set_yticklabels([])
    
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontweight="bold", fontsize=20)
        ax.grid(True, linestyle="--", color=gridline_color)
    else:
        ax.grid(True, linestyle="--", color=gridline_color)
        ax.set_xticklabels([])
    
        
    plt.tight_layout()
   
    if fig_path is not None:
        plt.savefig(os.path.join(fig_path, fig_name), bbox_inches="tight", format= "svg")
    
    
def plot_time_series(stationnames, path_to_data, filename, id_name,
                        daterange, color, label, ymax=None, ymin=None, ax=None, ylabel=None, xlabel=None,
                        fig_path=None, fig_name=None, method="Stacking", window=12):
    
    if ax is None:
       fig, ax = plt.subplots(1, 1, figsize= (20, 15), sharex=True)
       
    df = extract_time_series(stationnames, path_to_data, filename, id_name,
                             method, daterange,)
    
    
    df = df.rolling(window, min_periods=1, win_type="hann", center=True).mean()
    
    ax.plot(df["mean"], "--", color=color, label=label)
    
    
    #try with max and min to notice the difference with 5 years window
    
    ax.fill_between(df.index, df["mean"] - df["std"], df["mean"] + df["std"], color=color, 
                    alpha=0.2,)
    
    ax.xaxis.set_major_locator(YearLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.axhline(y=0, linestyle="--", color=grey, linewidth=2)
    
    ax.legend(frameon=True, fontsize=12, loc="lower left")
    if ymax is not None:
        
        ax.set_ylim([ymin, ymax])
        
        
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=20)
        ax.grid(True, linestyle="--", color=gridline_color)
    else:
        ax.grid(True, linestyle="--", color=gridline_color)
        ax.set_yticklabels([])
    
    # if xlabel is not None:
    #     ax.set_xlabel(xlabel, fontweight="bold", fontsize=20)
    #     ax.grid(True)
    # else:
    #     ax.grid(True)
    #     ax.set_xticklabels([])
    
        
    plt.tight_layout()
   
    if fig_path is not None:
        plt.savefig(os.path.join(fig_path, fig_name), bbox_inches="tight", format= "svg")

    

def plot_projection_comparison(stationnames, path_to_data,
                                  filename, id_name, method, stationloc_dir,
                                  daterange, datasets, variable, dataset_varname,
                                  ax=None, xlabel=None, ylabel=None, legend=True,
                                  figpath=None, figname=None, width=0.5, title=None,
                                  vmax=None, vmin=None, use_id=True):
    
   df = extract_comparison_data_means(stationnames, path_to_data, filename, id_name, 
                                      method, stationloc_dir, daterange, datasets, 
                                      variable, dataset_varname, use_id=use_id) 
   
   models_col_names = ["ESD", "MPIESM", "CESM5", "HadGEM2", "CORDEX"]
   
   
   if ax is None:
        fig,ax = plt.subplots(1,1, sharex=False, figsize=(18, 15))  
        
   colors = [Models_colors[c] for c in models_col_names] 
   mpl.rcParams["axes.prop_cycle"] = cycler("color", colors)
   
   if use_id:
       df.plot(kind="bar", rot=0, ax=ax, legend=legend, fontsize=20, width=width)
   else:
       df.plot(kind="bar", rot=45, ax=ax, legend=legend, fontsize=20, width=width)
   
   if vmax is not None:
       ax.set_ylim(vmin, vmax)
   if ylabel is not None:
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=20)
        ax.grid(True, linestyle="--", color=gridline_color)
   else:
        ax.grid(True, linestyle="--", color=gridline_color)
        ax.set_yticklabels([])
    
   if xlabel is not None:
        ax.set_xlabel(xlabel, fontweight="bold", fontsize=20)
        ax.grid(True, linestyle="--", color=gridline_color)
   else:
        ax.grid(True, linestyle="--", color=gridline_color)
        ax.set_xticklabels([])
        
   if title is not None:
       ax.set_title(title, fontsize=20, fontweight="bold", loc="left") 
        
        
        
   if legend ==True:    
        ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1), borderaxespad=0., frameon=True, fontsize=20)
   plt.tight_layout()
   plt.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.05)
   
   