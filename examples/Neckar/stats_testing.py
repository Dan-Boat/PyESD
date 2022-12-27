# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 12:48:43 2022

@author: dboateng
"""
import os 
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as colors


from read_data import *
from predictor_settings import *

from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator)




def extract_region(data, datarange, varname, minlat, maxlat, minlon, maxlon):
        
    data = data.get(varname).sel(time=datarange)
    
    if hasattr(data, "longitude"):
        data = data.rename({"longitude":"lon", "latitude":"lat"})
    
    #data = data.assign_coords({"lon": (((data.lon + 180) % 360) - 180)})
    
    data = data.where((data.lat >=minlat) & (data.lat <=maxlat), drop=True)
    data = data.where((data.lon >=minlon) & (data.lon <= maxlon), drop=True)
    
    return data

def _get_month(npdatetime64):
    
     """
     Returns the month for a given npdatetime64 object, 1 for January, 2 for
     February, ...
     """
     month =  npdatetime64.astype('datetime64[M]').astype(int) % 12 + 1
     
     return month
 
    
def plot_background(p, domain=None,ax=None, left_labels=True,
                    bottom_labels=True, plot_coastlines=True, plot_borders=False):
    """
    This funtion defines the plotting domain and also specifies the background. It requires 
    the plot handle from xarray.plot.imshow and other optional arguments 
    Parameters
    -------------
    p: TYPE: plot handle 
    DESCRIPTION: the plot handle after plotting with xarray.plot.imshow
    
    domian = TYPE:str 
    DESCRIPTION: defines the domain size, eg. "Europe", "Asia", "Africa"
                  "South America", "Alaska", "Tibet Plateau" or "Himalaya", "Eurosia",
                  "New Zealand", default: global
    """
    p.axes.set_global()                    # setting global axis 
    
    if plot_coastlines ==True:
        p.axes.coastlines(resolution = "50m")  # add coastlines outlines to the current axis
    
    if plot_borders == True:
        p.axes.add_feature(cfeature.BORDERS, edgecolor="black", linewidth = 0.3) #adding country boarder lines
    
    #setting domain size
    if domain is not None: 
        if domain == "Europe":   # Europe
            minLon = -15
            maxLon = 40
            minLat = 35
            maxLat = 65
        
    
        elif domain == "NH": # Northen Hemisphere
            minLon = -80
            maxLon = 60
            minLat = 20
            maxLat = 80
            
        elif domain == "West Africa":
            minLon = -25
            maxLon = 40
            minLat = -5
            maxLat = 35
            
        else:
            print("ERROR: invalid geographical domain passed in options")
        p.axes.set_extent([minLon, maxLon, minLat, maxLat], ccrs.PlateCarree())
    if domain is None: 
        p.axes.set_extent([-180, 180, -90, 90], ccrs.PlateCarree())
        
    

        
    # adding gridlines    
    gl= p.axes.gridlines(crs = ccrs.PlateCarree(), draw_labels = True, linewidth = 1,
                 edgecolor = "gray", linestyle = "--", color="gray", alpha=0.5)
    
    gl.top_labels = False                  # labesl at top
    gl.right_labels = False
    
    if left_labels == True:
        gl.left_labels = True
    else:
        gl.left_labels = False
    
    if bottom_labels == True:
        gl.bottom_labels =True
    else:
        gl.bottom_labels = False
        
    gl.xformatter = LongitudeFormatter()     # axis formatter
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {"fontsize": 20, "color": "black", "fontweight": "semibold"}   #axis style 
    gl.ylabel_style = {"fontsize": 20, "color": "black", "fontweight": "semibold"}
    
    
    
# stats testing
def StackArray(x,dim):
	'''return stacked array with only one dimension left
	INPUTS:
	   x  : xarray.DataArray or Dataset to be stacked
	   dim: sole dimension to remain after stacking
	OUTPUTS:
	   stacked: same as x, but stacked
	'''
	dims = []
	for d in x.dims:
		if d != dim:
			dims.append(d)
	return x.stack(stacked=dims)


def ComputeStat(i,sx,y,sy,test, return_score=True):
	'''This is part of StatTest, but for parmap.map to work, it has to be an
		independent function.
	'''
	from xarray import DataArray
	from scipy import stats
	# if sx and sy are the same, null hypothesis of the two being distinct
	#  should be rejected at all levels, i.e. p-value of them being
	#  equal should be 100%
	if isinstance(y,DataArray) and sx.shape == sy.shape and (np.all(sx.isel(stacked=i) == sy.isel(stacked=i))):
		ploc = 1.0
	else:
		if test == 'KS':
			sloc,ploc = stats.ks_2samp(sx.isel(stacked=i),sy.isel(stacked=i))
		elif test == 'MW':
			sloc,ploc = stats.mannwhitneyu(sx.isel(stacked=i),sy.isel(stacked=i),alternative='two-sided')
		elif test == 'WC':
			sloc,ploc = stats.wilcoxon(sx.isel(stacked=i),sy.isel(stacked=i))
		elif test == 'T':
			sloc,ploc = stats.ttest_1samp(sx.isel(stacked=i),y)
		elif test == 'sign': # not really a sig test, just checking sign agreement
                        # note that here a high p-value means significant, as it means
                        #  that a lot of members have the same sign
			lenx = len(sx.isel(stacked=i))
			if y is None: # check sx for same sign
				posx = np.sum(sx.isel(stacked=i) > 0)
				ploc = max(posx,lenx-posx)/lenx
			elif isinstance(y,float) or isinstance(y,int): # check sx for sign of y
				samex = np.sum( np.sign(sx.isel(stacked=i)) == np.sign(y) )
				ploc  = samex/lenx
			else: # check two ensembles for same sign
				# ensembles are not 1-by-1, so we can't check sign along dimension
				lenx = len(sx.isel(stacked=i))
				posx = np.sum(sx.isel(stacked=i) > 0)/lenx
				leny = len(sy.isel(stacked=i))
				posy = np.sum(sy.isel(stacked=i) > 0)/leny
				ploc = min(posx,posy)/max(posx,posy)
	return ploc, sloc


def StatTest(x,y,test,dim=None,parallel=False):
	'''Compute statistical test for significance between
	   two xr.DataArrays. Testing will be done along dimension with name `dim`
	   and the output p-value will have all dimensions except `dim`.
	   INPUTS:
	      x	 : xr.DataArray for testing.
	      y	 : xr.DataArray or scalar for testing against. Or None for single-ensemble sign test.
	      dim: dimension name along which to perform the test.
	      test:which test to use:
		    'KS' -> Kolmogorov-Smirnov
		    'MW' -> Mann-Whitney
		    'WC' -> Wilcoxon
		    'T'  -> T-test 1 sample with y=mean
                    'sign'->test against sign only.
		  parallel: Run the test in parallel? Requires the parmap package.
	   OUTPUTS:
	      pvalx: xr.DataArray containing the p-values.
		     Same dimension as x,y except `dim`.
	'''
	from xarray import DataArray
	if dim is None or len(x.dims) == 1:
		sx = x.expand_dims(stacked=[0])
		parallel = False
	else:
		sx = StackArray(x,dim)
	if parallel:
		import parmap
	nspace = len(sx.stacked)
	if isinstance(y,DataArray):
		if dim is None or len(y.dims) == 1:
			sy = y.expand_dims(stacked=[0])
		else:
			sy = StackArray(y,dim)
	else:
		sy = None
	if parallel:
		pval,sval = parmap.map(ComputeStat,list(range(nspace)),sx,y,sy,test)
	else:
		pval, sval = np.zeros(sx.stacked.shape), np.zeros(sx.stacked.shape)
		for i in range(nspace):
			pval[i], sval[i] = ComputeStat(i,sx,y,sy,test)
	if nspace > 1:
		pvalx, svalx = DataArray(pval,coords=[sx.stacked],name='pval').unstack('stacked'), DataArray(sval,coords=[sx.stacked],name='sval').unstack('stacked')
        
	else:
		pvalx, svalx = pval[0], sval[0]
	return pvalx, svalx

class MidpointNormalize(colors.Normalize):
    
    """
    At the moment its a bug to use divergence colormap and set the colorbar range midpoint 
    to zero if both vmax and vmin has different magnitude. This might be possible in 
    future development in matplotlib through colors.offsetNorm(). This class was original developed 
    by Joe Kingto and modified by Daniel Boateng. It sets the divergence color bar to a scale of 0-1 by dividing the midpoint to 0.5
    Use this class at your own risk since its non-standard practice for quantitative data.
    """
    
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))   

def plot_ks_stats(data, cmap, ax=None, vmax=None, vmin=None, levels=None, domain=None, center= True, output_name=None, 
                     output_format=None, level_ticks=None, title=None, path_to_store=None, left_labels= True, bottom_labels=True,
                     add_colorbar=True, hatches=None, fig=None, cbar_pos=None, use_colorbar_default=False, 
                     orientation = "horizontal", plot_projection=None, plot_stats=True, stats_results=None):
    """
    
    
    Returns
    -------
    None.
    """
    norm = MidpointNormalize(midpoint = 0)
    projection = ccrs.PlateCarree()
    
    if plot_projection is None:
        plot_projection = ccrs.PlateCarree()
        
    #generating plot using geoaxis predefined or from default
    if ax is None:
        fig, ax = plt.subplots(1, 1, sharex=False, figsize= (15, 13), subplot_kw= {"projection":plot_projection})
        
    if add_colorbar == True:    
        if cbar_pos is None:
            cbar_pos = [0.90, 0.30, 0.03, 0.45]
        
        if use_colorbar_default == False:
            
            
            cbar_ax = fig.add_axes(cbar_pos)   # axis for subplot colorbar # left, bottom, width, height
            
            if orientation == "vertical":
                cbar_ax.get_xaxis().set_visible(False)
                cbar_ax.yaxis.set_ticks_position('right')
                cbar_ax.set_yticklabels([])
                cbar_ax.tick_params(size=0)
            else:
                cbar_ax.get_yaxis().set_visible(False)
                cbar_ax.xaxis.set_ticks_position('bottom')
                cbar_ax.set_xticklabels([])
                cbar_ax.tick_params(size=0)
        
    
    
    if all(parameter is not None for parameter in [vmin, vmax, levels, level_ticks]):
        ticks = np.linspace(vmin, vmax, level_ticks)
        if vmin < 0:
            
            if center==True:
                if add_colorbar ==True:
                    
                    if use_colorbar_default == True:
                        
                        p = data.plot.imshow(ax =ax, cmap=cmap, vmin=vmin, vmax=vmax, center=0, 
                                        levels=levels, transform = projection, norm=norm, 
                                        cbar_kwargs= {"pad":0.1, "drawedges": True, "orientation": orientation, 
                                                      "shrink": 0.70, "format": "%.2f", "ticks":ticks}, extend= "neither",
                                        add_colorbar=True, add_labels=False)
                    else:
                        
            
                         p = data.plot.imshow(ax =ax, cmap=cmap, vmin=vmin, vmax=vmax, center=0, 
                                         levels=levels, transform = projection, norm=norm, 
                                         cbar_kwargs= {"pad":0.05, "drawedges": True, "orientation": orientation, 
                                                       "shrink": 0.30, "format": "%.2f", "ticks":ticks}, extend= "neither",
                                         add_colorbar=True, cbar_ax = cbar_ax, add_labels=False)
                else:
                    p = data.plot.imshow(ax =ax, cmap=cmap, vmin=vmin, vmax=vmax, center=0, 
                                    levels=levels, transform = projection, norm=norm, add_colorbar=False, add_labels=False) 
                                   
                    
            else:
                if add_colorbar == True:
                    if use_colorbar_default == True:
                        
                        p = data.plot.imshow(ax =ax, cmap=cmap, vmin=vmin, vmax=vmax, 
                                        levels=levels, transform = projection, 
                                        cbar_kwargs= {"pad":0.1, "drawedges": True, "orientation": orientation, 
                                                      "shrink": 0.70, "format": "%.2f", "ticks":ticks}, extend= "neither", 
                                        add_colorbar=True, add_labels=False)
                    else:
                        
                        p = data.plot.imshow(ax =ax, cmap=cmap, vmin=vmin, vmax=vmax, 
                                        levels=levels, transform = projection, 
                                        cbar_kwargs= {"pad":0.05, "drawedges": True, "orientation": orientation, 
                                                      "shrink": 0.30, "format": "%.2f", "ticks":ticks}, extend= "neither", 
                                        add_colorbar=True, cbar_ax = cbar_ax, add_labels=False)
                else:
                    p = data.plot.imshow(ax =ax, cmap=cmap, vmin=vmin, vmax=vmax, 
                                    levels=levels, transform = projection, add_colorbar=False, add_labels=False,)
                                    
        else:
            if add_colorbar == True:
                
                if use_colorbar_default == True:
                    p = data.plot.imshow(ax =ax, cmap=cmap, vmin=vmin, vmax=vmax, 
                                         levels=levels, transform = projection, 
                                         cbar_kwargs= {"pad":0.1, "drawedges": True, "orientation": orientation, 
                                                       "shrink": 0.70, "format": "%.2f", "ticks":ticks}, extend= "both",
                                         add_colorbar=True, add_labels=False)
                else:
                    
                    
                
                    p = data.plot.imshow(ax =ax, cmap=cmap, vmin=vmin, vmax=vmax, 
                                         levels=levels, transform = projection, 
                                         cbar_kwargs= {"pad":0.1, "drawedges": True, "orientation": orientation, 
                                                       "shrink": 0.70, "format": "%.2f", "ticks":ticks}, extend= "both",
                                         add_colorbar=True, cbar_ax = cbar_ax, add_labels=False)
            else:
                p = data.plot.imshow(ax =ax, cmap=cmap, vmin=vmin, vmax=vmax, 
                                     levels=levels, transform = projection, add_colorbar=False, add_labels=False)
                                     
    # when limits are not defined for the plot            
    else:
        p = data.plot.imshow(ax =ax, cmap=cmap, transform = projection, 
                                 cbar_kwargs= {"pad":0.1, "drawedges": True, "orientation": orientation, 
                                               "shrink": 0.70, "format": "%.2f", "ticks":ticks}, extend= "neither", 
                                 add_colorbar=True, cbar_ax = cbar_ax, add_labels=False)
    
    
    if add_colorbar == True:
        
        p.colorbar.set_label(label="KS statistic", size= 20, fontweight="bold")
        p.colorbar.ax.tick_params(labelsize=20, size=0,)
    
    # ploting background extent
    plot_background(p, domain= domain, left_labels=left_labels, bottom_labels=bottom_labels)
    
    
   
        
        
    if plot_stats == True:
        
        if hatches is not None:
            ax.contourf(stats_results.lon.data, stats_results.lat.data, stats_results.data, colors="none", hatches=[hatches])
        else:
            ax.contourf(stats_results.lon.data, stats_results.lat.data, stats_results.data, colors="none", hatches=["xx"])
            

    # fig.canvas.draw()   # the only way to apply tight_layout to matplotlib and cartopy is to apply canvas firt 
    # plt.tight_layout()
    # plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.30)
    
    
    if title is not None:
        ax.set_title(title, fontsize=20, weight="bold", loc="left")
        
    #optional if one plot is required, alternatively save from the control script
    if all(parameter is not None for parameter in [output_format, output_name, path_to_store]):
        plt.savefig(os.path.join(path_to_store, output_name + "." + output_format), format= output_format, bbox_inches="tight")
    else:
        print("The output would not be save on directory")
        

def ks_stats_amip_era5(varname, center=False, standardize=False, plot=False):
    
    
    data_cmip = extract_region(data=CMIP5_AMIP_R1, datarange=fullAMIP, varname=varname, minlat=30, maxlat=70, 
                                  minlon=-20, maxlon=50)
    
    data_era = extract_region(data=ERA5Data, datarange=fullAMIP, varname=varname, minlat=30, maxlat=70, 
                                  minlon=-20, maxlon=50)


    # convert data to the same physical units 
    if varname == "tp":
        data_cmip = data_cmip *60*60*24*30 #mm/month
        
        data_era = data_era * 1000 * 30  #mm/month

    
    # interpolate to the same coordinate (to cmip gridsize)
    
    data_era = data_era.interp(lat=data_cmip.lat).interp(lon=data_cmip.lon)


    # compute anomalies
    group_era = data_era.groupby("time.month")
    
    group_cmip = data_cmip.groupby("time.month")
    
    monthly_means_era = group_era.mean(dim="time")
    
    monthly_means_cmip = group_cmip.mean(dim="time")
    
    anomalies_era = group_era.apply(
               lambda x: x - monthly_means_era.sel(month=_get_month(x[0].time.values))
           )
    
    anomalies_cmip = group_cmip.apply(
               lambda x: x - monthly_means_era.sel(month=_get_month(x[0].time.values))
           )
    
    # apply standardize (optional for testing)
    if center ==True:
    # centered 
        anomalies_era = anomalies_era - anomalies_era.mean("time")
        
        anomalies_cmip = anomalies_cmip - anomalies_cmip.mean("time")
    
    if standardize ==True:
        anomalies_era = anomalies_era / anomalies_era.std(dim="time")
        
        anomalies_cmip = anomalies_cmip / anomalies_cmip.std(dim="time")

        
    
    pvalx, svalx = StatTest(x=anomalies_era, y=anomalies_cmip, test="KS", dim="time")
    
    sig_loc  = xr.where(pvalx < 0.05, pvalx, pvalx*np.nan)
    
    sig_loc = sig_loc.sortby("lon")
    svalx = svalx.sortby("lon")  
    
    return sig_loc, svalx



        
        
#plotting
path_to_store = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Neckar/plots"

varname = "u850"

sig_loc, svalx = ks_stats_amip_era5(varname=varname, center=False, standardize=False)
sig_loc_c, svalx_c = ks_stats_amip_era5(varname=varname, center=True, standardize=False)
sig_loc_s, svalx_s = ks_stats_amip_era5(varname=varname, center=True, standardize=True)



projection = ccrs.PlateCarree()
fig, (ax1,ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize=(28, 13), subplot_kw={"projection": projection})

plot_ks_stats(data=svalx, ax=ax1, cmap="YlGnBu", vmax=1, vmin=0, levels=22, level_ticks=6, plot_stats=True, 
              stats_results=sig_loc, domain="Europe", orientation="horizontal", cbar_pos = [0.35, 0.25, 0.35, 0.02],
              title= varname + " raw (ks_test(MIP-ESM, ERA5))", add_colorbar=False, fig=fig) #, path_to_store=path_to_store, output_format="svg", output_name="r850_raw.svg")

plot_ks_stats(data=svalx_c, ax=ax2, cmap="YlGnBu", vmax=1, vmin=0, levels=22, level_ticks=6, plot_stats=True, 
              stats_results=sig_loc_c, domain="Europe", orientation="horizontal", cbar_pos = [0.35, 0.25, 0.35, 0.02],
              title= varname + " centered (ks_test(MIP-ESM, ERA5))", fig=fig) #, path_to_store=path_to_store, output_format="svg", output_name="r850_raw.svg")

plot_ks_stats(data=svalx_s, ax=ax3, cmap="YlGnBu", vmax=1, vmin=0, levels=22, level_ticks=6, plot_stats=True, 
              stats_results=sig_loc_s, domain="Europe", orientation="horizontal", cbar_pos = [0.35, 0.25, 0.35, 0.02],
              title= varname + " standardize (ks_test(MIP-ESM, ERA5))", add_colorbar=False, fig=fig) #, path_to_store=path_to_store, output_format="svg", output_name="r850_raw.svg")

fig.canvas.draw()   # the only way to apply tight_layout to matplotlib and cartopy is to apply canvas firt 
plt.tight_layout()
plt.subplots_adjust(left=0.05, right=0.95, top=0.94, bottom=0.15)
plt.savefig(os.path.join(path_to_store, "u850_ks_test.svg"), format= "svg", bbox_inches="tight", dpi=300)


# projection = ccrs.PlateCarree()
# fig, ax = plt.subplots(1, 1, sharex=False, figsize= (15, 13), subplot_kw= {"projection":projection})
# score.plot(cmap="YlGnBu", transform=projection, add_colorbar=True)
# p = ax.contourf(stats.lon.data, stats.lat.data, stats.data, colors="none", hatches=["xx"])
# plot_background(p, domain="Europe", ax=ax)
plt.show()


