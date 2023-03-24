# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:58:59 2022

@author: dboateng
"""

# import modules 
import os 
import sys
import xarray as xr
import numpy as np 
import pandas as pd 
from eofs.xarray import Eof 
from sklearn import decomposition
import cartopy.crs as ccrs
import cartopy.feature as cfeature 
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from cartopy.util import add_cyclic_point


try:
    from .Predictor_Base import Predictor
    from .ESD_utils import map_to_xarray
except:
    from Predictor_Base import Predictor
    from ESD_utils import map_to_xarray


def eof_analysis(data, neofs, method="eof_package", apply_equal_wtgs=True, 
                 pcscaling=1):
    
    if hasattr(data, "longitude"):
        data = data.rename({"longitude":"lon", "latitude":"lat"})
    
    if apply_equal_wtgs == True:
        wtgs = np.sqrt(np.abs(np.cos(data.lat * np.pi / 180)))
        data = data * wtgs
    
    if True in data.isnull():
        data = data.dropna(dim="time")
        
    if method == "eof_package":
        
        print("The EoF Package implementation of EOF analysis is used! for the teleconnections")
        
        solver = Eof(data)
        eofs_cov = solver.eofsAsCovariance(neofs=neofs, pcscaling=pcscaling)
        
        eofs_cov = eofs_cov.sortby(eofs_cov.lon)
        
        pcs = solver.pcs(pcscaling=pcscaling, npcs=neofs)
        
    elif method == "sklearn_package":
        
        print("The sklearn implementation of EOF analysis is used! for the teleconnections")
        
        time_mean = data.mean(dim="time")
        
        deviations = data - time_mean

        pcs = np.empty(neofs)
        
        PCA = decomposition.PCA(n_components=neofs)
        
        X = np.reshape(deviations.values, (len(data.time), time_mean.size))
        
        PCA.fit(X)
        
        shape = [neofs] + list(np.shape(time_mean))
        
        np_eofs = np.reshape(PCA.components_, shape)
        pcs = PCA.explained_variance_
        
        ls_eofs= []
        for i in range(neofs):
            ls_eofs.append(map_to_xarray(X=np_eofs[i, ...], datarray=data))
            
        eofs_cov = xr.concat(ls_eofs, pd.Index(range(1, neofs+1), name="eof_number"))
        
        #eofs_cov = eofs_cov.sortby(eofs_cov.lon)
        
    
    else:
        raise ValueError("Define the method or defaut of EOF package is used")
        
        
        
    if apply_equal_wtgs == True:
        return eofs_cov, pcs, wtgs
    
    else:
        return eofs_cov, pcs 
        

def extract_region(data, datarange, varname, minlat, maxlat, minlon, maxlon):
        
    data = data.get(varname).sel(time=datarange)
    
    if hasattr(data, "longitude"):
        data = data.rename({"longitude":"lon", "latitude":"lat"})
    
    #data = data.assign_coords({"lon": (((data.lon + 180) % 360) - 180)})
    
    data = data.where((data.lat >=minlat) & (data.lat <=maxlat), drop=True)
    data = data.where((data.lon >=minlon) & (data.lon <= maxlon), drop=True)
    
    return data
        
# implementation of MEI (not tested)

class MEI(Predictor):
    def __init__(self, **kwargs):
        super().__init__(name="MEI", longname= "Multivariate ENSO Index", **kwargs)
        self.variables = ["t2m", "msl", "u10", "v10", "sst"] # replace temperature with outgoing long wave radiation!
        
    def _generate(self, datarange, dataset, fit, patterns_from, params_from):
        
        if hasattr(dataset, "expver"):
            data = dataset.drop("expver")
        
        
        bi_monthly={}
        params = self.params[params_from]
        
        if fit:
            params["nonan_indices"] = {}
            
        for v in self.variables:
            # load data and restrict to area of interest
            da = dataset.get(v)
            da = da.sel(time=datarange)
            da = da.where((da.latitude <= 30) & (da.latitude >= -30)
                            & (da.longitude <= 100) & (da.longitude >= -70), drop=True)

            # multiply by area weight (cos(lat)) -check if needed here
            da *= np.sqrt(np.cos(da.latitude*np.pi/180))
            
            if hasattr(da, "longitude"):
                da = da.rename({"longitude":"lon", "latitude":"lat"})
            # stack into one vector
            stacked = da.stack(z=['lat', 'lon'])

            if fit:
                # get the indices with non-nan entries (only relevant for sst)
                params["nonan_indices"][v] = np.where(~np.isnan(stacked.isel(time=0)))[0]
            stacked = stacked.isel(z=params["nonan_indices"][v])


            # get bimonthly season values
            b = stacked.rolling(time=2).mean()
            b[0,:] = stacked[0,:] # the first bimonthly season consist of only the first month
            bi_monthly[v] = b

        # this assumes that all dataarrays have the same time axis
        month = da.time.values.astype('datetime64[M]').astype(int) % 12
        
        # find EOF pattern for each bimonthly window
        if fit:
            self.patterns[dataset.name] = {}
            self.patterns[dataset.name]['eofs'] = []
            params["timeseries_means"] = {}
            params["timeseries_stds"] = {}
        mei = np.empty(len(month))
        for i in range(12):
            this_month = np.where(month == i)[0]
            normalized = []
            for v in bi_monthly:
                # select months
                ts = bi_monthly[v].isel(time=this_month)
                # normalize
                if fit:
                    params["timeseries_means"][v] = ts.mean(dim='time')
                    params["timeseries_stds"][v] = ts.std(dim='time')
                ts -= params["timeseries_means"][v]
                ts /= params["timeseries_stds"][v]
                normalized.append(ts)

            vector = xr.concat(normalized, dim='z')
            
            if fit:
                eof, _ = eof_analysis(data=vector, neofs=1, method="sklearn_package", apply_equal_wtgs=False)
                eof1 = eof.sel(eof_number=1) 
                # this guarantees that the EOFs of the different months have
                # the same direction
                if i != 0:
                    if np.dot(eof, eof1) < 0:
                        eof *= -1
                else:
                    eof1 = eof
                self.patterns[dataset.name]['eofs'].append(eof)
            index = np.dot(vector, self.patterns[patterns_from]['eofs'][i].squeeze())
            mei[this_month] = index
            
        # the standard deviation that is used to normalize the index should be the same as the one
        # found when constructing the patterns, that's why it's stored together with the patterns
        if fit:
            self.patterns[dataset.name]['std'] = np.std(mei)
        mei /= self.patterns[patterns_from]['std']

        mei_series = pd.Series(data=mei, name='MEI', index=da.time.values)

        return mei_series
            
            
        
        


class NAO(Predictor):
    def __init__(self, **kwargs):
        super().__init__(name="NAO", longname= "North Atlantic Oscilation", **kwargs)
        
    def _generate(self, datarange, dataset, fit, patterns_from, params_from):
        
        if hasattr(dataset, "expver"):
            data = dataset.drop("expver")
        
        params = self.params[params_from]
        
        data = extract_region(dataset, datarange, varname="msl", minlat=20, maxlat=80, 
                              minlon=-80, maxlon=60)
        
        # removing monthly cycle
        group = data.groupby("time.month")
            
        if fit:
            params["monthly_means"] = group.mean(dim="time")
            
            anomalies = group.apply(
            lambda x: x - params["monthly_means"].sel(month=_get_month(x[0].time.values))
        )
            params["monthly_means"] = group.mean(dim="time")
        
        
        #anomalies = group - params["monthly_means"]
        anomalies = anomalies.drop("month")
        
        if fit:
            params["std_field"] = anomalies.std(dim="time")
            
        anomalies /= params["std_field"]
        
        area_weighted = anomalies * np.sqrt(np.abs(np.cos(anomalies.lat*np.pi/180))) 
        
        
        method_name = "eof_package"
        
        if fit:
            
            self.patterns[dataset.name] = {}
            eofs, pcs, wtgs  = eof_analysis(data=anomalies, neofs=1, method=method_name, apply_equal_wtgs=True, 
                                                pcscaling=0)
            
            
            if hasattr(eofs, "eof_number"):
                self.patterns[dataset.name]["eof"] = eofs.sel(eof_number=1)
            
            else:
                self.patterns[dataset.name]["eof"] = eofs.sel(mode=0)
                
        if method_name == "sklearn_package":
            nao = (self.patterns[patterns_from]["eof"] * area_weighted).sum(dim=("lat", "lon"))
            
        else: 
            
            nao = pcs.sel(mode=0)
            
        nao.name = "NAO"
        
        
        if fit:
            self.patterns[dataset.name]["std"] = nao.std()
            
        nao /= self.patterns[patterns_from]["std"]
        
        nao_series = nao.to_series()
        
        
        return nao_series
    
    def plot_cov_matrix(ax=None):
        pass
        


class EA(Predictor):
    def __init__(self, **kwargs):
        super().__init__(name="EA", longname= "East Atlantic Oscilation", **kwargs)
    
    def _generate(self, datarange, dataset, fit, patterns_from, params_from):
        
        if hasattr(dataset, "expver"):
            data = dataset.drop("expver")
        
        params = self.params[params_from]
        
        data = extract_region(dataset, datarange, varname="msl", minlat=20, maxlat=80, 
                              minlon=-80, maxlon=60)
        
        # removing monthly cycle
        group = data.groupby("time.month")
            
        if fit:
            params["monthly_means"] = group.mean(dim="time")
            
            anomalies = group.apply(
            lambda x: x - params["monthly_means"].sel(month=_get_month(x[0].time.values))
        )
            params["monthly_means"] = group.mean(dim="time")
        
        
        #anomalies = group - params["monthly_means"]
        anomalies = anomalies.drop("month")
        
        if fit:
            params["std_field"] = anomalies.std(dim="time")
            
        anomalies /= params["std_field"]
        
        area_weighted = anomalies * np.sqrt(np.abs(np.cos(anomalies.lat*np.pi/180))) 
        
        method_name = "eof_package"
        
        if fit:
            
            self.patterns[dataset.name] = {}
            eofs, pcs, wtgs  = eof_analysis(data=anomalies, neofs=2, method=method_name, apply_equal_wtgs=True, 
                                                pcscaling=0)
            
            
            if hasattr(eofs, "eof_number"):
                self.patterns[dataset.name]["eof"] = eofs.sel(eof_number=2)
            
            else:
                self.patterns[dataset.name]["eof"] = eofs.sel(mode=1)
                
        if method_name == "sklearn_package":
            ea = (self.patterns[patterns_from]["eof"] * area_weighted).sum(dim=("lat", "lon"))
            
        else: 
            
            ea = pcs.sel(mode=1)
            
        ea.name = "EA"
        
        
        if fit:
            self.patterns[dataset.name]["std"] = ea.std()
            
        ea /= self.patterns[patterns_from]["std"]
        
        ea_series = ea.to_series()
        
        
        return ea_series   

    
    def plot_cov_matrix(ax=None):
        pass

class SCAN(Predictor):
    def __init__(self, **kwargs):
        super().__init__(name="SCAN", longname= "Scandinavian Oscilation", **kwargs)
    def _generate(self, datarange, dataset, fit, patterns_from, params_from):
        
        
        if hasattr(dataset, "expver"):
            data = dataset.drop("expver")
        
        params = self.params[params_from]
        
        data = extract_region(dataset, datarange, varname="msl", minlat=20, maxlat=80, 
                              minlon=-80, maxlon=60)
        
        # removing monthly cycle
        group = data.groupby("time.month")
            
        if fit:
            params["monthly_means"] = group.mean(dim="time")
            
            anomalies = group.apply(
            lambda x: x - params["monthly_means"].sel(month=_get_month(x[0].time.values))
        )
            params["monthly_means"] = group.mean(dim="time")
        
        
        #anomalies = group - params["monthly_means"]
        anomalies = anomalies.drop("month")
        
        if fit:
            params["std_field"] = anomalies.std(dim="time")
            
        anomalies /= params["std_field"]
        
        area_weighted = anomalies * np.sqrt(np.abs(np.cos(anomalies.lat*np.pi/180))) 
        
        method_name = "sklearn_package"
        
        if fit:
            
            self.patterns[dataset.name] = {}
            eofs, pcs, wtgs  = eof_analysis(data=anomalies, neofs=3, method=method_name, apply_equal_wtgs=True, 
                                                pcscaling=0)
            
            
            if hasattr(eofs, "eof_number"):
                self.patterns[dataset.name]["eof"] = eofs.sel(eof_number=3)
            
            else:
                self.patterns[dataset.name]["eof"] = eofs.sel(mode=2)
                
        if method_name == "sklearn_package":
            scan = (self.patterns[patterns_from]["eof"] * area_weighted).sum(dim=("lat", "lon"))
            
        else: 
            
            scan = pcs.sel(mode=2)
        
        scan.name = "SCAN"
        
        
        if fit:
            self.patterns[dataset.name]["std"] = scan.std()
            
        scan /= self.patterns[patterns_from]["std"]
        
        scan_series = scan.to_series()
        
        
        return scan_series    
        
    
    def plot_cov_matrix(ax=None):
        pass

class EAWR(Predictor):
    def __init__(self, **kwargs):
        super().__init__(name="EAWR", longname= "East Atlantic_West Russian Oscilation", **kwargs)
    def _generate(self, datarange, dataset, fit, patterns_from, params_from):
        
        if hasattr(dataset, "expver"):
            data = dataset.drop("expver")
        
        params = self.params[params_from]
        
        data = extract_region(dataset, datarange, varname="msl", minlat=20, maxlat=80, 
                              minlon=-80, maxlon=60)
        
        # removing monthly cycle
        group = data.groupby("time.month")
            
            
        if fit:
            params["monthly_means"] = group.mean(dim="time")
            
            anomalies = group.apply(
            lambda x: x - params["monthly_means"].sel(month=_get_month(x[0].time.values))
        )
            params["monthly_means"] = group.mean(dim="time")
        
        
        #anomalies = group - params["monthly_means"]
        anomalies = anomalies.drop("month")
        
        if fit:
            params["std_field"] = anomalies.std(dim="time")
            
        anomalies /= params["std_field"]
        
        area_weighted = anomalies * np.sqrt(np.abs(np.cos(anomalies.lat*np.pi/180))) 
        
        method_name = "sklearn_package"
        
        if fit:
            
            self.patterns[dataset.name] = {}
            eofs, pcs, wtgs  = eof_analysis(data=anomalies, neofs=4, method=method_name, apply_equal_wtgs=True, 
                                                pcscaling=0)
            
            
            if hasattr(eofs, "eof_number"):
                self.patterns[dataset.name]["eof"] = eofs.sel(eof_number=4)
            
            else:
                self.patterns[dataset.name]["eof"] = eofs.sel(mode=3)
                
        if method_name == "sklearn_package":
            ea_wr = (self.patterns[patterns_from]["eof"] * area_weighted).sum(dim=("lat", "lon"))
            
        else: 
            
            ea_wr = pcs.sel(mode=3)
        
        ea_wr.name = "EAWR"
        
        
        if fit:
            self.patterns[dataset.name]["std"] = ea_wr.std()
            
        ea_wr /= self.patterns[patterns_from]["std"]
        
        ea_wr_series = ea_wr.to_series()
        
        
        return ea_wr_series
    
    
    def plot_cov_matrix(ax=None):
        pass
    
def _get_month(npdatetime64):
    
     """
     Returns the month for a given npdatetime64 object, 1 for January, 2 for
     February, ...
     """
     month =  npdatetime64.astype('datetime64[M]').astype(int) % 12 + 1
     
     return month