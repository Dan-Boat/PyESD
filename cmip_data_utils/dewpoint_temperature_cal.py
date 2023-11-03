import xarray as xr
import numpy as np
import sys
import os


def safely_to_netcdf(da, fname):
    # to_netcdf sometimes fails after altering the existing file which leads to data loss
    da.to_netcdf('tmp.nc')
    os.rename('tmp.nc', fname)

levels = ["1000", "850", "700", "500", "250"]


for level in levels:
    t = xr.open_dataarray("t" + level + "_monthly.nc")
    r = xr.open_dataarray("r" + level + "_monthly.nc")
    
    # Magnus formula constants
    # following this here: https://en.wikipedia.org/wiki/Dew_point
    b = 18.678
    c = 257.14
    d = 234.5
    
    def gamma(rh, t):
        return np.log(rh/100*np.exp((b-t/d)*(t/(c+t))))
    
    def dewpoint_temp(r, t):
        g = gamma(r, t)
        return c*g/(b - g)
    
    dtd = t - dewpoint_temp(r, t)
    dtd.name = "dtd" + level
    
    safely_to_netcdf(dtd, "dtd" + level + "_monthly.nc")