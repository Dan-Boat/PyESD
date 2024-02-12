# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 17:02:27 2023

@author: dboateng
1. clip annual climatologies for Ghana based on the Shape file (if it works)
2. clip the future estimates based on CMIP GCMs and RCM model
"""


import xarray as xr
import geopandas as gpd
from rasterio.features import geometry_mask
from shapely.geometry import box
import matplotlib.pyplot as plt
from rasterio import features
from affine import Affine

import numpy as np

def clip_nc_with_shapefile(nc_path, shapefile_path, output_nc_path):
    # Read the NetCDF file
    ds = xr.open_dataset(nc_path)

    # Read the shapefile
    gdf = gpd.read_file(shapefile_path)

    # Get the bounding box of the shapefile
    bounds = gdf.total_bounds
    bounding_box = box(*bounds)

    # Mask the data using the shapefile
    mask = geometry_mask([bounding_box], ds.coords['longitude'], ds.coords['latitude'], invert=True)
    ds_clipped = ds.where(mask, drop=True)

    # Save the clipped data to a new NetCDF file
    ds_clipped.to_netcdf(output_nc_path)

    # Close the original NetCDF file
    ds.close()

    print("Clipping complete. Clipped data saved to:", output_nc_path)
    


def transform_from_latlon(lat, lon):
    """ input 1D array of lat / lon and output an Affine transformation
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale

def rasterize(shapes, coords, latitude='latitude', longitude='longitude',
              fill=np.nan, **kwargs):
    """Rasterize a list of (geometry, fill_value) tuples onto the given
    xray coordinates. This only works for 1d latitude and longitude
    arrays.

    usage:
    -----
    1. read shapefile to geopandas.GeoDataFrame
          `states = gpd.read_file(shp_dir+shp_file)`
    2. encode the different shapefiles that capture those lat-lons as different
        numbers i.e. 0.0, 1.0 ... and otherwise np.nan
          `shapes = (zip(states.geometry, range(len(states))))`
    3. Assign this to a new coord in your original xarray.DataArray
          `ds['states'] = rasterize(shapes, ds.coords, longitude='X', latitude='Y')`

    arguments:
    ---------
    : **kwargs (dict): passed to `rasterio.rasterize` function

    attrs:
    -----
    :transform (affine.Affine): how to translate from latlon to ...?
    :raster (numpy.ndarray): use rasterio.features.rasterize fill the values
      outside the .shp file with np.nan
    :spatial_coords (dict): dictionary of {"X":xr.DataArray, "Y":xr.DataArray()}
      with "X", "Y" as keys, and xr.DataArray as values

    returns:
    -------
    :(xr.DataArray): DataArray with `values` of nan for points outside shapefile
      and coords `Y` = latitude, 'X' = longitude.


    """
    transform = transform_from_latlon(coords[latitude], coords[longitude])
    out_shape = (len(coords[latitude]), len(coords[longitude]))
    raster = features.rasterize(shapes, out_shape=out_shape,
                                fill=fill, transform=transform,
                                dtype=float, **kwargs)
    spatial_coords = {latitude: coords[latitude], longitude: coords[longitude]}
    return xr.DataArray(raster, coords=spatial_coords, dims=(latitude, longitude))

def add_shape_coord_from_data_array(xr_da, shp_path, coord_name):
    """ Create a new coord for the xr_da indicating whether or not it 
         is inside the shapefile

        Creates a new coord - "coord_name" which will have integer values
         used to subset xr_da for plotting / analysis/

        Usage:
        -----
        precip_da = add_shape_coord_from_data_array(precip_da, "awash.shp", "awash")
        awash_da = precip_da.where(precip_da.awash==0, other=np.nan) 
    """
    # 1. read in shapefile
    shp_gpd = gpd.read_file(shp_path)

    # 2. create a list of tuples (shapely.geometry, id)
    #    this allows for many different polygons within a .shp file (e.g. States of US)
    shapes = [(shape, n) for n, shape in enumerate(shp_gpd.geometry)]

    # 3. create a new coord in the xr_da which will be set to the id in `shapes`
    xr_da[coord_name] = rasterize(shapes, xr_da.coords, 
                               longitude='longitude', latitude='latitude')

    return xr_da

if __name__ == "__main__":
    
    # set paths
    era_data_path="D:/Datasets/ERA5/monthly_1950_2021/tp_monthly.nc"
    path_shapefile="C:/Users/dboateng/Desktop/Datasets/Station/Ghana/Ghana_ShapeFile/gh_wgs16dregions.shp"
    output_netcdf_path = "C:/Users/dboateng/Desktop/Datasets/Station/Ghana/"

    # Call the function to clip the NetCDF file using the shapefile
    #clip_nc_with_shapefile(era_data_path, path_shapefile, output_netcdf_path)
    # tp_data = xr.open_dataset(era_data_path)
    # tp_data = tp_data["tp"]
    # tp_data = add_shape_coord_from_data_array(tp_data, path_shapefile, "ghana")
    # ghana_da = tp_data.where(tp_data.ghana==0, other=np.nan)
    
    import salem

    from salem.utils import get_demo_file
    
    ds = salem.open_xr_dataset(era_data_path)
    t2 = ds.tp.isel(time=2)

    #t2_sub = t2.salem.subset(corners=((77., 20.), (97., 35.)), crs=salem.wgs84)

    shdf = salem.read_shapefile(path_shapefile)
    #shdf1 = shdf.loc[shdf['CNTRY_NAME'].isin(["Ghana"])]  # GeoPandas' GeoDataFrame
    t2_sub = t2.salem.subset(shape=shdf, margin=2, crs=salem.wgs84)  # add 2 grid points
    t2_sub.salem.quick_map()
    t2_roi = t2_sub.salem.roi(shape=shdf)
    t2_roi.salem.quick_map()
    




