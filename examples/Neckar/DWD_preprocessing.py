# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 15:48:20 2022

@author: dboateng

This script preprocess DWD data downloaded for a subcatchment ("Neckar_bis_Enz")

"""

# import PyESD Pacakge
import sys
import os 

sys.path.append("C:/Users/dboateng/Desktop/Python_scripts/ESD_Package")

from Package.WeatherstationPreprocessing import extract_DWDdata_with_more_yrs, add_info_to_data


#define paths
path_to_data = "C:/Users/dboateng/Desktop/Datasets/Station/Neckar_Enz/Temperature/cdc_download_2022-03-17_13-38/data"
path_to_store_considered = "C:/Users/dboateng/Desktop/Datasets/Station/Neckar_Enz/Temperature/cdc_download_2022-03-17_13-38/considered"
path_to_store_processed = "C:/Users/dboateng/Desktop/Datasets/Station/Neckar_Enz/Temperature/cdc_download_2022-03-17_13-38/processed"

path_to_info = "C:/Users/dboateng/Desktop/Datasets/Station/Neckar_Enz/Temperature/cdc_download_2022-03-17_13-38/data/sdo_OBS_DEU_P1M_T2M.csv"

# sort data according to the requried number of years 
extract_DWDdata_with_more_yrs(path_to_data=path_to_data, path_to_store =path_to_store_considered,
                              min_yrs=40, glob_name="data*.csv", varname="Temperature")

# add station infos and save them in the processed folder

add_info_to_data(path_to_info=path_to_info, path_to_data=path_to_store_considered,
                 path_to_store = path_to_store_processed , glob_name="data*", 
                 varname="Temperature",)

