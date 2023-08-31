# -*- coding: utf-8 -*-
"""

@author: dboateng

This script post process the raw files downloaded from the DWD climate Data Store. 

The script uses the data_preprocess_utils module of pyESD.utils to extract the 
require datasets (stations with minmum 60 years of records) and structure them into the
format compatible with the pyESD 

"""

# import all the required modules
import sys
import os
from pyESD.data_preprocess_utils import extract_DWDdata_with_more_yrs, add_info_to_data


#set the paths to the raw datasets
main_path = "C:/Users/dboateng/Desktop/Datasets/Station/southern_germany"
data_files_path = os.path.join(main_path, "data")
path_data_considerd = os.path.join(main_path, "considered")
path_data_processed =os.path.join(main_path, "processed")
path_data_info = "C:/Users/dboateng/Desktop/Datasets/Station/southern_germany/data/sdo_OBS_DEU_P1M_RR.csv"

if not os.path.exists(path_data_considerd):
    os.makedirs(path_data_considerd)   
    
if not os.path.exists(path_data_processed):
    os.makedirs(path_data_processed)


# SORTING DATA WITH YEARS REQUIREMENT
# ===================================  
    
# extract all the datasets that meet the 60 years requirement
extract_DWDdata_with_more_yrs(path_to_data=data_files_path, path_to_store=path_data_considerd,
                              min_yrs=60, glob_name="data*.csv", varname="Precipitation",
                              start_date="1958-01-01", end_date="2022-12-01", data_freq="MS")
    
    

# FORMATTING PROCESSED DATA
# =========================

add_info_to_data(path_to_info=path_data_info, path_to_data=path_data_considerd,
                 path_to_store=path_data_processed, varname="Precipitation")

