# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 15:48:20 2022

@author: dboateng

This script preprocess DWD data downloaded for a subcatchment ("Neckar_bis_Enz")

"""

# import PyESD Pacakge
import sys
import os 
from pyESD.WeatherstationPreprocessing import extract_DWDdata_with_more_yrs, add_info_to_data


#DEFINING PATHS
#==============

path_to_catchment = "C:/Users/dboateng/Desktop/Datasets/Station/Neckar/"

path_to_prec = os.path.join(path_to_catchment, "Precipitation/")
path_to_prec_data = os.path.join(path_to_prec, "data/")
path_to_prec_processed = os.path.join(path_to_prec, "processed/")
path_to_prec_considered = os.path.join(path_to_prec, "considered/")
stn_info_prec = os.path.join(path_to_prec_data, "sdo_OBS_DEU_P1M_RR.csv")


path_to_temp = os.path.join(path_to_catchment, "Temperature/")
path_to_temp_data = os.path.join(path_to_temp, "data/")
path_to_temp_processed = os.path.join(path_to_temp, "processed/")
path_to_temp_considered = os.path.join(path_to_temp, "considered/")
stn_info_temp = os.path.join(path_to_temp_data, "sdo_OBS_DEU_P1M_T2M.csv")

# SORTING DATA WITH YEARS REQUIREMENT
# ===================================

extract_DWDdata_with_more_yrs(path_to_data=path_to_prec_data, 
                              path_to_store=path_to_prec_considered,
                              min_yrs=60, varname="Precipitation",)

extract_DWDdata_with_more_yrs(path_to_data=path_to_temp_data, 
                              path_to_store=path_to_temp_considered,
                              min_yrs=60, varname="Temperature",)


# FORMATTING PROCESSED DATA
# =========================

add_info_to_data(path_to_info=stn_info_prec, path_to_data=path_to_prec_considered, 
                  path_to_store=path_to_prec_processed, varname="Precipitation")


add_info_to_data(path_to_info=stn_info_temp, path_to_data=path_to_temp_considered, 
                 path_to_store=path_to_temp_processed, varname="Temperature")


