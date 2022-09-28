# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 11:28:52 2022

@author: dboateng
"""

import os
import pandas as pd 
import numpy as np 


from read_data import *

# from package
from pyESD.ESD_utils import load_all_stations, load_pickle, load_csv


# DIRECTORY SPECIFICATION
# =======================

def write_metrics(path_to_data, method, stationnames, path_to_save, varname,
                  filename_train = "validation_score_",
                  filename_test="test_score_"):
    
    train_score = load_all_stations(filename_train + method, path_to_data, stationnames)
    
    test_score = load_all_stations(filename_test + method, path_to_data, stationnames)
    
    df = pd.concat([train_score, test_score], axis=1, ignore_index=False)
    
    # save files
    
    df.to_csv(os.path.join(path_to_save, varname + "_train_test_metrics.csv"), index=True, header=True)


path_exp3 = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Neckar/experiment3"
path_to_save = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Neckar/plots"


prec_folder_name = "final_cache_Precipitation"
temp_folder_name = "final_cache_Temperature"


path_to_data_prec = os.path.join(path_exp3, prec_folder_name)
path_to_data_temp = os.path.join(path_exp3, temp_folder_name)


#Temperature

write_metrics(path_to_data_temp, "Stacking", stationnames_temp, path_to_save, 
              "Temperature")

# Precipitation

write_metrics(path_to_data_prec, "Stacking", stationnames_prec, path_to_save, 
              "Precipitation")


