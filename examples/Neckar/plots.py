# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:06:04 2022

@author: dboateng
"""
import os 
import sys
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
from collections import OrderedDict


sys.path.append("C:/Users/dboateng/Desktop/Python_scripts/ESD_Package")

from Package.ESD_utils import load_all_stations, load_pickle, load_csv
from Package.WeatherstationPreprocessing import read_weatherstationnames



temp_datadir = "C:/Users/dboateng/Desktop/Datasets/Station/Neckar_Enz/Temperature/cdc_download_2022-03-17_13-38/processed"

prec_datadir = "C:/Users/dboateng/Desktop/Datasets/Station/Neckar_Enz/Precipitation/cdc_download_2022-03-17_13/processed"

namedict_temp  = read_weatherstationnames(temp_datadir)
stationnames_temp = list(namedict_temp.values())

namedict_prec  = read_weatherstationnames(prec_datadir)
stationnames_prec = list(namedict_prec.values())


num_stations_temp = len(stationnames_temp)
num_stations_prec = len(stationnames_prec)

stationname_temp = stationnames_temp[1]
stationname_prec = stationnames_prec[1]


path_data_precipitation = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Neckar/experiment1/final_cache_Precipitation"
path_data_temperature = "C:/Users/dboateng/Desktop/Python_scripts/ESD_Package/examples/Neckar/experiment1/final_cache_Temperature"


selector_method = "Recursive"
selector_method_2 = "TreeBased"
selector_method_3 = "Sequential"

score = load_pickle(stationname_prec, "validation_score_" + selector_method, path_data_precipitation)
scores_rec = load_all_stations("validation_score_" + selector_method, path_data_precipitation, 
                           stationnames_prec[1:])

scores_tree = load_all_stations("validation_score_" + selector_method_2, path_data_precipitation, 
                           stationnames_prec[1:])

scores_seq = load_all_stations("validation_score_" + selector_method_3, path_data_precipitation, 
                           stationnames_prec[1:])

r2 = pd.DataFrame(index=stationnames_prec, columns=["Recursive", "TreeBased", "Sequential"])
r2["Recursive"] = scores_rec["test_r2"]
r2["TreeBased"] = scores_tree["test_r2"]
r2["Sequential"] = scores_seq["test_r2"]
print(score)


# to do 
# write a funciton that aranges the score in different dataframses 
# write funtion that plots the bar plots with error (use pandas bar plot with yerr)
#write a function that estimates the correlation btn predictors and predictand
#write a function for ploting with heat maps
#write a function that counts the number of predictor selected in all stations and store in csv file






