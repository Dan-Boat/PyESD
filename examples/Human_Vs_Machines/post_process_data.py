# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 17:23:00 2023

@author: dboateng

This script intend to organize the data required for testing the pyESD on daily output
"""
import os 
import pandas as pd 
import numpy as np 

stationnames = ["Beograd", "Kikinda", "Novi_Sad", "Palic", "Sombor", "Sremska_Mitrovica", "Vrsac",
                "Zrenjanin"]



path_to_data = "C:/Users/dboateng/Desktop/Datasets/Station/Vojvodina_new"

def preprocess(path_to_data, filename, stationname, is_train_data=True):
    
    path_to_station = os.path.join(path_to_data, stationname)
    
    data = pd.read_csv(os.path.join(path_to_station, filename), parse_dates=["date"], dayfirst=True,
                            index_col=["date"])
    
    # replace -999 with nans
    
    data = data.replace(-999, np.nan)
    
    
    #get columns
    predictors = data.columns.values.tolist()
    
    predictor_names = []
    
    for name in predictors[:-1]:
        predictor_names.append(name[5:])
    
    
    # get Y data 
    
    data_y = data["precipitation"]
    
    # get X data 
    data_X = data.drop(["precipitation"], axis=1)
    data_X.columns = predictor_names
    
    # Save datasets
    if is_train_data:
        data_X.to_csv(os.path.join(path_to_station, "train_X.csv"), index=True) 
        data_y.to_csv(os.path.join(path_to_station, "train_y.csv"), index=True) 
        
    else:
        data_X.to_csv(os.path.join(path_to_station, "test_X.csv"), index=True) 
        data_y.to_csv(os.path.join(path_to_station, "test_y.csv"), index=True) 

for station in stationnames:
    preprocess(path_to_data, filename="Train.csv", stationname=station, is_train_data=True)
    preprocess(path_to_data, filename="Test.csv", stationname=station, is_train_data=False)