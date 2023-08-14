# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 20:23:57 2023

@author: dboateng
"""

import os 
import pandas as pd 
import numpy as np 
from copy import copy

from pyESD.plot_utils import prediction_example_data
from pyESD.ESD_utils import load_csv, store_csv


stationnames = ["Beograd", "Kikinda", "Novi_Sad", "Palic", "Sombor", "Sremska_Mitrovica", "Vrsac",
                "Zrenjanin"]


path_to_store = "C:/Users/dboateng/Desktop/Datasets/Station/Vojvodina_new/plots/predicted"
path_to_send = "C:/Users/dboateng/Desktop/Datasets/Station/Vojvodina_new/plots/Send_to_Rob"

estimators = ["LassoLarsCV", "ARD", "RandomForest", "XGBoost", "Bagging", "AdaBoost", "RidgeCV", "Stacking"]

df = pd.DataFrame(columns=estimators)


for station in stationnames:
    for estimator in estimators:

        filename = "predictions_" + estimator
        data = load_csv(station, filename, path_to_store)
        test_predictions = data["pred_test"].dropna()
        
        np.savetxt(station + "_" + estimator + ".txt", test_predictions.values,
                   delimiter='\t')
        
    #     df[estimator] = test_predictions
    
    # store_csv(station, "test_predictions", df, path_to_store)