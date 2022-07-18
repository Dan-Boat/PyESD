# -*- coding: utf-8 -*-
"""
Created on Mon May 16 15:59:40 2022

@author: dboateng
"""


import pandas as pd 
import numpy as np 

dates = pd.date_range(start='2016-01', end= "2018-12", freq='MS')
df = pd.DataFrame({'num': np.arange(len(dates))}, index=dates)


by_season = df.resample("MS").mean()
print(by_season)
print(by_season.index.month)