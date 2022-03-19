# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 16:56:46 2022.

@author: dboateng
"""

import pandas as pd 

predictors = ["t2m", "tp","msl", "v10", "u10",'z500', 'z850', 'q850',"q500", "t850","t500", "r850", "r500",
               "vo850", "vo500", "pv850", "pv500", "u850", "u500", "v850", "v500", "d2m", "NAO", 
               "EA", "SCAN", "EAWR"]

predictors_without_indices = ["t2m", "tp","msl", "v10", "u10",'z500', 'z850', 'q850',"q500", "t850","t500", "r850", "r500",
               "vo850", "vo500", "pv850", "pv500", "u850", "u500", "v850", "v500", "d2m"]


from1958to2010 = pd.date_range(start="1958-01-01", end="2010-12-31", freq="MS")

from2011to2020 = pd.date_range(start="2011-01-01", end="2020-12-31", freq="MS")

from1958to2020 = pd.date_range(start="1958-01-01", end="2020-12-31", freq="MS")