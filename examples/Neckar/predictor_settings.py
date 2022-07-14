# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 16:56:46 2022.

@author: dboateng
"""

import pandas as pd 

predictors = ["NAO", "msl", "EAWR", "SCAN", "EA", "u850"]

# predictors = ["t2m", "tp","msl", "v10", "u10", 
#               "EAWR", "SCAN", "NAO", "EA",
#               "u250", "u850", "u500","u700", "u1000",
#               "v250", "v850", "v500","v700", "v1000",
#               "r250", "r850", "r500","r700", "r1000", 
#               "z250", "z500", "z700", "z850", "z1000", 
#               "t250", "t850", "t500","t700", "t1000",
#               ]
               

predictors_amip = ["t2m", "tp","msl", "v10", "u10", "r850", "r500",
               "u850", "u500", "v850", "v500", "d2m", "NAO", "t250", "u250", "u700", "u1000", "v250", "v700", "v1000", "r250", "r700", "r1000",
               "EA", "SCAN", "EAWR"]


from1958to2010 = pd.date_range(start="1958-01-01", end="2010-12-31", freq="MS")

from2011to2020 = pd.date_range(start="2011-01-01", end="2020-12-31", freq="MS")

from1958to2020 = pd.date_range(start="1958-01-01", end="2020-12-31", freq="MS")

fullAMIP = pd.date_range(start='1979-01-01', end='2008-12-31', freq='MS')

fullCMIP5 = pd.date_range(start='2006-01-01', end='2100-12-31', freq='MS')