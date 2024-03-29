# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 16:56:46 2022.

@author: dboateng
"""

import pandas as pd 


predictors = ["t2m", "tp","msl", "v10", "u10", 
              "EAWR", "SCAN", "NAO", "EA",
              "u250", "u850", "u500","u700", "u1000",
              "v250", "v850", "v500","v700", "v1000",
              "r250", "r850", "r500","r700", "r1000", 
              "z250", "z500", "z700", "z850", "z1000", 
              "t250", "t850", "t500","t700", "t1000",
              "dtd250", "dtd850", "dtd500","dtd700", 
              "dtd1000"
              ]
               

predictors_without_tp = ["t2m","msl", "v10", "u10", 
              "EAWR", "SCAN", "NAO", "EA",
              "u250", "u850", "u500","u700", "u1000",
              "v250", "v850", "v500","v700", "v1000",
              "r250", "r850", "r500","r700", "r1000", 
              "z250", "z500", "z700", "z850", "z1000", 
              "t250", "t850", "t500","t700", "t1000",
              "dtd250", "dtd850", "dtd500","dtd700", 
              "dtd1000"
              ]

predictors_without_t2m = ["tp","msl", "v10", "u10", 
              "EAWR", "SCAN", "NAO", "EA",
              "u250", "u850", "u500","u700", "u1000",
              "v250", "v850", "v500","v700", "v1000",
              "r250", "r850", "r500","r700", "r1000", 
              "z250", "z500", "z700", "z850", "z1000", 
              "t250", "t850", "t500","t700", "t1000",
              "dtd250", "dtd850", "dtd500","dtd700", 
              "dtd1000"
              ]


from1958to2010 = pd.date_range(start="1958-01-01", end="2010-12-31", freq="MS")

from2011to2020 = pd.date_range(start="2011-01-01", end="2020-12-31", freq="MS")

from1958to2020 = pd.date_range(start="1958-01-01", end="2020-12-31", freq="MS")

fullAMIP = pd.date_range(start='1979-01-01', end='2000-12-31', freq='MS')

fullCMIP5 = pd.date_range(start='2010-01-01', end='2100-12-31', freq='MS')

from2020to2040 = pd.date_range(start='2020-01-01', end='2040-12-31', freq='MS')
from2040to2060 = pd.date_range(start='2040-01-01', end='2060-12-31', freq='MS')
from2060to2080 = pd.date_range(start='2060-01-01', end='2080-12-31', freq='MS')
from2080to2100 = pd.date_range(start='2080-01-01', end='2100-12-31', freq='MS')


from2040to2070 = pd.date_range(start='2040-01-01', end='2070-12-31', freq='MS')
from2070to2100 = pd.date_range(start='2070-01-01', end='2100-12-31', freq='MS')