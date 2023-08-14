# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 21:05:43 2023

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



from1979to2010 = pd.date_range(start="1979-01-01", end="2010-12-31", freq="MS")

from2011to2018 = pd.date_range(start="2011-01-01", end="2018-12-31", freq="MS")