# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 12:13:25 2022

@author: dboateng
"""

import pandas as pd 


predictors = ["t2m", "tp","msl", "v10", "u10", 
              "u250", "u850", "u500","u700", "u1000",
              "v250", "v850", "v500","v700", "v1000",
              "r250", "r850", "r500","r700", "r1000", 
              "z250", "z500", "z700", "z850", "z1000", 
              "t250", "t850", "t500","t700", "t1000",
              "dtd250", "dtd850", "dtd500","dtd700", "dtd1000",
              "NAO"]



from2013to2017 = pd.date_range(start="2013-01-01", end="2017-12-31", freq="MS")


from1981to2017 = pd.date_range(start="1981-01-01", end="2017-12-31", freq="MS")

from1981to2012 = pd.date_range(start="1981-01-01", end="2012-12-31", freq="MS")

from1961to2012 = pd.date_range(start="1961-01-01", end="2012-12-31", freq="MS")

from1961to2017 = pd.date_range(start="1961-01-01", end="2017-12-31", freq="MS")

fullAMIP = pd.date_range(start='1979-01-01', end='2008-12-31', freq='MS')
fullAMIP2 = pd.date_range(start='1979-01-01', end='2014-12-31', freq='MS')

fullCMIP5 = pd.date_range(start='2010-01-01', end='2100-12-31', freq='MS')

from2020to2040 = pd.date_range(start='2020-01-01', end='2040-12-31', freq='MS')
from2040to2060 = pd.date_range(start='2040-01-01', end='2060-12-31', freq='MS')
from2060to2080 = pd.date_range(start='2060-01-01', end='2080-12-31', freq='MS')
from2080to2100 = pd.date_range(start='2080-01-01', end='2100-12-31', freq='MS')

from2040to2070 = pd.date_range(start='2041-01-01', end='2070-12-31', freq='MS')
from2070to2100 = pd.date_range(start='2070-01-01', end='2100-12-31', freq='MS')