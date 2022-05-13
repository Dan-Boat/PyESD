# -*- coding: utf-8 -*-
"""
Created on Fri May 13 14:22:11 2022

@author: dboateng
"""

import unittest
import numpy as np
from sklearn.utils.validation import check_is_fitted


from pyESD.ensemble_models import EnsembleRegressor
from pyESD.models import Regressors
from test_syn_data import generate_syn_data

X,y = generate_syn_data()


regressors = ["AdaBoost", "LassoLarsCV", "ARD"]
method = "Stacking"

