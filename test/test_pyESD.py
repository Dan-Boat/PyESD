# -*- coding: utf-8 -*-
"""
Created on Fri May 13 14:21:49 2022

@author: dboateng
"""

import unittest
import os 
import pandas as pd

from test_syn_data import generate_syn_data, Syn_predictor


from pyESD.ESD_utils import Dataset
from pyESD.StationOperator import StationOperator
from pyESD.predictand import PredictandTimeseries
from pyESD.standardizer import MonthlyStandardizer


X, y = generate_syn_data()

daterange = pd.date_range(start="1979-01-01", end="2000-01-01", freq="MS")

data = Dataset("test_data", {})

cachedir = os.path.join(os.path.dirname(__file__), ".predictors")


class Test_pyESD(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        
        self.SO = PredictandTimeseries(y, standardizer=MonthlyStandardizer())
    
    def test_score(self):        
        #set predictors 

        self.SO.set_predictors([
            Syn_predictor(X, 0, cachedir=cachedir), 
            Syn_predictor(X, 1, cachedir=cachedir), 
            Syn_predictor(X, 2, cachedir=cachedir),
            Syn_predictor(X, 3, cachedir=cachedir), 
            Syn_predictor(X, 4, cachedir=cachedir),
            ])


        self.SO.set_model(method="LassoLarsCV")
        self.SO.fit(daterange= daterange, predictor_dataset= data)
        scores = self.SO.evaluate(daterange= daterange, predictor_dataset= data)
        
        self.assertGreater(scores["r2"], 0.9, "Check if all component of the package is working")
        
if __name__ == '__main__': 
    unittest.main()