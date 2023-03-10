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


estimators = []

for i in range(len(regressors)):
    
    regressor = Regressors(method= regressors[i], cv=5)
    regressor.set_model()
    estimators.append((regressors[i], regressor.estimator))
    


class TestEnsembleRegressor(unittest.TestCase):
    
    @classmethod
    def setUp(self):
        self.ensemble = EnsembleRegressor(estimators=estimators, cv=5, 
                                          method=method)
        
        self.ensemble.fit(X,y)
        
    def test_fit(self):
        
        self.assertIsNone(check_is_fitted(self.ensemble.ensemble, "estimators_"), 
                          "The ensemble regressor might not be fitted")
        
    
    def test_score(self):
        score = self.ensemble.score(X,y)
        
        self.assertGreaterEqual(score, 0.99,
                                "The model is not well calibrating, check parameters")
    
    def test_predict(self):
        yhat = self.ensemble.predict(X)
        
        self.assertGreaterEqual(np.corrcoef(y,yhat)[0,1], 0.99,
                               "The model is not well calibrating, check parameters")
        pass
    
    def test_cross_val_score(self):
        
        val_score = self.ensemble.cross_val_score(X, y)
        
        self.assertGreaterEqual(np.mean(val_score), 0.99,
                                "The model is not well calibrating, check parameters")
        
if __name__ == '__main__': 
    unittest.main()