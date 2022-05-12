# -*- coding: utf-8 -*-
"""
Created on Thu May 12 14:06:41 2022

@author: dboateng
"""

import unittest
import numpy as np
import pandas as pd 
from sklearn.linear_model import LassoLarsCV
from sklearn.utils.validation import check_is_fitted


from pyESD.models import Regressors
from test_syn_data import generate_syn_data

X,y = generate_syn_data()


class TestRegressors(unittest.TestCase):
    
    
    @classmethod
    def setUp(self):
        self.regressor_lasso = Regressors(method="LassoLarsCV", cv=5)
        
        self.regressor_mlp = Regressors(method="MLPRegressor", cv=5, 
                                        hyper_method= "GridSearchCV")
    
    @classmethod
    def routine(self):
        self.regressor_lasso.set_model()
        self.regressor_lasso.fit(X,y)
        
        self.regressor_mlp.set_model()
        self.regressor_mlp.fit(X,y)
        
        
    def test_set_model(self):
        self.routine()
        
        self.assertTrue(hasattr(self.regressor_lasso, "estimator"),
                        "the model has not initialise the estimator")
        
        self.assertTrue(hasattr(self.regressor_mlp, "hyper"), 
                        "the set model has not set the hyperoptimize class")
        
    
    def test_fit(self):
        self.routine()
        
        self.assertIsNone(check_is_fitted(self.regressor_lasso.estimator, "coef_"),
                          "the regressor is not fitted or the set_model must be apply first")
        
        self.assertIsNone(check_is_fitted(self.regressor_mlp.estimator, "coefs_"),
                          "the regressor is not fitted or the set_model must be apply first")
    
    def test_score(self):
        self.routine()
        
        score = self.regressor_lasso.score(X,y)
        
        self.assertGreaterEqual(score, 0.99,
                                "The model is not well calibrating, check parameters")
    
        score = self.regressor_mlp.score(X,y)
        
        self.assertGreaterEqual(score, 0.99,
                                "The model is not well calibrating, check parameters")
        pass
    
    def test_predict(self):
        self.routine()
        
        yhat = self.regressor_lasso.predict(X)
        
        self.assertGreaterEqual(np.corrcoef(y,yhat)[0,1], 0.99,
                                "The model is not well calibrating, check parameters")
        
        
        pass
    
    def test_cross_val_score(self):
        pass


unittest.main()