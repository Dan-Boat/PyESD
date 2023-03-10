# -*- coding: utf-8 -*-
"""
Created on Fri May 13 14:23:23 2022

@author: dboateng
"""
import unittest
import numpy as np 

from pyESD.standardizer import MonthlyStandardizer, PCAScaling, StandardScaling

from test_syn_data import generate_syn_data

X,y = generate_syn_data()


class Test_Standardizer(unittest.TestCase):
    
    @classmethod
    def setUp(self):
        self.ms = MonthlyStandardizer(detrending=False)
        self.pca = PCAScaling(method="PCA")
        self.ss = StandardScaling(method="standardscaler")
        
        
    def test_fit_transform_ms(self):
        X_st = self.ms.fit_transform(X)
        
        self.assertFalse(np.all(X_st == X), "the monthly standardizer is not working, check fit")
        
    
    def test_fit_transform_pca(self):
        X_st = self.pca.fit_transform(X)
        
        self.assertFalse(np.all(X_st == X), "the monthly standardizer is not working, check fit")
    
    def test_fit_transform_ss(self):
        X_st = self.ss.fit_transform(X)
        
        self.assertFalse(np.all(X_st == X), "the monthly standardizer is not working, check fit")
        
        
        
if __name__ == '__main__': 
    unittest.main()