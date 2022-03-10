#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 16:52:13 2022

@author: dboateng
"""

# importing models
from sklearn.model_selection import KFold, LeaveOneOut, LeaveOneGroupOut, RepeatedKFold, TimeSeriesSplit
import numpy as np 

class Splitter():
    def __init__(self, method, shuffle=False, n_splits=5):
        self.method = method
        self.shuffle = shuffle
        self.n_splits = n_splits
        self.random_state = None
        
        if self.method == "Kfold":
            self.estimator = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        elif self.method == "LeaveOneOut":
            self.estimator = LeaveOneOut()
        elif self.method == "LeaveOneGroupOut":
            self.estimator = LeaveOneGroupOut()
        elif self.method == "RepeatedKFold":
            self.estimator = RepeatedKFold()
        elif self.method == "TimeSeriesSplit":
            self.estimator = TimeSeriesSplit()
        
        else: 
            raise ValueError("Invalid splitter method might have been defined")
            
            
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.method.get_n_splits(X, y, groups)
    
    def split(self, X, y=None, groups=None):
        return self.method.split(X, y, groups)
    
class MonthlyBooststrapper():
    def __init__(self, n_splits=500, test_size=0.1, block_size=12):
        self.n_splits = n_splits
        self.test_size = test_size
        self.block_size = int(block_size)
        
    def split(self, X, y, groups=None):
        """
        num_blocks * block_size = test_size*num_samples 
        --> n_blocks = test_size/block_size*num_samples

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        groups : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        num_samples = len(y)
        num_blocks = round(self.test_size/self.block_size * num_samples)
        
        for i in range(self.n_splits):
            test_mask = np.zeros(num_samples, dtype=np.bool)
            
            for k in range(num_blocks):
                train_mask = np.zeros(num_samples, dtype=np.bool)
                
                for j in range(num_samples - self.block_size):
                    train_mask[j] = not test_mask[j] and not test_mask[j+self.block_size]
                    
                train = np.where(train_mask)[0]
                rand = np.random.choice(train)
                test_mask[rand:rand + 12] = True
            train= np.where(~test_mask)
            test = np.where(test_mask)
            yield train, test
                    