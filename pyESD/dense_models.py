# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 12:26:01 2022

@author: dboateng

This module require further development of add deep learning models!
"""

# importing models 

import tensorflow as tf
import tensorflow.keras as keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


class DeepLearningRegressor():
    
    def __init__(self, method=None, optimizer="adam", loss="mean_squared_error", metrics=["RootMeanSquaredError"]):
        self.optimizer = optimizer 
        self.loss = loss
        self.metrics = metrics
        self.method = method      
        
        if self.method == None:
            print(".....Dense Neural Network is used as default regressor......")
            self.method = "Dense"
            
    def build_model(self):
        if self.method == "Dense":
            print("....using default design...edit the dense_model.py if the Package was installed with the edit option")
            
            self.model = Sequential()
            self.model.add(Dense(512, activation="relu", input_dim=13))
            self.model.add(Dense(256, activation="relu"))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(64, activation="relu"))
            self.model.add(Dense(1))
            
        elif self.method == "LSTM":
            
            raise NotImplementedError("LSTM is not implemented yet, check for future version")
            
        elif self.method == "CNN":
            raise NotImplementedError("CNN is not implemented yet, check for future version")
            
        else:
            raise ValueError("Not recognized method")

    
    def plot_network():
        pass
    
    def compile_model(self):
        return self.model.compile(optimizer=self.optimizer, loss = self.loss, metrics=self.metrics)
        
    
    def convert_to_sklearn_regressor(self, epochs=1000, verbose=False):
        self.estimator = tf.keras.wrappers.scikit_learn.KerasRegressor(self.model, epochs= epochs, verbose= verbose)
        self.estimator._estimator_type = "regressor"
        
        return self.estimator
        
    
    def fit(self, X, y):
        
        return self.estimator.fit(X,y)
    
    def predict(self, X):
        yhat = self.estimator.predict(X)
        return yhat

    
    
        
        