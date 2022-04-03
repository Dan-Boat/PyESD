#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 00:55:02 2021

@author: dboateng
"""

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from copy import deepcopy
import numpy as np
import pandas as pd
import scipy.stats as st


def add_seasonal_cycle(t, anomalies, mean):

    """
    Adds a seasonal cycle such that

        X = anomalies + mean_seasonal_cycle

    Parameters
    ----------
    t : numpy array of ints
        time in number of months
    res : numpy array
        Array of standardized values
    mean : array of shape 12 x #columns(res)
        Mean values for each month and each column in res

    Returns
    -------
    X : unstandardized values
    """
    N = len(t)
    M = anomalies.shape[1]
    for i in range(N):
        for j in range(M):
            anomalies[i,j] += mean[t[i] % 12,j]
    return anomalies


def remove_seasonal_cycle(t, X, mean):

    """
    Inverse operation to add_seasonal_cycle
    """
    N = len(t)
    M = X.shape[1]
    for i in range(N):
        for j in range(M):
            X[i,j] -= mean[t[i] % 12,j]
    return X


def get_annual_mean_cycle(t, X):
    N = len(t)
    M = X.shape[1]
    mean = np.zeros((12, M))
    counts = np.zeros((12, M), dtype=np.int)
    for i in range(N):
        for j in range(M):
            if not np.isnan(X[i,j]):
                mean[t[i] % 12,j] += X[i,j]
                counts[t[i] % 12,j] += 1
    for i in range(12):
        for j in range(M):
            mean[i,j] = mean[i,j]/counts[i,j]
    return mean


def get_mean_prediction(t, mean):
    N = len(t)
    mean_prediction = np.zeros(N)
    for i in range(N):
        mean_prediction[i] = mean[t[i] % 12, 0]
    return mean_prediction

class NoStandardizer(BaseEstimator, TransformerMixin):
    """
    This is just a dummy standardizer that does nothing.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

    def inverse_transform(self, X):
        return X


class MonthlyStandardizer(BaseEstimator, TransformerMixin):
    """
    Standardizes monthly data that has a seasonal cycle and possibly a linear trend.

    Since the seasonal cycle might affect the trend estimation, the
    seasonal cycle is removed first (by subtracting the mean annual cycle)
    and the trend is estimated by linear regression. Afterwards the data is
    scaled to variance 1.

    Parameters
    ----------
    detrending : bool, optional (default: False)
        Whether to remove a linear trend
    """

    def __init__(self, detrending=False, scaling=False):
        self.detrending = detrending
        self.scaling = scaling

    def fit(self, X, y=None):
        """
        Fits the standardizer to the provided data, i.e. calculates annual mean
        cycle and trends.

        Parameters
        ----------
        X : pd.DataFrame or pd.Series
            DataFrame or Series which holds the data
        y : dummy (optional, default: None)
            Not used

        Returns
        -------
        self
        """
        # be careful: X might contain NaNs
        t = X.index.values.astype('datetime64[M]').astype(int)
        values = np.array(X.values) # deep copy
        # X might be a Series or a DataFrame
        if values.ndim == 1:
            values = values[:,np.newaxis]
        N, M = values.shape

        # get seasonal mean
        self.mean = get_annual_mean_cycle(t, values)


        # get slope
        if self.detrending:
            # remove seasonal cycle
            values = remove_seasonal_cycle(t, values, self.mean)

            self.slopes = np.zeros(M)
            self.intercepts = np.zeros(M)
            for col in range(M):
                # remove NaNs
                nans = np.isnan(values[:,col])
                x = values[~nans,col]
                time = t[~nans]
                slope, intercept, r, p, stderr = st.linregress(time, x)
                self.slopes[col] = slope
                self.intercepts[col] = intercept
            # remove trend
            values -= self.intercepts + np.outer(t, self.slopes)

        if self.scaling:
            self.std = np.nanstd(values, axis=0)

        return self

    def transform(self, X, y=None):
        """
        Standardizes the values based on the previously calculated parameters
        Parameters
        ----------
        X : pd.DataFrame or pd.Series
            DataFrame or Series which holds the data
        y : dummy (optional, default: None)
            Not used

        Returns
        -------
        X_transformed : pd.DataFrame or pd.Series
            Transformed data
        """
        t = X.index.values.astype('datetime64[M]').astype(int)
        values = np.array(X.values) # deep copy
        # X might be a Series or a DataFrame
        if values.ndim == 1:
            values = values[:,np.newaxis]

        values = remove_seasonal_cycle(t, values, self.mean)
        if self.detrending:
            values -= self.intercepts + np.outer(t, self.slopes) # remove trend

        if self.scaling:
            values /= self.std

        if X.values.ndim == 1:
            return pd.Series(data=values[:,0], index=X.index, name=X.name)
        else:
            return pd.DataFrame(data=values, index=X.index, columns=X.columns)


    def inverse_transform(self, X, y=None):
        """
        De-standardizes the values based on the previously calculated parameters
        Parameters
        ----------
        X : pd.DataFrame or pd.Series
            DataFrame or Series which holds the standardized data
        y : dummy (optional, default: None)
            Not used

        Returns
        -------
        X_unstandardized : pd.DataFrame or pd.Series
            Unstandardized data
        """
        t = X.index.values.astype('datetime64[M]').astype(int)
        values = np.array(X.values)
        # X might be a Series or a DataFrame
        if values.ndim == 1:
            values = values[:,np.newaxis]

        if self.scaling:
            values *= self.std

        # add the trend
        if self.detrending:
            values += self.intercepts + np.outer(t, self.slopes)

        # add seasonal cycle
        values = add_seasonal_cycle(t, values, self.mean)


        if X.values.ndim == 1:
            return pd.Series(data=values[:,0], index=X.index, name=X.name)
        else:
            return pd.DataFrame(data=values, index=X.index, columns=X.columns)


class StandardScaling():
    
    def __init__(self, method=None, with_std=True, with_mean=True, unit_variance=False,
                 norm="l2"):
        self.method = method 
        self.with_std = with_std
        self.with_mean = with_mean
        self.unit_variance = unit_variance
        
        
        if self.method == None or "standardscaler":
            self.standardizer = StandardScaling(with_mean=self.with_mean,
                                                with_std=self.with_std)
        elif self.method == "robustscaler":
            self.standardizer = RobustScaler(with_centering=self.with_mean, with_scaling=self.with_std)
        
        elif self.method == "normalize":
            self.standardizer = Normalizer(norm=self.norm)
        
        elif self.method == "powertransformer":
            self.standardizer = PowerTransformer(method="yeo-johnson", standardize=True)
            
        elif self.method == "quantiletransformer":
            self.standardizer = QuantileTransformer(n_quantiles=1000, output_distribution="unifom")
        
        else:
            raise ValueError("The standardizer do not recognize the defined method")
                
        
    def fit(self, X, y=None):
        
        self.standardizer.fit(X=X, y=y)
        
        return self 
    
    def fit_transform(self, X, y=None):
        
        values = self.standardizer.fit_transform(X=X, y=y)
        
        if X.values.ndim == 1:
            return pd.Series(data=values[:,0], index=X.index, name=X.name)
        else:
            return pd.DataFrame(data=values, index=X.index, columns=X.columns)
        
    
    def inverse_transform(self, X):
        
        values = self.standardizer.inverse_transform(X)
        
        if X.values.ndim == 1:
            return pd.Series(data=values[:,0], index=X.index, name=X.name)
        else:
            return pd.DataFrame(data=values, index=X.index, columns=X.columns)
        
    
    def transform(self, X):
        values = self.standardizer.transform(X)
        
        if X.values.ndim == 1:
            return pd.Series(data=values[:,0], index=X.index, name=X.name)
        else:
            return pd.DataFrame(data=values, index=X.index, columns=X.columns)
        
    
        
class PCAScaling(TransformerMixin, BaseEstimator):
    
    def __init__(self):
        pass
    
    def fit(self):
        pass
    
    def fit_transform(self):
        pass
    
    def inverse_transform(self):
        pass
    
    def transform(self):
        pass 