# -*- coding: utf-8 -*-
"""
Created on Thu May 12 14:06:41 2022

@author: dboateng
"""

import unittest
import numpy as np
import pandas as pd 


from pyESD.models import Regressors

#test synthetic data (move to function in a different script)

np.random.seed(0)

daterange = pd.date_range(start="1979-01-01", end="2000-01-01", freq="MS")

X = np.random.randn(len(daterange), 5)

# offset one predictor by 20

X[:, 2] += 20

#define factors

m = np.asarray([0, 10, 20, -20, -5], dtype=float)

y = 4 + X.dot(m) + 0.01*np.random.randn(len(daterange))

# passing data into dataframe

X = pd.DataFrame(X, index=daterange)
y = pd.Series(y, index=daterange)


# rational testing

regressor = Regressors(method="LassoLarsCV", cv=5)
regressor.set_model()
regressor.fit(X,y)
score = regressor.score(X,y)
val_score = regressor.cross_val_score(X, y)
yhat = regressor.predict(X)
np.corrcoef(y,yhat)