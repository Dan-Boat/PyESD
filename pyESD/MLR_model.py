# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 17:28:48 2022

@author: dboateng

This module contains the regression routines. There are three layers for
bootstrapped forward selection regression:

- The ``BootstrappedRegression`` class is the outer layer. This implements the
  bootstrapping loop. This class has "regressor" member that implements the
  single regression step (i.e. a fit and a predict method). This can be the a
  ``ForwardSelection`` object, but can also be ``Lasso`` from sklearn or
  similar routines.
- The ``ForwardSelection`` class is the next layer. This class implements a
  Forward Selection loop. This again has a regressor object that has to
  implement ``get_coefs``, ``set_coefs``, and ``average_coefs``. Additionally
  the regressor object has to implement ``fit_active``, ``fit``, and
  ``predict``.
  An example of such a regressor object is ``MultipleLSRegression``.
  
"""

from copy import copy
import numpy as np
import pandas as pd
#import statsmodels.api as sm
from sklearn.model_selection import check_cv
from sklearn.linear_model import LinearRegression


__all__ =  ["BootstrappedRegression",
            "ForwardSelection",
            "BootstrappedForwardSelection",
            "MultipleLSRegression",
]

class MetaEstimator:
    """
    Meta estimators are classes that implement ``fit`` and ``predict``, but don't perform the
    regression themselves. They all have a member variable ``regressor`` that performs this task.
    This class contains getters and setters that are simply the ones of the regressor object.
    """
    
    def get_coefs(self):
        return self.regressor.get_coefs()

    def set_coefs(self, coefs):
        self.regressor.set_coefs(coefs)
        # set coef_ and intercept_ for ease of use
        if hasattr(self.regressor, 'coef_'):
            self.coef_ = self.regressor.coef_
        if hasattr(self.regressor, 'intercept_'):
            self.intercept_ = self.regressor.intercept_
        if hasattr(self.regressor, 'coefs_'):
            self.coefs_ = self.regressor.coefs_
        if hasattr(self.regressor, 'intercepts_'):
            self.intercepts_ = self.regressor.intercepts_

    def get_params(self):
        return self.regressor.get_params()

    def set_params(self, **params):
        self.regressor.set_params(**params)



###############################################################################################
# BootstrappedRegression and ForwardSelection
###############################################################################################

class BootstrappedRegression(MetaEstimator):
    """
    Performs a regression in a bootstrapping loop.

    This splits the data multiple times into training and test data and
    performs a regression for each split. In each loop
    the calculated parameters are stored. The final model uses the average of
    all predictors.
    If the model is a ``LinearModel`` from sklearn (i.e. it has the attributes
    ``coef_`` and ``intercept_``), the averaging routine does not have to be
    implemented. However, it can be implemented if something else than a
    arithmetic mean should be used (e.g. if only the average of robust
    predictors should be taken and everything else should be set to zero).

    Since this inherites from sklearn modules, it can to some extent be used
    interchangibly with other sklearn regressors.


    Parameters
    ----------
    regressor : regression object
        This should be an object similar to sklearn-like regressors that
        provides the methods ``fit(self, X_train, y_train, X_test, y_test)``
        and ``predict(self, X)``.
        This must also provide the methods ``get_coefs(self)``,
        ``set_coefs(self, coefs)``, and ``average_coefs(self, list_of_coefs)``.
        An example of this is ``ForwardSelection`` below.
        The regressor can also have a member variable ``additional_results``,
        which should be a dictionary of parameters that are calculated during
        fitting but not needed for predicting, for example metrics like the
        explained variance of predictors. In this case the regressor also needs
        the method ``average_additional_results(self, list_of_dicts)`` and
        ``set_additional_results(self, mean_additional_results)``.
    cv : integer or cross-validation generator (optional, default: None)
        This determines how the data are split:

        * If ``cv=None``, 3-fold cross-validation will be used.
        * If ``cv=n`` where ``n`` is an integer, n-fold cross-validation will be used.
        * If ``cv=some_object``, where ``some_object`` implements a
          ``some_object.split(X, y)`` method that returns indices for training
          and test set, this will be used. It is recommended to use
          ``YearlyBootstrapper()`` from ``stat_downscaling_tools.bootstrap``.

    Attributes
    ----------
    mean_coefs : type and shape depends on regressor, (only after fitting)
        Fitted coefficients (mean of all models where the coefficients
        were nonzero).
    cv_error : float (only after fitting)
        Mean of errors on test sets during bootstrapping loop.

    If the regressor object has the attributes ``intercept_`` and ``coef_``,
    these will also be set here.
    """

    def __init__(self, regressor, cv=None):
        self.regressor = regressor
        self.cv = cv
        self.has_additional_params = hasattr(self.regressor, 'additional_params')

    def fit(self, X, y):
        """
        Fits a model in a bootstrapping loop.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame of predictors
        y : pd.Series
            Series of predictand
        """
        
        #assign cross validation scheme
        cv = check_cv(self.cv, classifier=False)

        # number of splits in cross-validation:
        # when using cross-validaton procedures from sklearn, there is sometimes
        # no attribute cv.n_splits, therefore n_splits has to be get with the 
        # built-in method cv.get_n_splits()
        if not hasattr(cv, 'n_splits'):
            n_splits = cv.get_n_splits(X)
        # otherwise n_splits is just taken directly:
        else:
            n_splits = cv.n_splits

        cv_error = np.zeros(n_splits)
        coefs = []
        if self.has_additional_results:
            additional_results = []

        for k, (train, test) in enumerate(cv.split(X, y)):
            # standardize
            X_train = X.values[train]
            X_test = X.values[test]
            y_train = y.values[train]
            y_test = y.values[test]
            # Regression
            self.regressor.fit(X_train, y_train, X_test, y_test)
            # get coefficients and error
            coefs.append(self.regressor.get_coefs())
            if self.has_additional_results:
                additional_results.append(copy(self.regressor.additional_results))
            cv_error[k] = np.sqrt(np.mean((y_test - self.regressor.predict(X_test))**2))


        # average the coefficients and the additional parameters
        self.mean_coefs = self.regressor.average_coefs(coefs)
        self.regressor.set_coefs(self.mean_coefs)
        self.cv_error = np.mean(cv_error)
        if self.has_additional_results:
            mean_add_results = self.regressor.average_additional_results(additional_results)
            self.regressor.set_additional_results(mean_add_results)
        # set coef_ and intercept_ for ease of use
        if hasattr(self.regressor, 'coef_'):
            self.coef_ = self.regressor.coef_
        if hasattr(self.regressor, 'intercept_'):
            self.intercept_ = self.regressor.intercept_



    def predict(self, X):
        """
        Predicts values from previously fitted coefficients.

        If the input X is a pandas DataFrame or Series, a Series is returned,
        otherwise only a numpy array.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        y : pd.Series
        """
        y = self.regressor.predict(X)
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            return pd.Series(data=y, index=X.index)
        else:
            return y

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)


class ForwardSelection(MetaEstimator):
    """
    Performs a forward selection regression.

    This stepwise selects the next most promising candidate predictor and adds
    it to the model if it is good enough. The method is outlined in
    "Statistical Analysis in Climate Research" (von Storch, 1999).

    Since this object is intended to be used in the BootstrappedRegression
    class, it implements all necessary methods.

    Parameters
    ----------
    regressor : regression object
        This should be an object similar to sklearn-like regressors that
        provides the methods fit and predict. Furthermore, it must also provide
        the methods ``get_coefs``, ``set_coefs``, ``average_coefs``, and ``fit_active``.
        An example of this is ``MultipleLSRegression`` below.
    min_explained_variance : float, optional (default: 0.02)
        If inclusion of the staged predictor doesn't improve the explained
        variance on the test set by at least this amount, stop the selection
        process.

    Attributes
    ----------
    explaned_variances : numpy array



    """

    def __init__(self, regressor, min_explained_variance=0.02):
        self.regressor = regressor
        self.min_explained_variance = min_explained_variance
        self.additional_results = {}


    def fit(self, X_train, y_train, X_test, y_test):
        """
        Cross-validated forward selection. This fits a regression model
        according to the following algorithm:

        1) Start with yhat = mean(y), res = y - yhat, active = []
        2) for each predictor in inactive set:
               - add to active set
               - perform regression
               - get error and uncertainty of error (standard deviation)
               - remove from active set
        3) add predictor with lowest error on test set to active set
        4) if improvement was not good enough, abort and use previous model.

        Parameters
        ----------
        X_train : numpy array of shape #samples x #predictors
            Array that holds the values of the predictors (columns) at
            different times (rows) for the training dataset.
        y_train : numpy array of length #samples
            Training predictand data
        X_test : numpy array of shape #samples x #predictors
            Test predictor data
        y_test : numpy array of length #samples
            Test predictand data

        Returns
        -------
        exp_var : numpy array of length #predictors
            explained variance of each predictor
        """
        # get some memory
        n_samples, n_predictors = X_train.shape

        X_train = np.array(X_train) # deep copy
        X_test = np.array(X_test) # deep copy

        # I store the index time series in a dictionary, so I can easily
        # remove the ones we already have and at the same time keep the number
        # of the index so I can set the correct coefficient.
        X_inactive = {i:X_train[:,i] for i in range(n_predictors)}
        active = []

        # initial model: no active predictor
        self.fit_active(X_train, y_train, active)
        #X_test_active = _get_active(X_test, active)
        
        residual_test = y_test - self.regressor.predict(X_test)
        SST = np.mean(residual_test**2)
        explained_variance = 0
        exp_var_predictors = np.zeros(n_predictors)

        error = np.zeros(n_predictors)
        old_coefs = self.regressor.get_coefs()

        for k in range(n_predictors):
            # perform regression with all predictors in inactive set
            inactive_mse_test = []
            inactive_mse_train = []
            inactive_coefs = []
            inactive = []
            for idx in X_inactive:
                inactive.append(idx)
                active.append(idx)
                self.fit_active(X_train, y_train, active)
                inactive_coefs.append(self.regressor.get_coefs())
                residual_test = y_test - self.regressor.predict(X_test)  # turn the reset in  _validate_data to False in sklean base
                residual_train = y_train - self.regressor.predict(X_train)
                inactive_mse_test.append(np.mean(residual_train**2))
                inactive_mse_train.append(np.mean(residual_train**2))
                active.pop()
            # find best predictor and add to active set/remove from inactive set
            imax = np.argmin(inactive_mse_train)
            idxmax = inactive[imax]
            SSE = inactive_mse_test[imax]
            new_explained_variance = 1 - SSE/SST
            delta_exp_var = new_explained_variance - explained_variance
            if delta_exp_var < self.min_explained_variance:
                # abort
                self.regressor.set_coefs(old_coefs)
                break
            else:
                # set the current best parameters as old parameters and start again
                old_coefs = inactive_coefs[imax]
                self.regressor.set_coefs(old_coefs)
                active.append(idxmax)
                del X_inactive[idxmax]
                exp_var_predictors[idxmax] = delta_exp_var
                explained_variance = new_explained_variance

        # done we only need to set the explained variance
        self.additional_results['explained variances'] = exp_var_predictors


    def fit_active(self, X, y, active):
        """
        Fits using only the columns of X whose index is in ``active``.
        """
        X_active = _get_active(X, active)
        self.regressor.fit(X_active, y)
        self.regressor.set_expand_coefs(active, X.shape[1])

    def predict(self, X):
        return self.regressor.predict(X)
    
    def predict_active(self, X, active):
        X_active = _get_active(X, active)
        return self.regressor.predict(X_active)

    def set_additional_results(self, add_results):
        self.additional_results = copy(add_results)
        self.explained_variances = add_results['explained variances']

    def average_additional_results(self, list_of_params):
        n_params = len(list_of_params)
        n_predictors = len(list_of_params[0]['explained variances'])
        exp_var = np.zeros((n_params, n_predictors))
        for i, p in enumerate(list_of_params):
            exp_var[i,:] = p['explained variances']
        add_results = {'explained variances':robust_average(exp_var)}
        return add_results

    def average_coefs(self, list_of_coefs):
        return self.regressor.average_coefs(list_of_coefs)

###############################################################################################
# Regressors for Forward Selection
###############################################################################################


class LinearCoefsHandlerMixin:

    def get_coefs(self):
        """
        Returns all fitted coefficients in the same order as ``set_coefs`` needs them.
        """
        coefs = np.zeros(len(self.coef_) + 1)
        coefs[0:-1] = self.coef_
        coefs[-1] = self.intercept_
        return coefs

    def set_coefs(self, coefs):
        self.coef_ = coefs[0:-1]
        self.intercept_ = coefs[-1]

    def average_coefs(self, list_of_coefs):
        """
        Calculates the average of robust predictors, i.e. of those that are
        nonzero in at least 50% of the cases.
        """
        coefs = np.asarray(list_of_coefs)
        return robust_average(coefs)

    def get_params(self):
        return {}

    def set_params(self, **params):
        pass

    def fit_active(self, X, y, active):
        """
        Fits using only the columns of X whose index is in ``active``.
        """
        X_active = _get_active(X, active)
        self.fit(X_active, y)
        self.set_expand_coefs(active, X.shape[1])
        
    def predict_active(self, X, active):
        """
        Predict using only the columns of X whose index is in ``active``.
        """
        X_active = _get_active(X, active)
        self.predict(X_active)


class MultipleLSRegression(LinearCoefsHandlerMixin):
    """
    Implementation of multiple linear OLS regression to be used with
    ForwardSelection and BootstrappedRegression.
    The following methods are implemented:

    - ``fit``
    - ``predict``
    - ``get_coefs``
    - ``set_coefs``
    - ``average_coefs``
    - ``fit_active``
    """

    def __init__(self):
        self.lm = LinearRegression()

    def fit(self, X, y):
        if X.shape[1] == 0:
            # only intercept model
            self.intercept_ = np.mean(y)
        else:
            self.lm.fit(X, y)

    def predict(self, X):
        n, m = X.shape
        self.lm.coef_ = self.coef_
        self.lm.intercept_ = self.intercept_
        return self.lm.predict(X)

    def set_expand_coefs(self, active, n_predictors):
        """
        This will be called after ``fit``, since fit will often be called with only some of the
        predictors. This expands the current coefficients and expands them in a way such that
        ``predict`` can be called with all predictors.
        """
        # get a full coefficient vector back
        coefs = np.zeros(n_predictors)
        for i, idx in enumerate(active):
            coefs[idx] = self.lm.coef_[i]
        self.coef_ = coefs

        # get intercept_: if len(active) == 0, the intercept was already set
        if len(active) != 0:
            self.intercept_ = self.lm.intercept_


# class GammaRegression(LinearCoefsHandlerMixin):
#     """
#     Implementation of generalized linear gamma regression.
#     """

#     def __init__(self, family=sm.families.Gamma()):
#         self.family = family

#     def fit(self, X, y):
#         n, m = X.shape
#         G = np.zeros((n, m+1))
#         G[:,0:-1] = X
#         G[:,-1] = np.ones(n)
#         self.glm = sm.GLM(y, G, family=self.family)
#         gamma_results = self.glm.fit()
#         self.coef_ = gamma_results.params[0:-1]
#         self.intercept_ = gamma_results.params[-1]

#     def predict(self, X):
#         n, m = X.shape
#         G = np.zeros((n, m+1))
#         G[:,0:-1] = X
#         G[:,-1] = np.ones(n)
#         params = np.zeros(len(self.coef_) + 1)
#         params[0:-1] = self.coef_
#         params[-1] = self.intercept_
#         return self.glm.predict(params, exog=G)


#     def set_expand_coefs(self, active, n_predictors):
#         # get a full coefficient vector back
#         coefs = np.zeros(n_predictors)
#         for i, idx in enumerate(active):
#             coefs[idx] = self.coef_[i]
#         self.coef_ = coefs



###############################################################################################
# Some functions
###############################################################################################

def _get_active(X, active):
    """
    Returns a new matrix X_active with only the columns of X whose index is in ``active``.
    """
    Xnew = np.empty((X.shape[0], len(active)))
    for i, idx in enumerate(active):
        Xnew[:,i] = X[:,idx]
    return Xnew

def robust_average(coefs):
    """
    Takes the robust average of a coefficient matrix.

    Parameters
    ----------
    coefs : numpy 2d-array, n_coefs x n_predictors

    Returns
    -------
    mean_coefs : numpy array, length n_predictors
    """
    n_coefs, n_predictors = coefs.shape
    mean_coefs = np.zeros(n_predictors)
    for i in range(n_predictors):
        c = coefs[:,i] # one column
        if len(c[c != 0]) > 0.5*len(c):
            mean_coefs[i] = np.mean(c)
    return mean_coefs



###############################################################################################
# Easy to use classes
###############################################################################################

class BootstrappedForwardSelection(BootstrappedRegression):
    """
    This is an easy to use interface for BootstrappedRegression with ForwardSelection.

    Parameters
    ----------
    regressor : regression object
        This should be an object similar to sklearn-like regressors that
        provides the methods fit and predict. Furthermore, it must also provide
        the methods ``get_coefs``, ``set_coefs``, ``average_coefs``, and ``fit_active``.
        An example of this is ``MultipleLSRegression`` below.
    min_explained_variance : float, optional (default: 0.02)
        If inclusion of the staged predictor doesn't improve the explained
        variance on the test set by at least this amount, stop the selection
        process.
    cv : integer or cross-validation generator (optional, default: None)
        This determines how the data are split:

        * If ``cv=None``, 3-fold cross-validation will be used.
        * If ``cv=n`` where ``n`` is an integer, n-fold cross-validation will be used.
        * If ``cv=some_object``, where ``some_object`` implements a
          ``some_object.split(X, y)`` method that returns indices for training
          and test set, this will be used. It is recommended to use
          ``YearlyBootstrapper()`` from ``stat_downscaling_tools.bootstrap``.
    """

    def __init__(self, regressor, min_explained_variance=0.02, cv=None):
        self.regressor = ForwardSelection(regressor, min_explained_variance)
        self.cv = cv
        self.has_additional_results = True
