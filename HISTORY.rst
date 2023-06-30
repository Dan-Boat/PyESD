=======
History
=======

1.0.7 (2023-06-30)
-------------------

Bug fixes
~~~~~~~~~~
* Fix the issues with installation on pypi

New indicators
--------------
* added additional teleconnection indicies for Southern Hemisphere
* Test the framework with daily data but reqiure additional models like wet day classifiers
* missing data in the predictand values can be fill with the implemented impute methods

To do
------
* Extend the package to use spatial datasets
* Implement wet and dry day classifiers before model training
* implement the various calibration metric for daily data 
* Add simple MOS method (quantile mapping)