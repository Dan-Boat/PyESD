Installation
============

Install from PyPI
-----------------
- The standard version can be intalled with::
    ``pip install pyESD``

- The same version can be installed in editable mode with the e flag::
    ``pip install -e pyESD``

Install from Github
-------------------
The updated version (in development) can be clone from github:
    ``git clone git@github.com:Dan-Boat/PyESD.git``
    ``pip install .`` (in the clone folder)

It is recommended to install the package in an isolated environment.
Virtualenv or conda can be used to create a new environment. 
The package requires some dependencies that can be installed through the distro. If failed to be install
through pip installation, the following modules would require manual installation.


Dependencies
------------
- sklearn (``pip install -U scikit-learn``)
- xarray (``conda install -c conda-forge xarray dask netCDF4 bottleneck``)
- pandas
- seaborn 
- tensorflow (``pip install tensorflow``)
- matplotlib
- netCDF4
- eofs (``pip install eofs``)
- cartopy (``conda install -c conda-forge cartopy``) (not required for the modelling routines), optional
- xgboost (``pip install xgboost``)
- scikit-optimize (``pip install scikit-optimize``)

