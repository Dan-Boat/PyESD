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

Common issues
-------------
**Installing and Using Cartopy for Data Visualization**

Visualizing large-scale datasets with cartopy can be a bit challenging. However,
if you plan to use cartopy, keep in mind that it's not necessary for any of the downscaling 
routines. Here's a step-by-step guide to set it up:

1. Create a Cartopy Environment:
   First, create a new environment specifically for cartopy:
   
   .. code-block:: bash
   
      conda create --name cartopy_env

2. Install Cartopy:
   Once the environment is set up, install cartopy using the conda-forge channel:
   
   .. code-block:: bash
   
      conda install -c conda-forge cartopy

3. Clone Cartopy Environment:
   When creating an environment ("env_name") for installing pyESD, you can clone the previously created cartopy environment to include its settings:
   
   .. code-block:: bash
   
      conda create --name env_name --clone cartopy_env

Keep in mind that cartopy has specific dependencies that need to be installed on your system. 
For more detailed information about these dependencies and how to set up cartopy, you can refer 
to the cartopy website: `Cartopy Documentation <https://scitools.org.uk/cartopy/docs/latest/>`_. This will provide you with a comprehensive understanding of the setup process.


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

