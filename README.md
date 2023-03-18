**Python Package for Empirical Statistical Downscaling (v1.01) :sun_behind_rain_cloud: :cloud_with_snow: :cloud_with_rain: :fire: :thermometer:** 

**_PyESD_** is an open-source framework of the Perfect Prognosis approach of statistical downscaling of any climate-related variable such as precipitation, temperature, and wind speed using reanalysis products eg. ERA5 as predictors. The package features all the downscaling cycles including data preprocessing, predictor selection, constructions (eg. using transformers), model selection, training, validation and evaluation, and future prediction. The package serves as the means of downscaling General Circulation Models of future climate to high resolution relevant for climate impact assessment such as droughts, flooding, wildfire risk, and others. 
The main specialties of the pyESD include:

* Well designed in an OOP style that considers weather stations as individual objects and all the downscaling routines as attributes. This ensures fewer lines of code that cover the end-to-end downscaling of climate change variable products.
* PyESD features many machine learning algorithms and predictor selection techniques that can be experimented with toward the selection and design of robust transfer functions that can be coupled with GCM to generate future estimates of climate change
* And many other functionalities that are highlighted in the paper description of the package (to be submitted). 

The main component and the work flow of the package are summarised in the modeling outline:

![Model outline](./img/Model%20outline-model%20outline%20presentation.drawio.png)

**Installation :hammer_and_wrench:**
1. Install the standard version:
   `pip install pyESD` from PyPI or git clone git@github.com:Dan-Boat/PyESD.git
   `cd to the folder` | `pip install .`

2. Install in editable mode:
   `pip install -e pyESD` or `pip install -e .` in the package base folder clone from github
The installation might require some dependencies that must be installed if not successful from the distribution from PyPI: [cartopy](https://scitools.org.uk/cartopy/docs/latest/), [xarray](https://docs.xarray.dev/en/stable/), [sciki-learn](https://scikit-learn.org/stable/), [scipy](https://scipy.org/) and the other scientific frameworks such as NumPy, pandas, [Matplotlib](https://matplotlib.org/), and [seaborn](https://seaborn.pydata.org/)
3. Alternatively, to ensure the installation in an isolated environment, virtual python environment using `conda` or `virtualenv` can be used to create a separate env for the package installation
## Documentation :blue_book:

The package documentation is still in progress. The initial structure is accessible at [github-pages](https://dan-boat.github.io/PyESD/)

## Examples
The package has been used for downscaling precipitation and temperature for a catchment located in southwestern Germnany. We have also used it for generating future rainfall products for all the synoptic weather stations in Ghana. Their respective control scripts are located in the [examples folder](./pyESD/examples/). Generally, the control scripts follow the modeling workflow as shown in:
![Downscaling steps](./img/Model%20outline-Page-1.drawio.png). 
For instance, the downscaling framework show below can be experimented with to select the robust predictor selection method and emprical transfer function for a specific location and predictand variable.
![modeling framework](./img/Model%20outline-Page-2.drawio.png)

**Workflow demonstration**: To use the PP-ESD model to downscale climate model, weather station and reanalysis datasets are required. The predictors are loaded in as netCDF files and the predictand as csv file. Let assume that the various predictor variables are stored locally in the `era5_datadir `directory ``/home/daniel/ERA5/`` and the predictand variable eg. precipitation is stored in `station_dir` The files should have the same timestamp as the interested predictand variable
1. import all the required modules
```
from pyESD.Weatherstation import read_station_csv
from pyESD.standardizer import MonthlyStandardizer, StandardScaling
from pyESD.ESD_utils import store_pickle, store_csv
from pyESD.splitter import KFold
from pyESD.ESD_utils import Dataset
from pyESD.Weatherstation import read_weatherstationnames

import pandas as pd 
```
2. Read the datasets

```
ERA5Data = Dataset('ERA5', {
    't2m':os.path.join(era5_datadir, 't2m_monthly.nc'),
    'msl':os.path.join(era5_datadir, 'msl_monthly.nc'),
    'u10':os.path.join(era5_datadir, 'u10_monthly.nc'),
    'v10':os.path.join(era5_datadir, 'v10_monthly.nc'),
    'z250':os.path.join(era5_datadir, 'z250_monthly.nc'),
```
3. define potential predictors and radius of predictor construction, time range for model training and evaluation

```
radius = 100 #km
predictors = ["t2m", "tp","msl", "v10", "u10"]

from1958to2010 = pd.date_range(start="1958-01-01", end="2010-12-31", freq="MS") #training and validation

from2011to2020 = pd.date_range(start="2011-01-01", end="2020-12-31", freq="MS")  # testing trained data 
```

4. Read weather stations as objects and apply the downscaling cycle attributes. Note that running the model the first time for a specific location extract the regional means using the define radius and location of the station. The extracted means are stored in a pickel files in the directory called `predictor_dir`

```
variable = "Precipitation"
SO = read_station_csv(filename=station_dir, varname=variable)
        
        
# USING ERA5 DATA
# ================


#setting predictors 
SO.set_predictors(variable, predictors, predictordir, radius,)

#setting standardardizer
SO.set_standardizer(variable, standardizer=MonthlyStandardizer(detrending=False,
                                                               scaling=False))

scoring = ["neg_root_mean_squared_error",
            "r2", "neg_mean_absolute_error"]

#setting model
regressor = "RandomForest"
SO.set_model(variable, method=regressor, 
               daterange=from1958to2010,
               predictor_dataset=ERA5Data, 
               cv=KFold(n_splits=10),
               scoring = scoring)


# MODEL TRAINING (1958-2000)
# ==========================


SO.fit(variable, from1958to2010, ERA5Data, fit_predictors=True, predictor_selector=True, 
            selector_method="Recursive" , selector_regressor="ARD",
            cal_relative_importance=False)
   
score_1958to2010, ypred_1958to2010 = SO.cross_validate_and_predict(variable, from1958to2010, ERA5Data)
   
score_2011to2020 = SO.evaluate(variable, from2011to2020, ERA5Data)

ypred_1958to2010 = SO.predict(variable, from1958to2010, ERA5Data)
   
ypred_2011to2020 = SO.predict(variable, from2011to2020, ERA5Data)

ypred_2011to2020.plot()
plt.show()

```

## Package testing
The package is tested using the `unittest` framework with synthetic generated data. The testing scripts are located in the [test](./test) folder. Running  the various scripts with -v flag (higher level of verbose), would validate the modified version of the package. 

## Publications
The package description and application paper is currently under preparation (to be submitted to GMD)
Its application for weather station in Ghana was presented at the AGU22 [Link](http://dx.doi.org/10.22541/essoar.167458059.91056213/v1) and the paper is under preparation

Citation:
Upload on zenodo: https://doi.org/10.5281/zenodo.7748769

## Future versions
The pacakage is still in planing stage (v.0.0.1) (The stable version would be uploaded on pypi with the version number v.1.0.1)

## *Collaborators are welcomed: interms of model application, model improvement, documentation and expansion of the package!*
@ Daniel Boateng ([linkedin](https://www.linkedin.com/in/daniel-boateng-3892311b4/)) : **University of Tuebingen** :incoming_envelope: dannboateng@gmail.com
