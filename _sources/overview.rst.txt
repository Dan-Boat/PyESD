.. _overview:

.. # define a hard line break for HTML
.. |br| raw:: html

   <br />

.. # define a double hard line break for HTML
.. |brr| raw:: html

   <br /> <br />

Overview
========
Why is downscaling important?
------------------------------

Downscaling of climate information is crucial because of the widespread and diverse effects of 
human-caused climate change. To better understand climate change impacts, it is essential to generate 
accurate predictions about future climate conditions at a relevant scale for studying its effects and 
creating strategies to address them. General Circulation Models (GCMs) are physics-based numerical models 
that predict future climate patterns and their effects under different assumptions of radiative forcing.
However, they have limitations. While they can replicate many current and past atmospheric processes on
large scales, they struggle with representing smaller-scale processes, like local weather patterns, clouds, 
and certain climate variables, due to their coarse resolution. Additionally, they can't adequately capture local
and regional climate variations. To overcome these limitations, GCM simulations need to be downscaled, 
allowing us to predict regional climates more accurately.

What is the Perfect Prognosis?
------------------------------

Empirical Statistical Downscaling models fall into two categories: Model Output Statistics (MOS)
and Perfect Prognosis (PP). MOS uses GCM data directly to create a model with bias correction 
techniques for downscaling. However, it's inflexible because it's tied to specific GCM products.
On the other hand, PP-ESD trains the downscaling model using weather stations and large-scale
observations like reanalysis products and then connects the trained model to any GCM product 
for predicting downscaled future climates. PP establishes a relationship between larger observed 
patterns and local data, acting as a transfer function for predictions. While PP is more complex
to design and requires substantial modeling, it offers flexibility to work with various data sources.

What can PyESD do?
------------------

PyESD is an open-source Python package designed to perform Perfect Prognosis-based Empirical Statistical 
Downscaling. This package includes various modeling tools and processes. It can preprocess diverse datasets,
such as weather station data (from, e.g., German weather service, Ghana Met Agency), ERA5 analysis, CMIP5,
and CMIP6. PyESD constructs predictors and predictands (e.g., transformation and extracting teleconnection 
indices from climate variables), selects predictors using feature engineering techniques, chooses learning 
models or ensembles, predicts future climates, performs statistical analysis and provides visualizations. 
It's adaptable to different datasets, well-documented, and user-friendly. For a given weather station directory,
the stations can be loaded into the Station Operator (SO) object, then apply all the ESD routines: An example of
setting a model: 

>>> from pyESD.Weatherstation import read_station_csv
>>> from pyESD.standardizer import MonthlyStandardizer, StandardScaling
>>> SO = read_station_csv(filename=station_dir, varname=variable)
>>> SO.set_model(variable, method=regressor, scoring=scoring,
                     cv=TimeSeriesSplit(n_splits=10))

*Details about the modeling framework*

.. image:: ./imgs/outline1.png
   :width: 600
   :alt: Model Outline


What pyESD can't do?
---------------------

The current version of pyESD doesn't include Model Output Statistics models, can't work with spatial
predictand datasets like gridded weather stations, lacks spatial learning capabilities, and can't 
directly assess impacts. It heavily relies on machine learning algorithms and simple deep learning 
architectures. While pyESD can be used with daily datasets, this capability is still experimental. 
For instance, it's possible to create a classifier within the model to predict wet or dry conditions 
before the model learning to improve performance on rainfall occurrence and extreme events. 
The pre-processing in the package can handle various weather station data formats and can be easily adapted.
The developers of pyESD welcome suggestions for improvement of the software and the documentation as well.

Who developed pyESD?
--------------------

PyESD was developed by Daniel Boateng, a Ph.D. student at the University of Tübingen. Daniel developed the 
package alongside his Ph.D. project involving paleoclimate modeling using isotope-enabled GCMs and climate
dynamics. He's dedicated to open-source scientific software development and aims to enhance the 
reproducibility of research outcomes. He believes that science would be more fun if all research outputs 
were easily reproducible. He also created "pyClimat," another open-source package for analyzing and 
visualizing GCM model output, which has been open-source since the first day of his Ph.D. program.
Daniel believes in “Trusting The Process” (TTP) in all aspects of life.

.. image:: ./imgs/myself.jpg
   :width: 600
   :alt: Picture

