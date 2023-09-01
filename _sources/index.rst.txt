.. pyESD documentation master file, created by
   sphinx-quickstart on Thu Mar 16 16:31:41 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Akwaaba! Welcome to PyESD Documentation!
==============================================================
**Python framwork for Empirical Statistical Downscaling**

PyESD is an open-source framework of the Perfect Prognosis approach of statistical downscaling of 
any climate-related variable such as precipitation, temperature, and wind speed using reanalysis 
products eg. ERA5 as predictors. The package features all the downscaling cycles including data 
preprocessing, predictor selection, constructions (eg. using transformers), model selection, 
training, validation and evaluation, and future prediction. The package serves as the means 
of downscaling General Circulation Models of future climate to high resolution relevant for 
climate impact assessment such as droughts, flooding, wildfire risk, and others. The main 
specialties of the pyESD include:

- Well designed in an OOP style that considers weather stations as individual objects and all the
  downscaling routines as attributes. This ensures fewer lines of code that cover the end-to-end downscaling of climate change variable products.

- PyESD features many machine learning algorithms and predictor selection techniques that can 
  be experimented with toward the selection and design of robust transfer functions that can be coupled with GCM to generate future estimates of climate change.

- And many other functionalities that are highlighted in the paper description of the package (submitted to GMD)

.. image:: ./imgs/outline.png
   :width: 600
   :alt: Model Outline


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

Getting in touch with us
-------------------------

If you're interested in using pyESD for your weather station analysis, we've made sure that the modelling steps 
are user-friendly and applicable to weather stations worldwide. Whether you're new to pyESD and need help getting 
started, want to enhance or add new components, have found a bug, or simply want to discuss potential collaborations, 
you have several ways to reach out to us:

1. **Start a Discussion**: Have general questions about the scientific methods behind our tools? 
Need assistance with setting up experiments using pyESD? Looking for more information about features 
that might not be fully documented? You can initiate a _discussion on our GitHub page: `Start a Discussion
<https://github.com/Dan-Boat/PyESD/discussions>`_.

2. **Report an Issue**: If you encounter bugs in the source code, feel that certain features are missing, 
or have suggestions for techniques to improve, you can open an issue on our 
GitHub repository: `[Open an Issue]<https://github.com/Dan-Boat/PyESD/issues>. 
We're also interested in contributions to enhance our documentation.

3. **Email Us**: Feel free to reach out to us via email at dannboateng@gmail.com.

We're always thrilled to hear about the ways in which pyESD is being utilized. 
If you've incorporated pyESD into your research, activities, or teaching, please consider 
submitting a pull request to let us know. This helps us keep track of the various applications of pyESD.

Thank you for considering pyESD for your endeavors!

Some papers and preprints that have used pyESD
--------------------------------------------------

**Kindly make a pull request to let us know if you’ve used pyESD in your research, activities or teaching:**

1. Boateng, D. and Mutz, S. G.: pyESDv1.0.1: An open-source Python framework for empirical-statistical 
downscaling of climate information, Geoscientific Model Development Discussions, 
1–58, https://doi.org/10.5194/gmd-2023-67, 2023.

2. Arthur, F., Boateng, D., and Baidu, M.: Prediction of Rainfall Response to the 21st-century Climate Change
in Ghana using Machine Learning Empirical Statistical Downscaling, 2022, H25A-04, 2022.

**Citing pyESD**

```BibTeX

@article{boateng_pyesdv101_2023,
title = {{pyESDv1}.0.1: {An} open-source {Python} framework for empirical-statistical downscaling of climate information},
shorttitle = {{pyESDv1}.0.1},
url = {https://gmd.copernicus.org/preprints/gmd-2023-67/},
doi = {10.5194/gmd-2023-67},
language = {English},
urldate = {2023-07-24},
journal = {Geoscientific Model Development Discussions},
author = {Boateng, Daniel and Mutz, Sebastian G.},
month = apr,
year = {2023},
note = {Publisher: Copernicus GmbH},
pages = {1--58},
file = {Full Text PDF:/Users/danielboateng/Zotero/storage/89JHU897/Boateng and Mutz - 2023 - pyESDv1.0.1 An open-source Python framework for e.pdf:application/pdf},
}
```


Documentation
--------------

* :doc:`overview`
* :doc:`installation`
* :doc:`methods`
* :doc:`tutorials`
* :doc: `gallery`


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started:

   overview
   installation
   methods
   tutorials
   examples
   testing
   modules
   gallery


License
=======
pyESD is published under the MIT License (Copyright (c) 2023, Daniel Boateng)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`