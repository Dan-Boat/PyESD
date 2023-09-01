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

Documentation
--------------
**Getting Started**

* :doc:`overview`
* :doc:`installation`
* :doc:`methods`

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


License
=======
pyESD is published under the MIT License (Copyright (c) 2023, Daniel Boateng)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
.. note::
   The content of this website would be improve in future developments! However, we would acknowledge your help in making 
   the pyESD documentation better for wider usability. Please kindly raise a pull request with the improved version on github:
   ``https://github.com/Dan-Boat/PyESD``


.. warning::
    Please note that this page is under active delopement and would benefit from its extension! Thanks for the understanding