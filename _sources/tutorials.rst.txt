Using pyESD for Downscaling Climate Information
===============================================

In this section, we provide comprehensive examples of utilizing the pyESD package for 
downscaling climate information. We focus on analyzing precipitation records over 
southern Germany by utilizing data from 5 stations, randomly selected from a pool of over 
1500 stations available at the DWD Climate Data Center. We offer both the post-processed 
data for these stations and the corresponding scripts required to achieve the results showcased 
here. These examples are designed in the form of Jupyter Notebooks, complete with step-by-step 
instructions and explanations to assist users in adapting to pyESD's modeling routines.

Preprocessing Data for pyESD
-----------------------------

To preprocess the data downloaded from the DWD CDC for use with pyESD, we use the following code:

.. code-block:: python

   import sys
   import os
   from pyESD.data_preprocess_utils import extract_DWDdata_with_more_yrs, add_info_to_data

   # Set the paths to the raw datasets
   main_path = "C:/Users/dboateng/Desktop/Datasets/Station/southern_germany"
   data_files_path = os.path.join(main_path, "data")
   path_data_considered = os.path.join(main_path, "considered")
   path_data_processed = os.path.join(main_path, "processed")
   path_data_info = "C:/Users/dboateng/Desktop/Datasets/Station/southern_germany/data/sdo_OBS_DEU_P1M_RR.csv"

   if not os.path.exists(path_data_considered):
       os.makedirs(path_data_considered)   
       
   if not os.path.exists(path_data_processed):
       os.makedirs(path_data_processed)

   # Extract datasets meeting the 60-year requirement
   extract_DWDdata_with_more_yrs(path_to_data=data_files_path, path_to_store=path_data_considered,
                                 min_yrs=60, glob_name="data*.csv", varname="Precipitation",
                                 start_date="1958-01-01", end_date="2022-12-01", data_freq="MS")
    
   # Format processed data
   add_info_to_data(path_to_info=path_data_info, path_to_data=path_data_considered,


How does the used datasets look like?
-------------------------------------
`Its simple, check it out <https://nbviewer.org/github/Dan-Boat/PyESD/blob/main/examples/tutorials/data_structure.ipynb>`_

Selecting Predictors for ESD
-----------------------------

Choosing predictor variables for downscaling can be complex due to the multitude of climate 
variables that may be relevant. Predictors are crucial in determining the performance 
of the ESD model and must be selected judiciously for meaningful interpretation. 
The methods for predictor selection are not always straightforward and can be 
technique-dependent. In pyESD, we demonstrate how to utilize available methods to 
choose informative predictors for each station. Additionally, we compare the performance 
of different selection methods against a baseline training model (RidgeCV) to 
identify sets of predictors with the highest predictive skill based on the chosen method.

Including Large-Scale Teleconnection Indices
--------------------------------------------

One notable advantage of pyESD is its ability to incorporate atmospheric 
circulation indices, which play a role in explaining climate variability. 
We showcase how to include teleconnection indices as predictors to enhance the 
downscaling model's capabilities.

For further details and practical implementation, refer to the complete documentation 
and examples provided by pyESD.

How to preprocess data used by pyESD
-------------------------------------


How to select predictors for ESD
---------------------------------


Including large-scale teleconnection indices predictors
--------------------------------------------------------


How to select the optimal model for ESD
----------------------------------------


Coupling ESD to GCMs for future prediction
--------------------------------------------


Visualization and Data Analysis 
--------------------------------
