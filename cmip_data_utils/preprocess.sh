#!/bin/bash
#
# *********** PRE-PROCESS SCRIPT FOR CMIP MODEL OUTPUT MONTHLY *********
# 1. run the preprocess.sh to mergetime the downloaded files
# 2. run the rename_variables.sh to change the varnames to be consistent with ERA5 variables
# 3. run the sellevels.sh to select some vertical levels data used as predictors
# 4. run the preprocess_cmip5.py in the test of pyESD to extract the data_array from the datasets
# 5. move the dewpoint script to the monthly folder and run it to calculate the dewpoint depression variables for all levels
# --------------PATHS------------------------------
mPATH="/mnt/d/Datasets/CMIP6/CMIP"
experiment="AMIP"
Freq="Amon"
model_name="MPI-ESMI-2-LR"
other_paths="gn/v20190815"
variables=("pr" "tas" "vas" "uas" "psl" "va" "ua" "ta" "zg" "hur")
#
#************* Mergetime, renaming, interpolation?, selecting areas?************
for var in ${variables[@]}
do 

    path_to_data=${mPATH}/${experiment}/${model_name}/${Freq}/${var}/${other_paths}
    path_output=${mPATH}/${experiment}/${model_name}/postprocessed
    input=${path_to_data}/${var}_*.nc

    cdo mergetime ${input} ${path_output}/${var}_monthly.nc
done 