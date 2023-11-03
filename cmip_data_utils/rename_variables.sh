#!/bin/bash
#
# *********** RE-NAME for pyESD namelist format *********
#
#
# --------------PATHS------------------------------
mPATH="/mnt/d/Datasets/CMIP6/CMIP"
experiment="AMIP"
postprocessed_folder="postprocessed"
model_name="MPI-ESMI-2-LR"
variables=("pr" "tas" "vas" "uas" "psl" "va" "ua" "ta" "zg" "hur")
filenames=("tp" "t2m" "v10" "u10" "msl" "v" "u" "t" "z" "r")
#
#************* Mergetime, renaming, interpolation?, selecting areas?************
for i in {0..9}; do 
    input=${mPATH}/${experiment}/${model_name}/${postprocessed_folder}/${variables[$i]}_monthly.nc
    output=${mPATH}/${experiment}/${model_name}/${postprocessed_folder}/${filenames[$i]}_monthly.nc

    cdo chname,${variables[$i]},${filenames[$i]} ${input} ${output}
    /bin/rm ${input}
done