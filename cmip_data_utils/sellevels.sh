#!/bin/bash
#
# *********** Select levels for the atmospheric variables *********
#
#
# --------------PATHS------------------------------
mPATH="/mnt/d/Datasets/CMIP6/CMIP"
experiment="AMIP"
postprocessed_folder="postprocessed"
model_name="MPI-ESMI-2-LR"
variables=("v" "u" "t" "z" "r")
#
#************* Mergetime, renaming, interpolation?, selecting areas?************
#cdo sellevel,25000 -chname,v,v250 -sellevel,25000 v_monthly.nc v250_monthly.nc
for i in {0..4}; do 
    input=${mPATH}/${experiment}/${model_name}/${postprocessed_folder}/${variables[$i]}_monthly.nc
    output=${mPATH}/${experiment}/${model_name}/${postprocessed_folder}

    cdo sellevel,25000 -chname,${variables[$i]},${variables[$i]}250 ${input} ${output}/${variables[$i]}250_monthly.nc
    cdo sellevel,50000 -chname,${variables[$i]},${variables[$i]}500 ${input} ${output}/${variables[$i]}500_monthly.nc
    cdo sellevel,70000 -chname,${variables[$i]},${variables[$i]}700 ${input} ${output}/${variables[$i]}700_monthly.nc
    cdo sellevel,85000 -chname,${variables[$i]},${variables[$i]}850 ${input} ${output}/${variables[$i]}850_monthly.nc
    cdo sellevel,100000 -chname,${variables[$i]},${variables[$i]}1000 ${input} ${output}/${variables[$i]}1000_monthly.nc
done