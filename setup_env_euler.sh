#!/bin/bash
## based on iac wiki (firat)
## usage: source setup_env_euler.sh

## for pycles
module load openmpi/4.1.4
module load gcc/8.2.0
module load python/3.10.4
module load netcdf/4.9.0
module load hdf5/1.10.9

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/extras/CUPTI/lib64/ 
export DIR_HOME=/cluster/work/climate/dgrund
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${DIR_HOME}/libs

##if this is first time, create a virtual env and install necessary libs. 

VENV_DIR=/cluster/work/climate/dgrund/git/dana-grund/doctorate_code/pycles/.venv/pycles3.10
VENV_ACT=${VENV_DIR}/bin/activate

if test -f "$VENV_ACT"; then
   echo "found the virtual env, activating."
   source $VENV_ACT 
else
   echo "virtual env missing. Creating one at $VENV_DIR." python -m venv --system-site-packages $VENV_DIR
   source $VENV_ACT
   pip install --upgrade pip
   pip install -r requirements_euler.txt 
fi