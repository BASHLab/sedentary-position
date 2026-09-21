#!/bin/bash
# Source this to get the project Python environment.
#   source /work/hdd/bebr/Projects/IMU_pretraining/env.sh
#
# The venv is built on the system python module, so that module MUST be loaded
# before activating -- the venv's interpreter is a symlink into it.

module load python/3.13.5-gcc13.3.1
source /work/hdd/bebr/Projects/IMU_pretraining/.venv/bin/activate

# Keep pip/HF/torch caches off $HOME (small quota) and on project scratch.
export PIP_CACHE_DIR=/work/hdd/bebr/Projects/IMU_pretraining/.cache/pip
export HF_HOME=/work/hdd/bebr/Projects/IMU_pretraining/.cache/huggingface
export TORCH_HOME=/work/hdd/bebr/Projects/IMU_pretraining/.cache/torch
export MPLCONFIGDIR=/work/hdd/bebr/Projects/IMU_pretraining/.cache/matplotlib
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$MPLCONFIGDIR"
