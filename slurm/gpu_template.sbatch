#!/bin/bash
# Template for GPU work (pretraining). Delta GPU partitions:
#   gpuA40x4 (4x A40)   gpuA100x4 (4x A100 40G)   gpuA100x8   gpuH200x8
# Charge GPU jobs to bebr-delta-gpu, CPU-only jobs to bebr-delta-cpu.
#SBATCH --job-name=imu-pretrain
#SBATCH --account=bebr-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/work/hdd/bebr/Projects/IMU_pretraining/slurm/logs/%x-%j.out
#SBATCH --error=/work/hdd/bebr/Projects/IMU_pretraining/slurm/logs/%x-%j.err

set -euo pipefail
PROJ=/work/hdd/bebr/Projects/IMU_pretraining
mkdir -p "$PROJ/slurm/logs"
source "$PROJ/env.sh"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
nvidia-smi
srun python "$PROJ/scripts/train.py" "$@"
