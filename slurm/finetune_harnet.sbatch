#!/bin/bash
#SBATCH --job-name=harnet-ft
#SBATCH --account=bebr-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/work/hdd/bebr/Projects/IMU_pretraining/slurm/logs/%x-%j.out
#SBATCH --error=/work/hdd/bebr/Projects/IMU_pretraining/slurm/logs/%x-%j.err

set -euo pipefail
PROJ=/work/hdd/bebr/Projects/IMU_pretraining
mkdir -p "$PROJ/slurm/logs"
source "$PROJ/env.sh"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
srun python "$PROJ/scripts/finetune_harnet.py" "$@"
