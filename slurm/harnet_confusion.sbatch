#!/bin/bash
#SBATCH --job-name=harnet-confusion
#SBATCH --account=bebr-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/work/hdd/bebr/Projects/IMU_pretraining/slurm/logs/%x-%j.out
#SBATCH --error=/work/hdd/bebr/Projects/IMU_pretraining/slurm/logs/%x-%j.err

set -euo pipefail
PROJ=/work/hdd/bebr/Projects/IMU_pretraining
mkdir -p "$PROJ/slurm/logs"
source "$PROJ/env.sh"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
srun python "$PROJ/scripts/harnet_confusion.py" "$@"
