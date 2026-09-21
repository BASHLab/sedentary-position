#!/bin/bash
#SBATCH --job-name=posture-baseline
#SBATCH --account=bebr-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/work/hdd/bebr/Projects/IMU_pretraining/slurm/logs/%x-%j.out
#SBATCH --error=/work/hdd/bebr/Projects/IMU_pretraining/slurm/logs/%x-%j.err

set -euo pipefail
PROJ=/work/hdd/bebr/Projects/IMU_pretraining
mkdir -p "$PROJ/slurm/logs"
source "$PROJ/env.sh"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
srun python "$PROJ/scripts/posture_baseline.py"
