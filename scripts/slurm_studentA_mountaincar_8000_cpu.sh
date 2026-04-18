#!/bin/bash
# Submit with something like:
#   sbatch --account=def-yourname scripts/slurm_studentA_mountaincar_8000_cpu.sh
#
# Optional:
#   export ENV_ACTIVATE=/home/yourname/venvs/rl/bin/activate
#   sbatch --account=def-yourname scripts/slurm_studentA_mountaincar_8000_cpu.sh

#SBATCH --job-name=sa-mc8000-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-sa-mc8000-cpu-%j.out

set -euo pipefail

module purge
module load StdEnv/2023 python/3.11

cd /lustre06/project/6110103/fmoha077/project/elg5214_group_project8_project
mkdir -p logs outputs

source ~/venvs/rl_jax_cpu_env/bin/activate
echo "Running on: $(hostname)"


cd "${SLURM_SUBMIT_DIR:-$PWD}"
bash scripts/run_studentA_mountaincar_8000_cpu.sh "$@"
