#!/bin/bash
# Submit with something like:
#   sbatch --account=def-yourname scripts/slurm_studentA_mountaincar_8000_gpu.sh
#
# Optional:
#   export ENV_ACTIVATE=/home/yourname/venvs/rl/bin/activate
#   sbatch --account=def-yourname scripts/slurm_studentA_mountaincar_8000_gpu.sh

#SBATCH --job-name=sa-mc8000-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm-sa-mc8000-gpu-%j.out

set -euo pipefail

module purge
module load StdEnv/2023 python/3.11 cuda/12.2

if [[ -n "${ENV_ACTIVATE:-}" ]]; then
    # shellcheck disable=SC1090
    source "${ENV_ACTIVATE}"
fi

cd "${SLURM_SUBMIT_DIR:-$PWD}"
bash scripts/run_studentA_mountaincar_8000_gpu.sh "$@"
