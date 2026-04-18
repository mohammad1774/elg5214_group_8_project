#!/bin/bash
#SBATCH --job-name=sa-mc8000-gpu-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-sa-mc8000-gpu-%j.out
#SBATCH --account=def-n37chen
#SBATCH --gpus=a100_4g.20gb:1

set -euo pipefail

module purge
module load python/3.11

cd /lustre06/project/6110103/fmoha077/project/elg5214_group_project8_project 

mkdir -p logs outputs

source ~/venvs/rl_jax_gpu_env/bin/activate

echo "Running on: $(hostname)"
nvidia-smi

# Clear any ROCm-related environment variables that might interfere
unset HSA_OVERRIDE_GFX_VERSION
unset ROCR_VISIBLE_DEVICES
unset ROC_VISIBLE_DEVICES

# Force CUDA backend
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "Checking JAX CUDA support..."
python -c "
import jax
print('JAX devices:', jax.devices())
print('CUDA backend initialized successfully!')
"

bash scripts/run_studentA_mountaincar_8000_gpu.sh "$@"