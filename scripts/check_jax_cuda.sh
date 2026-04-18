#!/bin/bash
# Diagnostic script to check JAX CUDA setup and debug issues

set -e

echo "============================================"
echo "  JAX CUDA Diagnostic Check"
echo "============================================"
echo

# Clear any conflicting environment variables
unset HSA_OVERRIDE_GFX_VERSION
unset ROCR_VISIBLE_DEVICES
unset ROC_VISIBLE_DEVICES
export JAX_PLATFORMS=cuda

echo "1. Checking NVIDIA drivers and nvidia-smi:"
nvidia-smi --query-gpu=index,name,compute_cap --format=csv
echo

echo "2. Checking Python and JAX versions:"
python3 --version
python3 -c "import jax; print('JAX version:', jax.__version__)"
python3 -c "import jaxlib; print('JAXLib version:', jaxlib.__version__)"
echo

echo "3. Checking available JAX backends:"
python3 -c "
import jax
# Try to get backend info
try:
    cuda_device = jax.devices('cuda')
    print('CUDA devices found:', cuda_device)
except:
    print('No CUDA device - trying fallback')
    print('All devices:', jax.devices())
"
echo

echo "4. Attempting CUDA initialization with explicit device:"
python3 -c "
import os
os.environ['JAX_PLATFORMS'] = 'cuda'
import jax
print('JAX devices:', jax.devices())
print('Default device:', jax.default_device())
print('CUDA backend initialized successfully!')
"
echo

echo "5. Testing array creation on GPU:"
python3 -c "
import os
os.environ['JAX_PLATFORMS'] = 'cuda'
import jax
import jax.numpy as jnp
x = jnp.ones((1000, 1000))
y = jax.nn.softmax(x)
print('Array operations work on GPU:', y.shape)
print('Device location:', y.device())
"
echo

echo "============================================"
echo "✓ JAX CUDA setup appears to be working"
echo "============================================"
