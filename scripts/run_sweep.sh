#!/usr/bin/env bash

set -euo pipefail

LIBS="${1:-false}"

VENV_DIR="dl_env"
REQ_FILE="requirements.txt"
CONFIG_FILE="config.yaml"


# ═══════════════════════════════════════════════════════════════
#  STEP 1: Virtual Environment
# ═══════════════════════════════════════════════════════════════

echo "###################################################################"
echo "==> Initialising Environment"

if [ ! -d "$VENV_DIR" ]; then 
    echo "==> Creating Virtual Environment : $VENV_DIR"
    sudo apt update
    sudo apt install -y software-properties-common
    sudo add-apt-repository -y universe
    sudo apt update
    sudo apt install -y python3-venv    
    python3 -m venv "$VENV_DIR"
else
    echo "==> Virtual Env already exists: $VENV_DIR"
fi 
echo "###################################################################"

if [ -f "$VENV_DIR/Scripts/activate" ]; then 
    source "$VENV_DIR/Scripts/activate"
elif [ -f "$VENV_DIR/bin/activate" ]; then 
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: Could not find venv activate script in $VENV_DIR"
    exit 1
fi 


echo "###################################################################"

echo "==> Checking for requirements.txt file "

if [ "$LIBS" = true ]; then
    if [ -f "$REQ_FILE" ]; then 
        echo "###################################################################"
        echo "==> uprading pip"
        python -m pip install --upgrade pip 
        sudo apt update
        sudo apt install -y software-properties-common
        sudo add-apt-repository -y universe
        sudo apt update
        sudo apt install -y python3-venv

        echo "==> installing dependencies from "
        
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
        pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128

        pip install -r "$REQ_FILE"
        echo "###################################################################"
    else 
        echo "Warning: $REQ_FILE not found."
    fi 
else
    echo "Skipping Dependency Installation (libs=false)"
fi 

echo "###################################################################"
echo "-> Testing the Target device of JAX"
python3 -c "import jax; print(jax.devices()); print(jax.default_backend())"
echo "###################################################################"


# ═══════════════════════════════════════════════════════════════
#  STEP 3: Device Check
# ═══════════════════════════════════════════════════════════════

echo "==> Step 3: JAX Device Check"
python3 -c "
import jax
print(f'Backend : {jax.default_backend()}')
print(f'Devices : {jax.devices()}')
"
echo "###################################################################"
