#!/bin/bash
# One-time environment for PIRL on Plaksha HPC (run on the login node; needs no GPU):
#   bash hpc/setup_env.sh > logs/setup_env.log 2>&1 &
# Reuses the user's Miniforge on /scratch and creates a separate env "pirl".
# Safe to re-run: finished steps are skipped.
set -eo pipefail

SCRATCH_DIR="/scratch/$USER"
CONDA_DIR="$SCRATCH_DIR/miniforge3"
ENV_NAME="pirl"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"   # covers Ada (8.9), Hopper (9.0), Blackwell (12.0)

source "$CONDA_DIR/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS="/tmp/${USER}_conda_pkgs"   # extract on local disk (Lustre Errno 14)
mkdir -p "$CONDA_PKGS_DIRS"
LOCAL_STAGE="/tmp/${USER}_stage_$$"
trap 'rm -rf "$CONDA_PKGS_DIRS" "$LOCAL_STAGE"' EXIT

# pip into a local staging prefix, then cp into the env: big pip writes to Lustre can fail with Errno 14.
pip_local() {
    local site="lib/python$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')/site-packages"
    rm -rf "$LOCAL_STAGE"
    pip install --no-cache-dir --prefix "$LOCAL_STAGE" "$@"
    [ -d "$LOCAL_STAGE/$site" ] && cp -a "$LOCAL_STAGE/$site/." "$CONDA_PREFIX/$site/"
    [ -d "$LOCAL_STAGE/bin" ] && cp -a "$LOCAL_STAGE/bin/." "$CONDA_PREFIX/bin/"
    rm -rf "$LOCAL_STAGE"
}

if ! conda env list | grep -qE "^$ENV_NAME\s"; then
    echo "== creating env $ENV_NAME"
    conda create -y -n "$ENV_NAME" -c conda-forge python=3.11 pip
fi
conda activate "$ENV_NAME"

if ! python -c "import torch" 2>/dev/null; then
    echo "== installing torch ($TORCH_INDEX)"
    pip_local torch --index-url "$TORCH_INDEX"
fi
if ! python -c "import stable_baselines3, gymnasium, scipy, matplotlib" 2>/dev/null; then
    echo "== installing numpy scipy matplotlib pillow gymnasium stable-baselines3"
    pip_local numpy scipy matplotlib pillow gymnasium stable-baselines3
fi
python -c "import torch, stable_baselines3, gymnasium; print('OK torch', torch.__version__, 'cuda', torch.version.cuda, '| sb3', stable_baselines3.__version__)"
