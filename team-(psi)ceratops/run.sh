#!/bin/bash
#SBATCH --job-name=yquantum-catqubit
#SBATCH --output=/home/valentm/hackathons/YQuantum-2026/slurm_out/%x-%j.out
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --gpus=v100:1

# YQuantum 2026 — Cat Qubit Optimization on Palmetto
#
# Usage:
#   sbatch run.sh              # submit to SLURM (Palmetto)
#   bash run.sh                # run locally (skips SLURM headers)
#
# Benchmark sweep:
#   3 optimizers (CMA-ES, REINFORCE, PPO-Clip)
#   x 3 rewards  (proxy, vacuum, spectral)
#   x 3 drifts   (Kerr, frequency, SNR degradation)
#   = 27 combinations @ MEDIUM Hilbert space (dim=75, 60 epochs, pop=8)

set -euo pipefail

PROJECT_DIR="/home/valentm/hackathons/YQuantum-2026"
cd "$PROJECT_DIR"

# Ensure slurm_out exists BEFORE submitting (also safe if re-run)
mkdir -p slurm_out results figures

# --- Load modules (must happen before any python/uv calls) ---
source /etc/profile.d/modules.sh 2>/dev/null || true
module load anaconda3/2023.09-0

echo "============================================="
echo "  YQuantum 2026 — Cat Qubit Optimization"
echo "============================================="
echo "  Host:      $(hostname)"
echo "  Started:   $(date)"
echo "  Directory: $PROJECT_DIR"

# --- GPU info (if available) ---
if command -v nvidia-smi &>/dev/null; then
    echo "  GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
fi

# --- SLURM info (if running under SLURM) ---
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "  SLURM Job: $SLURM_JOB_NAME ($SLURM_JOB_ID)"
    echo "  Node:      $SLURM_NODELIST"
fi
echo "============================================="
echo

# --- Environment setup ---
# Prefer uv for fast installs; fall back to pip
if command -v uv &>/dev/null; then
    echo "[setup] Using uv for environment management"

    # Create venv if it doesn't exist
    if [ ! -d ".venv" ]; then
        echo "[setup] Creating virtual environment..."
        uv venv --python 3.11
    fi

    # Activate
    source .venv/bin/activate

    # Install/sync requirements (idempotent, fast with uv)
    echo "[setup] Installing requirements..."
    uv pip install -r requirements.txt
else
    echo "[setup] uv not found, falling back to pip"

    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
    fi

    source .venv/bin/activate
    pip install -r requirements.txt
fi

echo "[setup] Python: $(python --version)"
echo "[setup] Venv:   $VIRTUAL_ENV"
echo

# --- Verify critical imports ---
python -c "
import dynamiqs, jax, cmaes, optax
print(f'[verify] dynamiqs={dynamiqs.__version__}, jax={jax.__version__}, cmaes={cmaes.__version__}')
print(f'[verify] JAX devices: {jax.devices()}')
"
echo

# --- Run benchmark ---
# MEDIUM profile: na=15, nb=5 (dim=75), 60 epochs, pop=8
# Hardcoded sweep: cmaes,reinforce,ppo x proxy,vacuum,spectral x kerr,frequency,snr
# --no-interactive: headless mode for SLURM (save plots to disk only)
echo "[run] Starting benchmark: 3 optimizers x 3 rewards x 3 drifts = 27 combos..."
python -u run.py \
    --profile medium \
    --no-interactive

echo
echo "============================================="
echo "  Finished: $(date)"
echo "  Results:  $PROJECT_DIR/results/"
echo "  Figures:  $PROJECT_DIR/figures/"
echo "============================================="
