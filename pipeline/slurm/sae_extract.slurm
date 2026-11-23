#!/bin/bash
# GemmaScope activations for the SAE analysis (Figure 18). Submit from the repository root:
#   sbatch pipeline/slurm/sae_extract.slurm
#SBATCH --job-name=sae_extract
# (set partition / qos for your cluster)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=data/logs/%x_%j.out
#SBATCH --error=data/logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p data/logs
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
PY="${PY:-python}"
nvidia-smi -L
$PY pipeline/sae/extract_sae_acts.py "$@"
