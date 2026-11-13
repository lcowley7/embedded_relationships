#!/bin/bash
#SBATCH --job-name=ds_infer
# (set partition / qos for your cluster)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=data/logs/%x_l%j.out
#SBATCH --error=data/logs/%x_l%j.err
# Generic single-layer steering inference. Env: DATASET, CONFIG_PREFIX, LAYER.
# 26 vectors x 12 heldout x 15 factors ~ 1-2h; 6h is a safe ceiling that backfills well.
set -euo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then PROJECT="$SLURM_SUBMIT_DIR"; else PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
cd "$PROJECT"
mkdir -p data/logs
: "${DATASET:?DATASET env var required (e.g. frequency_v2)}"
: "${CONFIG_PREFIX:?CONFIG_PREFIX env var required (e.g. diffmean_frequency_v2_g2-2b)}"
: "${LAYER:?LAYER env var required}"
CONFIG="ax_bench/axbench/axbench/sweep/frh/${CONFIG_PREFIX}_l${LAYER}.yaml"
DS_DIR="data/axbench/${DATASET}"

source ax_bench/axbench/.venv/bin/activate
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
echo "[ds_infer] dataset=$DATASET layer=$LAYER config=$CONFIG"
torchrun --standalone --nproc_per_node=1 ax_bench/axbench/axbench/scripts/inference.py --config "$CONFIG" --mode steering
