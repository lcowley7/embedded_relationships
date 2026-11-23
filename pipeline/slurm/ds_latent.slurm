#!/bin/bash
#SBATCH --job-name=ds_latent
# (set partition / qos for your cluster)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=data/logs/%x_l%j.out
#SBATCH --error=data/logs/%x_l%j.err
# Generic per-layer max_act calibration. Env: DATASET, CONFIG_PREFIX, LAYER.
set -euo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then PROJECT="$SLURM_SUBMIT_DIR"; else PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
cd "$PROJECT"
mkdir -p data/logs
: "${DATASET:?DATASET env var required (e.g. h2o_v1)}"
: "${CONFIG_PREFIX:?CONFIG_PREFIX env var required (e.g. diffmean_h2o_v1_g2-2b)}"
: "${LAYER:?LAYER env var required}"
CONFIG="ax_bench/axbench/axbench/sweep/frh/${CONFIG_PREFIX}_l${LAYER}.yaml"
DS_DIR="data/axbench/${DATASET}"

source ax_bench/axbench/.venv/bin/activate
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
echo "[ds_latent] dataset=$DATASET layer=$LAYER"
python pipeline/axbench/compute_max_act.py --layer "$LAYER" --data-dir "$DS_DIR"
