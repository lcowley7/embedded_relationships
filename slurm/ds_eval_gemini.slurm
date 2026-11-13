#!/bin/bash
#SBATCH --job-name=ds_eval
# (set partition for your cluster; needs outbound internet for the Gemini API)
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=data/logs/%x_l%j.out
#SBATCH --error=data/logs/%x_l%j.err
# Cheap Gemini judge (gemini-3.1-flash-lite), API-only, no GPU. Scores every factor
# incl. 0.0 -> steering-success-vs-alpha=0. One job does all 26 vectors. Env: DATASET, CONFIG_PREFIX, LAYER.
# NOTE: needs outbound internet to the Gemini API.
set -euo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then PROJECT="$SLURM_SUBMIT_DIR"; else PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
cd "$PROJECT"
mkdir -p data/logs
: "${DATASET:?DATASET env var required (e.g. frequency_v2)}"
: "${CONFIG_PREFIX:?CONFIG_PREFIX env var required (e.g. diffmean_frequency_v2_g2-2b)}"
: "${LAYER:?LAYER env var required}"
CONFIG="ax_bench/axbench/axbench/sweep/frh/${CONFIG_PREFIX}_l${LAYER}.yaml"
DS_DIR="data/axbench/${DATASET}"

PY=python
echo "[ds_eval] dataset=$DATASET layer=$LAYER (gemini judge)"
$PY olmo/gemini_judge.py \
    --steering-data "$DS_DIR/diffmean_l${LAYER}/inference/steering_data.parquet" \
    --concept-index "$DS_DIR/concept_index.parquet" \
    --out-dir "$DS_DIR/diffmean_l${LAYER}/eval" \
    --model gemini-3.1-flash-lite --concurrency 48
