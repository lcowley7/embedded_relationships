#!/bin/bash
# Steering runs for the two 32B models (OLMo-3.1-32B, Qwen3-32B), as used for the paper's
# cross-model steering comparison (analysis/models/cross_model_steering.py).
#
# The 32B models were steered on a SUBSET of Gemma's grid: 9 of its 15 factors and its first 6 of 12
# held-out prompts, with the same 512 generated tokens. Gemma's full 15 x 12 grid would have taken
# ~63 h per model per layer (~0.65 generations/s per 32B model at batch size 48), so
# cross_model_steering.py re-scores Gemma on exactly this subset for a like-for-like comparison.
# Running steer.py with its defaults (15 factors x 12 prompts) does NOT reproduce these runs.
#
# Order: the paper's main layer (L25, depth-matched to Gemma L10) for both models, then the other
# depth-matched layers. steer.py checkpoints and resumes, so re-running continues an interrupted run.
#
# Usage (from the repository root, on a GPU machine with the model weights available):
#   bash pipeline/bigmodel/run_steer.sh
# Reads data/axbench/<ds>/ and the vectors in data/models/<tag>/axbench/ (pipeline/bigmodel/extract.py);
# writes data/models/<tag>/axbench/<ds>/diffmean_l<L>/inference/. Set ROOT to use a scratch disk instead.
set -uo pipefail
PY="${PY:-python}"
ROOTARG=""; [ -n "${ROOT:-}" ] && ROOTARG="--root $ROOT"
export PYTHONUNBUFFERED=1
GRID="--factors 0.0 0.4 1.0 1.6 2.0 2.5 3.0 4.0 5.0 --max-prompts 6"

for spec in "olmo31_32b 25" "qwen3_32b 25" "olmo31_32b 12" "qwen3_32b 12" \
            "olmo31_32b 37" "qwen3_32b 37" "olmo31_32b 49" "qwen3_32b 49"; do
    set -- $spec
    echo "=== $(date -u +%FT%TZ) steer $1 layer $2"
    $PY pipeline/bigmodel/steer.py --tag "$1" $ROOTARG --layer "$2" --datasets all $GRID \
        || echo "!!! steer $1 L$2 exited non-zero"
done
echo "=== $(date -u +%FT%TZ) STEER DONE"
